import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import paths

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from third_party.func_flow_control_worldarena import CondGenerator


def _normalize_gripper_column(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    min_val, max_val = float(np.nanmin(values)), float(np.nanmax(values))
    if min_val >= -0.05 and max_val <= 1.05:
        return np.clip(values, 0.0, 1.0)
    return np.clip((values - min_val) / (max_val - min_val + 1e-8), 0.0, 1.0)


def _read_qpos_from_hdf5(hdf5_path: Path) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as f:
        left_arm = f["/joint_action/left_arm"][:]
        right_arm = f["/joint_action/right_arm"][:]
        left_gripper = _normalize_gripper_column(f["/joint_action/left_gripper"][:])
        right_gripper = _normalize_gripper_column(f["/joint_action/right_gripper"][:])
        qpos = np.concatenate([left_arm, left_gripper[:, None], right_arm, right_gripper[:, None]], axis=-1)
    return qpos.astype(np.float32)


def _try_get_depth_hw_from_hdf5(hdf5_path: Path) -> Optional[Tuple[int, int]]:
    aliases = ("head_camera", "front_camera", "cam_high")
    with h5py.File(hdf5_path, "r") as f:
        obs = f.get("/observation")
        if obs is None:
            return None
        for alias in aliases:
            if alias not in obs:
                continue
            node = obs[alias]
            data = None
            if isinstance(node, h5py.Group):
                for key in ("depth", "image", "rgb", "image_bit", "images"):
                    if key in node:
                        data = np.asarray(node[key])
                        break
            else:
                data = np.asarray(node)
            if data is None or data.ndim < 2:
                continue
            if data.ndim == 2:
                return (int(data.shape[0]), int(data.shape[1]))
            return (int(data.shape[-3]), int(data.shape[-2]))
    return None


def _load_blosc_depth(depth_blosc_path: Path, hdf5_path: Path, fallback_hw: Tuple[int, int]) -> np.ndarray:
    try:
        import blosc
    except ImportError as exc:  # pragma: no cover
        raise ImportError("blosc is required. Please install with `pip install blosc`.") from exc

    payload = depth_blosc_path.read_bytes()
    raw = blosc.decompress(payload)
    flat = np.frombuffer(raw, dtype=np.float16)

    hw = _try_get_depth_hw_from_hdf5(hdf5_path) or fallback_hw
    h, w = hw
    frame_size = h * w
    if frame_size <= 0 or flat.size % frame_size != 0:
        raise ValueError(
            f"Cannot reshape depth from {depth_blosc_path}. "
            f"float16_count={flat.size}, inferred_hw={hw}"
        )

    num_frames = flat.size // frame_size
    depth = flat.reshape(num_frames, h, w).astype(np.float32)
    if num_frames == 1:
        return depth[0]
    return depth[0]


def _find_episode_pairs(data_root: Path) -> list[tuple[Path, Path, Path]]:
    pairs: list[tuple[Path, Path, Path]] = []
    for depth_path in sorted(data_root.glob("**/depth_ref/*.blosc")):
        root = depth_path.parent.parent
        hdf5_path = data_root / "data" / "fixed_scene_task" / f"{depth_path.stem}.hdf5"
        if hdf5_path.exists():
            pairs.append((root, depth_path, hdf5_path))
    return pairs


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    return arr


def _read_video_meta(video_path: Path) -> Tuple[float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    if fps <= 1e-6 or width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid source video metadata: path={video_path}, fps={fps}, width={width}, height={height}"
        )
    return fps, width, height


def _write_cond_video(
    cond_video: Iterable[list[np.ndarray]],
    save_path: Path,
    fps: float,
    target_hw: Tuple[int, int],
) -> None:
    frames = list(cond_video)
    if len(frames) == 0:
        return

    target_h, target_w = target_hw
    n_views = len(frames[0])
    if n_views <= 0:
        return

    first_row = [_to_uint8_rgb(x) for x in frames[0]]
    h, w = first_row[0].shape[:2]
    if h != target_h or (w * n_views) != target_w:
        if target_w % n_views != 0:
            raise ValueError(
                f"Source video width {target_w} is not divisible by view count {n_views}, cannot align output"
            )
        per_view_w = target_w // n_views
    else:
        per_view_w = w
    row_w = per_view_w * n_views

    save_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(save_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (row_w, target_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {save_path}")

    try:
        for frame_views in frames:
            resized_views = []
            for x in frame_views:
                rgb = _to_uint8_rgb(x)
                if rgb.shape[0] != target_h or rgb.shape[1] != per_view_w:
                    rgb = cv2.resize(rgb, (per_view_w, target_h), interpolation=cv2.INTER_LINEAR)
                resized_views.append(rgb)
            row = np.concatenate(resized_views, axis=1)
            writer.write(cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 3D condition videos from depth_ref + hdf5 actions")
    parser.add_argument("--data_root", required=True, type=str, help="Dataset root path")
    parser.add_argument("--robot_type", default="agilex", choices=["piper", "agilex"])
    parser.add_argument("--rendermask_dilate", type=int, default=2)
    parser.add_argument("--urdf", type=str, default=None)
    parser.add_argument("--mesh_dir", type=str, default=None)
    parser.add_argument("--use_gpu", action="store_true", help="Use torch.cdist on GPU")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fallback_h", type=int, default=240)
    parser.add_argument("--fallback_w", type=int, default=320)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    generator_kwargs = dict(
        rendermask_dilate_iterations=args.rendermask_dilate,
        robot_type=args.robot_type,
        load_da3_model=False,
    )
    if args.urdf is not None:
        generator_kwargs["urdf_path"] = args.urdf
    if args.mesh_dir is not None:
        generator_kwargs["gripper_mesh_dir"] = args.mesh_dir
    generator = CondGenerator(**generator_kwargs)

    pairs = _find_episode_pairs(data_root)
    if not pairs:
        raise ValueError(f"No episode pair found under {data_root} (expect depth_ref/*.blosc and data/fixed_scene_task/*.hdf5)")

    pbar = tqdm(pairs, desc="process_3d_condition", unit="episode")
    fallback_hw = (args.fallback_h, args.fallback_w)
    for root, depth_path, hdf5_path in pbar:
        stem = depth_path.stem
        video_path = root / "video" / f"{stem}.mp4"
        replay_save = root / "replay_3d_condition" / f"{stem}.mp4"
        scene_save = root / "scene_3d_condition" / f"{stem}.mp4"

        if (not args.overwrite) and replay_save.exists() and scene_save.exists():
            continue

        try:
            fps, src_w, src_h = _read_video_meta(video_path)
            current_depth = _load_blosc_depth(depth_path, hdf5_path, fallback_hw)
            current_future_action = _read_qpos_from_hdf5(hdf5_path)
            cond_video1, cond_video2, _ = generator.depth_point2double_cond(
                current_depth=current_depth,
                current_future_action=current_future_action,
                current_obs=None,
                use_gpu=args.use_gpu,
            )
            _write_cond_video(cond_video1, replay_save, fps=fps, target_hw=(src_h, src_w))
            _write_cond_video(cond_video2, scene_save, fps=fps, target_hw=(src_h, src_w))
        except Exception as exc:
            print(f"[WARN] skip {depth_path}: {exc}")


if __name__ == "__main__":
    main(parse_args())
