import argparse
from pathlib import Path

import paths

import cv2
import numpy as np
from tqdm import tqdm

from third_party.func_flow_control_urdf import CondGenerator


def save_blosc_file(path: Path, sem_feature: np.ndarray) -> None:
    try:
        import blosc
    except ImportError as exc:  # pragma: no cover
        raise ImportError("blosc is required. Please install with `pip install blosc`.") from exc

    sem_feature = sem_feature.astype(np.float16)
    compressed_array = blosc.compress(sem_feature.tobytes())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(compressed_array)


def _synthetic_rig_extrinsics() -> dict[str, np.ndarray]:
    front = np.eye(4, dtype=np.float32)

    left = np.eye(4, dtype=np.float32)
    left[0, 3] = -0.10

    right = np.eye(4, dtype=np.float32)
    right[0, 3] = 0.10

    return {"front": front, "left": left, "right": right}


def _build_non_degenerate_extrinsics(cond_generator: CondGenerator) -> dict[str, np.ndarray]:
    default_qpos = np.zeros(14, dtype=np.float32)
    try:
        extrinsics = cond_generator.forward_kinematics(default_qpos)
        return {k: np.asarray(v, dtype=np.float32) for k, v in extrinsics.items()}
    except Exception:
        return _synthetic_rig_extrinsics()


def _normalize_depth_frame(depth_frame: np.ndarray, depth_far_m: float, depth_percentile: float) -> np.ndarray:
    depth_arr = np.asarray(depth_frame, dtype=np.float32)
    valid = depth_arr[np.isfinite(depth_arr) & (depth_arr > 0)]

    if valid.size == 0:
        return depth_arr

    robust_far = float(np.percentile(valid, depth_percentile))
    if robust_far <= 1e-8:
        return depth_arr

    scale = depth_far_m / robust_far
    depth_norm = (depth_arr * scale).astype(np.float32)
    return depth_norm


def _resize_depth_frame(depth_frame: np.ndarray, target_hw: tuple[int, int] = (240, 320)) -> np.ndarray:
    target_h, target_w = target_hw
    if depth_frame.shape[:2] == (target_h, target_w):
        return depth_frame.astype(np.float32)
    return cv2.resize(depth_frame.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _infer_image_depth_ref_frame(
    cond_generator: CondGenerator,
    image_path: Path,
    depth_far_m: float,
    depth_percentile: float,
) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    extrinsics_input = _build_non_degenerate_extrinsics(cond_generator)
    current_obs = [rgb, rgb, rgb]
    depths, _, _ = cond_generator.forward_DA3(current_obs, extrinsics_input)

    depth_frame = np.asarray(depths["front"], dtype=np.float32)
    depth_norm = _normalize_depth_frame(
        depth_frame=depth_frame,
        depth_far_m=depth_far_m,
        depth_percentile=depth_percentile,
    )
    depth_norm = _resize_depth_frame(depth_norm, target_hw=(240, 320))
    return depth_norm[None, ...].astype(np.float32)


def _find_image_files(data_root: Path) -> list[Path]:
    return sorted(data_root.glob("first_frame/fixed_scene_task/*.png"))


def run(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root_path)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root_path not found: {data_root}")

    image_paths = _find_image_files(data_root)
    if not image_paths:
        raise ValueError(f"No png images found under: {data_root}/first_frame/fixed_scene_task")

    cond_generator_kwargs = dict(device=args.device, load_da3_model=True)
    for arg_name, kwarg_name in (
        ("da3_model_path", "model_path"),
        ("urdf_path", "urdf_path"),
        ("gripper_mesh_dir", "gripper_mesh_dir"),
    ):
        value = getattr(args, arg_name)
        if value:
            cond_generator_kwargs[kwarg_name] = value
    cond_generator = CondGenerator(**cond_generator_kwargs)

    pbar = tqdm(image_paths, desc="DA3 worldarena testdata depth", unit="image")
    for image_path in pbar:
        depth_dir = data_root / "depth_ref"
        save_path = depth_dir / f"{image_path.stem}.blosc"

        if save_path.exists() and (not args.overwrite):
            continue

        normal_depth_frames = _infer_image_depth_ref_frame(
            cond_generator=cond_generator,
            image_path=image_path,
            depth_far_m=args.depth_far_m,
            depth_percentile=args.depth_percentile,
        )
        save_blosc_file(save_path, normal_depth_frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process worldarena test images with DA3 and save normalized depth frames as blosc files.",
    )
    parser.add_argument("--data_root_path", type=str, required=True, help="worldarena dataset root")
    parser.add_argument("--device", type=str, default="cuda", help="DA3 device, e.g. cuda:0")
    parser.add_argument("--da3_model_path", type=str, default=None, help="DA3 model directory")
    parser.add_argument("--urdf_path", type=str, default=None, help="Robot URDF path")
    parser.add_argument("--gripper_mesh_dir", type=str, default=None, help="Robot mesh directory")
    parser.add_argument("--depth_far_m", type=float, default=1.02, help="target far depth after scaling")
    parser.add_argument("--depth_percentile", type=float, default=95.0, help="robust percentile for scaling")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing depth blosc files")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
