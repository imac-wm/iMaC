import argparse
import os
import multiprocessing
import shutil
import subprocess
from multiprocessing import Process

import numpy as np
from decord import VideoReader
from tqdm import tqdm

from giga_datasets import load_dataset
from giga_datasets import utils as gd_utils
from third_party.video_cond.func_flow_control import CondGenerator


DEFAULT_DATASET_PATH_LIST = [
    "/shared_disk/users/zhenyu.wu/datasets/models--open-gigaai--CVPR-2026-WorldModel-Track-Dataset/task1/train",
    "/shared_disk/users/zhenyu.wu/datasets/models--open-gigaai--CVPR-2026-WorldModel-Track-Dataset/task2/train"
]


def _depth_to_uint16_mm(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth frame must be [H, W], got shape={depth.shape}")
    depth_mm = np.rint(np.clip(depth, 0.0, 65.535) * 1000.0).astype(np.uint16)
    return depth_mm


def _write_depth_video(depth_list, save_path, fps=16):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if len(depth_list) == 0:
        raise ValueError(f"No depth frames to write for: {save_path}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode 16-bit single-channel depth mp4.")

    depth_u16 = [_depth_to_uint16_mm(depth) for depth in depth_list]
    h, w = depth_u16[0].shape
    for idx, frame in enumerate(depth_u16):
        if frame.shape != (h, w):
            raise ValueError(f"Depth frame size mismatch at index {idx}: {frame.shape} vs {(h, w)}")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "gray16le",
        "-s",
        f"{w}x{h}",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx265",
        "-x265-params",
        "lossless=1",
        "-pix_fmt",
        "gray16le",
        save_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in depth_u16:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        ret_code = proc.wait()
        if ret_code != 0:
            stderr = proc.stderr.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg failed to write {save_path}, return code={ret_code}\n{stderr}")
    finally:
        if proc.poll() is None:
            proc.kill()


def _load_rgb_views(data_dict):
    # front = VideoReader(data_dict['front_video_path'])
    # left = VideoReader(data_dict['cam_left_wrist_video_path'])
    # right = VideoReader(data_dict['cam_right_wrist_video_path'])
    front = VideoReader(data_dict["cam_high_video_path"])
    left = VideoReader(data_dict['cam_left_wrist_video_path'])
    right = VideoReader(data_dict['cam_right_wrist_video_path'])
    length = len(front)
    assert len(left) == length and len(right) == length
    return front, left, right, length


def inference(dataset_path, da3_model_path, device, world_size, rank):
    dataset = load_dataset(dataset_path)
    cond_generator = CondGenerator(model_path=da3_model_path, device=device)
    all_data_list = range(len(dataset))
    data_list = gd_utils.split_data(all_data_list, world_size, rank)

    for i in tqdm(data_list):
        data_dict = dataset[i]
        front_reader, left_reader, right_reader, length = _load_rgb_views(data_dict)

        qpos = np.asarray(data_dict['qpos'])
        assert qpos.shape[0] == length, f"qpos length mismatch: {qpos.shape[0]} vs {length}"

        depth_front, depth_left, depth_right = [], [], []
        for t in range(length):
            current_obs = [
                front_reader[t].asnumpy(),
                left_reader[t].asnumpy(),
                right_reader[t].asnumpy(),
            ]
            extrinsics = cond_generator.forward_kinematics(qpos[t])
            depths, _, _ = cond_generator.forward_DA3(current_obs, extrinsics)

            for cam_name, container in [
                ('front', depth_front),
                ('left', depth_left),
                ('right', depth_right),
            ]:
                depth = depths[cam_name]
                container.append(np.asarray(depth, dtype=np.float32))

        # _write_depth_video(depth_front, data_dict['front_depth_path'])
        # _write_depth_video(depth_left, data_dict['left_depth_path'])
        # _write_depth_video(depth_right, data_dict['right_depth_path'])

        _write_depth_video(depth_front, data_dict['cam_high_depth_path'])
        _write_depth_video(depth_left, data_dict['cam_left_wrist_depth_path'])
        _write_depth_video(depth_right, data_dict['cam_right_wrist_depth_path'])


def process_dataset(dataset_path, da3_model_path, gpu_ids):
    process_list = []
    world_size = len(gpu_ids)

    for i in range(world_size):
        device = f'cuda:{gpu_ids[i]}'
        rank = i
        process = Process(target=inference, args=(dataset_path, da3_model_path, device, world_size, rank))
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Depth process failed with exit code {process.exitcode}.")


def _parse_gpu_ids(gpu_ids):
    if len(gpu_ids) == 1 and "," in gpu_ids[0]:
        gpu_ids = gpu_ids[0].split(",")
    return [int(gpu_id) for gpu_id in gpu_ids]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process dataset videos with DA3 and write depth videos."
    )
    parser.add_argument(
        "--dataset_path",
        "--data_path",
        nargs="+",
        default=DEFAULT_DATASET_PATH_LIST,
        help="Dataset path(s) to process. You can pass one or multiple paths.",
    )
    parser.add_argument(
        "--da3_model_path",
        default=os.environ.get("DA3_MODEL_PATH"),
        help="DA3 model directory. Defaults to the DA3_MODEL_PATH environment variable.",
    )
    parser.add_argument(
        "--gpu_ids",
        nargs="+",
        default=["3", "4", "5", "6"],
        help="GPU ids to use, e.g. --gpu_ids 0 1 2 3 or --gpu_ids 0,1,2,3.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    args = parse_args()

    if not args.da3_model_path:
        raise ValueError("DA3 model path is required. Pass --da3_model_path or set DA3_MODEL_PATH.")
    os.environ["DA3_MODEL_PATH"] = args.da3_model_path

    gpu_ids = _parse_gpu_ids(args.gpu_ids)

    for dataset_path in args.dataset_path:
        process_dataset(dataset_path, args.da3_model_path, gpu_ids)
