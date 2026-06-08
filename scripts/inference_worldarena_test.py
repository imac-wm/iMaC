import argparse
import importlib
import multiprocessing
import os
import time
from multiprocessing import Process
import glob

import paths

import imageio
import numpy as np
import torch
from accelerate.utils import set_seed
from decord import VideoReader
from diffusers import AutoencoderKLWan
from giga_datasets import image_utils
from giga_datasets import utils as gd_utils
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from tqdm import tqdm

from iMac.models.transformer_worldarena_3d import WanTransformer3DModelWorldArena3D
from iMac.model_config import model_config
from iMac.pipelines.pipeline_worldarena import WorldArenaPipeline


def load_video(video, sample_frames):
    sample_indexes = np.linspace(0, len(video) - 1, sample_frames, dtype=int)
    images = [video[index] for index in sample_indexes]
    return images, sample_indexes






def _load_config(config_path: str):
    module_name, attr_name = config_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _get_model_config(config_path: str):
    cfg = _load_config(config_path)
    return cfg.get("models", {})

def get_ref_image(image, dst_size):
    height, width = image.height, image.width
    dst_width, dst_height = image_utils.get_image_size((width, height), dst_size, mode="area", multiple=32)
    image = F.resize(image, (dst_height, dst_width), InterpolationMode.BILINEAR)
    return image


def inference_worldarena_test(device, args, world_size=1, rank=0):
    torch.cuda.set_device(device)
    device = "cuda"

    model_cfg = _get_model_config(args.config_path)
    use_scene_3d_condition = model_cfg.get("use_scene_3d_condition", True)
    use_replay_3d_condition = model_cfg.get("use_replay_3d_condition", True)
    scene_3d_weight = model_cfg.get("scene_3d_weight", 1.0)
    replay_3d_weight = model_cfg.get("replay_3d_weight", 1.0)

    model_id = args.model_id
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
    transformer = WanTransformer3DModelWorldArena3D.from_pretrained(args.transformer_model_path).to(torch.bfloat16)
    pipe = WorldArenaPipeline.from_pretrained(model_id, vae=vae, transformer=transformer, torch_dtype=torch.bfloat16)
    pipe.to(device)

    data_list = []
    for data_dir in args.dataset_paths:
        img_path_list = glob.glob(os.path.join(data_dir, "first_frame", "fixed_scene_task", "*.png"))
        for img_path in img_path_list:
            episode_name = os.path.basename(img_path).replace(".png", "")
            replay_condition_path = os.path.join(data_dir, "condition", "head_camera", f"{episode_name}.mp4")
            scene_3d_condition_path = os.path.join(data_dir, "scene_3d_condition", f"{episode_name}.mp4")
            replay_3d_condition_path = os.path.join(data_dir, "replay_3d_condition", f"{episode_name}.mp4")
            required_paths = [replay_condition_path]
            if use_scene_3d_condition:
                required_paths.append(scene_3d_condition_path)
            if use_replay_3d_condition:
                required_paths.append(replay_3d_condition_path)
            if not all(os.path.exists(p) for p in required_paths):
                print(f"Skip {episode_name}: missing condition files")
                continue
            data_list.append(
                {
                    "task_name": "fixed_scene_task",
                    "episode_name": episode_name,
                    "first_frame_path": img_path,
                    "condition_video_path": replay_condition_path,
                    "scene_3d_condition_path": scene_3d_condition_path,
                    "replay_3d_condition_path": replay_3d_condition_path,
                    "dataset": os.path.basename(data_dir),
                }
            )
    data_list = gd_utils.split_data(data_list, world_size, rank)

    def process_video(video_reader, dst_size, sample_indexes=None, sample_frames=None):
        if sample_indexes is None:
            assert sample_frames is not None
            images, sample_indexes = load_video(video_reader, sample_frames)
        else:
            images = [video_reader[index] for index in sample_indexes]
        height, width = images[0].height, images[0].width
        dst_width, dst_height = image_utils.get_image_size((width, height), dst_size, mode="area", multiple=32)
        input_images = [F.resize(img, (dst_height, dst_width), InterpolationMode.BILINEAR) for img in images]
        return input_images, sample_indexes

    seed_list = args.seed if isinstance(args.seed, list) else [args.seed]

    for data_dict in tqdm(data_list):
        task_name = data_dict["task_name"]
        episode_name = data_dict["episode_name"]
        save_name = f"{task_name}/{episode_name}"

        first_frame_path = data_dict.get("first_frame_path", data_dict.get("img_path"))
        if first_frame_path is None:
            raise KeyError("Missing first_frame_path/img_path in test sample for ref image.")

        replay_condition_vr = VideoReader(data_dict["condition_video_path"])
        scene_3d_condition_vr = VideoReader(data_dict["scene_3d_condition_path"]) if use_scene_3d_condition else None
        replay_3d_condition_vr = VideoReader(data_dict["replay_3d_condition_path"]) if use_replay_3d_condition else None

        replay_condition_images = [Image.fromarray(replay_condition_vr[i].asnumpy()) for i in range(len(replay_condition_vr))]
        scene_3d_condition_images = [Image.fromarray(scene_3d_condition_vr[i].asnumpy()) for i in range(len(scene_3d_condition_vr))] if use_scene_3d_condition else None
        replay_3d_condition_images = [Image.fromarray(replay_3d_condition_vr[i].asnumpy()) for i in range(len(replay_3d_condition_vr))] if use_replay_3d_condition else None

        lengths = [len(replay_condition_images)]
        if scene_3d_condition_images is not None:
            lengths.append(len(scene_3d_condition_images))
        if replay_3d_condition_images is not None:
            lengths.append(len(replay_3d_condition_images))
        video_length = min(lengths)
        if video_length <= 0:
            continue

        sample_indexes = np.linspace(0, video_length - 1, args.sub_frames + 1, dtype=int)
        replay_condition_images, _ = process_video(replay_condition_images, args.dst_size, sample_indexes=sample_indexes)
        if scene_3d_condition_images is not None:
            scene_3d_condition_images, _ = process_video(scene_3d_condition_images, args.dst_size, sample_indexes=sample_indexes)
        if replay_3d_condition_images is not None:
            replay_3d_condition_images, _ = process_video(replay_3d_condition_images, args.dst_size, sample_indexes=sample_indexes)
        ref_image = get_ref_image(Image.open(first_frame_path).convert("RGB"), args.dst_size)

        for inference_id, seed in enumerate(seed_list):
            if inference_id == 0:
                set_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
            start_time = time.time()
            output_images = pipe(
                replay=replay_condition_images,
                scene_3d=scene_3d_condition_images,
                replay_3d=replay_3d_condition_images,
                scene_3d_weight=scene_3d_weight,
                replay_3d_weight=replay_3d_weight,
                height=args.dst_size[1],
                width=args.dst_size[0],
                num_frames=args.sub_frames + 1,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                image=ref_image,
                generator=generator,
                output_type="pil",
                prompt="",
            ).frames[0]
            torch.cuda.synchronize()
            print(f"[{rank}] {save_name} seed={seed} inference time: {time.time() - start_time:.2f}s")

            vis_images, gen_images = [], []
            for k in range(len(replay_condition_images)):
                vis_image = [ref_image, output_images[k], replay_condition_images[k]]
                if scene_3d_condition_images is not None:
                    vis_image.append(scene_3d_condition_images[k])
                if replay_3d_condition_images is not None:
                    vis_image.append(replay_3d_condition_images[k])
                vis_images.append(image_utils.concat_images_grid(vis_image, cols=1, pad=2))
                gen_images.append(output_images[k])

            suffix = "" if len(seed_list) == 1 else f"_seed{seed}"
            for subdir, images in [("concat", vis_images), ("gen", gen_images)]:
                path = os.path.join(args.save_dir, subdir, f"{save_name}{suffix}.mp4")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                imageio.mimsave(path, images, fps=args.save_fps)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer_model_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True, help="Python config path, e.g. iMac.configs.0501_worldarena_3d_r1c120.config")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--dataset_paths", type=str, required=True, help="Comma-separated WorldArena test dataset root paths.")
    parser.add_argument("--model_id", type=str, default=model_config["wan2.2-5b-diffusers"])
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument(
        "--seed",
        type=str,
        default="1024",
        help="Single seed like 1024 or multi-seed list like 102,1024,10246.",
    )
    parser.add_argument("--sub_frames", type=int, default=120)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--dst_size", type=int, nargs=2, default=[320, 256], metavar=("W", "H"))
    parser.add_argument("--save_fps", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    args.dst_size = tuple(args.dst_size)
    args.seed = [int(x) for x in args.seed.split(",") if x.strip() != ""]
    args.dataset_paths = [x.strip() for x in args.dataset_paths.split(",") if x.strip() != ""]
    if len(args.dataset_paths) == 0:
        raise ValueError("--dataset_paths must provide at least one path.")
    if len(args.seed) == 0:
        raise ValueError("--seed must provide at least one integer seed.")
    multiprocessing.set_start_method("spawn")
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
    world_size = len(gpu_ids)
    process_list = []
    for i, gpu in enumerate(gpu_ids):
        process = Process(
            target=inference_worldarena_test,
            args=(f"cuda:{gpu}", args, world_size, i),
        )
        process.start()
        process_list.append(process)
    for process in process_list:
        process.join()


if __name__ == "__main__":
    main()
