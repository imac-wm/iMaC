import json
import multiprocessing
import os
import time
from multiprocessing import Process
import argparse
import importlib
import re

import paths

import imageio
import numpy as np
import torch
from accelerate.utils import set_seed
from decord import VideoReader
from diffusers import AutoencoderKLWan
from giga_datasets import image_utils
from giga_datasets import load_dataset
from giga_datasets import utils as gd_utils
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from tqdm import tqdm

from iMac.models.transformer_worldarena_3d import WanTransformer3DModelWorldArena3D
from iMac.model_config import WORLDARENA_DATA_DIR, model_config
from iMac.pipelines.pipeline_worldarena import WorldArenaPipeline


def load_video(video, sample_frames):
    sample_indexes = np.linspace(0, len(video) - 1, sample_frames, dtype=int)
    images = [video[index] for index in sample_indexes]
    return images, sample_indexes


def _parse_episode_filter(episodes_arg: str):
    if not episodes_arg:
        return set()
    out = set()
    for x in episodes_arg.split(","):
        x = x.strip()
        if not x:
            continue
        out.add(int(x))
    return out




def _load_config(config_path: str):
    module_name, attr_name = config_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _get_model_config(config_path: str):
    cfg = _load_config(config_path)
    return cfg.get("models", {})

def _episode_id_from_name(episode_name: str):
    m = re.search(r"(\d+)$", str(episode_name))
    if m is None:
        return None
    return int(m.group(1))


def inference_worldarena(
    device,
    transformer_model_path,
    save_dir,
    config_path,
    dataset_dir,
    pretrained_model_path,
    world_size=1,
    rank=0,
    episode_filter_arg="",
):
    torch.cuda.set_device(device)
    device = "cuda"
    model_cfg = _get_model_config(config_path)
    use_scene_3d_condition = model_cfg.get("use_scene_3d_condition", True)
    use_replay_3d_condition = model_cfg.get("use_replay_3d_condition", True)
    scene_3d_weight = model_cfg.get("scene_3d_weight", 1.0)
    replay_3d_weight = model_cfg.get("replay_3d_weight", 1.0)

    seed = 1024
    dst_size = (320, 256)
    sub_frames = 120
    num_inference_steps = 30
    guidance_scale = 0.0
    dtype = torch.bfloat16

    model_id = pretrained_model_path
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    transformer = WanTransformer3DModelWorldArena3D.from_pretrained(transformer_model_path).to(dtype)
    pipe = WorldArenaPipeline.from_pretrained(model_id, vae=vae, transformer=transformer, torch_dtype=dtype)
    pipe.to(device)

    dataset = load_dataset([dataset_dir])
    data_list = gd_utils.split_data(list(range(len(dataset))), world_size, rank)
    allowed_episode_ids = _parse_episode_filter(episode_filter_arg)

    def process_video(video_reader, sample_indexes=None, sample_frames=None):
        if sample_indexes is None:
            assert sample_frames is not None
            images, sample_indexes = load_video(video_reader, sample_frames)
        else:
            images = [video_reader[index] for index in sample_indexes]
        height, width = images[0].height, images[0].width
        dst_width, dst_height = image_utils.get_image_size((width, height), dst_size, mode="area", multiple=32)
        input_images = [F.resize(img, (dst_height, dst_width), InterpolationMode.BILINEAR) for img in images]
        return input_images, sample_indexes

    for idx in tqdm(data_list):
        set_seed(seed)
        data_dict = dataset[idx]
        task_name = data_dict["task_name"]
        episode_name = data_dict["episode_name"]
        episode_id = _episode_id_from_name(episode_name)
        if allowed_episode_ids and (episode_id is None or episode_id not in allowed_episode_ids):
            continue
        save_name = f"{task_name}/{episode_name}"

        front_images_vr = VideoReader(data_dict["gt_video_path"])
        replay_condition_vr = VideoReader(data_dict["condition_video_path"])
        scene_3d_condition_vr = VideoReader(data_dict["scene_3d_condition_path"]) if use_scene_3d_condition else None
        replay_3d_condition_vr = VideoReader(data_dict["replay_3d_condition_path"]) if use_replay_3d_condition else None

        front_images = [Image.fromarray(front_images_vr[i].asnumpy()) for i in range(len(front_images_vr))]
        replay_condition_images = [Image.fromarray(replay_condition_vr[i].asnumpy()) for i in range(len(replay_condition_vr))]
        scene_3d_condition_images = [Image.fromarray(scene_3d_condition_vr[i].asnumpy()) for i in range(len(scene_3d_condition_vr))] if use_scene_3d_condition else None
        replay_3d_condition_images = [Image.fromarray(replay_3d_condition_vr[i].asnumpy()) for i in range(len(replay_3d_condition_vr))] if use_replay_3d_condition else None

        lengths = [len(front_images), len(replay_condition_images)]
        if scene_3d_condition_images is not None:
            lengths.append(len(scene_3d_condition_images))
        if replay_3d_condition_images is not None:
            lengths.append(len(replay_3d_condition_images))
        video_length = min(lengths)
        if video_length <= 0:
            continue

        sample_indexes = np.linspace(0, video_length - 1, sub_frames + 1, dtype=int)
        input_images, _ = process_video(front_images, sample_indexes=sample_indexes)
        replay_condition_images, _ = process_video(replay_condition_images, sample_indexes=sample_indexes)
        if scene_3d_condition_images is not None:
            scene_3d_condition_images, _ = process_video(scene_3d_condition_images, sample_indexes=sample_indexes)
        if replay_3d_condition_images is not None:
            replay_3d_condition_images, _ = process_video(replay_3d_condition_images, sample_indexes=sample_indexes)
        ref_image = input_images[0]

        generator = torch.Generator(device=device).manual_seed(seed)
        start_time = time.time()
        output_images = pipe(
            replay=replay_condition_images,
            scene_3d=scene_3d_condition_images,
            replay_3d=replay_3d_condition_images,
            scene_3d_weight=scene_3d_weight,
            replay_3d_weight=replay_3d_weight,
            height=dst_size[1],
            width=dst_size[0],
            num_frames=sub_frames + 1,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            image=ref_image,
            generator=generator,
            output_type="pil",
            prompt="",
        ).frames[0]
        torch.cuda.synchronize()
        print(f"[{rank}] {save_name} inference time: {time.time() - start_time:.2f}s")

        vis_images, gt_images, gen_images = [], [], []
        vis_output_replay_images = []
        vis_output_scene_3d_images = []
        vis_output_replay_3d_images = []
        for k in range(len(input_images)):
            vis_image = [input_images[k], output_images[k], replay_condition_images[k]]
            if scene_3d_condition_images is not None:
                vis_image.append(scene_3d_condition_images[k])
            if replay_3d_condition_images is not None:
                vis_image.append(replay_3d_condition_images[k])
            vis_images.append(image_utils.concat_images_grid(vis_image, cols=1, pad=2))
            vis_output_replay_images.append(image_utils.concat_images_grid([output_images[k], replay_condition_images[k]], cols=1, pad=2))
            if scene_3d_condition_images is not None:
                vis_output_scene_3d_images.append(image_utils.concat_images_grid([output_images[k], scene_3d_condition_images[k]], cols=1, pad=2))
            if replay_3d_condition_images is not None:
                vis_output_replay_3d_images.append(image_utils.concat_images_grid([output_images[k], replay_3d_condition_images[k]], cols=1, pad=2))
            gt_images.append(input_images[k])
            gen_images.append(output_images[k])

        save_items = [
            ("concat", vis_images),
            ("gt", gt_images),
            ("gen", gen_images),
            ("concat_replay_gen", vis_output_replay_images),
        ]
        if scene_3d_condition_images is not None:
            save_items.append(("concat_scene_3d_gen", vis_output_scene_3d_images))
        if replay_3d_condition_images is not None:
            save_items.append(("concat_replay_3d_gen", vis_output_replay_3d_images))

        for subdir, images in save_items:
            path = os.path.join(save_dir, subdir, f"{save_name}.mp4")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            imageio.mimsave(path, images, fps=30)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformer_model_path",
        type=str,
        required=True,
        help="Path to trained WorldArena transformer checkpoint directory.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Output directory for concat/gt/gen videos.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help="Comma separated gpu ids, e.g. '0' or '0,1,2,3'.",
    )
    parser.add_argument("--config_path", type=str, required=True, help="Python config path, e.g. iMac.configs.0501_worldarena_3d_r1c120.config")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=os.path.join(WORLDARENA_DATA_DIR, "val"),
        help="Packed WorldArena validation dataset directory.",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=model_config["wan2.2-5b-diffusers"],
        help="Wan2.2 diffusers model directory.",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="",
        help="Optional episode ids to run, comma separated, e.g. '40,41,49'. Empty means run all.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    multiprocessing.set_start_method("spawn")
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
    world_size = len(gpu_ids)
    process_list = []
    for i, gpu in enumerate(gpu_ids):
        process = Process(
            target=inference_worldarena,
            args=(
                f"cuda:{gpu}",
                args.transformer_model_path,
                args.save_dir,
                args.config_path,
                args.dataset_dir,
                args.pretrained_model_path,
                world_size,
                i,
                args.episodes,
            ),
        )
        process.start()
        process_list.append(process)
    for process in process_list:
        process.join()


if __name__ == "__main__":
    main()
