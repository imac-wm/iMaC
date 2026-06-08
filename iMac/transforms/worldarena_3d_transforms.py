import copy
import random

import numpy as np
import torch
from decord import VideoReader
from giga_datasets import video_utils
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

# from giga_train import TRANSFORMS, PromptTransform
from giga_train import TRANSFORMS
from decord import VideoReader
import decord
import h5py
from PIL import Image
from typing import List, Tuple, Optional, Union
import re
import ftfy
import html
from transformers import AutoTokenizer, UMT5EncoderModel

from giga_datasets import Dataset, FileWriter, PklWriter, load_dataset
from giga_datasets import utils as gd_utils
import pandas as pd
# from giga_models.models.diffusion.cosmos import T5TextEncoder
from decord import VideoReader as DecordVideoReader
from torchvision.io import VideoReader as TorchVideoReader
import torch

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


def save_video_per_frame(images, path):
    import cv2
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(path, fourcc, 16, (width, height))
    for img in images:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video_writer.write(img_bgr)
    video_writer.release()

import os
import pickle
@TRANSFORMS.register
class WorldArena3DTransforms:
    def __init__(
        self,
        is_train=False,
        dst_size=None,
        num_frames=1,
        fps=16,
        image_cfg=None,
        max_stride=None,
        sub_frames=1,
        num_views=1,
        default_prompt_embeds_path=None,
        use_replay_condition=True,
        use_scene_3d_condition=True,
        use_replay_3d_condition=True,
        **kwargs,
    ):
        if "use_replay_cond" in kwargs:
            use_replay_condition = kwargs.pop("use_replay_cond")
        if "use_scene_3d_cond" in kwargs:
            use_scene_3d_condition = kwargs.pop("use_scene_3d_cond")
        if "use_replay_3d_cond" in kwargs:
            use_replay_3d_condition = kwargs.pop("use_replay_3d_cond")
        # keep forward-compatibility with extra config keys from runner/builders
        self.fps = fps
        self.is_train = is_train
        self.normalize = transforms.Normalize([0.5], [0.5])
        self.dst_size = dst_size
        self.num_frames = num_frames
        self.image_cfg = image_cfg
        self.mask_generator = MaskGenerator(**image_cfg['mask_generator'])
        self.max_stride = max_stride
        self.sub_frames = sub_frames

        assert default_prompt_embeds_path is not None
        self.default_prompt_embeds = torch.load(default_prompt_embeds_path)['condition_dict']['prompt_embeds'].squeeze(0)
        # self.default_prompt_embeds = torch.load("/data/ode_depth_pairs/00000.pt")['condition_dict']['prompt_embeds'].squeeze(0)
        self.num_views = num_views
        self.use_replay_condition = use_replay_condition
        self.use_scene_3d_condition = use_scene_3d_condition
        self.use_replay_3d_condition = use_replay_3d_condition

    def __call__(self, data_dict):
        # org_file = data_dict['org_data_path']
        dst_width, dst_height = data_dict['video_info']
        if self.num_views == 1:
            front_images = DecordVideoReader(data_dict['gt_video_path'])
            candidate_lengths = [len(front_images)]
            replay_condition_images = None
            scene_3d_condition_images = None
            replay_3d_condition_images = None

            if self.use_replay_condition:
                replay_condition_images = DecordVideoReader(data_dict['condition_video_path'])
                candidate_lengths.append(len(replay_condition_images))
            if self.use_scene_3d_condition:
                scene_3d_condition_images = DecordVideoReader(data_dict['scene_3d_condition_path'])
                candidate_lengths.append(len(scene_3d_condition_images))
            if self.use_replay_3d_condition:
                replay_3d_condition_images = DecordVideoReader(data_dict['replay_3d_condition_path'])
                candidate_lengths.append(len(replay_3d_condition_images))

            video_legnth = min(candidate_lengths)
            assert video_legnth > 0

        if self.max_stride is not None:
            stride = random.randint(1, self.max_stride)
            start_frame = random.randint(0, max(0, video_legnth - stride * (self.num_frames - 1) - 1))
            end_frame = start_frame + stride * (self.num_frames - 1)
            end_frame = min(video_legnth, end_frame)
            sample_indexes = np.linspace(start_frame, end_frame, num=self.num_frames, dtype=int)
            # print(sample_indexes)
        else:
            sample_indexes = np.linspace(0, video_legnth - 1, self.num_frames, dtype=int)
        # print(sample_indexes)
        def get_input_images(video):
            if isinstance(video, VideoReader):
                input_images = video_utils.sample_video(video, sample_indexes, method=2)
            else:
                input_images = video[sample_indexes]
            data_dict['input_fps'] = self.fps
            input_images = torch.from_numpy(input_images).permute(0, 3, 1, 2).contiguous()
            height = input_images.shape[2]
            width = input_images.shape[3]
            # assert height == dst_height
            # assert width == dst_width
            input_images = F.resize(input_images, (dst_height, dst_width), InterpolationMode.BILINEAR)
            input_images = input_images / 255.0
            input_images = self.normalize(input_images)
            return input_images

        if self.num_views == 1:
            data_dict['input_front_images'] = get_input_images(front_images)
            if replay_condition_images is not None:
                data_dict['input_replay_condition_images'] = get_input_images(replay_condition_images)
            if scene_3d_condition_images is not None:
                data_dict['input_scene_3d_condition_images'] = get_input_images(scene_3d_condition_images)
            if replay_3d_condition_images is not None:
                data_dict['input_replay_3d_condition_images'] = get_input_images(replay_3d_condition_images)

        if self.image_cfg is not None:
            ref_masks, ref_latent_masks = self.mask_generator.get_mask(data_dict['input_front_images'].shape[0])
            ref_masks = ref_masks[:, None, None, None]
            ref_latent_masks = ref_latent_masks[None, :, None, None]
            ref_images = copy.deepcopy(data_dict['input_front_images'])
            ref_images = ref_images * ref_masks
            data_dict['input_front_ref_images'] = ref_images
            data_dict['input_front_ref_masks'] = ref_latent_masks

        # image = input_images[0].permute(1, 2, 0).cpu().numpy()
        # image = (image * 0.5 + 0.5) * 255
        # image = image.astype(np.uint8)
        # image = Image.fromarray(image)
        if self.is_train:
            new_data_dict = {}
            if 'input_fps' in data_dict:
                new_data_dict['fps'] = data_dict['input_fps']
            if 'input_front_images' in data_dict:
                new_data_dict['front_images'] = data_dict['input_front_images']
            if 'input_front_ref_images' in data_dict:
                new_data_dict['front_ref_images'] = data_dict['input_front_ref_images']
                new_data_dict['front_ref_masks'] = data_dict['input_front_ref_masks']
            if 'input_replay_condition_images' in data_dict:
                new_data_dict['replay_condition_images'] = data_dict['input_replay_condition_images']
            if 'input_scene_3d_condition_images' in data_dict:
                new_data_dict['scene_3d_condition_images'] = data_dict['input_scene_3d_condition_images']
            if 'input_replay_3d_condition_images' in data_dict:
                new_data_dict['replay_3d_condition_images'] = data_dict['input_replay_3d_condition_images']
            if (
                'input_replay_condition_images' in data_dict
                or 'input_scene_3d_condition_images' in data_dict
                or 'input_replay_3d_condition_images' in data_dict
            ):
                new_data_dict['prompt_embeds'] = self.default_prompt_embeds

        else:
            assert False
        keys = list(new_data_dict.keys())
        for key in keys:
            if new_data_dict[key] is None:
                new_data_dict.pop(key)
        return new_data_dict

class MaskGenerator:
    def __init__(self, max_ref_frames, factor=8, start=1):
        assert max_ref_frames > 0 and (max_ref_frames - 1) % factor == 0
        self.max_ref_frames = max_ref_frames
        self.factor = factor
        self.start = start
        self.max_ref_latents = 1 + (max_ref_frames - 1) // factor
        assert self.start <= self.max_ref_latents

    def get_mask(self, num_frames):
        assert num_frames > 0 and (num_frames - 1) % self.factor == 0 and num_frames >= self.max_ref_frames
        num_latents = 1 + (num_frames - 1) // self.factor
        num_ref_latents = random.randint(self.start, self.max_ref_latents)
        if num_ref_latents > 0:
            num_ref_frames = 1 + (num_ref_latents - 1) * self.factor
        else:
            num_ref_frames = 0
        ref_masks = torch.zeros((num_frames,), dtype=torch.float32)
        ref_masks[:num_ref_frames] = 1
        ref_latent_masks = torch.zeros((num_latents,), dtype=torch.float32)
        ref_latent_masks[:num_ref_latents] = 1
        return ref_masks, ref_latent_masks
