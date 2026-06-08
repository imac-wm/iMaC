import argparse
import multiprocessing
from multiprocessing import Process

import paths

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from PIL import Image

from iMac.image_utils import concat_images_grid
from iMac.model_config import DATA_DIR, model_config
from iMac.models import WanConditionRNDMixStageTwoModel
from iMac.pipelines import BaselineWMPipeline
from iMac.utils import resize_with_pad, split_data

# Reuse the stage-one inference data loading / interaction loop.
from inference_stage_one import InferenceStageOneEngine as BaseStageOneEngine
from inference_stage_one import VideoReader, imageio, json, os, pickle, tqdm, inference as stage_one_inference


class InferenceRNDMixStageTwoEngine(BaseStageOneEngine):
    def __init__(
        self,
        transformer_model_path,
        device,
        dtype=torch.bfloat16,
        num_views=3,
        mode='offline',
        seed=1024,
        da3_model_path=None,
        da3_urdf_path=None,
        da3_gripper_mesh_dir=None,
        da3_device="cuda",
        normalize_metric_depth_frames_min=0.08,
        normalize_metric_depth_frames_max=1.2,
        normalize_metric_depth_frames_use_relative=False,
        normalize_metric_depth_frames_use_sqrt=False,
        metric_depth_rgb_encoding="linear",
        metric_depth_rgb_lambda=-500.0,
        metric_depth_rgb_c=0.53,
    ):
        assert mode in ['offline', 'online'], f"mode must be offline or online, but got {mode}"
        torch.cuda.set_device(device)
        device = "cuda"
        model_id = model_config['wan2.2-5b-diffusers']
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
        transformer = WanConditionRNDMixStageTwoModel.from_pretrained(transformer_model_path).to(dtype)
        self.mode = mode
        self.pipe = BaselineWMPipeline.from_pretrained(model_id, vae=vae, transformer=transformer, torch_dtype=dtype)
        self.pipe.to(device)

        self.dst_size = (224, 224)
        self.wm_frame_per_time = 8
        self.num_inference_steps = 30
        self.guidance_scale = 0.0
        self.num_views = num_views
        self.normalize_metric_depth_frames_min = float(normalize_metric_depth_frames_min)
        self.normalize_metric_depth_frames_max = float(normalize_metric_depth_frames_max)
        self.normalize_metric_depth_frames_use_relative = bool(normalize_metric_depth_frames_use_relative)
        self.normalize_metric_depth_frames_use_sqrt = bool(normalize_metric_depth_frames_use_sqrt)
        self.metric_depth_rgb_encoding = str(metric_depth_rgb_encoding)
        self.metric_depth_rgb_lambda = float(metric_depth_rgb_lambda)
        self.metric_depth_rgb_c = float(metric_depth_rgb_c)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        from third_party.func_flow_control_urdf import CondGenerator

        da3_kwargs = dict(device=da3_device)
        if da3_model_path is not None:
            da3_kwargs["model_path"] = da3_model_path
        if da3_urdf_path is not None:
            da3_kwargs["urdf_path"] = da3_urdf_path
        if da3_gripper_mesh_dir is not None:
            da3_kwargs["gripper_mesh_dir"] = da3_gripper_mesh_dir
        self.da3_generator = CondGenerator(**da3_kwargs)

    @staticmethod
    def _stack_rgb_depth(rgb_image, depth_image=None):
        if depth_image is None:
            depth_image = rgb_image
        if isinstance(rgb_image, Image.Image):
            rgb_image = np.asarray(rgb_image)
        if isinstance(depth_image, Image.Image):
            depth_image = np.asarray(depth_image)
        stacked = np.concatenate([rgb_image, depth_image], axis=0)
        return Image.fromarray(stacked.astype(np.uint8))

    @staticmethod
    def _metric_depth_to_vision_banana_rgb(depth, lambda_param=-500.0, c=0.53):
        if lambda_param >= -1:
            raise ValueError("Vision Banana metric-depth RGB encoding requires lambda < -1.")
        if c <= 0:
            raise ValueError("Vision Banana metric-depth RGB encoding requires c > 0.")

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth.mean(axis=-1)
        if np.nanmax(depth) > 65.535:
            depth = depth / 1000.0
        valid = np.isfinite(depth) & (depth >= 0)
        clipped = np.where(valid, depth, 0.0).astype(np.float32)
        normalized = 1.0 - np.power(1.0 - clipped / (lambda_param * c), lambda_param + 1.0)
        normalized = np.clip(normalized, 0.0, np.nextafter(1.0, 0.0))
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        scaled = normalized * (len(vertices) - 1)
        index = np.minimum(np.floor(scaled).astype(np.int32), len(vertices) - 2)
        frac = (scaled - index)[..., None]
        rgb = vertices[index] * (1.0 - frac) + vertices[index + 1] * frac
        rgb_u8 = np.rint(rgb * 255.0).astype(np.uint8)
        rgb_u8[~valid] = 0
        return rgb_u8

    def _metric_depth_to_uint8_rgb(self, depth):
        if self.metric_depth_rgb_encoding in ("vision_banana", "rgb_cube", "depth2rgb"):
            return self._metric_depth_to_vision_banana_rgb(
                depth,
                lambda_param=self.metric_depth_rgb_lambda,
                c=self.metric_depth_rgb_c,
            )
        if self.metric_depth_rgb_encoding not in ("linear", "legacy"):
            raise ValueError(f"Unsupported metric depth RGB encoding: {self.metric_depth_rgb_encoding}")

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth.mean(axis=-1)
        if np.nanmax(depth) > 65.535:
            depth = depth / 1000.0
        if self.normalize_metric_depth_frames_use_relative:
            d_min = np.nanmin(depth)
            d_max = np.nanmax(depth)
            if not np.isfinite(d_min) or not np.isfinite(d_max) or d_max <= d_min:
                depth = np.zeros_like(depth, dtype=np.float32)
            else:
                depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.clip(depth, self.normalize_metric_depth_frames_min, self.normalize_metric_depth_frames_max)
            if self.normalize_metric_depth_frames_use_sqrt:
                depth = np.sqrt(depth) / np.sqrt(max(self.normalize_metric_depth_frames_max, 1e-8))
            else:
                depth = depth / max(self.normalize_metric_depth_frames_max, 1e-8)
        depth_u8 = np.rint(np.clip(depth, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.stack([depth_u8] * 3, axis=-1)

    def build_depth_bootstrap_da3(self, img_front, img_left, img_right, qpos):
        current_obs = [np.asarray(img_front), np.asarray(img_left), np.asarray(img_right)]
        extrinsics = self.da3_generator.forward_kinematics(np.asarray(qpos))
        depths, _, _ = self.da3_generator.forward_DA3(current_obs, extrinsics)
        depth_images = []
        for cam in ("front", "left", "right"):
            depth_rgb = self._metric_depth_to_uint8_rgb(depths[cam])
            depth_images.append(self.resize_image(Image.fromarray(depth_rgb)))
        return concat_images_grid(depth_images, cols=3, pad=0)

    @staticmethod
    def _ensure_condition_frames_size(frames, target_size, name):
        checked_frames = []
        for frame in frames:
            if not isinstance(frame, Image.Image):
                frame = Image.fromarray(np.asarray(frame).astype(np.uint8))
            if frame.mode != "RGB":
                frame = frame.convert("RGB")
            if frame.size != target_size:
                raise ValueError(f"{name} frame size must be {target_size}, got {frame.size}")
            checked_frames.append(frame)
        return checked_frames

    def _resize_condition_frames(self, frames, target_size):
        resized_frames = []
        for frame in frames:
            if not isinstance(frame, Image.Image):
                frame = Image.fromarray(np.asarray(frame).astype(np.uint8))
            if frame.mode != "RGB":
                frame = frame.convert("RGB")
            if frame.size != target_size:
                frame = Image.fromarray(
                    resize_with_pad(np.asarray(frame), target_size[1], target_size[0]).astype(np.uint8)
                )
            resized_frames.append(frame)
        return resized_frames

    def wm_inference_per_time(self, replay_condition, ref_image, replay_3d_condition=None, scene_3d_condition=None):
        # RND-mix stage-two uses vertically stacked RGB+Depth as the denoising input.
        ref_stacked = ref_image
        if ref_stacked.size[1] == self.dst_size[1]:
            ref_stacked = self._stack_rgb_depth(ref_stacked)

        # replay condition follows training transform behavior:
        # duplicate replay itself along height to match stacked [448, 672] resolution.
        replay_condition_stacked = [self._stack_rgb_depth(frame, frame) for frame in replay_condition]
        condition_size = (self.dst_size[0] * self.num_views, self.dst_size[1] * 2)
        if replay_3d_condition is not None:
            replay_3d_condition = self._ensure_condition_frames_size(
                replay_3d_condition, condition_size, "replay_3d_condition"
            )
        if scene_3d_condition is not None:
            scene_3d_condition = self._ensure_condition_frames_size(
                scene_3d_condition, condition_size, "scene_3d_condition"
            )

        output_images = self.pipe(
            replay=replay_condition_stacked,
            depth=replay_condition_stacked,
            replay_3d=replay_3d_condition,
            scene_3d=scene_3d_condition,
            replay_weight=1.0,
            replay_3d_weight=1.0,
            scene_3d_weight=1.0,
            height=self.dst_size[1] * 2,
            width=self.dst_size[0] * self.num_views,
            num_frames=self.wm_frame_per_time + 1,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            image=ref_stacked,
            generator=self.generator,
            output_type='pil',
            prompt=''
        ).frames[0]
        return output_images

    def _split_six_grid(self, stacked_img):
        if isinstance(stacked_img, Image.Image):
            stacked_img = np.asarray(stacked_img)
        h = stacked_img.shape[0] // 2
        w = stacked_img.shape[1] // 3
        rgb = stacked_img[:h]
        dep = stacked_img[h:]
        rgb_views = [rgb[:, i * w : (i + 1) * w] for i in range(3)]
        dep_views = [dep[:, i * w : (i + 1) * w] for i in range(3)]
        return rgb_views, dep_views

    def _depth_u8_to_metric(self, depth_u8):
        depth_u8 = np.asarray(depth_u8, dtype=np.float32)
        if self.metric_depth_rgb_encoding in ("vision_banana", "rgb_cube", "depth2rgb"):
            metric = self._vision_banana_rgb_to_metric(
                depth_u8,
                lambda_param=self.metric_depth_rgb_lambda,
                c=self.metric_depth_rgb_c,
            )
            metric = np.nan_to_num(
                metric,
                nan=self.normalize_metric_depth_frames_min,
                posinf=self.normalize_metric_depth_frames_max,
                neginf=self.normalize_metric_depth_frames_min,
            )
            return np.clip(
                metric,
                self.normalize_metric_depth_frames_min,
                self.normalize_metric_depth_frames_max,
            )
        if self.metric_depth_rgb_encoding not in ("linear", "legacy"):
            raise ValueError(f"Unsupported metric depth RGB encoding: {self.metric_depth_rgb_encoding}")

        if depth_u8.ndim == 3:
            depth_u8 = depth_u8.mean(axis=-1)
        depth_norm = np.clip(depth_u8 / 255.0, 0.0, 1.0)
        if self.normalize_metric_depth_frames_use_relative:
            raise ValueError("Cannot uniquely invert relative depth normalization during inference.")
        if self.normalize_metric_depth_frames_use_sqrt:
            metric = np.square(depth_norm) * max(self.normalize_metric_depth_frames_max, 1e-8)
        else:
            metric = depth_norm * max(self.normalize_metric_depth_frames_max, 1e-8)
        metric = np.clip(metric, self.normalize_metric_depth_frames_min, self.normalize_metric_depth_frames_max)
        return metric

    @staticmethod
    def _vision_banana_rgb_to_metric(depth_rgb, lambda_param=-500.0, c=0.53):
        if lambda_param >= -1:
            raise ValueError("Vision Banana metric-depth RGB encoding requires lambda < -1.")
        if c <= 0:
            raise ValueError("Vision Banana metric-depth RGB encoding requires c > 0.")

        rgb = np.asarray(depth_rgb, dtype=np.float32)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=-1)
        rgb = np.clip(rgb[..., :3] / 255.0, 0.0, 1.0)
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        best_dist = None
        best_p = None
        for idx in range(len(vertices) - 1):
            start = vertices[idx]
            direction = vertices[idx + 1] - start
            denom = float(np.dot(direction, direction))
            t = np.clip(np.sum((rgb - start) * direction, axis=-1) / denom, 0.0, 1.0)
            projected = start + t[..., None] * direction
            dist = np.sum((rgb - projected) ** 2, axis=-1)
            p = (idx + t) / (len(vertices) - 1)
            if best_dist is None:
                best_dist = dist
                best_p = p
            else:
                take = dist < best_dist
                best_dist = np.where(take, dist, best_dist)
                best_p = np.where(take, p, best_p)
        p = np.clip(best_p, 0.0, np.nextafter(1.0, 0.0))
        metric = lambda_param * c * (1.0 - np.power(1.0 - p, 1.0 / (lambda_param + 1.0)))
        return metric.astype(np.float32)

    @staticmethod
    def _pad_or_trim_actions(qpos_chunk, k):
        current_future_action = np.asarray(qpos_chunk)
        if current_future_action.shape[0] == 0:
            raise ValueError("qpos_chunk must contain at least one action")
        if current_future_action.shape[0] > k:
            current_future_action = current_future_action[:k]
        elif current_future_action.shape[0] < k:
            pad = np.repeat(current_future_action[-1:], k - current_future_action.shape[0], axis=0)
            current_future_action = np.concatenate([current_future_action, pad], axis=0)
        return current_future_action

    def _build_3d_cond_from_stacked_ref(self, ref_stacked, qpos_chunk, k=None):
        if k is None:
            k = self.wm_frame_per_time + 1
        rgb_views, dep_views = self._split_six_grid(ref_stacked)
        current_depth = [self._depth_u8_to_metric(d) for d in dep_views]
        current_obs = rgb_views
        current_future_action = self._pad_or_trim_actions(qpos_chunk, k)
        cond_video1, cond_video2, _ = self.da3_generator.depth_point2double_cond(
            current_depth=current_depth,
            current_future_action=current_future_action,
            current_obs=current_obs,
            use_gpu=True,
        )
        return cond_video1, cond_video2

    def _get_current_ref_stacked(self, ref_stacked):
        return ref_stacked

    def _format_3d_condition_frames(self, cond_video, k, target_h, target_w):
        one_h = target_h // 2
        one_w = target_w // self.num_views
        if len(cond_video) == 0:
            zero_views = [
                np.zeros((one_h, one_w, 3), dtype=np.uint8)
                for _ in range(self.num_views)
            ]
            cond_video = [zero_views]

        cond_video = cond_video[:k]
        if len(cond_video) < k:
            cond_video += [cond_video[-1]] * (k - len(cond_video))

        frames = []
        for views in cond_video:
            padded_views = []
            for view in views[: self.num_views]:
                view = np.asarray(view)
                if view.ndim == 2:
                    view = np.repeat(view[..., None], 3, axis=-1)
                padded_views.append(resize_with_pad(view.astype(np.uint8), one_h, one_w))
            while len(padded_views) < self.num_views:
                padded_views.append(np.zeros((one_h, one_w, 3), dtype=np.uint8))
            three_view = np.concatenate(padded_views, axis=1)
            frame = np.concatenate([three_view, three_view], axis=0)
            frames.append(Image.fromarray(frame.astype(np.uint8)))
        return frames

    def _stack_3d_condition_views(self, cond_video, k, target_size):
        return self._format_3d_condition_frames(cond_video, k, target_size[1], target_size[0])

    def crop_three_view_images(self, ref_image):
        # For stacked frames [448, 672], policy / next-step RGB ref uses top half.
        if ref_image.size[1] == self.dst_size[1] * 2:
            ref_image = ref_image.crop((0, 0, self.dst_size[0] * 3, self.dst_size[1]))
        return super().crop_three_view_images(ref_image)

    def wm_inference(self, ref_images, action_images, qpos_seq):
        img_front = self.resize_image(ref_images['front'])
        img_left = self.resize_image(ref_images['left'])
        img_right = self.resize_image(ref_images['right'])
        ref_rgb = concat_images_grid([img_front, img_left, img_right], cols=3, pad=0)
        ref_depth = self.build_depth_bootstrap_da3(img_front, img_left, img_right, qpos_seq[0])
        ref_stacked = self._stack_rgb_depth(ref_rgb, ref_depth)

        front_replay_images = self.resize_images(action_images['front_replay'])
        left_replay_images = self.resize_images(action_images['left_replay'])
        right_replay_images = self.resize_images(action_images['right_replay'])

        replay_images = []
        action_chunk = min(len(front_replay_images), len(left_replay_images), len(right_replay_images), len(qpos_seq))
        for i in range(action_chunk):
            replay_image = concat_images_grid(
                [front_replay_images[i], left_replay_images[i], right_replay_images[i]], cols=3, pad=0
            )
            replay_images.append(replay_image)

        wm_inference_time = (action_chunk - 1) // self.wm_frame_per_time
        if action_chunk % self.wm_frame_per_time != 0:
            print(f"Warning: action_chunk {action_chunk} is not divisible by wm_frame_per_time {self.wm_frame_per_time}")

        all_output_images = []
        front_depth_condition_images = []
        replay_condition_images = []
        replay_3d_condition_images = []
        scene_3d_condition_images = []
        for step in tqdm(range(wm_inference_time)):
            start = step * self.wm_frame_per_time
            end = (step + 1) * self.wm_frame_per_time + 1
            action_images_chunk = replay_images[start:end]
            replay_condition_chunk = self.build_replay_condition(action_images_chunk, self.wm_frame_per_time + 1)
            qpos_chunk = qpos_seq[start:end]
            current_ref_stacked = self._get_current_ref_stacked(ref_stacked)
            cond_video1, cond_video2 = self._build_3d_cond_from_stacked_ref(
                current_ref_stacked, qpos_chunk, self.wm_frame_per_time + 1
            )
            model_condition_size = (self.dst_size[0] * self.num_views, self.dst_size[1] * 2)
            replay_3d_chunk = self._stack_3d_condition_views(
                cond_video1, len(replay_condition_chunk), model_condition_size
            )
            scene_3d_chunk = self._stack_3d_condition_views(
                cond_video2, len(replay_condition_chunk), model_condition_size
            )
            output_stacked = self.wm_inference_per_time(
                replay_condition_chunk,
                current_ref_stacked,
                replay_3d_condition=replay_3d_chunk,
                scene_3d_condition=scene_3d_chunk,
            )

            current_rgb = [img.crop((0, 0, self.dst_size[0] * 3, self.dst_size[1])) for img in output_stacked]
            current_depth = [img.crop((0, self.dst_size[1], self.dst_size[0] * 3, self.dst_size[1] * 2)) for img in output_stacked]

            if step == wm_inference_time - 1:
                all_output_images.extend(current_rgb)
                front_depth_condition_images.extend(current_depth)
                replay_condition_images.extend(replay_condition_chunk)
                replay_3d_condition_images.extend(replay_3d_chunk)
                scene_3d_condition_images.extend(scene_3d_chunk)
            else:
                all_output_images.extend(current_rgb[:-1])
                front_depth_condition_images.extend(current_depth[:-1])
                replay_condition_images.extend(replay_condition_chunk[:-1])
                replay_3d_condition_images.extend(replay_3d_chunk[:-1])
                scene_3d_condition_images.extend(scene_3d_chunk[:-1])

            ref_stacked = output_stacked[-1]

        condition_images_dict = {
            'front_depth': front_depth_condition_images,
            'replay_condition': replay_condition_images,
            'replay_3d_condition': replay_3d_condition_images,
            'scene_3d_condition': scene_3d_condition_images,
        }
        return all_output_images, condition_images_dict

    def interaction(self, ref_images, state, task, max_interactions=15, pos_lookahead_step=24):
        img_front = self.resize_image(ref_images['front'])
        img_left = self.resize_image(ref_images['left'])
        img_right = self.resize_image(ref_images['right'])
        assert pos_lookahead_step % self.wm_frame_per_time == 0

        all_output_images = []
        front_depth_condition_images = []
        replay_condition_images = []
        replay_3d_condition_images = []
        scene_3d_condition_images = []
        for step in tqdm(range(max_interactions)):
            print("Interaction step {}".format(step))
            actions = self.get_action(img_front, img_left, img_right, state, task)
            actions = actions[:pos_lookahead_step]
            future_state = np.concatenate([state[None, :], actions], axis=0)
            action_images = self.render_qpos(future_state)
            output_images, condition_images_dict = self.wm_inference(ref_images, action_images, future_state)
            all_output_images.extend(output_images)
            front_depth_condition_images.extend(condition_images_dict['front_depth'])
            replay_condition_images.extend(condition_images_dict['replay_condition'])
            replay_3d_condition_images.extend(condition_images_dict['replay_3d_condition'])
            scene_3d_condition_images.extend(condition_images_dict['scene_3d_condition'])

            state = future_state[-1]
            img_front, img_left, img_right = self.crop_three_view_images(output_images[-1])
            ref_images = {
                'front': img_front,
                'left': img_left,
                'right': img_right,
            }
            assert (
                len(all_output_images)
                == len(front_depth_condition_images)
                == len(replay_condition_images)
                == len(replay_3d_condition_images)
                == len(scene_3d_condition_images)
            )

        condition_images_dict = {
            'front_depth': front_depth_condition_images,
            'replay_condition': replay_condition_images,
            'replay_3d_condition': replay_3d_condition_images,
            'scene_3d_condition': scene_3d_condition_images,
        }
        return all_output_images, condition_images_dict


def save_condition_videos(output_dir, episode_name, condition_images_dict, save_length, fps=24):
    if save_length <= 0:
        return
    condition_video_specs = {
        "replay_image": condition_images_dict["replay_condition"],
        "replay_3d": condition_images_dict["replay_3d_condition"],
        "scene_3d": condition_images_dict["scene_3d_condition"],
    }
    for name, frames in condition_video_specs.items():
        if len(frames) == 0:
            continue
        save_path = os.path.join(output_dir, f"{name}_{episode_name}.mp4")
        imageio.mimsave(save_path, frames[:save_length], fps=fps)


def inference(args, device, world_size, rank):
    # Keep the same data loading/saving logic as stage-one script with minimal changes.
    mode = args.mode
    if mode == 'offline':
        eval_data_dir = os.path.join(args.data_dir, args.task, 'video_quality')
    elif mode == 'online':
        eval_data_dir = os.path.join(args.data_dir, args.task, 'evaluator')
    else:
        raise ValueError(f"mode {mode} is not supported.")

    episode_list = sorted(os.listdir(eval_data_dir))
    if args.max_episodes is not None:
        episode_list = episode_list[: args.max_episodes]
    data_list = split_data(episode_list, world_size, rank)
    inference_engine = InferenceRNDMixStageTwoEngine(
        args.transformer_model_path,
        device=device,
        mode=mode,
        seed=args.seed,
        da3_model_path=args.da3_model_path,
        da3_urdf_path=args.da3_urdf_path,
        da3_gripper_mesh_dir=args.da3_gripper_mesh_dir,
        da3_device=args.da3_device,
        normalize_metric_depth_frames_min=args.normalize_metric_depth_frames_min,
        normalize_metric_depth_frames_max=args.normalize_metric_depth_frames_max,
        normalize_metric_depth_frames_use_relative=args.normalize_metric_depth_frames_use_relative,
        normalize_metric_depth_frames_use_sqrt=args.normalize_metric_depth_frames_use_sqrt,
        metric_depth_rgb_encoding=args.metric_depth_rgb_encoding,
        metric_depth_rgb_lambda=args.metric_depth_rgb_lambda,
        metric_depth_rgb_c=args.metric_depth_rgb_c,
    )
    if mode == 'online':
        inference_engine.activate_policy(
            args.policy_ckpt_dir,
            args.policy_norm_stats_path,
            args.policy_type,
            args.compile_policy,
        )
        inference_engine.activate_simulator_client(args.simulator_ip, args.simulator_port)
    output_dir = os.path.join(args.output_dir, 'video_quality_eval' if mode == 'offline' else 'evaluator_test', args.task)
    os.makedirs(output_dir, exist_ok=True)
    completed = 0
    for episode_name in data_list:
        save_path = os.path.join(output_dir, f'{episode_name}.mp4')
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            completed += 1
    print(
        f"[rank {rank}/{world_size}] {args.task}: {completed}/{len(data_list)} episodes already done in {output_dir}",
        flush=True,
    )

    for episode_name in data_list:
        episode_dir = os.path.join(eval_data_dir, episode_name)
        if not os.path.isdir(episode_dir):
            continue
        save_path = os.path.join(output_dir, f'{episode_name}.mp4')
        concat_save_path = os.path.join(output_dir, f'concat_{episode_name}.mp4')
        if (
            args.skip_existing
            and os.path.exists(save_path)
            and os.path.getsize(save_path) > 0
        ):
            print(f"Skip existing episode: {save_path}", flush=True)
            continue

        if mode == 'offline':
            cam_high = Image.open(os.path.join(episode_dir, 'cam_high.png')).convert('RGB')
            cam_left_wrist = Image.open(os.path.join(episode_dir, 'cam_left_wrist.png')).convert('RGB')
            cam_right_wrist = Image.open(os.path.join(episode_dir, 'cam_right_wrist.png')).convert('RGB')
            front_replay_images = VideoReader(os.path.join(episode_dir, 'simulator_cam_high.mp4'))
            left_replay_images = VideoReader(os.path.join(episode_dir, 'simulator_cam_left_wrist.mp4'))
            right_replay_images = VideoReader(os.path.join(episode_dir, 'simulator_cam_right_wrist.mp4'))
            action_images = {
                'front_replay': [Image.fromarray(front_replay_images[i].asnumpy()) for i in range(len(front_replay_images))],
                'left_replay': [Image.fromarray(left_replay_images[i].asnumpy()) for i in range(len(left_replay_images))],
                'right_replay': [Image.fromarray(right_replay_images[i].asnumpy()) for i in range(len(right_replay_images))],
            }
            ref_images = {'front': cam_high, 'left': cam_left_wrist, 'right': cam_right_wrist}
            qpos_seq = np.asarray(pickle.load(open(os.path.join(episode_dir, 'traj.pkl'), 'rb')))
            all_output_images, condition_images_dict = inference_engine.wm_inference(ref_images, action_images, qpos_seq)
        else:
            cam_high = Image.open(os.path.join(episode_dir, 'cam_high.png')).convert('RGB')
            cam_left_wrist = Image.open(os.path.join(episode_dir, 'cam_left_wrist.png')).convert('RGB')
            cam_right_wrist = Image.open(os.path.join(episode_dir, 'cam_right_wrist.png')).convert('RGB')
            ref_images = {'front': cam_high, 'left': cam_left_wrist, 'right': cam_right_wrist}
            initial_state = pickle.load(open(os.path.join(episode_dir, 'initial_state.pkl'), 'rb'))
            prompt = json.load(open(os.path.join(episode_dir, 'meta.json')))['prompt']
            all_output_images, condition_images_dict = inference_engine.interaction(
                ref_images,
                initial_state,
                prompt,
                args.max_interactions,
                args.pos_lookahead_step,
            )

        front_depth_condition_images = condition_images_dict['front_depth']
        replay_condition_images = condition_images_dict['replay_condition']
        replay_3d_condition_images = condition_images_dict['replay_3d_condition']
        scene_3d_condition_images = condition_images_dict['scene_3d_condition']
        vis_images = []
        save_length = min(
            len(all_output_images),
            len(front_depth_condition_images),
            len(replay_condition_images),
            len(replay_3d_condition_images),
            len(scene_3d_condition_images),
        )
        for k in range(save_length):
            vis_image = [
                all_output_images[k],
                front_depth_condition_images[k],
                replay_condition_images[k],
                replay_3d_condition_images[k],
                scene_3d_condition_images[k],
            ]
            vis_image = concat_images_grid(vis_image, cols=1, pad=2)
            vis_images.append(vis_image)

        imageio.mimsave(save_path, all_output_images, fps=24)
        imageio.mimsave(concat_save_path, vis_images, fps=24)
        save_condition_videos(output_dir, episode_name, condition_images_dict, save_length, fps=24)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transformer_model_path', type=str, default='wan2.2-5b-diffusers')
    parser.add_argument('--device_list', type=str, default='0,1,2,3')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--mode', type=str, default='offline')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--task', type=str, default='task1')
    parser.add_argument('--output_dir', type=str, default='outputs/rnd_wm_mix_stage_two')
    parser.add_argument('--da3_model_path', type=str, default=None)
    parser.add_argument('--da3_urdf_path', type=str, default=None)
    parser.add_argument('--da3_gripper_mesh_dir', type=str, default=None)
    parser.add_argument('--da3_device', type=str, default='cuda')
    parser.add_argument('--normalize_metric_depth_frames_min', type=float, default=0.08)
    parser.add_argument('--normalize_metric_depth_frames_max', type=float, default=1.2)
    parser.add_argument('--normalize_metric_depth_frames_use_relative', action='store_true')
    parser.add_argument(
        '--normalize_metric_depth_frames_use_sqrt',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument('--metric_depth_rgb_encoding', type=str, default='vision_banana', choices=['linear', 'legacy', 'vision_banana', 'rgb_cube', 'depth2rgb'])
    parser.add_argument('--metric_depth_rgb_lambda', type=float, default=-500.0)
    parser.add_argument('--metric_depth_rgb_c', type=float, default=0.53)
    parser.add_argument('--simulator_ip', type=str, default='127.0.0.1')
    parser.add_argument('--simulator_port', type=str, default='9151')
    parser.add_argument('--policy_ckpt_dir', type=str, default=None)
    parser.add_argument('--policy_norm_stats_path', type=str, default=None)
    parser.add_argument('--policy_type', type=str, default='gigabrain', choices=['gigabrain', 'pi0', 'pi05'])
    parser.add_argument('--compile_policy', action='store_true')
    parser.add_argument('--max_interactions', type=int, default=15)
    parser.add_argument('--pos_lookahead_step', type=int, default=24)
    parser.add_argument('--max_episodes', type=int, default=None)
    parser.add_argument('--skip_existing', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.policy_ckpt_dir is None:
        args.policy_ckpt_dir = model_config[f'cvpr-2026-worldmodel-track-model-{args.task}']

    if args.policy_norm_stats_path is None:
        args.policy_norm_stats_path = os.path.join(
            model_config[f'cvpr-2026-worldmodel-track-model-{args.task}'], 'norm_stat_gigabrain.json'
        )

    devices = args.device_list.split(',')
    multiprocessing.set_start_method('spawn')
    process_list = []
    world_size = len(devices)
    for i in range(world_size):
        device = f'cuda:{devices[i]}'
        process = Process(target=inference, args=(args, device, world_size, i))
        process.start()
        process_list.append(process)
    for process in process_list:
        process.join()

    print('RND-mix stage-two inference done')


"""
python scripts/inference_rnd_mix_stage_two.py \
    --transformer_model_path /path/to/stage_two/transformer \
    --device_list 0 \
    --seed 1024 \
    --mode online \
    --task task1 \
    --output_dir outputs/baseline_rnd_mix_stage_two_task1_0527 \
    --normalize_metric_depth_frames_min 0.08 \
    --normalize_metric_depth_frames_max 1.2 \
    --normalize_metric_depth_frames_use_sqrt \
    --metric_depth_rgb_encoding vision_banana \
    --metric_depth_rgb_lambda -500.0 \
    --metric_depth_rgb_c 0.53 \
    --policy_type gigabrain \
    --simulator_ip 127.0.0.1 \
    --simulator_port 9151 \
    --policy_ckpt_dir /path/to/policy \
    --policy_norm_stats_path /path/to/policy/norm_stat_gigabrain.json
"""

# offline evaluation example:
"""
python scripts/inference_rnd_mix_stage_two.py \
    --transformer_model_path /path/to/stage_two/transformer \
    --device_list 6 \
    --seed 1024 \
    --mode offline \
    --task task4 \
    --output_dir outputs/baseline_rnd_mix_stage_two_0602 \
    --normalize_metric_depth_frames_min 0.08 \
    --normalize_metric_depth_frames_max 1.2 \
    --normalize_metric_depth_frames_use_sqrt \
    --metric_depth_rgb_encoding vision_banana \
    --metric_depth_rgb_lambda -500.0 \
    --metric_depth_rgb_c 0.53 \
    --policy_type gigabrain
"""
