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
from iMac.models import WanConditionRNDMixStageOneModel
from iMac.pipelines import BaselineWMPipeline
from iMac.utils import split_data

# Reuse the stage-one inference data loading / interaction loop.
from inference_stage_one import InferenceStageOneEngine as BaseStageOneEngine
from inference_stage_one import VideoReader, imageio, json, os, pickle, tqdm, inference as stage_one_inference


class InferenceRNDMixStageOneEngine(BaseStageOneEngine):
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
        normalize_metric_depth_frames_min=0.0,
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
        transformer = WanConditionRNDMixStageOneModel.from_pretrained(transformer_model_path).to(dtype)
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

    def wm_inference_per_time(self, replay_condition, ref_image):
        # RND-mix stage-one uses vertically stacked RGB+Depth as the denoising input.
        ref_stacked = ref_image
        if ref_stacked.size[1] == self.dst_size[1]:
            ref_stacked = self._stack_rgb_depth(ref_stacked)

        # replay condition follows training transform behavior:
        # duplicate replay itself along height to match stacked [448, 672] resolution.
        replay_condition_stacked = [self._stack_rgb_depth(frame, frame) for frame in replay_condition]

        output_images = self.pipe(
            replay=replay_condition_stacked,
            depth=replay_condition_stacked,
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
        for step in tqdm(range(wm_inference_time)):
            start = step * self.wm_frame_per_time
            end = (step + 1) * self.wm_frame_per_time + 1
            action_images_chunk = replay_images[start:end]
            replay_condition_chunk = self.build_replay_condition(action_images_chunk, self.wm_frame_per_time + 1)
            output_stacked = self.wm_inference_per_time(replay_condition_chunk, ref_stacked)

            current_rgb = [img.crop((0, 0, self.dst_size[0] * 3, self.dst_size[1])) for img in output_stacked]
            current_depth = [img.crop((0, self.dst_size[1], self.dst_size[0] * 3, self.dst_size[1] * 2)) for img in output_stacked]

            if step == wm_inference_time - 1:
                all_output_images.extend(current_rgb)
                front_depth_condition_images.extend(current_depth)
                replay_condition_images.extend(replay_condition_chunk)
            else:
                all_output_images.extend(current_rgb[:-1])
                front_depth_condition_images.extend(current_depth[:-1])
                replay_condition_images.extend(replay_condition_chunk[:-1])

            ref_stacked = output_stacked[-1]

        condition_images_dict = {
            'front_depth': front_depth_condition_images,
            'replay_condition': replay_condition_images,
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

            state = future_state[-1]
            img_front, img_left, img_right = self.crop_three_view_images(output_images[-1])
            ref_images = {
                'front': img_front,
                'left': img_left,
                'right': img_right,
            }
            assert len(all_output_images) == len(front_depth_condition_images) == len(replay_condition_images)

        condition_images_dict = {
            'front_depth': front_depth_condition_images,
            'replay_condition': replay_condition_images,
        }
        return all_output_images, condition_images_dict


def inference(args, device, world_size, rank):
    # Keep the same data loading/saving logic as stage-one script with minimal changes.
    mode = args.mode
    if mode == 'offline':
        eval_data_dir = os.path.join(args.data_dir, args.task, 'video_quality')
    elif mode == 'online':
        eval_data_dir = os.path.join(args.data_dir, args.task, 'evaluator')
    else:
        raise ValueError(f"mode {mode} is not supported.")

    episode_list = os.listdir(eval_data_dir)
    data_list = split_data(episode_list, world_size, rank)
    inference_engine = InferenceRNDMixStageOneEngine(
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

    for episode_name in data_list:
        episode_dir = os.path.join(eval_data_dir, episode_name)
        if not os.path.isdir(episode_dir):
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
        vis_images = []
        save_length = min(len(all_output_images), len(front_depth_condition_images), len(replay_condition_images))
        for k in range(save_length):
            vis_image = [all_output_images[k], front_depth_condition_images[k], replay_condition_images[k]]
            vis_image = concat_images_grid(vis_image, cols=1, pad=2)
            vis_images.append(vis_image)

        save_path = os.path.join(output_dir, f'{episode_name}.mp4')
        concat_save_path = os.path.join(output_dir, f'concat_{episode_name}.mp4')
        imageio.mimsave(save_path, all_output_images, fps=24)
        imageio.mimsave(concat_save_path, vis_images, fps=24)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transformer_model_path', type=str, default='wan2.2-5b-diffusers')
    parser.add_argument('--device_list', type=str, default='0,1,2,3')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--mode', type=str, default='offline')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--task', type=str, default='task1')
    parser.add_argument('--output_dir', type=str, default='outputs/rnd_wm_mix_stage_one')
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

    print('RND-mix stage-one inference done')


"""
This script performs inference for the RND-mix stage-one model, which generates video predictions based on a single reference frame and replay conditions. The script is designed to run in both offline and online modes, with data loading and saving logic adapted accordingly. The inference engine utilizes a pretrained transformer model and an optional DA3 generator for depth bootstrap. The output includes generated video frames as well as condition images for visualization.

python scripts/inference_rnd_mix_stage_one.py \
--transformer_model_path /path/to/stage_one/transformer \
--mode offline \
--device_list 1,2 \
--task task2 \
--output_dir outputs/baseline_rnd_mix_stage_one_task2_0519 \
--normalize_metric_depth_frames_min 0.08 \
--normalize_metric_depth_frames_max 1.2 \
--normalize_metric_depth_frames_use_sqrt \
--metric_depth_rgb_encoding vision_banana \
--metric_depth_rgb_lambda -500.0 \
--metric_depth_rgb_c 0.53
"""
