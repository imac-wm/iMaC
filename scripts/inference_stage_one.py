import paths
import json
import types
from iMac.pipelines import BaselineWMPipeline, GigaBrain0Pipeline
from iMac.model_config import model_config, DATA_DIR
from iMac.policy_utils import get_policy as build_policy
from diffusers import AutoencoderKLWan
import torch
from iMac.models import WanConditionStageOneModel
from iMac.utils import resize_with_pad, split_data
from iMac.image_utils import concat_images_grid
from iMac.sockets import RobotInferenceClient
from PIL import Image
import numpy as np
import argparse
import multiprocessing
import os
from multiprocessing import Process
import pickle
from einops import rearrange
from typing import Any
from tqdm import tqdm
import imageio
from decord import VideoReader


def get_policy(
        ckpt_dir: str,
        tokenizer_model_path: str,
        fast_tokenizer_path: str,
        embodiment_id: int,
        norm_stats_path: str,
        delta_mask: list[bool],
        original_action_dim: int,
        depth_img_prefix_name: str | None = None,
        policy_type: str = 'gigabrain',
        compile_policy: bool = False,
) -> GigaBrain0Pipeline:
    return build_policy(
        ckpt_dir=ckpt_dir,
        tokenizer_model_path=tokenizer_model_path,
        fast_tokenizer_path=fast_tokenizer_path,
        embodiment_id=embodiment_id,
        norm_stats_path=norm_stats_path,
        delta_mask=delta_mask,
        original_action_dim=original_action_dim,
        depth_img_prefix_name=depth_img_prefix_name,
        policy_type=policy_type,
        compile_policy=compile_policy,
    )


def make_infer_data(camera_high, camera_left, camera_right, task_name, qpos):
    assert qpos.shape == (14,)
    camera_high_chw = rearrange(camera_high, 'h w c -> c h w')
    camera_left_chw = rearrange(camera_left, 'h w c -> c h w')
    camera_right_chw = rearrange(camera_right, 'h w c -> c h w')
    observation = {
        'observation.state': torch.tensor(qpos, dtype=torch.float32),
        'observation.images.cam_high': torch.from_numpy(camera_high_chw),
        'observation.images.cam_left_wrist': torch.from_numpy(camera_left_chw),
        'observation.images.cam_right_wrist': torch.from_numpy(camera_right_chw),
        'task': task_name,
    }
    return observation


class InferenceStageOneEngine:
    def __init__(self, transformer_model_path, device, dtype=torch.bfloat16, num_views=3, mode='offline', seed=1024):
        assert mode in ['offline', 'online'], f"mode must be offline or online, but got {mode}"
        torch.cuda.set_device(device)
        device = "cuda"
        print(f"Loading model from {transformer_model_path}")
        model_id = model_config['wan2.2-5b-diffusers']
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
        transformer = WanConditionStageOneModel.from_pretrained(transformer_model_path).to(dtype)
        self.mode = mode
        self.pipe = BaselineWMPipeline.from_pretrained(model_id, vae=vae, transformer=transformer, torch_dtype=dtype)
        self.pipe.to(device)

        self.dst_size = (224, 224)
        self.wm_frame_per_time = 8
        self.num_inference_steps = 30
        self.guidance_scale = 0.0
        self.num_views = num_views
        self.generator = torch.Generator(device=device).manual_seed(seed)


    def activate_policy(self, policy_ckpt_dir, norm_stats_path, policy_type='gigabrain', compile_policy=False):
        tokenizer_model_path = model_config['paligemma']
        fast_tokenizer_path = model_config['fast-tokenizer']
        self.policy = get_policy(
            ckpt_dir=policy_ckpt_dir,
            norm_stats_path=norm_stats_path,
            tokenizer_model_path=tokenizer_model_path,
            fast_tokenizer_path=fast_tokenizer_path,
            embodiment_id=0,
            delta_mask=[True, True, True, True, True, True, False, True, True, True, True, True, True, False],
            original_action_dim=14,
            depth_img_prefix_name=None,
            policy_type=policy_type,
            compile_policy=compile_policy,
        )

    def activate_simulator_client(self, sim_ip, sim_port):
        self.sim_api = RobotInferenceClient(host=sim_ip, port=sim_port)

    def wm_inference_per_time(self, replay, depth, ref_image):
        output_images = self.pipe(
            replay=replay,
            depth=depth,
            height=self.dst_size[1],
            width=self.dst_size[0] * self.num_views,
            num_frames=self.wm_frame_per_time + 1,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            image=ref_image,
            generator=self.generator,
            output_type='pil',
            prompt=''
        ).frames[0]
        return output_images

    def resize_image(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = Image.fromarray(resize_with_pad(image, self.dst_size[0], self.dst_size[1]))
        return image

    def resize_images(self, images):
        images = [self.resize_image(image) for image in images]
        return images

    def build_replay_condition(self, replay_seq, k):
        replay_seq = list(replay_seq)
        if len(replay_seq) == 0:
            raise ValueError("replay_seq must contain at least one frame")
        if len(replay_seq) > k:
            replay_seq = replay_seq[:k]
        elif len(replay_seq) < k:
            replay_seq = replay_seq + [replay_seq[-1]] * (k - len(replay_seq))
        return replay_seq

    def wm_inference(self, ref_images, action_images, qpos_seq):
        img_front = ref_images['front']
        img_left = ref_images['left']
        img_right = ref_images['right']
        img_front = self.resize_image(img_front)
        img_left = self.resize_image(img_left)
        img_right = self.resize_image(img_right)
        ref_image = concat_images_grid([img_front, img_left, img_right], cols=3, pad=0)

        front_replay_images = action_images['front_replay']
        left_replay_images = action_images['left_replay']
        right_replay_images = action_images['right_replay']
        front_replay_images = self.resize_images(front_replay_images)
        left_replay_images = self.resize_images(left_replay_images)
        right_replay_images = self.resize_images(right_replay_images)

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
        depth_condition_images = []
        replay_condition_images = []
        for step in tqdm(range(wm_inference_time)):
            start = step * self.wm_frame_per_time
            end = (step + 1) * self.wm_frame_per_time + 1
            action_images_chunk = replay_images[start:end]
            qpos_chunk = np.asarray(qpos_seq[start:end])
            replay_chunk = self.build_replay_condition(action_images_chunk, self.wm_frame_per_time + 1)
            output_images = self.wm_inference_per_time(replay_chunk, replay_chunk, ref_image)

            if step == wm_inference_time - 1:
                depth_condition_images.extend(replay_chunk)
                replay_condition_images.extend(replay_chunk)
            else:
                output_images = output_images[:-1]
                depth_condition_images.extend(replay_chunk[:-1])
                replay_condition_images.extend(replay_chunk[:-1])
            all_output_images.extend(output_images)

            ref_image = output_images[-1]
            img_front, img_left, img_right = self.crop_three_view_images(ref_image)
            assert len(all_output_images) == len(depth_condition_images) == len(replay_condition_images)

        condition_images_dict = {
            'depth': depth_condition_images,
            'replay': replay_condition_images,
        }
        return all_output_images, condition_images_dict

    def get_action(self, img_front, img_left, img_right, state, task_name):
        if isinstance(img_front, Image.Image):
            img_front = np.array(img_front)
            img_left = np.array(img_left)
            img_right = np.array(img_right)
        img_front = resize_with_pad(img_front, 224, 224).astype(np.float32) / 255.0
        img_left = resize_with_pad(img_left, 224, 224).astype(np.float32) / 255.0
        img_right = resize_with_pad(img_right, 224, 224).astype(np.float32) / 255.0
        obs = make_infer_data(img_front, img_left, img_right, task_name, state)
        action = self.policy.inference(obs)
        return action

    def render_qpos(self, action):
        render_frames = self.sim_api.inference({"action": action})
        front_replay_images = VideoReader(render_frames['sim_front_rgb'])
        left_replay_images = VideoReader(render_frames['sim_left_rgb'])
        right_replay_images = VideoReader(render_frames['sim_right_rgb'])
        front_replay_images = [Image.fromarray(front_replay_images[i].asnumpy()) for i in range(len(front_replay_images))]
        left_replay_images = [Image.fromarray(left_replay_images[i].asnumpy()) for i in range(len(left_replay_images))]
        right_replay_images = [Image.fromarray(right_replay_images[i].asnumpy()) for i in range(len(right_replay_images))]
        return {
            'front_replay': front_replay_images,
            'left_replay': left_replay_images,
            'right_replay': right_replay_images,
        }

    def crop_three_view_images(self, ref_image):
        img_front = ref_image.crop((0, 0, self.dst_size[0], self.dst_size[1]))
        img_left = ref_image.crop((self.dst_size[0], 0, self.dst_size[0] * 2, self.dst_size[1]))
        img_right = ref_image.crop((self.dst_size[0] * 2, 0, self.dst_size[0] * 3, self.dst_size[1]))
        return img_front, img_left, img_right

    def interaction(self, ref_images, state, task, max_interactions=15, pos_lookahead_step=24):
        img_front = ref_images['front']
        img_left = ref_images['left']
        img_right = ref_images['right']
        img_front = self.resize_image(img_front)
        img_left = self.resize_image(img_left)
        img_right = self.resize_image(img_right)
        assert pos_lookahead_step % self.wm_frame_per_time == 0

        depth_condition_images = []
        all_output_images = []
        replay_condition_images = []
        for step in tqdm(range(max_interactions)):
            print("Interaction step {}".format(step))
            actions = self.get_action(img_front, img_left, img_right, state, task)
            actions = actions[:pos_lookahead_step]
            future_state = np.concatenate([state[None, :], actions], axis=0)
            action_images = self.render_qpos(future_state)
            output_images, condition_images_dict = self.wm_inference(ref_images, action_images, future_state)
            all_output_images.extend(output_images)
            depth_condition_images.extend(condition_images_dict['depth'])
            replay_condition_images.extend(condition_images_dict['replay'])

            state = future_state[-1]
            ref_image = output_images[-1]
            img_front, img_left, img_right = self.crop_three_view_images(ref_image)
            ref_images = {
                'front': img_front,
                'left': img_left,
                'right': img_right,
            }
            assert len(all_output_images) == len(depth_condition_images) == len(replay_condition_images)

        condition_images_dict = {
            'depth': depth_condition_images,
            'replay': replay_condition_images,
        }
        return all_output_images, condition_images_dict


def inference(args, device, world_size, rank):
    mode = args.mode
    if mode == 'offline':
        eval_data_dir = os.path.join(args.data_dir, args.task, 'video_quality')
    elif mode == 'online':
        eval_data_dir = os.path.join(args.data_dir, args.task, 'evaluator')
    else:
        raise ValueError(f"mode {mode} is not supported.")

    episode_list = os.listdir(eval_data_dir)
    data_list = split_data(episode_list, world_size, rank)
    inference_engine = InferenceStageOneEngine(args.transformer_model_path, device=device, mode=mode, seed=args.seed)
    if mode == 'online':
        inference_engine.activate_policy(args.policy_ckpt_dir, args.policy_norm_stats_path, args.policy_type, args.compile_policy)
        inference_engine.activate_simulator_client(args.simulator_ip, args.simulator_port)
    output_dir = os.path.join(args.output_dir, 'video_quality_eval' if mode == 'offline' else 'evaluator_test', args.task)
    os.makedirs(output_dir, exist_ok=True)

    for episode_name in data_list:
        episode_dir = os.path.join(eval_data_dir, episode_name)
        print("Episode name: {}".format(episode_name), episode_dir)
        if not os.path.isdir(episode_dir):
            continue

        if mode == 'offline':
            cam_high = Image.open(os.path.join(episode_dir, 'cam_high.png')).convert('RGB')
            cam_left_wrist = Image.open(os.path.join(episode_dir, 'cam_left_wrist.png')).convert('RGB')
            cam_right_wrist = Image.open(os.path.join(episode_dir, 'cam_right_wrist.png')).convert('RGB')

            front_replay_images = VideoReader(os.path.join(episode_dir, 'simulator_cam_high.mp4'))
            left_replay_images = VideoReader(os.path.join(episode_dir, 'simulator_cam_left_wrist.mp4'))
            right_replay_images = VideoReader(os.path.join(episode_dir, 'simulator_cam_right_wrist.mp4'))
            front_replay_images = [Image.fromarray(front_replay_images[i].asnumpy()) for i in range(len(front_replay_images))]
            left_replay_images = [Image.fromarray(left_replay_images[i].asnumpy()) for i in range(len(left_replay_images))]
            right_replay_images = [Image.fromarray(right_replay_images[i].asnumpy()) for i in range(len(right_replay_images))]

            qpos_seq = pickle.load(open(os.path.join(episode_dir, 'traj.pkl'), 'rb'))
            qpos_seq = np.asarray(qpos_seq)

            action_images = {
                'front_replay': front_replay_images,
                'left_replay': left_replay_images,
                'right_replay': right_replay_images,
            }
            ref_images = {
                'front': cam_high,
                'left': cam_left_wrist,
                'right': cam_right_wrist,
            }
            all_output_images, condition_images_dict = inference_engine.wm_inference(ref_images, action_images, qpos_seq)

        elif mode == 'online':
            cam_high = Image.open(os.path.join(episode_dir, 'cam_high.png')).convert('RGB')
            cam_left_wrist = Image.open(os.path.join(episode_dir, 'cam_left_wrist.png')).convert('RGB')
            cam_right_wrist = Image.open(os.path.join(episode_dir, 'cam_right_wrist.png')).convert('RGB')
            ref_images = {
                'front': cam_high,
                'left': cam_left_wrist,
                'right': cam_right_wrist,
            }
            initial_state = pickle.load(open(os.path.join(episode_dir, 'initial_state.pkl'), 'rb'))
            prompt = json.load(open(os.path.join(episode_dir, 'meta.json')))['prompt']
            all_output_images, condition_images_dict = inference_engine.interaction(
                ref_images, initial_state, prompt, args.max_interactions, args.pos_lookahead_step
            )

        depth_condition_images = condition_images_dict['depth']
        replay_condition_images = condition_images_dict['replay']
        vis_images = []
        save_length = min(len(all_output_images), len(depth_condition_images), len(replay_condition_images))
        for k in range(save_length):
            vis_image = [all_output_images[k], depth_condition_images[k], replay_condition_images[k]]
            vis_image = concat_images_grid(vis_image, cols=1, pad=2)
            vis_images.append(vis_image)
        save_path = os.path.join(output_dir, '{}.mp4'.format(episode_name))
        concat_save_path = os.path.join(output_dir, 'concat_{}.mp4'.format(episode_name))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    parser.add_argument('--output_dir', type=str, default='outputs/baseline_3d_wm_stage_one')

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
        rank = i
        process = Process(target=inference, args=(args, device, world_size, rank))
        process.start()
        process_list.append(process)
    for process in process_list:
        process.join()

    print('Inference done')
