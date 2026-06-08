import copy
import os

from giga_train.utils import load_state_dict

from .baseline_wm_rnd_mix_stage_one_trainer import BaselineWMRNDMixStageOneTrainer
from ..models import WanConditionRNDMixStageTwoModel
from ..utils import resize_with_pad
import numpy as np
import torch
import torch.nn as nn
from third_party.func_flow_control_urdf import CondGenerator


class BaselineWMRNDMixStageTwoTrainer(BaselineWMRNDMixStageOneTrainer):
    transformer_cls = WanConditionRNDMixStageTwoModel

    def _resolve_stage_one_transformer_checkpoint(self, model_config):
        stage_one_model_dir = model_config.get("stage_one_model_dir", None)
        stage_one_project_dir = model_config.get("stage_one_project_dir", None)
        stage_one_checkpoint = model_config.get("stage_one_checkpoint", None)
        if stage_one_model_dir is None and stage_one_project_dir is None and stage_one_checkpoint is None:
            return None

        def ensure_transformer_dir(path):
            if path is None:
                return None
            path = os.path.normpath(path)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
                return path
            transformer_dir = os.path.join(path, "transformer")
            if os.path.isdir(transformer_dir) and os.path.exists(os.path.join(transformer_dir, "config.json")):
                return transformer_dir
            raise FileNotFoundError(f"Cannot resolve transformer checkpoint from: {path}")

        def find_checkpoint_root(path):
            if path is None:
                return None
            path = os.path.normpath(path)
            direct_root = os.path.isdir(path) and any(name.startswith("checkpoint") for name in os.listdir(path))
            if direct_root:
                return path
            model_root = os.path.join(path, "model")
            nested_root = os.path.isdir(model_root) and any(name.startswith("checkpoint") for name in os.listdir(model_root))
            if nested_root:
                return model_root
            return None

        if stage_one_checkpoint is not None and os.path.isabs(stage_one_checkpoint):
            return ensure_transformer_dir(stage_one_checkpoint)

        checkpoint_root = find_checkpoint_root(stage_one_model_dir)
        if checkpoint_root is None:
            checkpoint_root = find_checkpoint_root(stage_one_project_dir)

        if checkpoint_root is not None:
            if stage_one_checkpoint is None:
                checkpoints = [d for d in os.listdir(checkpoint_root) if d.startswith("checkpoint")]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))
                if len(checkpoints) == 0:
                    raise FileNotFoundError(f"No stage-one checkpoint found under {checkpoint_root}")
                return ensure_transformer_dir(os.path.join(checkpoint_root, checkpoints[-1]))
            if stage_one_checkpoint.startswith("checkpoint"):
                return ensure_transformer_dir(os.path.join(checkpoint_root, stage_one_checkpoint))

        return ensure_transformer_dir(stage_one_checkpoint)

    def _load_stage_one_weights(self, transformer, model_config):
        checkpoint_dir = self._resolve_stage_one_transformer_checkpoint(model_config)
        if checkpoint_dir is None:
            return transformer
        stage_one_strict = model_config.get("stage_one_strict", False)
        self.logger.info(f"Load stage-one RND-mix weights from {checkpoint_dir}")
        state_dict = load_state_dict(checkpoint_dir)
        message = transformer.load_state_dict(state_dict, strict=stage_one_strict)
        if self.is_main_process and not stage_one_strict:
            self.logger.info(message)
        return transformer

    def _materialize_meta_parameters(self, module):
        for child in module.modules():
            for name, param in list(child._parameters.items()):
                if param is None or not getattr(param, "is_meta", False):
                    continue
                child._parameters[name] = torch.nn.Parameter(
                    torch.zeros(param.shape, dtype=param.dtype, device="cpu"),
                    requires_grad=param.requires_grad,
                )

    def _build_zero_conv_like_patch(self, transformer):
        patch_embedding = transformer.patch_embedding
        conv = nn.Conv3d(
            in_channels=patch_embedding.in_channels,
            out_channels=patch_embedding.out_channels,
            kernel_size=patch_embedding.kernel_size,
            stride=patch_embedding.stride,
            padding=patch_embedding.padding,
            dilation=patch_embedding.dilation,
            groups=patch_embedding.groups,
            bias=patch_embedding.bias is not None,
            padding_mode=patch_embedding.padding_mode,
            device=patch_embedding.weight.device,
            dtype=patch_embedding.weight.dtype,
        )
        nn.init.zeros_(conv.weight)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)
        return conv

    def _init_3d_embedding_by_mode(self, transformer, mode):
        mode = str(mode).lower()
        if mode in ("copy_replay_image", "copy_replay", "copy"):
            return copy.deepcopy(transformer.replay_embedding)
        if mode in ("zero", "zeros", "all_zero"):
            return self._build_zero_conv_like_patch(transformer)
        raise ValueError(
            f"Unsupported 3D condition init mode: {mode}. "
            "Please use one of [copy_replay_image, zero]."
        )

    def _materialize_new_3d_modules(self, transformer):
        self._materialize_meta_parameters(transformer)
        transformer.replay_3d_embedding = self._init_3d_embedding_by_mode(transformer, self.replay_3d_cond_init)
        transformer.scene_3d_embedding = self._init_3d_embedding_by_mode(transformer, self.scene_3d_cond_init)
        return transformer

    def get_models(self, model_config):
        self.replay_3d_transition_epochs = model_config.get("replay_3d_transition_epochs", 20)
        self.cond_modality = model_config.get("cond_modality", "3D")
        self.cond_name = model_config.get("cond_name", "gripper_interact")
        self.replay_3d_cond_init = model_config.get("replay_3d_cond_init", "copy_replay_image")
        self.scene_3d_cond_init = model_config.get("scene_3d_cond_init", "zero")
        self.use_start_end_ref_image = model_config.get("use_start_end_ref_image", False)
        generator_cfg = model_config.get("generator", {})
        depth_point2double_cfg = generator_cfg.get("depth_point2double_cond", {})
        self.depth_point2double_cond_kwargs = {
            "use_gpu": depth_point2double_cfg.get("use_gpu", False),
        }
        self.gpu_dist_chunk_size = int(depth_point2double_cfg.get("gpu_dist_chunk_size", 1024))
        self.load_da3_model = bool(generator_cfg.get("load_da3_model", False))
        self.cond_generator_paths = {
            key: generator_cfg.get(key)
            for key in ("model_path", "urdf_path", "gripper_mesh_dir")
            if generator_cfg.get(key)
        }
        self.depth_norm_min = float(model_config.get("normalize_metric_depth_frames_min", 0.0))
        self.depth_norm_max = float(model_config.get("normalize_metric_depth_frames_max", 1.2))
        self.depth_norm_use_sqrt = bool(model_config.get("normalize_metric_depth_frames_use_sqrt", False))
        self.depth_norm_use_relative = bool(model_config.get("normalize_metric_depth_frames_use_relative", False))
        self.metric_depth_rgb_encoding = str(model_config.get("metric_depth_rgb_encoding", "linear"))
        self.metric_depth_rgb_lambda = float(model_config.get("metric_depth_rgb_lambda", -500.0))
        self.metric_depth_rgb_c = float(model_config.get("metric_depth_rgb_c", 0.53))
        model = super().get_models(model_config)
        self._materialize_new_3d_modules(model["transformer"])
        cond_generator_device = str(self.device) if self.depth_point2double_cond_kwargs["use_gpu"] else "cpu"
        self.cond_generator = CondGenerator(
            device=cond_generator_device,
            gpu_dist_chunk_size=self.gpu_dist_chunk_size,
            load_da3_model=self.load_da3_model,
            **self.cond_generator_paths,
        )
        self._load_stage_one_weights(model["transformer"], model_config)
        return model

    def _split_views_video(self, video_chw):
        one_w = video_chw.shape[-1] // 3
        one_h = video_chw.shape[-2] // 2
        rgb = video_chw[..., :one_h, :]
        depth = video_chw[..., one_h:, :]
        rgb_front, rgb_left, rgb_right = rgb[..., :one_w], rgb[..., one_w : 2 * one_w], rgb[..., 2 * one_w : 3 * one_w]
        depth_front, depth_left, depth_right = (
            depth[..., :one_w],
            depth[..., one_w : 2 * one_w],
            depth[..., 2 * one_w : 3 * one_w],
        )
        return (rgb_front, rgb_left, rgb_right), (depth_front, depth_left, depth_right)

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

    def _denorm_depth_to_metric(self, depth_chw):
        # inverse of _normalize_rgb_like_frames: [-1,1] -> [0,255]
        depth_u8_chw = ((depth_chw.float() * 0.5) + 0.5) * 255.0
        depth_u8_chw = depth_u8_chw.clamp(0.0, 255.0)
        if self.metric_depth_rgb_encoding in ("vision_banana", "rgb_cube", "depth2rgb"):
            depth_rgb = depth_u8_chw.permute(1, 2, 0).cpu().numpy().astype(np.float32)
            metric = self._vision_banana_rgb_to_metric(
                depth_rgb,
                lambda_param=self.metric_depth_rgb_lambda,
                c=self.metric_depth_rgb_c,
            )
            metric = np.nan_to_num(metric, nan=self.depth_norm_min, posinf=self.depth_norm_max, neginf=self.depth_norm_min)
            metric = np.clip(metric, self.depth_norm_min, self.depth_norm_max)
            return metric.astype(np.float32)
        if self.metric_depth_rgb_encoding not in ("linear", "legacy"):
            raise ValueError(f"Unsupported metric depth RGB encoding: {self.metric_depth_rgb_encoding}")

        depth_u8 = depth_u8_chw.mean(dim=0)
        # inverse of _normalize_metric_depth_frames:
        #   uint8 -> [0,1] normalized depth -> metric depth
        depth_norm = (depth_u8 / 255.0).cpu().numpy().astype(np.float32)
        if self.depth_norm_use_relative:
            raise ValueError("Cannot uniquely invert relative depth normalization during stage-two conditioning.")
        if self.depth_norm_use_sqrt:
            metric = np.square(depth_norm) * max(self.depth_norm_max, 1e-8)
        else:
            metric = depth_norm * max(self.depth_norm_max, 1e-8)
        metric = np.clip(metric, self.depth_norm_min, self.depth_norm_max)
        return metric.astype(np.float32)

    def _format_3d_condition_frames(self, cond_video, k, target_h, target_w):
        one_h = target_h // 2
        one_w = target_w // 3
        frames = []
        for views in cond_video[:k]:
            padded_views = []
            for view in views[:3]:
                view = np.asarray(view)
                if view.ndim == 2:
                    view = np.repeat(view[..., None], 3, axis=-1)
                padded_views.append(resize_with_pad(view.astype(np.uint8), one_h, one_w))
            while len(padded_views) < 3:
                padded_views.append(np.zeros((one_h, one_w, 3), dtype=np.uint8))
            three_view = np.concatenate(padded_views, axis=1)
            frames.append(np.concatenate([three_view, three_view], axis=0))
        return frames

    def _build_3d_conditions(self, current_ref, qpos_seq, replay_seq, k):
        (cur_front, cur_left, cur_right), (dep_front, dep_left, dep_right) = self._split_views_video(current_ref)
        bsz = qpos_seq.shape[0]
        replay_3d_batches, scene_3d_batches = [], []
        target_h, target_w = current_ref.shape[-2], current_ref.shape[-1]
        for b in range(bsz):
            current_obs = [
                ((cur_front[b] + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy(),
                ((cur_left[b] + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy(),
                ((cur_right[b] + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy(),
            ]
            current_depth = [
                self._denorm_depth_to_metric(dep_front[b]),
                self._denorm_depth_to_metric(dep_left[b]),
                self._denorm_depth_to_metric(dep_right[b]),
            ]
            qpos_np = qpos_seq[b].detach().cpu().numpy()
            current_future_action = qpos_np
            if current_future_action.shape[0] > k:
                current_future_action = current_future_action[:k]
            elif current_future_action.shape[0] < k:
                pad = np.repeat(current_future_action[-1:], k - current_future_action.shape[0], axis=0)
                current_future_action = np.concatenate([current_future_action, pad], axis=0)
            cond_video1, cond_video2, _ = self.cond_generator.depth_point2double_cond(
                current_depth=current_depth,
                current_future_action=current_future_action,
                current_obs=current_obs,
                **self.depth_point2double_cond_kwargs,
            )
            if len(cond_video1) == 0:
                cond_video1 = [[np.zeros_like(current_obs[0]), np.zeros_like(current_obs[1]), np.zeros_like(current_obs[2])]]
            if len(cond_video2) == 0:
                cond_video2 = [[np.zeros_like(current_obs[0]), np.zeros_like(current_obs[1]), np.zeros_like(current_obs[2])]]
            cond_video1 = cond_video1[:k]
            cond_video2 = cond_video2[:k]
            if len(cond_video1) < k:
                cond_video1 += [cond_video1[-1]] * (k - len(cond_video1))
            if len(cond_video2) < k:
                cond_video2 += [cond_video2[-1]] * (k - len(cond_video2))
            replay_frames = self._format_3d_condition_frames(cond_video1, k, target_h, target_w)
            scene_frames = self._format_3d_condition_frames(cond_video2, k, target_h, target_w)
            replay_tensor = torch.from_numpy(np.stack(replay_frames)).permute(0, 3, 1, 2).float() / 127.5 - 1.0
            scene_tensor = torch.from_numpy(np.stack(scene_frames)).permute(0, 3, 1, 2).float() / 127.5 - 1.0
            replay_3d_batches.append(replay_tensor)
            scene_3d_batches.append(scene_tensor)
        replay_3d = torch.stack(replay_3d_batches, dim=0).to(current_ref.device, dtype=current_ref.dtype)
        scene_3d = torch.stack(scene_3d_batches, dim=0).to(current_ref.device, dtype=current_ref.dtype)
        return replay_3d, scene_3d

    def _get_replay_3d_schedule(self):
        current_epoch = float(getattr(self, "cur_epoch", 0))
        progress = min(max(current_epoch / float(max(self.replay_3d_transition_epochs, 1)), 0.0), 1.0)
        return 1.0, progress

    def _set_ref_frames(self, ref_frames, latest_frames, num_ref_images, first_ref_frame=None):
        if self.use_start_end_ref_image and first_ref_frame is not None and num_ref_images > 1:
            ref_frames[:, :1] = first_ref_frame
            ref_frames[:, 1:num_ref_images] = latest_frames[:, -(num_ref_images - 1):]
        else:
            ref_frames[:, :num_ref_images] = latest_frames[:, -num_ref_images:]
        return ref_frames

    def _get_current_ref_frame(self, ref_frames, num_ref_images):
        if num_ref_images <= 0:
            raise ValueError("num_ref_images must be positive.")
        return ref_frames[:, num_ref_images - 1]

    def forward_step(self, batch_dict):
        sub_latents = self.sub_frames // self.vae_scale_factor_temporal + 1
        front_ref_masks = batch_dict["front_ref_masks"][:, :, :sub_latents]
        front_ref_images = batch_dict["front_ref_images"][:, : self.sub_frames + 1]
        front_images = batch_dict["front_images"]
        replay = batch_dict.get("replay_condition", batch_dict.get("replay"))
        qpos = batch_dict["qpos"]
        prompt_embeds = batch_dict["prompt_embeds"]
        num_ref_images = (torch.sum(batch_dict["front_ref_masks"]).int() - 1).item() * 4 + 1
        start_ref_image = front_ref_images[:, :1].clone()
        replay_w, replay3d_w = self._get_replay_3d_schedule()
        loss_dict = {}
        loss_weight_per_roll = 1.0 / self.rollout_step
        for i in range(self.rollout_step):
            start_frame, end_frame = i * self.sub_frames, (i + 1) * self.sub_frames + 1
            replay_slice = replay[:, start_frame:end_frame]
            with torch.no_grad():
                current_ref = self._get_current_ref_frame(front_ref_images, num_ref_images)
                replay_3d_cond, scene_3d_cond = self._build_3d_conditions(
                    current_ref,
                    qpos[:, start_frame:end_frame],
                    replay_slice,
                    self.sub_frames + 1,
                )
            roll_dict = {
                "front_ref_masks": front_ref_masks,
                "front_ref_images": front_ref_images,
                "prompt_embeds": prompt_embeds,
                "front_images": front_images[:, start_frame:end_frame],
                "replay": replay_slice,
                "replay_3d_cond": replay_3d_cond,
                "scene_3d_cond": scene_3d_cond,
            }
            condition = self.prepare_conditioning(roll_dict)
            condition["replay_3d_latents"] = self.forward_vae(roll_dict["replay_3d_cond"])
            condition["scene_3d_latents"] = self.forward_vae(roll_dict["scene_3d_cond"])
            condition["replay_weight"] = replay_w
            condition["replay_3d_weight"] = replay3d_w
            condition["scene_3d_weight"] = 1.0
            latents = condition["x0"]
            _, sigma = self.get_timestep_and_sigma(latents.shape[0], latents.ndim)
            noise = torch.randn_like(latents)
            target = noise - latents
            noisy_latents = noise * sigma + latents * (1 - sigma)
            model_pred, pred_x0 = self.denoise_net(self.transformer, noisy_latents, sigma, condition, add_ref_aug=True, return_x0=True)
            loss_dict[f"roll_{i}"] = (((model_pred.float() - target.float()) * condition["first_frame_mask"]) ** 2) * loss_weight_per_roll
            with torch.no_grad():
                tensor_video = self.vae_decode(latents=pred_x0.detach(), sign=f"rollout_step_{i}", return_tensor=True)
                front_ref_images = self._set_ref_frames(front_ref_images, tensor_video.transpose(1, 2), num_ref_images, first_ref_frame=start_ref_image)
        return loss_dict

    def denoise_net(self, transformer, xt, sigma, condition, add_ref_aug=False, return_x0=False):
        t = sigma * self.timestep_scale
        ref_latents = condition["ref_latents"]
        first_frame_mask = condition["first_frame_mask"]
        prompt_embeds = condition["prompt_embeds"]
        replay_condition_latents = condition.get("replay_condition_latents", condition.get("replay_latents", None))
        if replay_condition_latents is None:
            raise KeyError("condition must contain replay_condition_latents or replay_latents")
        if add_ref_aug:
            ref_latents = ref_latents + (torch.randn_like(ref_latents) * (torch.rand(1).item() * self.ref_aug_strength))
        input_noisy_latents = (1 - first_frame_mask) * ref_latents + first_frame_mask * xt
        temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
        timestep = temp_ts.unsqueeze(0).expand(xt.shape[0], -1)
        model_pred = transformer(
            hidden_states=input_noisy_latents.to(self.dtype),
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            replay_condition_latents=replay_condition_latents,
            replay_3d_latents=condition.get("replay_3d_latents", None),
            scene_3d_latents=condition.get("scene_3d_latents", None),
            replay_weight=float(condition.get("replay_weight", 1.0)),
            replay_3d_weight=float(condition.get("replay_3d_weight", 0.0)),
            scene_3d_weight=float(condition.get("scene_3d_weight", 1.0)),
        )[0]
        if return_x0:
            pred_x0 = self.repare_first_frame(xt - model_pred * sigma, condition)
            return model_pred, pred_x0
        return model_pred, None
