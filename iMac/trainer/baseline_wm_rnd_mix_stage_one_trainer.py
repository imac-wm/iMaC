import copy
import functools
import os

import imageio
import torch
import torch.utils.checkpoint as torch_checkpoint
from PIL import ImageDraw
from diffusers.models import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
from giga_train import ModuleDict

from .baseline_wm_stage_one_trainer import BaselineWMStageOneTrainer
from ..models import WanConditionRNDMixStageOneModel


class BaselineWMRNDMixStageOneTrainer(BaselineWMStageOneTrainer):
    transformer_cls = WanConditionRNDMixStageOneModel

    @property
    def transformer(self):
        return functools.partial(self.model, "transformer")

    def get_models(self, model_config):
        pretrained = model_config.pretrained
        self.flow_shift = model_config.flow_shift
        self.ref_aug_strength = 0.1
        self.expand_timesteps = model_config.get("expand_timesteps", False)
        self.view_interval = 100
        self.view_dir = model_config.view_dir
        self.sub_frames = model_config.sub_frames
        self.rollout_step = model_config.rollout
        self.timestep_scale = 1000
        self.enable_rollout_depth_recon_loss = bool(model_config.get("enable_rollout_depth_recon_loss", False))
        self.rollout_depth_recon_loss_weight = float(model_config.get("rollout_depth_recon_loss_weight", 1.0))
        self.rollout_depth_recon_use_checkpoint = bool(model_config.get("rollout_depth_recon_use_checkpoint", True))
        self.rollout_video_save_start_epoch = int(model_config.get("rollout_video_save_start_epoch", -1))

        model = dict()
        vae_pretrained = model_config.get("vae_pretrained", os.path.join(pretrained, "vae"))
        vae_dtype = model.get("vae_dtype", self.dtype)
        vae = AutoencoderKLWan.from_pretrained(vae_pretrained)
        vae.requires_grad_(False)
        vae.to(self.device, dtype=vae_dtype)
        self.vae = vae
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(self.device, dtype=vae_dtype)
        self.latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(self.device, dtype=vae_dtype)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        transformer_pretrained = model_config.transformer_model_path if getattr(model_config, "transformer", None) else os.path.join(pretrained, "transformer")
        if model_config.get("unpretrain", False):
            transformer = self.transformer_cls.from_config(transformer_pretrained, torch_dtype=self.dtype)
        else:
            transformer = self.transformer_cls.from_pretrained(transformer_pretrained, torch_dtype=self.dtype)
            transformer.replay_embedding = copy.deepcopy(transformer.patch_embedding)
        model.update(transformer=transformer)

        checkpoint = model_config.get("checkpoint", None)
        strict = model_config.get("strict", True)
        self.load_checkpoint(checkpoint, list(model.values()), strict=strict)
        model = ModuleDict(model)
        model.train()
        return model

    @staticmethod
    def _split_rgb_depth(video_btchw):
        h = video_btchw.shape[-2] // 2
        return video_btchw[..., :h, :], video_btchw[..., h:, :]

    def _decode_depth_for_recon_loss(self, pred_x0):
        pred_decode_latents = pred_x0.to(self.vae.dtype)
        pred_decode_latents = pred_decode_latents / self.latents_std + self.latents_mean

        def _decode_fn(latents):
            return self.vae.decode(latents, return_dict=False)[0]

        if self.rollout_depth_recon_use_checkpoint and torch.is_grad_enabled():
            pred_video = torch_checkpoint.checkpoint(_decode_fn, pred_decode_latents, use_reentrant=False)
        else:
            pred_video = _decode_fn(pred_decode_latents)
        return pred_video.transpose(1, 2)

    def _save_rollout_debug_video(self, video_btchw, sign, overlay_text=None):
        if video_btchw is None:
            return
        save_dir = os.path.join(self.view_dir, "images", "{}".format(self.cur_step))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "{}.mp4".format(sign))
        tensor_video = video_btchw.permute(0, 2, 1, 3, 4).contiguous()
        video = self.video_processor.postprocess_video(tensor_video, output_type='pil')[0]
        if overlay_text is not None:
            annotated = []
            for frame in video:
                draw = ImageDraw.Draw(frame)
                draw.rectangle((4, 4, 180, 26), fill=(0, 0, 0))
                draw.text((8, 8), overlay_text, fill=(255, 255, 255))
                annotated.append(frame)
            video = annotated
        imageio.mimsave(save_path, video, fps=16)

    def _save_prevae_debug_inputs(self, batch_dict):
        if "front_ref_images" in batch_dict:
            self._save_rollout_debug_video(batch_dict["front_ref_images"].detach(), "input_front_ref_images")
        if "front_images" in batch_dict:
            self._save_rollout_debug_video(batch_dict["front_images"].detach(), "input_front_images_full")
        if "front_depth_images" in batch_dict:
            self._save_rollout_debug_video(batch_dict["front_depth_images"].detach(), "input_front_depth_images")

    def _should_save_rollout_video(self):
        if self.rollout_video_save_start_epoch < 0:
            return False
        if self.cur_epoch < self.rollout_video_save_start_epoch:
            return False
        return self.process_index == 0 and (self.cur_step % self.view_interval == 0 or self.cur_step == 1)

    def rollout(self, batch_dict, idx):
        transformer = self.transformer
        condition = self.prepare_conditioning(batch_dict)
        front_images_x0 = condition["x0"]
        replay_condition_latents = condition["replay_condition_latents"]
        if self._should_save_rollout_video():
            self.vae_decode(front_images_x0, sign=f"input_rollout_step_{idx}")
            self.vae_decode(replay_condition_latents, sign=f"replay_condition_step_{idx}")
        _, sigma = self.get_timestep_and_sigma(front_images_x0.shape[0], front_images_x0.ndim)
        sigma_value = float(sigma.flatten()[0].item())
        noise = torch.randn_like(front_images_x0)
        target = noise - front_images_x0
        noisy_latents = noise * sigma + front_images_x0 * (1 - sigma)
        if self._should_save_rollout_video():
            noisy_video = self.vae_decode(latents=noisy_latents.detach(), return_tensor=True).transpose(1, 2)
            self._save_rollout_debug_video(noisy_video, f"roll_{idx}_noisy_video", overlay_text=f"sigma={sigma_value:.3f}")
        model_pred, pred_x0 = self.denoise_net(
            transformer,
            noisy_latents,
            sigma,
            condition,
            add_ref_aug=True,
            return_x0=True,
        )
        pred_x0 = pred_x0.detach()
        loss = ((model_pred.float() - target.float()) * condition["first_frame_mask"]) ** 2
        return loss, pred_x0

    def forward_step(self, batch_dict):
        if "replay" not in batch_dict and "replay_condition" in batch_dict:
            batch_dict = dict(batch_dict)
            batch_dict["replay"] = batch_dict["replay_condition"]
        loss_dict = super().forward_step(batch_dict)
        if not self.enable_rollout_depth_recon_loss:
            return loss_dict

        # add depth reconstruction loss using decoded rollout target in the mixed (RGB+Depth) frame layout
        condition = self.prepare_conditioning(batch_dict)
        front_images_x0 = condition["x0"]
        _, sigma = self.get_timestep_and_sigma(front_images_x0.shape[0], front_images_x0.ndim)
        noise = torch.randn_like(front_images_x0)
        noisy_latents = noise * sigma + front_images_x0 * (1 - sigma)
        _, pred_x0 = self.denoise_net(self.transformer, noisy_latents, sigma, condition, add_ref_aug=False, return_x0=True)
        pred_video = self._decode_depth_for_recon_loss(pred_x0)
        _, pred_depth = self._split_rgb_depth(pred_video)
        front_depth_images_x0 = batch_dict["front_depth_images"].to(dtype=pred_depth.dtype, device=pred_depth.device)
        depth_recon_loss = (pred_depth.float() - front_depth_images_x0.float()).pow(2).mean()
        loss_dict["depth_recon"] = depth_recon_loss * self.rollout_depth_recon_loss_weight

        if self._should_save_rollout_video():
            self._save_prevae_debug_inputs(batch_dict)
            self._save_rollout_debug_video(pred_video.detach(), "rollout_noisy_video")
        return loss_dict

    def prepare_conditioning(self, batch_dict):
        if "replay_condition" in batch_dict and "replay" not in batch_dict:
            batch_dict = dict(batch_dict)
            batch_dict["replay"] = batch_dict["replay_condition"]
        condition = super().prepare_conditioning(batch_dict)
        if "replay_latents" in condition:
            condition["replay_condition_latents"] = condition.pop("replay_latents")
        if "x0" in condition:
            condition["front_images_x0"] = condition["x0"]
        return condition

    def denoise_net(self, transformer, xt, sigma, condition, add_ref_aug=False, return_x0=False):
        if "replay_condition_latents" in condition and "replay_latents" not in condition:
            condition = dict(condition)
            condition["replay_latents"] = condition["replay_condition_latents"]
        return super().denoise_net(transformer, xt, sigma, condition, add_ref_aug=add_ref_aug, return_x0=return_x0)

    def vae_decode(self, latents=None, images=None, sign=None, return_tensor=False):
        # Avoid the base trainer's unconditional visualization-on-first-step behavior,
        # which is very memory heavy at 448x672 and can trigger OOM.
        if return_tensor:
            latents = latents.to(self.vae.dtype)
            latents = latents / self.latents_std + self.latents_mean
            with torch.no_grad():
                return self.vae.decode(latents, return_dict=False)[0].detach()

        if self._should_save_rollout_video():
            return super().vae_decode(latents=latents, images=images, sign=sign, return_tensor=return_tensor)
        return None
