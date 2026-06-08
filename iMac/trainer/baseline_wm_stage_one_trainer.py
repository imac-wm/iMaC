import copy
import functools
import os
import random

import torch
from diffusers.models import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
from giga_train import ModuleDict

from .baseline_wm_trainer import BaselineWMTrainer
from ..models import WanConditionWMStageOneModel


class BaselineWMStageOneTrainer(BaselineWMTrainer):
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
            transformer = WanConditionWMStageOneModel.from_config(transformer_pretrained, torch_dtype=self.dtype)
        else:
            transformer = WanConditionWMStageOneModel.from_pretrained(transformer_pretrained, torch_dtype=self.dtype)
            transformer.replay_embedding = copy.deepcopy(transformer.patch_embedding)
        model.update(transformer=transformer)

        checkpoint = model_config.get("checkpoint", None)
        strict = model_config.get("strict", True)
        self.load_checkpoint(checkpoint, list(model.values()), strict=strict)
        model = ModuleDict(model)
        model.train()
        return model

    @property
    def transformer(self):
        return functools.partial(self.model, "transformer")

    def prepare_conditioning(self, batch_dict):
        condition = dict()
        front_images = batch_dict["front_images"]
        replay = batch_dict["replay"]

        latents = self.forward_vae(front_images)
        replay_latents = self.forward_vae(replay)
        num_ref_images = (torch.sum(batch_dict["front_ref_masks"]).int() - 1).item() * 4 + 1
        num_ref_latent_frames = (num_ref_images - 1) // self.vae_scale_factor_temporal + 1
        num_latent_frames = latents.shape[2]
        latent_height = latents.shape[-2]
        latent_width = latents.shape[-1]
        first_frame_mask = torch.ones(
            1, 1, num_latent_frames, latent_height, latent_width, dtype=latents.dtype, device=latents.device
        )
        first_frame_mask[:, :, :num_ref_latent_frames] = 0
        front_ref_images = batch_dict["front_ref_images"][:, :num_ref_images]
        ref_latents = self.forward_vae(front_ref_images)

        condition["ref_latents"] = ref_latents
        condition["replay_latents"] = replay_latents
        condition["first_frame_mask"] = first_frame_mask
        condition["x0"] = latents
        condition["prompt_embeds"] = batch_dict["prompt_embeds"].to(dtype=latents.dtype, device=latents.device)
        return condition

    def denoise_net(self, transformer, xt, sigma, condition, add_ref_aug=False, return_x0=False):
        t = sigma * self.timestep_scale
        ref_latents = condition["ref_latents"]
        first_frame_mask = condition["first_frame_mask"]
        prompt_embeds = condition["prompt_embeds"]
        replay_latents = condition["replay_latents"]
        if add_ref_aug:
            noisy_ref_latents = torch.randn_like(ref_latents)
            aug_noise = random.random() * self.ref_aug_strength
            ref_latents = ref_latents + aug_noise * noisy_ref_latents
        input_noisy_latents = (1 - first_frame_mask) * ref_latents + first_frame_mask * xt
        temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
        timestep = temp_ts.unsqueeze(0).expand(xt.shape[0], -1)
        input_noisy_latents = input_noisy_latents.to(self.dtype)
        model_pred = transformer(
            hidden_states=input_noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            replay_latents=replay_latents,
        )[0]
        if return_x0:
            pred_x0 = xt - model_pred * sigma
            pred_x0 = self.repare_first_frame(pred_x0, condition)
            return model_pred, pred_x0
        return model_pred, None

    def rollout(self, batch_dict, idx):
        transformer = self.transformer
        condition = self.prepare_conditioning(batch_dict)
        latents = condition["x0"]
        replay_latents = condition["replay_latents"]
        self.vae_decode(latents, sign=f"input_rollout_step_{idx}")
        self.vae_decode(replay_latents, sign=f"replay_step_{idx}")
        _, sigma = self.get_timestep_and_sigma(latents.shape[0], latents.ndim)
        noise = torch.randn_like(latents)
        target = noise - latents
        noisy_latents = noise * sigma + latents * (1 - sigma)
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
        sub_latents = self.sub_frames // self.vae_scale_factor_temporal + 1
        front_ref_masks = batch_dict["front_ref_masks"][:, :, :sub_latents]
        front_ref_images = batch_dict["front_ref_images"][:, : self.sub_frames + 1]
        front_images = batch_dict["front_images"]
        replay = batch_dict["replay"]
        prompt_embeds = batch_dict["prompt_embeds"]
        num_ref_images = (torch.sum(batch_dict["front_ref_masks"]).int() - 1).item() * 4 + 1
        loss_dict = {}
        loss_weight_per_roll = 1.0 / self.rollout_step
        for i in range(self.rollout_step):
            start_frame = i * self.sub_frames
            end_frame = (i + 1) * self.sub_frames + 1
            roll_dict = {
                "front_ref_masks": front_ref_masks,
                "front_ref_images": front_ref_images,
                "prompt_embeds": prompt_embeds,
                "front_images": front_images[:, start_frame:end_frame],
                "replay": replay[:, start_frame:end_frame],
            }
            loss, pred_x0 = self.rollout(roll_dict, i)
            loss_dict[f"roll_{i}"] = loss * loss_weight_per_roll
            with torch.no_grad():
                tensor_video = self.vae_decode(latents=pred_x0, sign=f"rollout_step_{i}", return_tensor=True)
                front_ref_images[:, :num_ref_images] = tensor_video.transpose(1, 2)[:, -num_ref_images:]
        return loss_dict
