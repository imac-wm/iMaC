from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

from .tranformer_wan import logger
from .transformer_wan_rnd_mix_stage_one import WanConditionRNDMixStageOneModel


class WanConditionRNDMixStageTwoModel(WanConditionRNDMixStageOneModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_3d_embedding = nn.Conv3d(
            self.config.in_channels,
            self.patch_embedding.out_channels,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        )
        self.scene_3d_embedding = nn.Conv3d(
            self.config.in_channels,
            self.patch_embedding.out_channels,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        front_depth_condition_latents=None,
        replay_condition_latents=None,
        replay_latents=None,
        replay_3d_latents=None,
        scene_3d_latents=None,
        replay_weight: float = 1.0,
        replay_3d_weight: float = 1.0,
        scene_3d_weight: float = 1.0,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        elif attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)
        hidden_states = self.patch_embedding(hidden_states)

        if replay_condition_latents is None and replay_latents is not None:
            replay_condition_latents = replay_latents
        if replay_condition_latents is not None:
            hidden_states = hidden_states + self.replay_embedding(replay_condition_latents) * replay_weight
        if replay_3d_latents is not None:
            hidden_states = hidden_states + self.replay_3d_embedding(replay_3d_latents) * replay_3d_weight
        if scene_3d_latents is not None:
            hidden_states = hidden_states + self.scene_3d_embedding(scene_3d_latents) * scene_3d_weight

        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1)) if ts_seq_len is not None else timestep_proj.unflatten(1, (6, -1))
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        for block in self.blocks:
            hidden_states = (
                self._gradient_checkpointing_func(block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                if (torch.is_grad_enabled() and self.gradient_checkpointing)
                else block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
            )

        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift, scale = shift.squeeze(2), scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
