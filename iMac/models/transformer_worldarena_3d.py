import torch
import torch.nn as nn

from .transformer_wan_wae import WanTransformer3DModel_depth


class WanTransformer3DModelWorldArena3D(WanTransformer3DModel_depth):
    """WorldArena 3D condition variant with scene/replay 3D embeddings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        in_channels = self.config.in_channels
        patch_size = self.config.patch_size

        if hasattr(self, "depth_embedding"):
            del self.depth_embedding

        self.replay_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.scene_3d_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.replay_3d_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image=None,
        return_dict: bool = True,
        attention_kwargs=None,
        replay_latents=None,
        scene_3d_latents=None,
        replay_3d_latents=None,
        scene_3d_weight: float = 1.0,
        replay_3d_weight: float = 1.0,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            attention_kwargs.pop("scale", 1.0)

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        if replay_latents is not None:
            hidden_states = hidden_states + self.replay_embedding(replay_latents)
        if scene_3d_latents is not None:
            hidden_states = hidden_states + self.scene_3d_embedding(scene_3d_latents) * scene_3d_weight
        if replay_3d_latents is not None:
            hidden_states = hidden_states + self.replay_3d_embedding(replay_3d_latents) * replay_3d_weight

        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)
        return {"sample": output}
