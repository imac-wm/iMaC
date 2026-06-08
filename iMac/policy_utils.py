import json
import types
from typing import Any, Literal

import torch

from .pipelines import GigaBrain0Pipeline, Pi0Pipeline

PolicyType = Literal["gigabrain", "pi0", "pi05"]


def _load_norm_stats(norm_stats_path: str) -> dict[str, Any]:
    with open(norm_stats_path, "r") as f:
        return json.load(f)["norm_stats"]


def _get_norm_stats_entry(norm_stats_data: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        if key in norm_stats_data:
            return norm_stats_data[key]
    available_keys = ", ".join(sorted(norm_stats_data.keys()))
    raise KeyError(f"None of norm stats keys {keys} were found. Available keys: {available_keys}")


def _bind_inference_method(pipe: GigaBrain0Pipeline | Pi0Pipeline) -> GigaBrain0Pipeline | Pi0Pipeline:
    def inference(self, data: dict[str, Any]) -> torch.Tensor:
        images = {
            "observation.images.cam_high": data["observation.images.cam_high"],
            "observation.images.cam_left_wrist": data["observation.images.cam_left_wrist"],
            "observation.images.cam_right_wrist": data["observation.images.cam_right_wrist"],
        }
        if getattr(self, "enable_depth_img", False) and "observation.depth_images.cam_high" in data:
            images["observation.depth_images.cam_high"] = data["observation.depth_images.cam_high"]
        if getattr(self, "enable_depth_img", False) and "observation.depth_images.cam_left_wrist" in data:
            images["observation.depth_images.cam_left_wrist"] = data["observation.depth_images.cam_left_wrist"]
        if getattr(self, "enable_depth_img", False) and "observation.depth_images.cam_right_wrist" in data:
            images["observation.depth_images.cam_right_wrist"] = data["observation.depth_images.cam_right_wrist"]

        return self(images, data["task"], data["observation.state"])

    pipe.inference = types.MethodType(inference, pipe)
    return pipe


def get_policy(
    ckpt_dir: str,
    tokenizer_model_path: str,
    norm_stats_path: str,
    original_action_dim: int,
    policy_type: PolicyType = "gigabrain",
    device: str | torch.device = "cuda",
    fast_tokenizer_path: str | None = None,
    embodiment_id: int = 0,
    delta_mask: list[bool] | None = None,
    depth_img_prefix_name: str | None = None,
    compile_policy: bool = False,
) -> GigaBrain0Pipeline | Pi0Pipeline:
    norm_stats_data = _load_norm_stats(norm_stats_path)
    if policy_type in {"pi0", "pi05"}:
        pipe = Pi0Pipeline(
            model_path=ckpt_dir,
            tokenizer_model_path=tokenizer_model_path,
            state_norm_stats=_get_norm_stats_entry(norm_stats_data, "observation.state", "state"),
            action_norm_stats=_get_norm_stats_entry(norm_stats_data, "action", "actions"),
            original_action_dim=original_action_dim,
        )
    elif policy_type == "gigabrain":
        if fast_tokenizer_path is None:
            raise ValueError("fast_tokenizer_path is required for GigaBrain0Pipeline")
        if delta_mask is None:
            raise ValueError("delta_mask is required for GigaBrain0Pipeline")
        pipe = GigaBrain0Pipeline(
            model_path=ckpt_dir,
            tokenizer_model_path=tokenizer_model_path,
            fast_tokenizer_path=fast_tokenizer_path,
            embodiment_id=embodiment_id,
            state_norm_stats=norm_stats_data["observation.state"],
            action_norm_stats=norm_stats_data["action"],
            delta_mask=delta_mask,
            original_action_dim=original_action_dim,
            depth_img_prefix_name=depth_img_prefix_name,
        )
    else:
        raise ValueError(f"Unsupported policy_type: {policy_type}")

    pipe.to(device)
    if compile_policy:
        pipe.compile()
    return _bind_inference_method(pipe)
