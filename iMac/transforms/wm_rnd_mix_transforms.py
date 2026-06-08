import copy
import random
import subprocess

import numpy as np
import torch
from decord import VideoReader
from decord import VideoReader as DecordVideoReader
from giga_datasets import video_utils
from torchvision import transforms

from giga_train import TRANSFORMS

from ..model_config import DEFAULT_PROMPT_EMBEDDING_PATH
from ..utils import resize_with_pad
from .wm_transforms import MaskGenerator


@TRANSFORMS.register
class WMRNDMixTransforms:
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
        _normalize_metric_depth_frames_min=0.0,
        _normalize_metric_depth_frames_max=1.2,
        _depth_is_metric=True,
        _normalize_metric_depth_frames_use_relative=False,
        _normalize_metric_depth_frames_use_sqrt=False,
        _metric_depth_rgb_encoding="linear",
        _metric_depth_rgb_lambda=-500.0,
        _metric_depth_rgb_c=0.53,
        _debug_fixed_sampling=False,
        _debug_sample_start=0,
        _debug_sample_stride=1,
    ):
        self.fps = fps
        self.is_train = is_train
        self.normalize = transforms.Normalize([0.5], [0.5])
        self.dst_size = dst_size
        self.num_frames = num_frames
        self.image_cfg = image_cfg
        self.mask_generator = MaskGenerator(**image_cfg["mask_generator"])
        self.max_stride = max_stride
        self.sub_frames = sub_frames
        self.default_prompt_embeds = torch.load(DEFAULT_PROMPT_EMBEDDING_PATH)["prompt_embeds"]
        self.num_views = num_views
        self.normalize_metric_depth_frames_min = float(_normalize_metric_depth_frames_min)
        self.normalize_metric_depth_frames_max = float(_normalize_metric_depth_frames_max)
        self.depth_is_metric = bool(_depth_is_metric)
        self.normalize_metric_depth_frames_use_relative = bool(_normalize_metric_depth_frames_use_relative)
        self.normalize_metric_depth_frames_use_sqrt = bool(_normalize_metric_depth_frames_use_sqrt)
        self.metric_depth_rgb_encoding = str(_metric_depth_rgb_encoding)
        self.metric_depth_rgb_lambda = float(_metric_depth_rgb_lambda)
        self.metric_depth_rgb_c = float(_metric_depth_rgb_c)
        self.debug_fixed_sampling = bool(_debug_fixed_sampling)
        self.debug_sample_start = int(_debug_sample_start)
        self.debug_sample_stride = max(1, int(_debug_sample_stride))

    @staticmethod
    def _normalize_rgb_like_frames(input_images):
        input_images = np.asarray(input_images)
        if input_images.dtype != np.uint8:
            input_images = np.clip(np.round(input_images), 0.0, 255.0).astype(np.uint8)
        # avoid double padding when frames are already at target resolution
        if input_images.shape[1] != 224 or input_images.shape[2] != 224:
            input_images = resize_with_pad(input_images, 224, 224)
        input_images = torch.from_numpy(input_images).permute(0, 3, 1, 2).contiguous()
        return (input_images / 255.0 - 0.5) / 0.5

    @staticmethod
    def _metric_depth_to_vision_banana_rgb(depth_frames, lambda_param=-500.0, c=0.53):
        if lambda_param >= -1:
            raise ValueError("Vision Banana metric-depth RGB encoding requires lambda < -1.")
        if c <= 0:
            raise ValueError("Vision Banana metric-depth RGB encoding requires c > 0.")

        valid = np.isfinite(depth_frames) & (depth_frames >= 0)
        clipped = np.where(valid, depth_frames, 0.0).astype(np.float32)
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
        scaled = normalized[..., 0] * (len(vertices) - 1)
        index = np.minimum(np.floor(scaled).astype(np.int32), len(vertices) - 2)
        frac = (scaled - index)[..., None]
        rgb = vertices[index] * (1.0 - frac) + vertices[index + 1] * frac
        rgb_u8 = np.rint(rgb * 255.0).astype(np.uint8)
        rgb_u8[~valid[..., 0]] = 0
        return rgb_u8

    def _normalize_metric_depth_frames(self, depth_frames):
        depth_frames = np.asarray(depth_frames)
        if depth_frames.ndim == 3:
            depth_frames = depth_frames[..., None]
        if depth_frames.shape[-1] > 1:
            depth_frames = depth_frames[..., :1]
        depth_frames = depth_frames.astype(np.float32)

        # DA3 depth videos are written by process_depth_DA3.py as uint16 millimeters.
        # Convert to meters before feeding downstream modules, then map to [0, 1].
        if np.nanmax(depth_frames) > 65.535:
            depth_frames = depth_frames / 1000.0
        if self.metric_depth_rgb_encoding in ("vision_banana", "rgb_cube", "depth2rgb"):
            depth_rgb = self._metric_depth_to_vision_banana_rgb(
                depth_frames,
                lambda_param=self.metric_depth_rgb_lambda,
                c=self.metric_depth_rgb_c,
            )
            return WMRNDMixTransforms._normalize_rgb_like_frames(depth_rgb)
        if self.metric_depth_rgb_encoding not in ("linear", "legacy"):
            raise ValueError(f"Unsupported metric depth RGB encoding: {self.metric_depth_rgb_encoding}")

        if self.normalize_metric_depth_frames_use_relative:
            d_min = np.nanmin(depth_frames)
            d_max = np.nanmax(depth_frames)
            if not np.isfinite(d_min) or not np.isfinite(d_max) or d_max <= d_min:
                depth_frames = np.zeros_like(depth_frames, dtype=np.float32)
            else:
                depth_frames = (depth_frames - d_min) / (d_max - d_min)
        else:
            depth_frames = np.clip(
                depth_frames, self.normalize_metric_depth_frames_min, self.normalize_metric_depth_frames_max
            )
            if self.normalize_metric_depth_frames_use_sqrt:
                if self.normalize_metric_depth_frames_max <= 0:
                    depth_frames = np.zeros_like(depth_frames, dtype=np.float32)
                else:
                    depth_frames = np.sqrt(depth_frames) / np.sqrt(self.normalize_metric_depth_frames_max)
            else:
                if self.normalize_metric_depth_frames_max <= 0:
                    depth_frames = np.zeros_like(depth_frames, dtype=np.float32)
                else:
                    depth_frames = depth_frames / self.normalize_metric_depth_frames_max
        depth_frames = np.repeat(depth_frames, 3, axis=-1)
        return WMRNDMixTransforms._normalize_rgb_like_frames(depth_frames * 255.0)

    @staticmethod
    def _probe_video_resolution(video_path):
        cmd_info = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            video_path,
        ]
        res = subprocess.check_output(cmd_info).decode("utf-8").strip().split("x")
        return int(res[0]), int(res[1])

    @staticmethod
    def _probe_video_num_frames(video_path):
        cmd_info = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            video_path,
        ]
        output = subprocess.check_output(cmd_info).decode("utf-8").strip()
        return int(output)

    def _read_gray16le_depth_video(self, video_path, frame_indices=None):
        width, height = self._probe_video_resolution(video_path)
        cmd_read = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            video_path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray16le",
            "-vcodec",
            "rawvideo",
            "-",
        ]
        process = subprocess.Popen(cmd_read, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frames = []
        frame_size = width * height * 2
        target_indices = None if frame_indices is None else sorted(int(i) for i in frame_indices)
        target_set = None if target_indices is None else set(target_indices)
        max_target = None if target_indices is None or len(target_indices) == 0 else target_indices[-1]
        frame_id = 0
        try:
            while True:
                raw_frame = process.stdout.read(frame_size)
                if not raw_frame or len(raw_frame) != frame_size:
                    break
                if target_set is None or frame_id in target_set:
                    frame = np.frombuffer(raw_frame, dtype=np.uint16).reshape((height, width))
                    frames.append(frame.astype(np.float32) / 1000.0)
                frame_id += 1
                if max_target is not None and frame_id > max_target:
                    break
            # Drain pipes and ensure subprocess resources are released.
            process.stdout.close()
            process.stderr.read()
            process.stderr.close()
            process.wait(timeout=5)
        except Exception:
            process.kill()
            process.wait()
            raise
        if len(frames) == 0:
            raise RuntimeError(f"No decoded depth frames from {video_path}")
        if target_indices is not None and len(frames) != len(target_indices):
            raise RuntimeError(
                f"Decoded {len(frames)} depth frames but expected {len(target_indices)} for {video_path}"
            )
        return np.stack(frames, axis=0)

    def __call__(self, data_dict):
        if self.num_views == 1:
            front_images = DecordVideoReader(data_dict["cam_high_video_path"])
            replay_images = DecordVideoReader(data_dict["cam_high_replay_path"])
            depth_path = data_dict["cam_high_depth_path"]
            if self.depth_is_metric:
                depth_images = depth_path
                depth_length = self._probe_video_num_frames(depth_path)
            else:
                depth_images = DecordVideoReader(depth_path)
                depth_length = len(depth_images)
            video_length = min(len(front_images), depth_length, len(replay_images))
        elif self.num_views == 3:
            front_view_images = DecordVideoReader(data_dict["cam_high_video_path"])
            left_view_images = DecordVideoReader(data_dict["cam_left_wrist_video_path"])
            right_view_images = DecordVideoReader(data_dict["cam_right_wrist_video_path"])
            front_depth_path = data_dict["cam_high_depth_path"]
            left_depth_path = data_dict["cam_left_wrist_depth_path"]
            right_depth_path = data_dict["cam_right_wrist_depth_path"]
            if self.depth_is_metric:
                front_depth_images = front_depth_path
                left_depth_images = left_depth_path
                right_depth_images = right_depth_path
                front_depth_length = self._probe_video_num_frames(front_depth_path)
                left_depth_length = self._probe_video_num_frames(left_depth_path)
                right_depth_length = self._probe_video_num_frames(right_depth_path)
            else:
                front_depth_images = DecordVideoReader(front_depth_path)
                left_depth_images = DecordVideoReader(left_depth_path)
                right_depth_images = DecordVideoReader(right_depth_path)
                front_depth_length = len(front_depth_images)
                left_depth_length = len(left_depth_images)
                right_depth_length = len(right_depth_images)
            front_replay_images = DecordVideoReader(data_dict["cam_high_simulator_path"])
            left_replay_images = DecordVideoReader(data_dict["cam_left_wrist_simulator_path"])
            right_replay_images = DecordVideoReader(data_dict["cam_right_wrist_simulator_path"])
            video_length = min(
                len(front_view_images),
                len(left_view_images),
                len(right_view_images),
                front_depth_length,
                left_depth_length,
                right_depth_length,
                len(front_replay_images),
                len(left_replay_images),
                len(right_replay_images),
            )
        else:
            raise ValueError(f"Unsupported num_views: {self.num_views}")

        if self.debug_fixed_sampling:
            start_frame = max(0, min(self.debug_sample_start, video_length - 1))
            stride = self.debug_sample_stride
            last_frame = start_frame + stride * (self.num_frames - 1)
            if last_frame > video_length - 1:
                start_frame = max(0, (video_length - 1) - stride * (self.num_frames - 1))
                last_frame = min(video_length - 1, start_frame + stride * (self.num_frames - 1))
            sample_indexes = np.linspace(start_frame, last_frame, num=self.num_frames, dtype=int)
        elif self.max_stride is not None:
            stride = random.randint(1, self.max_stride)
            start_frame = random.randint(0, max(0, video_length - stride * (self.num_frames - 1) - 1))
            end_frame = min(video_length - 1, start_frame + stride * (self.num_frames - 1))
            sample_indexes = np.linspace(start_frame, end_frame, num=self.num_frames, dtype=int)
        else:
            sample_indexes = np.linspace(0, video_length - 1, self.num_frames, dtype=int)

        def get_input_images(video):
            safe_indexes = np.clip(sample_indexes, 0, len(video) - 1)
            if isinstance(video, VideoReader):
                input_images = video_utils.sample_video(video, safe_indexes, method=2)
            else:
                input_images = video[safe_indexes]
            data_dict["input_fps"] = self.fps
            return self._normalize_rgb_like_frames(input_images)

        def get_input_depth_images(video):
            if isinstance(video, str):
                depth_num_frames = self._probe_video_num_frames(video)
            else:
                depth_num_frames = len(video)
            safe_indexes = np.clip(sample_indexes, 0, depth_num_frames - 1)
            data_dict["input_fps"] = self.fps

            if isinstance(video, str):
                requested_indexes = np.asarray(safe_indexes, dtype=int)
                unique_indexes = np.unique(requested_indexes)
                decoded_images = self._read_gray16le_depth_video(video, frame_indices=unique_indexes)
                index_to_pos = {int(idx): pos for pos, idx in enumerate(unique_indexes.tolist())}
                input_images = np.stack([decoded_images[index_to_pos[int(idx)]] for idx in requested_indexes], axis=0)
            elif isinstance(video, VideoReader):
                input_images = video_utils.sample_video(video, safe_indexes, method=2)
            else:
                input_images = video[safe_indexes]
            input_images = np.asarray(input_images)
            if self.depth_is_metric:
                return self._normalize_metric_depth_frames(input_images)
            return self._normalize_rgb_like_frames(input_images)

        if self.num_views == 1:
            data_dict["input_front_images"] = get_input_images(front_images)
            data_dict["input_depth_images"] = get_input_depth_images(depth_images)
            data_dict["input_replay_images"] = get_input_images(replay_images)
        else:
            front_images = get_input_images(front_view_images)
            left_images = get_input_images(left_view_images)
            right_images = get_input_images(right_view_images)
            data_dict["input_front_images"] = torch.cat([front_images, left_images, right_images], dim=-1)
            front_depth = get_input_depth_images(front_depth_images)
            left_depth = get_input_depth_images(left_depth_images)
            right_depth = get_input_depth_images(right_depth_images)
            data_dict["input_depth_images"] = torch.cat([front_depth, left_depth, right_depth], dim=-1)
            front_replay = get_input_images(front_replay_images)
            left_replay = get_input_images(left_replay_images)
            right_replay = get_input_images(right_replay_images)
            data_dict["input_replay_images"] = torch.cat([front_replay, left_replay, right_replay], dim=-1)

        ref_masks, ref_latent_masks = self.mask_generator.get_mask(data_dict["input_front_images"].shape[0])
        ref_masks = ref_masks[:, None, None, None]
        ref_latent_masks = ref_latent_masks[None, :, None, None]
        data_dict["input_front_ref_images"] = copy.deepcopy(data_dict["input_front_images"]) * ref_masks
        data_dict["input_front_ref_depth_images"] = copy.deepcopy(data_dict["input_depth_images"]) * ref_masks
        data_dict["input_front_ref_masks"] = ref_latent_masks

        if not self.is_train:
            raise AssertionError("WMRNDMixTransforms is train-only.")

        new_data_dict = {
            "fps": data_dict.get("input_fps", self.fps),
            "front_images": torch.cat([data_dict["input_front_images"], data_dict["input_depth_images"]], dim=-2),
            "front_depth_images": data_dict["input_depth_images"],
            "front_ref_images": torch.cat([data_dict["input_front_ref_images"], data_dict["input_front_ref_depth_images"]], dim=-2),
            "front_ref_depth_images": data_dict["input_front_ref_depth_images"],
            "front_ref_masks": data_dict["input_front_ref_masks"],
            # Keep replay condition spatial shape aligned with stacked front image
            # ([224, 672] -> [448, 672]) to avoid patch-shape mismatch in transformer.
            "replay_condition": torch.cat([data_dict["input_replay_images"], data_dict["input_replay_images"]], dim=-2),
            "prompt_embeds": self.default_prompt_embeds,
        }
        if "qpos" in data_dict and data_dict["qpos"] is not None:
            qpos_seq = data_dict["qpos"]
            if isinstance(qpos_seq, torch.Tensor):
                new_data_dict["qpos"] = qpos_seq[sample_indexes].float()
            else:
                new_data_dict["qpos"] = torch.from_numpy(np.asarray(qpos_seq)[sample_indexes]).float()
        return new_data_dict
