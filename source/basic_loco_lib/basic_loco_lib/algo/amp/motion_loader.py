# Copyright (c) 2025-2026, The IsaacNewton Project Developers.
# AMP motion data loader - loads npz files and precomputes body-relative features.

from __future__ import annotations

import copy
import numpy as np
import os
import torch
from collections.abc import Sequence

from isaaclab.utils.string import resolve_matching_names

from .motion_labels import AMP_LABEL_DIM, AMP_LABEL_NAMES, label_from_name, label_index


class _Arr:
    """Stand-in for IsaacLab's ProxyArray: exposes ``.torch``."""

    __slots__ = ("torch",)

    def __init__(self, tensor: torch.Tensor):
        self.torch = tensor


class _MotionData:
    """ArticulationData-shaped view of one motion clip."""

    def __init__(self, tensors: dict[str, torch.Tensor], default_joint_pos: torch.Tensor):
        num_frames = tensors["body_pos_w"].shape[0]
        self.body_pos_w = _Arr(tensors["body_pos_w"])
        self.body_quat_w = _Arr(tensors["body_quat_w"])
        self.body_link_lin_vel_w = _Arr(tensors["body_lin_vel_w"])
        self.body_link_ang_vel_w = _Arr(tensors["body_ang_vel_w"])
        self.joint_pos = _Arr(tensors["joint_pos"])
        self.joint_vel = _Arr(tensors["joint_vel"])
        self.default_joint_pos = _Arr(default_joint_pos.unsqueeze(0).expand(num_frames, -1))
        self.default_joint_vel = _Arr(torch.zeros_like(self.default_joint_pos.torch))


class _MotionAsset:
    def __init__(self, body_names: Sequence[str], joint_names: Sequence[str], data: _MotionData):
        self.body_names = list(body_names)
        self.joint_names = list(joint_names)
        self.data = data


class _MotionScene:
    def __init__(self, asset: _MotionAsset):
        self._asset = asset

    def __getitem__(self, _name: str) -> _MotionAsset:
        return self._asset


class MotionEnv:
    """Minimal env surface for the AMP observation terms; ``num_envs`` is the frame count."""

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        body_names: Sequence[str],
        joint_names: Sequence[str],
        default_joint_pos: torch.Tensor,
        device: str,
        step_dt: float = 0.02,
    ):
        self.device = device
        self.step_dt = step_dt
        self.physics_dt = step_dt
        self.num_envs = int(tensors["body_pos_w"].shape[0])
        self.scene = _MotionScene(
            _MotionAsset(body_names, joint_names, _MotionData(tensors, default_joint_pos))
        )


def _resolve_entity_cfg(cfg, asset: _MotionAsset):
    """Resolve a SceneEntityCfg's names against ``asset`` (motion clip ordering)."""
    cfg = copy.deepcopy(cfg)
    if cfg.body_names is not None:
        names = [cfg.body_names] if isinstance(cfg.body_names, str) else list(cfg.body_names)
        ids, resolved = resolve_matching_names(names, asset.body_names, preserve_order=cfg.preserve_order)
        cfg.body_ids, cfg.body_names = ids, resolved
    if cfg.joint_names is not None:
        names = [cfg.joint_names] if isinstance(cfg.joint_names, str) else list(cfg.joint_names)
        ids, resolved = resolve_matching_names(names, asset.joint_names, preserve_order=cfg.preserve_order)
        cfg.joint_ids, cfg.joint_names = ids, resolved
    return cfg


def amp_features_from_group(group_cfg, motion_env: MotionEnv) -> torch.Tensor:
    """Concatenate the group's ObsTerms by calling each ``func`` on ``motion_env``."""
    chunks: list[torch.Tensor] = []
    for term in vars(group_cfg).values():
        func = getattr(term, "func", None)
        if func is None:
            continue
        params = {}
        for key, value in (term.params or {}).items():
            if hasattr(value, "body_names"):
                value = _resolve_entity_cfg(value, motion_env.scene["robot"])
            params[key] = value
        chunks.append(func(motion_env, **params))
    if not chunks:
        raise ValueError("[AMPLoader] AMP observation group is empty")
    return torch.cat(chunks, dim=-1)


_NPZ_KEYS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_pos", "joint_vel")


class AMPLoader:
    def __init__(self, motion_file: str,
                 group_cfg=None,
                 default_joint_pos: torch.Tensor | None = None,
                 num_frames: int = 2,
                 device: str = "cuda:0",
                 labels: Sequence[str] = (),
                 weights: Sequence[float] = ()):
        """Load AMP motion data.

        Args:
            motion_file: Path to a single .npz file or a directory of .npz files.
            group_cfg: The env's AMP observation group (``env.cfg.observations.amp``). Every
                term in it is replayed on the motion data, in declaration order.
            default_joint_pos: Robot default joint positions, used by ``*_rel`` joint terms.
            num_frames: Number of consecutive frames per sampled expert window.
            device: Torch device.
            labels: Per-clip style label enabling label conditioning (empty disables it).
            weights: Optional per-clip sampling weight.
        """
        assert os.path.exists(motion_file), f"Invalid path: {motion_file}"
        if group_cfg is None:
            raise ValueError("AMPLoader needs the env AMP observation group: group_cfg=env.cfg.observations.amp")
        self.num_frames = max(1, int(num_frames))

        # Resolve a motion file or directory.
        if os.path.isfile(motion_file):
            motion_files = [motion_file]
            motion_names = [os.path.splitext(os.path.basename(motion_file))[0]]
        elif os.path.isdir(motion_file):
            motion_names = []
            motion_files = []
            for root, _dirs, files in os.walk(motion_file):
                for filename in sorted(files):
                    if filename.endswith('.npz'):
                        motion_names.append(os.path.splitext(filename)[0])
                        motion_files.append(os.path.join(root, filename))
            motion_files, motion_names = zip(*sorted(zip(motion_files, motion_names))) if motion_files else ([], [])
            motion_files, motion_names = list(motion_files), list(motion_names)
            assert len(motion_files) > 0, f"No npz files found in directory: {motion_file}"
        else:
            raise ValueError(f"Path is neither a file nor a directory: {motion_file}")

        self.motion_names = motion_names
        self._obs_list: list[torch.Tensor] = []
        self._lengths: list[int] = []

        for motion_idx, (motion_name, motion_path) in enumerate(zip(motion_names, motion_files)):
            print(f"Processing motion {motion_idx+1}/{len(motion_files)}: {motion_name}")
            data = np.load(motion_path)
            if motion_idx == 0:
                self.fps = data["fps"]

            tensors = {
                key: torch.tensor(data[key], dtype=torch.float32, device=device) for key in _NPZ_KEYS
            }
            num_joints = int(tensors["joint_pos"].shape[-1])
            if default_joint_pos is None:
                joint_default = torch.zeros(num_joints, dtype=torch.float32, device=device)
            else:
                joint_default = torch.as_tensor(default_joint_pos, dtype=torch.float32, device=device).reshape(-1)
                if joint_default.numel() != num_joints:
                    raise ValueError(
                        f"default_joint_pos has {joint_default.numel()} entries but "
                        f"'{motion_name}' has {num_joints} joints"
                    )

            motion_env = MotionEnv(
                tensors=tensors,
                body_names=data["body_names"].tolist(),
                joint_names=data["joint_names"].tolist(),
                default_joint_pos=joint_default,
                device=device,
                step_dt=1.0 / float(np.asarray(data["fps"]).reshape(-1)[0]),
            )
            features = amp_features_from_group(group_cfg, motion_env)
            if features.shape[0] != tensors["body_pos_w"].shape[0]:
                raise ValueError(
                    f"AMP features produced {features.shape[0]} frames but '{motion_name}' has "
                    f"{tensors['body_pos_w'].shape[0]}"
                )
            self._obs_list.append(features)
            self._lengths.append(int(features.shape[0]))

        self._observations = torch.cat(self._obs_list, dim=0)
        self.time_step_total = int(self._observations.shape[0])
        self.motion_total_time = self.time_step_total / self.fps

        self._motion_lengths = torch.tensor(self._lengths, device=device)
        self._motion_offsets = self._motion_lengths.cumsum(0) - self._motion_lengths

        self._setup_labels(labels, weights)

    @property
    def observation_dim(self) -> int:
        return int(self._observations.shape[-1])

    @property
    def window_dim(self) -> int:
        return self.observation_dim * self.num_frames

    def _setup_labels(self, labels: Sequence[str], weights: Sequence[float]) -> None:
        names = [str(label) if label else label_from_name(name) for label, name in zip(labels, self.motion_names)]
        names += [label_from_name(name) for name in self.motion_names[len(names):]]
        names = [name if name in AMP_LABEL_NAMES else "forward" for name in names]
        self.motion_labels = names
        self.label_dim = AMP_LABEL_DIM if len(labels) > 0 else 0

        device = self._observations.device
        self.motion_label_ids = torch.tensor([label_index(name) for name in names], dtype=torch.long, device=device)
        self._label_onehot = torch.nn.functional.one_hot(self.motion_label_ids, num_classes=AMP_LABEL_DIM).to(
            dtype=torch.float32
        )
        clip_weights = torch.ones(len(names), dtype=torch.float32, device=device)
        if len(weights) > 0:
            clip_weights[: len(weights)] = torch.as_tensor(list(weights), dtype=torch.float32, device=device)
        self.motion_weights = clip_weights / clip_weights.sum()
        self.last_label_fractions = torch.zeros(AMP_LABEL_DIM, device=device)
        self._missing_labels: set[int] = set()

    def sample(self, batch_size: int, label_probs: torch.Tensor | None = None):
        device = self._observations.device
        motion_ids = torch.multinomial(self.motion_weights, batch_size, replacement=True)
        if self.label_dim > 0 and label_probs is not None:
            labels = torch.as_tensor(label_probs, device=device, dtype=torch.float32)
            label_ids = labels.argmax(dim=1)
            matching = self.motion_label_ids.unsqueeze(0) == label_ids.unsqueeze(1)
            clip_probs = self.motion_weights.unsqueeze(0).expand_as(matching)
            mask = torch.where(matching, clip_probs, torch.zeros_like(clip_probs))
            empty = mask.sum(dim=1) <= 0
            if bool(empty.any()):
                for missing in torch.unique(label_ids[empty]).tolist():
                    if missing not in self._missing_labels:
                        self._missing_labels.add(missing)
                        print(
                            f"[AMPLoader] no expert clip labelled '{AMP_LABEL_NAMES[missing]}', "
                            "falling back to the full clip pool"
                        )
                mask = torch.where(empty.unsqueeze(1), clip_probs, mask)
            motion_ids = torch.multinomial(mask, 1).squeeze(1)
            self.last_label_fractions = (
                torch.bincount(label_ids, minlength=self.label_dim).to(torch.float32) / max(batch_size, 1)
            )

        span = (self._motion_lengths[motion_ids] - self.num_frames).clamp(min=1)
        frame_ids = (torch.rand(batch_size, device=device) * span).long()
        idxs = self._motion_offsets[motion_ids] + frame_ids
        labels = self._label_onehot[motion_ids] if self.label_dim > 0 else None
        return self._window(idxs), labels

    def _window(self, idxs: torch.Tensor) -> torch.Tensor:
        """Stack ``num_frames`` consecutive frames starting at ``idxs`` (oldest first)."""
        return torch.cat([self._observations[idxs + k] for k in range(self.num_frames)], dim=-1)
