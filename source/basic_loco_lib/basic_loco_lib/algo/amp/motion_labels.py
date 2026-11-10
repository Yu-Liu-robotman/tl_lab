from __future__ import annotations

import os

import torch

AMP_LABEL_NAMES = (
    "forward",
    "backward",
    "turn_left",
    "turn_right",
    "side_left",
    "side_right",
    "stand",
)
AMP_LABEL_DIM = len(AMP_LABEL_NAMES)

COMMAND_MODE_STANDING = 0
COMMAND_MODE_TURNING = 1
COMMAND_MODE_LATERAL = 2
COMMAND_MODE_MIXED = 3

_NAME_MAP = {
    "walk": "forward",
    "forward": "forward",
    "backward": "backward",
    "recovery": "forward",
    "jog": "forward",
    "run": "forward",
    "sprint": "forward",
    "turn_left": "turn_left",
    "turn_right": "turn_right",
    "left_turn": "turn_left",
    "right_turn": "turn_right",
    "turn": "turn_left",
    "turning": "turn_left",
    "yaw": "turn_left",
    "side_left": "side_left",
    "side_right": "side_right",
    "left_side": "side_left",
    "right_side": "side_right",
    "lateral_left": "side_left",
    "lateral_right": "side_right",
    "side": "side_left",
    "lateral": "side_left",
    "strafe": "side_left",
    "stand": "stand",
    "idle": "stand",
    "static": "stand",
}


def label_index(name: str) -> int:
    return AMP_LABEL_NAMES.index(name if name in AMP_LABEL_NAMES else "forward")


def label_from_name(name: str) -> str:
    stem = os.path.splitext(os.path.basename(str(name)))[0].lower().replace("-", "_").replace(".", "_")
    if "turn" in stem and "right" in stem:
        return "turn_right"
    if "turn" in stem and "left" in stem:
        return "turn_left"
    if ("side" in stem or "lateral" in stem) and "right" in stem:
        return "side_right"
    if ("side" in stem or "lateral" in stem) and "left" in stem:
        return "side_left"
    for token in stem.split("_"):
        if token in _NAME_MAP:
            return _NAME_MAP[token]
    for key, value in _NAME_MAP.items():
        if key in stem:
            return value
    return "forward"


def command_to_motion_label(
    command: torch.Tensor,
    command_mode: torch.Tensor | None = None,
    velocity_threshold: float = 0.1,
    standing_threshold: float = 0.1,
) -> torch.Tensor:
    command = torch.as_tensor(command)
    label_ids = torch.where(
        command[:, 0] >= 0.0,
        torch.full((command.shape[0],), label_index("forward"), device=command.device, dtype=torch.long),
        torch.full((command.shape[0],), label_index("backward"), device=command.device, dtype=torch.long),
    )
    if command_mode is not None:
        mode = torch.as_tensor(command_mode, device=command.device)
        turn_left = torch.full_like(label_ids, label_index("turn_left"))
        turn_right = torch.full_like(label_ids, label_index("turn_right"))
        side_left = torch.full_like(label_ids, label_index("side_left"))
        side_right = torch.full_like(label_ids, label_index("side_right"))
        label_ids = torch.where(
            mode == COMMAND_MODE_TURNING,
            torch.where(command[:, 2] >= 0.0, turn_left, turn_right),
            label_ids,
        )
        label_ids = torch.where(
            mode == COMMAND_MODE_LATERAL,
            torch.where(command[:, 1] >= 0.0, side_left, side_right),
            label_ids,
        )
        standing = mode == COMMAND_MODE_STANDING
    else:
        standing = torch.zeros_like(label_ids, dtype=torch.bool)
    magnitude = torch.linalg.vector_norm(command[:, :2], dim=1) + command[:, 2].abs()
    lateral_dominant = (command[:, 1].abs() >= velocity_threshold) & (command[:, 1].abs() > command[:, 0].abs())
    label_ids = torch.where(
        lateral_dominant,
        torch.where(
            command[:, 1] >= 0.0,
            torch.full_like(label_ids, label_index("side_left")),
            torch.full_like(label_ids, label_index("side_right")),
        ),
        label_ids,
    )
    label_ids = torch.where(
        standing | (magnitude <= standing_threshold), torch.full_like(label_ids, label_index("stand")), label_ids
    )
    return torch.nn.functional.one_hot(label_ids, num_classes=AMP_LABEL_DIM).to(dtype=torch.float32)
