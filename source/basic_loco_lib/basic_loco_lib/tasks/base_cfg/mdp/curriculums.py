from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _linear_ramp(env: ManagerBasedRLEnv, start: float, end: float, ramp_steps: int) -> float:
    if ramp_steps <= 0:
        return float(end)
    t = min(float(env.common_step_counter) / float(ramp_steps), 1.0)
    return float(start) + (float(end) - float(start)) * t


def push_velocity_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    event_name: str = "push_robot",
    nominal_range: dict[str, float] | None = None,
    start_scale: float = 0.25,
    end_scale: float = 1.0,
    ramp_steps: int = 60_000,
) -> torch.Tensor:
    if nominal_range is None:
        nominal_range = {"x": 0.90, "y": 0.70, "roll": 0.40, "pitch": 0.50, "yaw": 0.50}
    scale = _linear_ramp(env, start_scale, end_scale, ramp_steps)
    cfg = env.event_manager.get_term_cfg(event_name)
    cfg.params["velocity_range"] = {
        key: (-abs(float(v)) * scale, abs(float(v)) * scale) for key, v in nominal_range.items()
    }
    return torch.tensor(scale, device=env.device)
