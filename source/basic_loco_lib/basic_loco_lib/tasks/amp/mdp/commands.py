"""Velocity command with mutually exclusive standing / turning / lateral modes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass

from basic_loco_lib.algo.amp.motion_labels import (
    COMMAND_MODE_LATERAL,
    COMMAND_MODE_MIXED,
    COMMAND_MODE_STANDING,
    COMMAND_MODE_TURNING,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["AmpVelocityCommand", "AmpVelocityCommandCfg"]


@configclass
class AmpVelocityCommandCfg(UniformVelocityCommandCfg):
    """Velocity command that additionally tags each env with a command mode."""

    class_type: type | str = "{DIR}.commands:AmpVelocityCommand"

    rel_turning_envs: float = 0.0
    """Probability of a pure in-place turning command."""

    rel_lateral_envs: float = 0.0
    """Probability of a pure lateral command."""

    turning_deadband: float = 0.2
    """Minimum |ang_vel_z| for pure turning commands."""

    lateral_deadband: float = 0.1
    """Minimum |lin_vel_y| for pure lateral commands."""

    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        pure_turn_ang_vel: tuple[float, float] = (0.0, 0.0)
        pure_lateral_lin_vel: tuple[float, float] = (0.0, 0.0)

    ranges: Ranges = MISSING


class AmpVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command that enforces pure turn / pure lateral command buckets."""

    cfg: AmpVelocityCommandCfg

    def __init__(self, cfg: AmpVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.command_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _sample_outside_deadband(
        self, count: int, bounds: tuple[float, float], deadband: float
    ) -> torch.Tensor:
        low, high = bounds
        if count == 0:
            return torch.empty(0, device=self.device)
        if low >= -deadband or high <= deadband:
            return torch.empty(count, device=self.device).uniform_(low, high)
        negative_span = -deadband - low
        positive_span = high - deadband
        positive_probability = positive_span / (negative_span + positive_span)
        magnitude = torch.rand(count, device=self.device)
        negative_value = low + magnitude * negative_span
        positive_value = deadband + magnitude * positive_span
        return torch.where(
            torch.rand(count, device=self.device) < positive_probability, positive_value, negative_value
        )

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

        env_index = torch.arange(self.num_envs, device=self.device)[env_ids]
        draws = torch.rand(len(env_index), device=self.device)
        standing_end = self.cfg.rel_standing_envs
        turning_end = standing_end + self.cfg.rel_turning_envs
        lateral_end = turning_end + self.cfg.rel_lateral_envs
        standing = draws < standing_end
        turning = (draws >= standing_end) & (draws < turning_end)
        lateral = (draws >= turning_end) & (draws < lateral_end)

        self.command_mode[env_index] = COMMAND_MODE_MIXED
        self.command_mode[env_index[standing]] = COMMAND_MODE_STANDING
        self.command_mode[env_index[turning]] = COMMAND_MODE_TURNING
        self.command_mode[env_index[lateral]] = COMMAND_MODE_LATERAL

        self.is_standing_env[env_index] = standing
        self.is_heading_env[env_index[standing | turning | lateral]] = False
        self.vel_command_b[env_index[standing]] = 0.0

        turning_ids = env_index[turning]
        self.vel_command_b[turning_ids, :2] = 0.0
        self.vel_command_b[turning_ids, 2] = self._sample_outside_deadband(
            len(turning_ids), self.cfg.ranges.pure_turn_ang_vel, self.cfg.turning_deadband
        )

        lateral_ids = env_index[lateral]
        self.vel_command_b[lateral_ids, 0] = 0.0
        self.vel_command_b[lateral_ids, 2] = 0.0
        self.vel_command_b[lateral_ids, 1] = self._sample_outside_deadband(
            len(lateral_ids), self.cfg.ranges.pure_lateral_lin_vel, self.cfg.lateral_deadband
        )
