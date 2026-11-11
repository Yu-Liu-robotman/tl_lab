"""Action terms for the AMP task: per-env actuation delay."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class DelayedJointPositionAction(JointPositionAction):
    """Joint-position action whose applied target lags the policy output.

    Every env keeps a constant actuation lag in ``[min_delay, max_delay]`` control
    steps, resampled on reset (the reference project's ``DelayBuffer``). The policy
    output is still stored in ``_raw_actions``, so ``mdp.last_action`` stays
    undelayed while the articulation receives the stale command.
    """

    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._max_delay = max(int(cfg.max_delay), 0)
        self._min_delay = min(max(int(cfg.min_delay), 0), self._max_delay)
        self._steps = 0
        self._rows = torch.arange(self.num_envs, device=self.device)
        self._lags = torch.full((self.num_envs,), self._max_delay, dtype=torch.long, device=self.device)
        self._ring = torch.zeros(
            self._max_delay + 1, self.num_envs, self.action_dim, device=self.device
        )
        self.reset(self._rows)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if self._max_delay <= 0:
            return
        # ``ActionManager`` hands terms either a tensor, ``slice(None)`` or ``None``
        if env_ids is None:
            rows = self._rows
        elif isinstance(env_ids, slice):
            rows = self._rows[env_ids]
        else:
            rows = torch.as_tensor(env_ids, device=self.device)
        self._ring[:, rows] = 0.0
        self._lags[rows] = torch.randint(
            self._min_delay, self._max_delay + 1, (len(rows),), device=self.device
        )

    def _delayed(self, actions: torch.Tensor) -> torch.Tensor:
        if self._max_delay <= 0:
            return actions
        size = self._max_delay + 1
        self._ring[self._steps % size] = actions
        applied = self._ring[(self._steps - self._lags) % size, self._rows]
        self._steps += 1
        return applied

    def process_actions(self, actions: torch.Tensor):
        applied = self._delayed(actions)
        super().process_actions(applied)
        # Restore the policy output: only the command sent to the robot is delayed.
        self._raw_actions[:] = actions


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Joint-position action with a per-env actuation delay (``max_delay=0`` disables it)."""

    class_type: type = DelayedJointPositionAction
    min_delay: int = 0
    max_delay: int = 2
