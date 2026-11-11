"""Left-right mirror symmetry for the Tlibot Z1 mimic task."""

from __future__ import annotations

import math

import torch

from basic_loco_lib.tasks.amp.mdp.symmetry import _body_perm, _joint_perm_sign, _mirror_bodies_vec

__all__ = ["compute_symmetric_states_z1"]

_VEC_SIGN = {
    "base_ang_vel": (-1.0, 1.0, -1.0),
    "base_lin_vel": (1.0, -1.0, 1.0),
    "motion_anchor_pos_b": (1.0, -1.0, 1.0),
}

# anchor_ori_b / body_ori are matrix_from_quat(...)[..., :2].reshape(..., 6):
# element (i, j) scales by s_i * s_j with S = diag(1, -1, 1).
_ORI_SIGN = (1.0, -1.0, -1.0, 1.0, 1.0, -1.0)

_VEC_TERMS = ("base_ang_vel", "base_lin_vel", "motion_anchor_pos_b")
_ORI_TERMS = ("motion_anchor_ori_b",)
_MOTION_COMMAND_TERMS = ("motion_command", "command")
_ACTION_JOINT_TERMS = ("joint_pos_rel", "joint_vel_rel", "joint_pos", "joint_vel", "last_action", "actions")
_GROUP_ALIAS = {"actor": "policy", "policy": "actor"}


def _asset(env_unwrapped):
    return env_unwrapped.scene["robot"]


def _motion_joint_names(env_unwrapped, num_joints: int) -> list[str]:
    names = list(_asset(env_unwrapped).joint_names)
    if len(names) != num_joints:
        raise ValueError(f"motion command has {num_joints} joints but the robot has {len(names)}")
    return names


def _action_joint_names(env_unwrapped) -> list[str]:
    term = env_unwrapped.action_manager.get_term("JointPositionAction")
    return list(term._joint_names)


def _tracked_body_names(env_unwrapped) -> list[str]:
    return list(env_unwrapped.command_manager.get_term("motion").cfg.body_names)


def _env_group(env_unwrapped, key: str) -> str:
    groups = env_unwrapped.cfg.observations
    if hasattr(groups, key):
        return key
    alias = _GROUP_ALIAS.get(key)
    if alias is not None and hasattr(groups, alias):
        return alias
    return key


def _mirror_term(env_unwrapped, term: str, x: torch.Tensor) -> torch.Tensor:
    if term in _VEC_TERMS:
        sign = torch.tensor(_VEC_SIGN[term], dtype=x.dtype, device=x.device)
        return x * sign
    if term in _ORI_TERMS:
        sign = torch.tensor(_ORI_SIGN, dtype=x.dtype, device=x.device)
        return x * sign
    if term in _MOTION_COMMAND_TERMS:
        num_joints = x.shape[-1] // 2
        perm, sign = _joint_perm_sign(_motion_joint_names(env_unwrapped, num_joints), x.device)
        halves = x.reshape(*x.shape[:-1], 2, num_joints)
        return (halves[..., perm] * sign).reshape(x.shape)
    if term in _ACTION_JOINT_TERMS:
        perm, sign = _joint_perm_sign(_action_joint_names(env_unwrapped), x.device)
        return x[..., perm] * sign
    if term == "body_pos":
        names = _tracked_body_names(env_unwrapped)
        return _mirror_bodies_vec(x, len(names), _body_perm(names, x.device), (1.0, -1.0, 1.0))
    if term == "body_ori":
        names = _tracked_body_names(env_unwrapped)
        num_bodies = len(names)
        shape = x.shape
        blocks = x.reshape(*shape[:-1], num_bodies, 6)[..., _body_perm(names, x.device), :]
        sign = torch.tensor(_ORI_SIGN, dtype=x.dtype, device=x.device)
        return (blocks * sign).reshape(shape)
    raise KeyError(f"no mirror rule for observation term '{term}'")


def _mirror_obs_group(env_unwrapped, group: str, obs: torch.Tensor) -> torch.Tensor:
    manager = env_unwrapped.observation_manager
    group_cfg = getattr(env_unwrapped.cfg.observations, group)
    history = max(int(getattr(group_cfg, "history_length", 1) or 1), 1)

    term_names = list(manager.active_terms[group])
    term_dims = [int(math.prod(d)) for d in manager.group_obs_term_dim[group]]
    if sum(term_dims) != obs.shape[-1]:
        raise ValueError(
            f"group '{group}': term dims {term_dims} sum to {sum(term_dims)} but obs has {obs.shape[-1]}"
        )

    mirrored = []
    start = 0
    for name, dim in zip(term_names, term_dims):
        if dim % history != 0:
            raise ValueError(f"group '{group}' term '{name}': dim {dim} is not divisible by history {history}")
        frame_dim = dim // history
        block = obs[..., start : start + dim].reshape(*obs.shape[:-1], history, frame_dim)
        block = _mirror_term(env_unwrapped, name, block)
        mirrored.append(block.reshape(*obs.shape[:-1], dim))
        start += dim
    return torch.cat(mirrored, dim=-1)


@torch.no_grad()
def compute_symmetric_states_z1(env, obs=None, actions: torch.Tensor | None = None):
    """rsl_rl symmetry hook: duplicate the batch, then overwrite the second half with its mirror."""
    if obs is None and actions is None:
        return None, None

    env_unwrapped = env.unwrapped

    obs_aug = None
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for group in obs.keys():
            obs_aug[group][batch_size:] = _mirror_obs_group(env_unwrapped, _env_group(env_unwrapped, group), obs[group])

    actions_aug = None
    if actions is not None:
        perm, sign = _joint_perm_sign(_action_joint_names(env_unwrapped), actions.device)
        actions_aug = torch.cat([actions, actions[..., perm] * sign], dim=0)

    return obs_aug, actions_aug