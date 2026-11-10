"""Left-right mirror symmetry for the Tlibot Z1 AMP task.

Only the Z1 29-DoF robot is supported (23 action joints; the wrist joints are
frozen by ``RobotEnvCfg.ignored_action_joints``).

The mirror is the reflection about the sagittal (x-z) plane, i.e. ``y -> -y``::

    position / linear velocity :   v = (x, y, z)  ->  (x, -y, z)
    angular velocity           :   w = (x, y, z)  ->  (-x, y, -z)     (pseudo-vector)
    orientation                :   R             ->  M R M,  M = diag(1, -1, 1)
    joint                      :   q_j           ->  s_j * q_partner(j)
                                   (left <-> right; hip/ankle/shoulder roll & yaw
                                    and torso yaw/roll change sign)

Observation groups are mirrored **per term**, so the term-major history layout
produced by ``concatenate_terms=True`` + ``history_length`` is preserved: a term
of total size ``history * frame_dim`` is reshaped to ``(B, history, frame_dim)``
and every frame is mirrored with the same rule.

Used by :class:`rsl_rl`'s symmetry extension::

    RslRlSymmetryCfg(
        use_data_augmentation=True,
        data_augmentation_func="basic_loco_lib.tasks.amp.mdp.symmetry:compute_symmetric_states_z1",
    )
"""

from __future__ import annotations

import math

import torch

__all__ = ["compute_symmetric_states_z1"]

# Joint name fragments whose rotation axis flips under a left-right mirror
# (roll and yaw axes); pitch joints keep their sign.
_NEGATED_JOINT_KEYS = (
    "hip_roll",
    "hip_yaw",
    "ankle_roll",
    "shoulder_roll",
    "shoulder_yaw",
    "torso_yaw",
    "torso_roll",
)

# Plain vector terms: component-wise sign of the mirrored vector.
_VEC_SIGN = {
    "command": (1.0, -1.0, -1.0),          # (vx, vy, wz)
    "projected_gravity": (1.0, -1.0, 1.0),
    "base_ang_vel": (-1.0, 1.0, -1.0),
    "base_lin_vel": (1.0, -1.0, 1.0),
}

# Joint-space terms (mirror = left/right swap + per-joint sign).
_JOINT_TERMS = ("joint_pos", "joint_vel", "actions")

# Per-body 3-vector terms.
_BODY_VEC_SIGN = {
    "body_pos_b": (1.0, -1.0, 1.0),
    "body_lin_vel_b": (1.0, -1.0, 1.0),
    "body_ang_vel_b": (-1.0, 1.0, -1.0),
}

# Per-body 6D orientation term (first two columns of the rotation matrix).
_BODY_ORI_TERMS = ("body_ori_b",)


# --------------------------------------------------------------------------- #
# name tables
# --------------------------------------------------------------------------- #
def _partner(name: str) -> str:
    """Left <-> right counterpart of a joint/body name (identity if unpaired)."""
    if "_left_" in name:
        return name.replace("_left_", "_right_")
    if "_right_" in name:
        return name.replace("_right_", "_left_")
    return name


def _joint_perm_sign(names: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
    """``mirrored[..., j] = sign[j] * x[..., perm[j]]`` for a joint-space term."""
    index = {n: i for i, n in enumerate(names)}
    perm = torch.arange(len(names), device=device)
    sign = torch.ones(len(names), device=device)
    for j, name in enumerate(names):
        partner = _partner(name)
        if partner not in index:
            raise KeyError(f"mirror partner of joint '{name}' is missing from {names}")
        perm[j] = index[partner]
        if any(key in name for key in _NEGATED_JOINT_KEYS):
            sign[j] = -1.0
    return perm, sign


def _body_perm(names: list[str], device) -> torch.Tensor:
    """Left <-> right permutation over the body list (unpaired bodies stay put)."""
    index = {n: i for i, n in enumerate(names)}
    perm = torch.arange(len(names), device=device)
    for j, name in enumerate(names):
        partner = _partner(name)
        if partner in index:
            perm[j] = index[partner]
    return perm


def _resolve_names(env_unwrapped, names, preserve_order: bool) -> list[str]:
    """Resolve regex/exact joint names against the robot in the order the term uses."""
    asset = env_unwrapped.scene["robot"]
    _, resolved = asset.find_joints(list(names), preserve_order=preserve_order)
    return list(resolved)


def _resolve_body_names(env_unwrapped, names, preserve_order: bool) -> list[str]:
    """Resolve body names against the robot in the order the term uses."""
    asset = env_unwrapped.scene["robot"]
    _, resolved = asset.find_bodies(list(names), preserve_order=preserve_order)
    return list(resolved)


def _term_cfg(env_unwrapped, group: str, term: str):
    group_cfg = getattr(env_unwrapped.cfg.observations, group, None)
    return getattr(group_cfg, term, None)


def _obs_joint_names(env_unwrapped, group: str) -> list[str]:
    cfg = _term_cfg(env_unwrapped, group, "joint_pos")
    asset_cfg = (getattr(cfg, "params", None) or {}).get("asset_cfg") if cfg is not None else None
    if asset_cfg is None or not getattr(asset_cfg, "joint_names", None):
        raise ValueError(f"cannot resolve joint names for observations.{group}.joint_pos")
    return _resolve_names(env_unwrapped, asset_cfg.joint_names, getattr(asset_cfg, "preserve_order", False))


def _body_names(env_unwrapped, group: str, term: str) -> list[str]:
    cfg = _term_cfg(env_unwrapped, group, term)
    body_cfg = (getattr(cfg, "params", None) or {}).get("body_cfg") if cfg is not None else None
    if body_cfg is None or not getattr(body_cfg, "body_names", None):
        raise ValueError(f"cannot resolve body names for observations.{group}.{term}")
    return _resolve_body_names(env_unwrapped, body_cfg.body_names, getattr(body_cfg, "preserve_order", False))


def _action_joint_names(env_unwrapped) -> list[str]:
    """Joint names of the (single) joint-position action term, in its own order."""
    cfg = getattr(env_unwrapped.cfg.actions, "joint_pos", None)
    if cfg is None:  # fall back to the first action term that looks like a joint action
        for term in vars(env_unwrapped.cfg.actions).values():
            if hasattr(term, "joint_names") and hasattr(term, "scale"):
                cfg = term
                break
    if cfg is None:
        raise ValueError("cannot resolve the action term joint names")
    return _resolve_names(env_unwrapped, cfg.joint_names, getattr(cfg, "preserve_order", False))


# --------------------------------------------------------------------------- #
# per-term mirroring
# --------------------------------------------------------------------------- #
def _mirror_bodies_vec(x: torch.Tensor, n_bodies: int, perm: torch.Tensor, axis_sign) -> torch.Tensor:
    """Mirror a ``n_bodies * 3`` position/velocity block (left/right swap + y flip)."""
    shape = x.shape
    xb = x.reshape(*shape[:-1], n_bodies, 3)
    xb = xb[..., perm, :] * torch.as_tensor(axis_sign, dtype=x.dtype, device=x.device)
    return xb.reshape(shape)


def _mirror_bodies_6d(x: torch.Tensor, n_bodies: int, perm: torch.Tensor) -> torch.Tensor:
    """Mirror a ``n_bodies * 6`` orientation block (6D = first two rotation columns)."""
    shape = x.shape
    d = x.reshape(*shape[:-1], n_bodies, 6)[..., perm, :]
    a1 = torch.stack([d[..., 0], d[..., 2], d[..., 4]], dim=-1)  # first column
    a2 = torch.stack([d[..., 1], d[..., 3], d[..., 5]], dim=-1)  # second column
    a3 = torch.cross(a1, a2, dim=-1)
    rot = torch.stack([a1, a2, a3], dim=-1)                      # (..., 3, 3)
    m = torch.tensor([1.0, -1.0, 1.0], dtype=x.dtype, device=x.device)
    rot = rot * m[..., :, None] * m[..., None, :]                # M R M
    out = torch.stack(
        [rot[..., 0, 0], rot[..., 0, 1], rot[..., 1, 0], rot[..., 1, 1], rot[..., 2, 0], rot[..., 2, 1]],
        dim=-1,
    )
    return out.reshape(shape)


def _mirror_term(env_unwrapped, group: str, term: str, x: torch.Tensor) -> torch.Tensor:
    """Mirror one term block of shape ``(B, history, frame_dim)`` (or ``(B, frame_dim)``)."""
    if term in _VEC_SIGN:
        sign = torch.tensor(_VEC_SIGN[term], dtype=x.dtype, device=x.device)
        return x * sign
    if term in _JOINT_TERMS:
        perm, sign = _joint_perm_sign(_obs_joint_names(env_unwrapped, group), x.device)
        return x[..., perm] * sign
    if term in _BODY_VEC_SIGN:
        names = _body_names(env_unwrapped, group, term)
        return _mirror_bodies_vec(x, len(names), _body_perm(names, x.device), _BODY_VEC_SIGN[term])
    if term in _BODY_ORI_TERMS:
        names = _body_names(env_unwrapped, group, term)
        return _mirror_bodies_6d(x, len(names), _body_perm(names, x.device))
    raise KeyError(f"no mirror rule for observation term '{term}' (group '{group}')")


def _mirror_obs_group(env_unwrapped, group: str, obs: torch.Tensor) -> torch.Tensor:
    """Mirror one concatenated observation group, preserving the per-term layout."""
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
        block = _mirror_term(env_unwrapped, group, name, block)
        mirrored.append(block.reshape(*obs.shape[:-1], dim))
        start += dim
    return torch.cat(mirrored, dim=-1)


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_symmetric_states_z1(env, obs=None, actions: torch.Tensor | None = None):
    """Augment observations/actions with their left-right mirror image.

    Signature follows rsl_rl's ``data_augmentation_func``: the returned tensors
    hold the originals first and the mirrored samples after them (batch doubled).

    Args:
        env: Vectorized environment (``unwrapped`` gives the ManagerBasedRLEnv).
        obs: Observation ``TensorDict`` with the groups used by the runner
            (``actor`` / ``critic`` / ``amp`` for AMP).
        actions: Action tensor of shape ``(B, num_actions)``.

    Returns:
        ``(obs_aug, actions_aug)``; each is ``None`` when its input is ``None``.
    """
    if obs is None and actions is None:
        return None, None

    env_unwrapped = env.unwrapped

    obs_aug = None
    if obs is not None:
        batch_size = obs.batch_size[0]
        # Duplicate the whole batch first; ``amp`` and any other group that the
        # PPO update does not read (the discriminator uses the live env) simply
        # stays duplicated, while actor/critic are overwritten with the mirror.
        obs_aug = obs.repeat(2)
        for group in obs.keys():
            if group in ("actor", "critic"):
                obs_aug[group][batch_size:] = _mirror_obs_group(env_unwrapped, group, obs[group])

    actions_aug = None
    if actions is not None:
        perm, sign = _joint_perm_sign(_action_joint_names(env_unwrapped), actions.device)
        actions_aug = torch.cat([actions, actions[..., perm] * sign], dim=0)

    return obs_aug, actions_aug
