"""Auto-export the sim2sim bundle produced during training.

``scripts/rsl_rl/train.py`` patches ``runner.save`` so every checkpoint save also
refreshes ``<log_dir>/exported/`` with

* ``policy.pt`` / ``policy.onnx`` -- deploy-ready (single file, frame-stacked
  ``obs_history`` input),
* ``params.yaml`` -- everything ``deploy_mujoco`` needs that is *derived from the
  training config* (joint order, PD gains, action scale, obs scales, command
  ranges). MJCF paths / viewer settings are deployment-only and stay in the
  hand-written task yaml.

Keeping ``params.yaml`` machine-generated means the deploy side can never drift
from the checkpoint it loads.
"""

from __future__ import annotations

import os
from math import prod

import numpy as np
import yaml

__all__ = ["export_policy_artifacts", "collect_deploy_params", "make_deploy_ready_onnx"]


# --------------------------------------------------------------------------- #
# small conversion helpers (messages can come from torch tensors or warp proxies)
# --------------------------------------------------------------------------- #
def _first_row(value) -> np.ndarray:
    """Convert a per-env buffer (tensor / warp proxy / list) to env 0's values."""
    if value is None:
        return np.zeros(0)
    if hasattr(value, "torch"):  # warp ProxyArray
        value = value.torch
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    arr = np.asarray(value, dtype=float)
    if arr.ndim > 1:
        arr = arr[0]
    return arr.reshape(-1)


def _scalar(value, default: float = 1.0) -> float:
    if value is None:
        return default
    if hasattr(value, "torch"):
        value = value.torch
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    arr = np.asarray(value, dtype=float).reshape(-1)
    return float(arr[0]) if arr.size else default


# --------------------------------------------------------------------------- #
# ONNX post-processing (same transform as scripts/rsl_rl/play.py)
# --------------------------------------------------------------------------- #
def make_deploy_ready_onnx(env, onnx_path: str) -> None:
    """Re-save the exported ONNX as ONE self-contained file with a simple input.

    The input becomes a plain stack of obs frames, oldest first::

        [frame(t-H+1) | ... | frame(t)]    frame = all obs terms, one step

    IsaacLab instead feeds the history grouped per obs term; the fixed
    permutation between the two layouts is baked in as a single Gather node.
    """
    import onnx
    from onnx import helper, numpy_helper

    history, term_dims = 1, []
    if obs_man := getattr(env.unwrapped, "observation_manager", None):  # None for Direct envs
        policy_cfg = getattr(env.unwrapped.cfg.observations, "policy", None)
        policy_group = "policy"
        if policy_cfg is None:
            policy_cfg = env.unwrapped.cfg.observations.actor
            policy_group = "actor"
        history = int(policy_cfg.history_length or 0)
        if history <= 1:
            # Warp tasks: history lives in per-term wrappers instead of the group.
            hs = {getattr(t.func, "history_length", 1) for t in vars(policy_cfg).values() if hasattr(t, "func")}
            if len(hs) > 1:
                raise ValueError(
                    f"mixed per-term history lengths {sorted(hs)} in the policy group; "
                    "the deploy-side frame-stacked layout requires a uniform H."
                )
            history = int(hs.pop()) if hs else 1
        if history > 1:
            term_dims = [prod(d) // history for d in obs_man.group_obs_term_dim[policy_group]]

    model = onnx.load(onnx_path)  # merges any .data sidecar into memory
    graph = model.graph
    input_name = graph.input[0].name
    if history > 1:
        frame, bounds = sum(term_dims), np.cumsum([0] + term_dims)
        perm = np.concatenate(
            [(np.arange(history)[:, None] * frame + np.arange(s, s + d)).ravel() for s, d in zip(bounds[:-1], term_dims)]
        )
        graph.input[0].name = "obs_history"
        graph.initializer.append(numpy_helper.from_array(perm, name="history_perm"))
        policy_input_name = f"{input_name}_reordered"
        for node in graph.node:
            for index, node_input in enumerate(node.input):
                if node_input == input_name:
                    node.input[index] = policy_input_name
        graph.node.insert(0, helper.make_node("Gather", ["obs_history", "history_perm"], [policy_input_name], axis=1))
    else:
        graph.input[0].name = "obs_history"
        for node in graph.node:
            for index, node_input in enumerate(node.input):
                if node_input == input_name:
                    node.input[index] = "obs_history"

    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)
    if os.path.isfile(onnx_path + ".data"):
        os.remove(onnx_path + ".data")
    layout = f"obs_history = {history} stacked frames (term dims {term_dims})" if history > 1 else "obs_history = single frame"
    print(f"[INFO] exported {os.path.basename(onnx_path)}: single file, {layout}")


# --------------------------------------------------------------------------- #
# params.yaml
# --------------------------------------------------------------------------- #
def _policy_history_length(env) -> int:
    obs_cfg = getattr(env.cfg.observations, "policy", None)
    if obs_cfg is None:
        obs_cfg = getattr(env.cfg.observations, "actor", None)
    history = int(getattr(obs_cfg, "history_length", 1) or 1)
    if history <= 1:  # warp tasks: per-term history wrappers
        hs = {getattr(t.func, "history_length", 1) for t in vars(obs_cfg).values() if hasattr(t, "func")}
        if len(hs) == 1:
            history = int(hs.pop())
    return max(history, 1)


def _action_scale_per_joint(env, lab_names: list[str]) -> np.ndarray:
    """Resolve the action term's ``scale`` (float or dict) into one value per joint."""
    action_cfg = None
    for term in vars(env.cfg.actions).values():
        if hasattr(term, "joint_names") and hasattr(term, "scale"):
            action_cfg = term
            break
    if action_cfg is None:
        return np.ones(len(lab_names))

    scale = action_cfg.scale
    if isinstance(scale, dict):
        from isaaclab.utils.string import resolve_matching_names_values

        ids, _, values = resolve_matching_names_values(scale, lab_names)
        out = np.ones(len(lab_names))
        out[ids] = np.asarray(values, dtype=float)
        return out
    return np.full(len(lab_names), float(scale))


def _action_joint_names(env, asset) -> list[str]:
    """Joint names in the exact order the policy's action vector uses.

    Mirrors ``JointAction.__init__`` (``asset.find_joints(cfg.joint_names,
    preserve_order=cfg.preserve_order)``) so the exported order always matches the
    action term, including when only a subset of joints is actuated.
    """
    action_cfg = None
    for term in vars(env.cfg.actions).values():
        if hasattr(term, "joint_names") and hasattr(term, "scale"):
            action_cfg = term
            break
    if action_cfg is None:
        return list(getattr(asset, "joint_names", None) or asset.data.joint_names)
    _, names = asset.find_joints(
        action_cfg.joint_names, preserve_order=getattr(action_cfg, "preserve_order", False)
    )
    return list(names)


def _obs_joint_names(env, asset, group: str = "policy") -> list[str]:
    """Joints covered by the joint-position term, in the order the policy sees them.

    Velocity tasks name the term ``joint_pos_rel``; AMP tasks use ``joint_pos``.
    """
    defaults = list(getattr(asset, "joint_names", None) or asset.data.joint_names)
    obs_cfg = getattr(env.cfg.observations, group, None) or env.cfg.observations.actor
    term = getattr(obs_cfg, "joint_pos_rel", None)
    if term is None:
        term = getattr(obs_cfg, "joint_pos", None)
    asset_cfg = (getattr(term, "params", None) or {}).get("asset_cfg") if term is not None else None
    joint_names = getattr(asset_cfg, "joint_names", None) if asset_cfg is not None else None
    if not joint_names:
        return defaults
    _, names = asset.find_joints(joint_names, preserve_order=getattr(asset_cfg, "preserve_order", False))
    return list(names)


def _term_scale(obs_cfg, names: tuple[str, ...], default: float = 1.0) -> float:
    """First ``scale`` found among ``names`` on an ObsGroup (term names differ per task)."""
    for name in names:
        term = getattr(obs_cfg, name, None)
        if term is not None:
            return _scalar(getattr(term, "scale", default), default)
    return default


def collect_deploy_params(env) -> dict:
    """Everything ``deploy_mujoco`` needs that follows from the training config."""
    asset = env.scene["robot"]
    lab_names = list(getattr(asset, "joint_names", None) or asset.data.joint_names)

    # Joint order the MuJoCo model is expected to use (the robot's SDK/URDF order).
    sdk_names = getattr(env.cfg.scene.robot, "joint_sdk_names", None)
    mj_names = list(sdk_names) if sdk_names else list(lab_names)
    missing = [n for n in mj_names if n not in lab_names]
    if missing:
        raise ValueError(f"joint_sdk_names entries not found on the articulation: {missing}")
    mj_to_lab = [lab_names.index(n) for n in mj_names]
    order = np.asarray(mj_to_lab, dtype=int)

    # Which mj joint each policy action output drives. Differs from the full lab
    # order when the training config freezes joints (``ignored_action_joints``).
    sdk_index = {name: i for i, name in enumerate(mj_names)}
    action_names = _action_joint_names(env, asset)
    try:
        action_mj_indices = [sdk_index[name] for name in action_names]
    except KeyError as exc:
        raise ValueError(f"action joint {exc.args[0]!r} is not listed in joint_sdk_names") from None

    # Joint-state observations may cover a different (sub)set than the actions;
    # with ``ignored_action_joints`` both are the same reduced subset.
    obs_names = _obs_joint_names(env, asset)
    try:
        obs_mj_indices = [sdk_index[name] for name in obs_names]
    except KeyError as exc:
        raise ValueError(f"observation joint {exc.args[0]!r} is not listed in joint_sdk_names") from None

    # Active physics backend decides which joint permutation a policy uses.
    backend_name = type(env.cfg.sim.physics).__name__.lower()
    backend = "newton" if "newton" in backend_name else "physx"

    cmd_cfg = getattr(env.cfg.commands, "base_velocity", None)
    if cmd_cfg is None:  # AMP tasks name the velocity command term ``twist``
        cmd_cfg = getattr(env.cfg.commands, "twist", None)
    ranges = getattr(cmd_cfg, "limit_ranges", None) or getattr(cmd_cfg, "ranges", None)
    command_ranges = {}
    if ranges is not None:
        command_ranges = {
            "vx": [float(v) for v in ranges.lin_vel_x],
            "vy": [float(v) for v in ranges.lin_vel_y],
            "wz": [float(v) for v in ranges.ang_vel_z],
        }

    policy_cfg = getattr(env.cfg.observations, "policy", None)
    if policy_cfg is None:  # AMP tasks expose the actor group instead of ``policy``
        policy_cfg = getattr(env.cfg.observations, "actor", None)
    ang_vel_scale = _term_scale(policy_cfg, ("base_ang_vel",))
    joint_vel_scale = _term_scale(policy_cfg, ("joint_vel_rel", "joint_vel"))

    def to_mj(values: np.ndarray) -> list[float]:
        return [float(v) for v in values[order]]

    return {
        "backend": backend,
        "obs_history_length": _policy_history_length(env),
        "mj_joint_names": mj_names,
        "joint_orders": {backend: [int(i) for i in np.argsort(mj_to_lab)]},
        "action_joint_names": action_names,
        "action_mj_indices": [int(i) for i in action_mj_indices],
        "obs_joint_names": obs_names,
        "obs_joint_mj_indices": [int(i) for i in obs_mj_indices],
        "command_ranges": command_ranges,
        "ang_vel_scale": ang_vel_scale,
        "joint_vel_scale": joint_vel_scale,
        "stiffness": to_mj(_first_row(asset.data.joint_stiffness)),
        "damping": to_mj(_first_row(asset.data.joint_damping)),
        "default_joint_pos": to_mj(_first_row(asset.data.default_joint_pos)),
        "action_scale": to_mj(_action_scale_per_joint(env, lab_names)),
    }


def _tidy(obj):
    """Trim float32 noise (e.g. -0.029999999329447746 -> -0.03) for readability."""
    if isinstance(obj, float):
        return float(f"{obj:.6g}")
    if isinstance(obj, list):
        return [_tidy(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _tidy(v) for k, v in obj.items()}
    return obj


def _write_params(params: dict, out_dir: str) -> str:
    path = os.path.join(out_dir, "params.yaml")
    header = (
        "# AUTO-GENERATED by basic_loco_lib.utils.deploy_export -- do not edit.\n"
        "# Regenerated on every checkpoint save; deployment-only keys (xml_path, policy,\n"
        "# viewer, initial_pos, ...) stay in the hand-written task yaml.\n"
    )
    with open(path, "w") as f:
        f.write(header)
        yaml.dump(_tidy(params), f, default_flow_style=None, sort_keys=False, width=200)
    return path


def export_policy_artifacts(runner, env, out_dir: str) -> str:
    """Export ``policy.pt`` / ``policy.onnx`` / ``params.yaml`` into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    runner.export_policy_to_jit(path=out_dir, filename="policy.pt")
    runner.export_policy_to_onnx(path=out_dir, filename="policy.onnx")
    onnx_path = os.path.join(out_dir, "policy.onnx")
    make_deploy_ready_onnx(env, onnx_path)
    params_path = _write_params(collect_deploy_params(env), out_dir)
    print(f"[INFO] deploy bundle written: {out_dir} (policy.pt, policy.onnx, params.yaml)")
    return params_path
