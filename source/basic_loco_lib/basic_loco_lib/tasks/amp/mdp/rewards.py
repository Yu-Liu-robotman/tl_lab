"""Torch AMP rewards matching the Warp task's frames, scales, and delay masks."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, quat_apply_yaw

from basic_loco_lib.algo.amp.motion_labels import (
    COMMAND_MODE_STANDING, # stand command mode
    COMMAND_MODE_TURNING,  # turn command mode -W
    COMMAND_MODE_LATERAL,  # lateral command mode -Y
    COMMAND_MODE_MIXED     # mixed command mode -X (forward/backward/run)
)

def _command_mode(env, command_name: str) -> torch.Tensor | None:
    """Command label maintained by ``AmpVelocityCommand`` (None for plain velocity terms)."""
    manager = getattr(env, "command_manager", None)
    if manager is None:
        return None
    try:
        term = manager.get_term(command_name)
    except Exception:
        return None
    return getattr(term, "command_mode", None)

def track_anchor_linear_velocity(
    env,
    command_name: str,
    std: float,
    mask_delay: bool = True,
    delay_env_rew_ratio: float = 0.0,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset = env.scene[anchor_cfg.name]
    anchor_id = anchor_cfg.body_ids[0]
    command = env.command_manager.get_command(command_name)
    command_local = torch.zeros_like(command)
    command_local[:, :2] = command[:, :2]
    command_world = quat_apply_yaw(asset.data.body_quat_w.torch[:, anchor_id], command_local)
    error = command_world - asset.data.body_lin_vel_w.torch[:, anchor_id]
    reward = torch.exp(-error.square().sum(dim=-1) / std**2)
    return _mask_by_delay(env, reward, delay_env_rew_ratio) if mask_delay else reward


def track_anchor_angular_velocity(
    env,
    command_name: str,
    std: float,
    mask_delay: bool = True,
    delay_env_rew_ratio: float = 0.0,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset = env.scene[anchor_cfg.name]
    anchor_id = anchor_cfg.body_ids[0]
    velocity = asset.data.body_ang_vel_w.torch[:, anchor_id]
    local_velocity = quat_apply_inverse(asset.data.body_quat_w.torch[:, anchor_id], velocity)
    command = env.command_manager.get_command(command_name)
    error = (command[:, 2] - velocity[:, 2]).square() + local_velocity[:, :2].square().sum(dim=-1)
    reward = torch.exp(-error / std**2)
    return _mask_by_delay(env, reward, delay_env_rew_ratio) if mask_delay else reward


def track_root_height(env, std: float, mask_delay: bool = True, delay_env_rew_ratio: float = 3.5) -> torch.Tensor:
    asset = env.scene["robot"]
    error = asset.data.root_pos_w.torch[:, 2] - asset.cfg.init_state.pos[2]
    reward = torch.exp(-error.square() / std**2)
    if mask_delay:
        # Height tracking is a recovery-only reward in the Warp task.
        delay = _active_delay(env)
        reward = torch.where(delay, reward * delay_env_rew_ratio, 0.0)
    return reward


def body_ang_vel_xy_l2(
    env,
    std: float,
    mask_delay: bool = True,
    delay_env_rew_ratio: float = 0.0,
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("pelvis",)),
) -> torch.Tensor:
    asset = env.scene[body_cfg.name]
    body_id = body_cfg.body_ids[0]
    local_velocity = quat_apply_inverse(
        asset.data.body_quat_w.torch[:, body_id], asset.data.body_ang_vel_w.torch[:, body_id]
    )
    reward = torch.exp(-local_velocity[:, :2].square().sum(dim=-1) / std**2)
    return _mask_by_delay(env, reward, delay_env_rew_ratio) if mask_delay else reward


def feet_not_lifted_when_commanded(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "twist",
    command_threshold: float = 0.1,
    min_contact_time: float = 0.02,
    velocity_ratio: float = 0.5,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_contact_time = contact_sensor.data.current_contact_time
    if current_contact_time is None:
        raise RuntimeError("Contact sensor needs current_contact_time enabled.")

    command = env.command_manager.get_command(command_name)
    command_speed = torch.linalg.norm(command[:, :2], dim=1)
    is_commanded = command_speed > command_threshold

    base_speed = torch.linalg.norm(env.scene["robot"].data.root_lin_vel_w.torch[:, :2], dim=1)
    not_moving = base_speed < velocity_ratio * command_speed.clamp_min(command_threshold)

    feet_contact_time = current_contact_time[:, sensor_cfg.body_ids]
    both_feet_in_contact = (feet_contact_time > min_contact_time).all(dim=-1)

    return (is_commanded & not_moving & both_feet_in_contact).float()


def _active_delay(env) -> torch.Tensor:
    manager = env.termination_manager
    if hasattr(manager, "_delay_env_mask"):
        return manager._delay_env_mask & (manager._delay_counters > 0)
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _mask_by_delay(env, reward: torch.Tensor, delay_rew_ratio: float) -> torch.Tensor:
    return torch.where(_active_delay(env), reward * delay_rew_ratio, reward)

# -- Joint deviation when standing
def joint_deviation_l1_when_standing(
    env,
    asset_cfg=SceneEntityCfg("robot"),
    command_name: str = "twist",
    command_threshold: float = 0.1,
    stand_cmd: bool = True,
    vel_threshold: float = 0.25,
    angvel_threshold: float = 0.8,
) -> torch.Tensor:
    """Keep selected joints near their defaults while the robot is standing still.

    ``stand_cmd=True`` uses the command label maintained by ``AmpVelocityCommand``
    (falling back to a command-magnitude test); ``stand_cmd=False`` always uses the
    magnitude test. ``vel_threshold`` / ``angvel_threshold`` additionally gate on the
    root speed, so the term fades out while the robot is still braking.
    """
    from basic_loco_lib.tasks.velocity import mdp

    if stand_cmd:
        active = _standing_mask(env, command_name, command_threshold).float()
    else:
        command = env.command_manager.get_command(command_name)
        magnitude = torch.linalg.vector_norm(command[:, :2], dim=1) + command[:, 2].abs()
        active = (magnitude < command_threshold).float()

    if vel_threshold > 0.0 or angvel_threshold > 0.0:
        asset = env.scene[asset_cfg.name]
        if vel_threshold > 0.0:
            speed = torch.linalg.vector_norm(asset.data.root_lin_vel_b.torch[:, :2], dim=-1)
            active = active * (speed < vel_threshold).float()
        if angvel_threshold > 0.0:
            rate = torch.linalg.vector_norm(asset.data.root_ang_vel_b.torch, dim=-1)
            active = active * (rate < angvel_threshold).float()

    return mdp.joint_deviation_l1(env, asset_cfg) * active

# -- Base height above terrain with deadband
def base_height_above_terrain_deadband(
    env,
    target_height: float,
    deadband: float = 0.03,
    sensor_cfg: SceneEntityCfg | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    root_z = asset.data.root_pos_w.torch[:, 2]
    if sensor_cfg is not None:
        sensor = env.scene[sensor_cfg.name]
        hits = sensor.data.ray_hits_w.torch[..., 2]
        finite = torch.isfinite(hits)
        safe = torch.where(finite, hits, torch.zeros_like(hits))
        terrain_z = safe.sum(dim=1) / finite.sum(dim=1).clamp(min=1)
    else:
        terrain_z = env.scene.env_origins[:, 2]
    return ((root_z - terrain_z - target_height).abs() - deadband).clamp(min=0.0).square()

def _feet_positions_b(asset, feet_cfg: SceneEntityCfg) -> torch.Tensor:
    feet_w = asset.data.body_pos_w.torch[:, feet_cfg.body_ids, :]          # [N, K, 3]
    root_w = asset.data.root_pos_w.torch[:, None, :]                       # [N, 1, 3]
    root_quat_w = asset.data.root_quat_w.torch                             # [N, 4]

    rel_w = feet_w - root_w
    n, k, _ = rel_w.shape
    quat_rep = root_quat_w[:, None, :].expand(-1, k, -1).reshape(-1, 4)
    rel_w_flat = rel_w.reshape(-1, 3)
    feet_b = quat_apply_inverse(quat_rep, rel_w_flat).reshape(n, k, 3)
    return feet_b


def _standing_mask(env, command_name: str = "twist", command_threshold: float = 0.1) -> torch.Tensor:
    """站立掩码：优先用 AmpVelocityCommand 的 command_mode，否则退回指令幅值。"""
    mode = _command_mode(env, command_name)
    if mode is not None:
        return mode == COMMAND_MODE_STANDING
    cmd = env.command_manager.get_command(command_name)
    return torch.linalg.vector_norm(cmd[:, :2], dim=1) + cmd[:, 2].abs() <= command_threshold


def _stand_gate(
    env,
    sensor_cfg: SceneEntityCfg,
    torso_cfg: SceneEntityCfg,
    command_name: str = "twist",
    command_threshold: float = 0.1,
    planar_speed_scale: float = 0.2,
    torso_ang_vel_scale: float = 0.5,
    contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    """站立稳定软权重 ∈ [0,1]：站立指令 × 低基座平面速度 × 低躯干角速度 × 双脚触地。"""
    v2 = env.scene["robot"].data.root_lin_vel_w.torch[:, :2].square().sum(-1)
    w2 = env.scene[torso_cfg.name].data.body_ang_vel_w.torch[:, torso_cfg.body_ids[0]].square().sum(-1)
    forces = env.scene.sensors[sensor_cfg.name].data.net_normal_forces_w_history.torch  # [N,H,S,3]
    feet_f = torch.linalg.vector_norm(forces[:, :, sensor_cfg.body_ids], dim=-1).amax(dim=1)  # [N,K]
    contact = (feet_f > contact_force_threshold).all(dim=1)  # 只用双脚，不用全机身
    return (
        _standing_mask(env, command_name, command_threshold).float()
        * torch.exp(-v2 / planar_speed_scale**2)
        * torch.exp(-w2 / torso_ang_vel_scale**2)
        * contact.float()
    )

def _joint_err(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    ids = asset_cfg.joint_ids
    return data.joint_pos.torch[:, ids] - data.default_joint_pos.torch[:, ids]


def stable_stand_hip_roll_norm(
    env, asset_cfg: SceneEntityCfg, torso_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    command_name: str = "twist", command_threshold: float = 0.1,
    planar_speed_scale: float = 0.2, torso_ang_vel_scale: float = 0.5,
    contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    gate = _stand_gate(env, sensor_cfg, torso_cfg, command_name, command_threshold,
                       planar_speed_scale, torso_ang_vel_scale, contact_force_threshold)
    return torch.linalg.vector_norm(_joint_err(env, asset_cfg), dim=1) * gate


def stable_stand_feet_fore_aft_l1(
    env, feet_cfg: SceneEntityCfg, torso_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    position_deadband: float = 0.02, command_name: str = "twist", command_threshold: float = 0.1,
    planar_speed_scale: float = 0.2, torso_ang_vel_scale: float = 0.5,
    contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    feet_b = _feet_positions_b(env.scene[feet_cfg.name], feet_cfg)  # [N,2,3]
    err = torch.relu(feet_b[:, :, 0].abs() - position_deadband).sum(dim=1)
    gate = _stand_gate(env, sensor_cfg, torso_cfg, command_name, command_threshold,
                       planar_speed_scale, torso_ang_vel_scale, contact_force_threshold)
    return err * gate


def stable_stand_feet_lateral_center_l1(
    env, feet_cfg: SceneEntityCfg, torso_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    center_deadband: float = 0.01, command_name: str = "twist", command_threshold: float = 0.1,
    planar_speed_scale: float = 0.2, torso_ang_vel_scale: float = 0.5,
    contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    feet_b = _feet_positions_b(env.scene[feet_cfg.name], feet_cfg)
    center = 0.5 * (feet_b[:, 0, 1] + feet_b[:, 1, 1])
    err = torch.relu(center.abs() - center_deadband)
    gate = _stand_gate(env, sensor_cfg, torso_cfg, command_name, command_threshold,
                       planar_speed_scale, torso_ang_vel_scale, contact_force_threshold)
    return err * gate


def stand_torso_pitch_l2(
    env, asset_cfg: SceneEntityCfg, command_name: str = "twist", command_threshold: float = 0.1,
) -> torch.Tensor:
    quat_w = env.scene[asset_cfg.name].data.body_quat_w.torch[:, asset_cfg.body_ids[0]]
    g = torch.tensor(env.cfg.sim.gravity, device=env.device, dtype=quat_w.dtype)
    g = g / g.norm().clamp_min(1e-6)
    gravity_b = quat_apply_inverse(quat_w, g.unsqueeze(0).expand(env.num_envs, -1))
    return gravity_b[:, 0].square() * _standing_mask(env, command_name, command_threshold).float()


def zero_command_joint_l2(
    env, asset_cfg: SceneEntityCfg, command_name: str = "twist", threshold: float = 0.05,
) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    magnitude = torch.linalg.vector_norm(cmd[:, :2], dim=1) + cmd[:, 2].abs()
    return _joint_err(env, asset_cfg).square().sum(dim=1) * (magnitude < threshold).float()