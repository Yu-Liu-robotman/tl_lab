"""Torch AMP rewards matching the Warp task's frames, scales, and delay masks."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, quat_apply_yaw

import isaaclab.utils.math as math_utils
from basic_loco_lib.tasks.velocity import mdp

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
    env, asset_cfg=SceneEntityCfg("robot"), command_name: str = "twist",
    command_threshold: float = 0.1, stand_cmd: bool = True,
    vel_threshold: float = 0.25, angvel_threshold: float = 0.8,
    tolerance: float = 0.0,
    require_contact: bool = False,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
    min_contact_time: float = 0.15,
) -> torch.Tensor:
    """Keep joints near default while standing, with a deadband so balance work is free."""
    if stand_cmd:
        active = _standing_mask(env, command_name, command_threshold).float()
    else:
        command = env.command_manager.get_command(command_name)
        magnitude = torch.linalg.vector_norm(command[:, :2], dim=1) + command[:, 2].abs()
        active = (magnitude < command_threshold).float()

    asset = env.scene[asset_cfg.name]
    if vel_threshold > 0.0:
        speed = torch.linalg.vector_norm(asset.data.root_lin_vel_b.torch[:, :2], dim=-1)
        active = active * (speed < vel_threshold).float()
    if angvel_threshold > 0.0:
        rate = torch.linalg.vector_norm(asset.data.root_ang_vel_b.torch, dim=-1)
        active = active * (rate < angvel_threshold).float()
    if require_contact:
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        ct = sensor.data.current_contact_time[:, sensor_cfg.body_ids]
        active = active * (ct > min_contact_time).all(dim=-1).float()

    q = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    q0 = asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    return (torch.abs(q - q0) - tolerance).clamp(min=0.0).sum(dim=-1) * active

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

def body_orientation_l1(
    env,
    asset_cfg,
    tolerance: float = 0.02,
    pitch_target: float = 0.0,
    command_name: str = "twist",
    command_threshold: float = 0.1,
    vel_threshold: float = 0.0,
    angvel_threshold: float = 0.0,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids, :]
    num_envs, num_bodies = quat.shape[0], quat.shape[1]
    gravity_w = torch.tensor(env.cfg.sim.gravity, device=quat.device, dtype=quat.dtype)
    gravity_w = gravity_w / gravity_w.norm().clamp_min(1e-6)
    gravity_b = math_utils.quat_apply_inverse(
        quat.reshape(-1, 4), gravity_w.unsqueeze(0).expand(num_envs * num_bodies, -1)
    ).reshape(num_envs, num_bodies, 3)
    pitch = torch.asin(gravity_b[..., 0].clamp(-1.0, 1.0))
    roll = torch.asin(gravity_b[..., 1].clamp(-1.0, 1.0))
    lean = torch.relu(torch.abs(pitch - pitch_target) - tolerance) + torch.relu(torch.abs(roll) - tolerance)
    active = _standing_mask(env, command_name, command_threshold).float()
    if vel_threshold > 0.0:    
        speed = torch.linalg.vector_norm(asset.data.root_lin_vel_b.torch[:, :2], dim=-1)
        active = active * (speed < vel_threshold).float()
    if angvel_threshold > 0.0: 
        rate = torch.linalg.vector_norm(asset.data.root_ang_vel_b.torch, dim=-1)
        active = active * (rate < angvel_threshold).float()
    return lean.mean(dim=-1) * active


def foot_step_home_exp(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    default_foot_xy=((0.0, 0.115), (0.0, -0.115)),   # base 系 (x, y)，需要实测
    std: float = 0.05,
    min_air_time: float = 0.06,
    command_name: str = "twist",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """站立时迈出的一步，落点接近默认站位就给奖励（奖励"抬脚回位"而不是"拖脚回位"）。"""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids].float()
    air_time = sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]

    asset = env.scene[asset_cfg.name]
    rel = asset.data.body_pos_w.torch[:, asset_cfg.body_ids] - asset.data.root_pos_w.torch[:, None]
    quat = asset.data.root_quat_w.torch[:, None].expand(-1, rel.shape[1], -1).reshape(-1, 4)
    feet_b = math_utils.quat_apply_inverse(quat, rel.reshape(-1, 3)).reshape(rel.shape)

    ref = torch.tensor(default_foot_xy, device=rel.device, dtype=rel.dtype)
    err = torch.linalg.vector_norm(feet_b[:, :, :2] - ref, dim=-1)
    landed = (air_time > min_air_time).float() * first_contact
    reward = torch.exp(-err / std) * landed

    active = _standing_mask(env, command_name, command_threshold).float()
    return reward.sum(dim=-1) * active