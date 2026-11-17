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

def _standing_mask(env, command_name: str = "twist", command_threshold: float = 0.1) -> torch.Tensor:
    """站立掩码：优先用 AmpVelocityCommand 的 command_mode，否则退回指令幅值。"""
    mode = _command_mode(env, command_name)
    if mode is not None:
        return mode == COMMAND_MODE_STANDING
    cmd = env.command_manager.get_command(command_name)
    return torch.linalg.vector_norm(cmd[:, :2], dim=1) + cmd[:, 2].abs() <= command_threshold

def _turning_mask(env, command_name: str = "twist", command_threshold: float = 0.15) -> torch.Tensor:
    """原地转向掩码：优先用 command_mode，否则退回 |wz| 阈值。"""
    mode = _command_mode(env, command_name)
    if mode is not None:
        return mode == COMMAND_MODE_TURNING
    return env.command_manager.get_command(command_name)[:, 2].abs() > command_threshold

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

def adaptive_flat_orientation_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    gravity_xy = asset.data.projected_gravity_b.torch[:, :2]
    reward = torch.sum(torch.square(gravity_xy), dim=1)
    scale = _standing_mask(env).float() * 3.0
    return reward * scale

def _huber(e: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
    """|e|<delta 时退化成 L2（梯度随误差趋零，不振荡），否则同 L1。"""
    a = e.abs()
    return torch.where(a < delta, 0.5 * a.square() / delta, a - 0.5 * delta)

def arm_swing_natural(
    env, sh_cfg, hip_cfg,                       # joint_ids: [L, R] shoulder_pitch / hip_pitch
    swing_gain: float = 1.0, max_swing: float = 0.30, lead_time: float = 0.0,
    smooth_alpha: float = 0.18, arm_vel_gain: float = 0.03,
    elbow_cfg=None, elbow_ratio: float = 0.35, elbow_sign: float = 1.0, elbow_vel_gain: float = 0.01,
    swing_bias: float = 0.0, max_swing_back: float | None = None, max_swing_fwd: float | None = None,
    center_alpha: float = 0.0, huber: float = 0.05, gate_width: float = 0.10,
    elbow_lag: float = 0.0, command_name: str = "twist", command_threshold: float = 0.1,
) -> torch.Tensor:
    """肩 pitch 与同侧 hip pitch 反相摆动：去直流 + tanh 软饱和 + 平滑，肘可滞后跟随。"""
    asset = env.scene[sh_cfg.name]
    for cfg in (sh_cfg, hip_cfg):
        assert not isinstance(cfg.joint_ids, slice) and len(cfg.joint_ids) == 2, "需要 [left, right] 两个关节"

    q0, q, qv = asset.data.default_joint_pos.torch, asset.data.joint_pos.torch, asset.data.joint_vel.torch
    d_sh, v_sh = q[:, sh_cfg.joint_ids] - q0[:, sh_cfg.joint_ids], qv[:, sh_cfg.joint_ids]
    d_hip, v_hip = q[:, hip_cfg.joint_ids] - q0[:, hip_cfg.joint_ids], qv[:, hip_cfg.joint_ids]

    rb = getattr(env, "reset_buf", None)
    rid = rb.nonzero(as_tuple=False).flatten() if rb is not None else None

    def _ema(name, x, alpha, zero=False):
        prev = getattr(env, name, None)
        if prev is None or prev.shape != x.shape or prev.device != x.device:
            prev = torch.zeros_like(x) if zero else x.clone()
        elif rid is not None and rid.numel():
            prev = prev.clone()
            prev[rid] = 0.0 if zero else x[rid]
        out = (1.0 - alpha) * prev + alpha * x
        setattr(env, name, out)
        return out

    sig = d_hip + lead_time * _ema("_arm_hip_vel", v_hip, 0.2, zero=True)
    if center_alpha > 0.0:
        sig = sig - _ema("_arm_hip_mean", sig, center_alpha, zero=True)

    lo = -max_swing if max_swing_fwd is None else -abs(max_swing_fwd)
    hi = max_swing if max_swing_back is None else abs(max_swing_back)
    amp = max(0.5 * (abs(lo) + abs(hi)), 1e-6)
    tgt = amp * torch.tanh((-swing_gain * sig + swing_bias) / amp)

    speed = torch.linalg.vector_norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
    x = ((speed - command_threshold) / max(gate_width, 1e-6)).clamp(0.0, 1.0)
    tgt = tgt * (x * x * (3.0 - 2.0 * x)).unsqueeze(-1)

    tgt = _ema("_arm_swing_tgt", tgt, smooth_alpha)

    loss = _huber(d_sh - tgt, huber).sum(-1) + arm_vel_gain * v_sh.square().sum(-1)
    if elbow_cfg is not None:
        e_tgt = _ema("_arm_elbow_tgt", tgt, elbow_lag) if elbow_lag > 0.0 else tgt
        d_el, v_el = q[:, elbow_cfg.joint_ids] - q0[:, elbow_cfg.joint_ids], qv[:, elbow_cfg.joint_ids]
        loss = loss + _huber(d_el - elbow_sign * elbow_ratio * e_tgt, huber).sum(-1) \
                    + elbow_vel_gain * v_el.square().sum(-1)
    return loss

def arm_clear_body(
    env, asset_cfg,                             # joint_ids: L roll, R roll, L yaw, R yaw, L elbow, R elbow
    roll_offset: float = 0.05, roll_gain: float = 10.0,
    elbow_deadband: float = 0.0, elbow_vel_gain: float = 0.0,
    command_name: str = "twist", command_threshold: float = 0.1,
    yaw_gain: float = 0.15,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    assert not isinstance(asset_cfg.joint_ids, slice) and len(asset_cfg.joint_ids) == 6, "需要 6 个关节"

    dq = asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    dqd = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    off = torch.tensor([roll_offset, -roll_offset], device=dq.device, dtype=dq.dtype)
    roll_yaw = roll_gain * (dq[:, :2] - off).abs().sum(-1) + yaw_gain * dq[:, 2:4].abs().sum(-1)
    elbow = torch.relu(dq[:, 4:].abs() - elbow_deadband).sum(-1) if elbow_deadband > 0.0 else 0.0
    damp = elbow_vel_gain * dqd[:, 4:].square().sum(-1)
    active = (torch.linalg.vector_norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > command_threshold).float()
    return (roll_yaw + elbow + damp) * active

def track_ang_vel_z_exp(
    env, std: float, command_name: str = "twist",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """偏航角速度指令的指数核跟踪奖励；指令接近 0 时用更小的 std 把"别乱转"拉紧。"""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)          # [N, 3] = (vx, vy, wz)
    ang_vel_error = (command[:, 2] - asset.data.root_ang_vel_b.torch[:, 2]).square()

    total_command = torch.linalg.vector_norm(command[:, :2], dim=1) + command[:, 2].abs()
    adaptive_std = torch.where(
        total_command < 0.3,
        torch.full_like(total_command, 0.05),
        torch.full_like(total_command, std),
    )
    return torch.exp(-ang_vel_error / adaptive_std**2)

def feet_not_lifted_when_turning(
    env, sensor_cfg: SceneEntityCfg, command_name: str = "twist",
    command_threshold: float = 0.15, min_contact_time: float = 0.15,
    velocity_ratio: float = 0.5,
) -> torch.Tensor:
    """被要求原地转向，但实际几乎没转、且双脚都长时间贴地 → 罚（既不转也不抬脚）。"""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    ct = sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]     # [N, 2]

    cmd = env.command_manager.get_command(command_name)
    wz_cmd = cmd[:, 2].abs()
    is_commanded = _turning_mask(env, command_name, command_threshold)      # 用转向桶判断

    wz = env.scene["robot"].data.root_ang_vel_b.torch[:, 2].abs()
    not_turning = wz < velocity_ratio * wz_cmd.clamp_min(command_threshold)

    both_planted = (ct > min_contact_time).all(dim=-1)
    return (is_commanded & not_turning & both_planted).float()

def feet_air_time_turning(
    env, sensor_cfg: SceneEntityCfg, command_name: str = "twist",
    command_threshold: float = 0.15, min_air_time: float = 0.08, air_cap: float = 0.30,
) -> torch.Tensor:
    """原地转向时，每迈出一步（离地够久再落地）给奖励，按离地时长线性计分。"""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids].float()
    air = sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    step = torch.clamp(air, max=air_cap) * (air > min_air_time).float() * first_contact
    return step.sum(dim=-1) * _turning_mask(env, command_name, command_threshold).float()


def body_flat_orientation_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["torso_roll_link"]),
    pitch_offset: float = 0.0,
    roll_offset: float = 0.0,
    tolerance: float = 0.0,
    relative_to_root: bool = True,
    command_name: str = "twist",
    command_threshold: float = 0.1,
    only_when_standing: bool = False,
) -> torch.Tensor:

    asset = env.scene[asset_cfg.name]
    body_q = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]          # [N,B,4] xyzw
    n, b = body_q.shape[:2]
    body_q = body_q.reshape(-1, 4)
    if relative_to_root:
        root_q = asset.data.root_quat_w.torch[:, None, :].expand(n, b, 4).reshape(-1, 4)
        body_q = math_utils.quat_mul(math_utils.quat_conjugate(root_q), body_q)

    g_w = torch.tensor([0.0, 0.0, -1.0], device=body_q.device, dtype=body_q.dtype)
    g_b = quat_apply_inverse(body_q, g_w.expand(body_q.shape[0], 3)).reshape(n, b, 3)

    pitch = torch.asin(g_b[..., 0].clamp(-1.0, 1.0))      # 前倾为正
    roll = torch.asin(-g_b[..., 1].clamp(-1.0, 1.0))      # 右倾为正

    loss = ((pitch - pitch_offset).abs() - tolerance).clamp(min=0.0).square()
    loss += (roll - roll_offset).abs().sub(tolerance).clamp(min=0.0).square()
    loss = loss.sum(dim=1)
    if only_when_standing:
        loss = loss * _standing_mask(env, command_name, command_threshold).float()
    return loss