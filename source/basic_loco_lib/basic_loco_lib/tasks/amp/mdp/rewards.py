"""Torch AMP rewards matching the Warp task's frames, scales, and delay masks."""

from __future__ import annotations

import torch
import functools          # ← 新增

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
def joint_deviation_l2_when_standing(
    env, asset_cfg=SceneEntityCfg("robot"), command_name: str = "twist",
    command_threshold: float = 0.1, stand_cmd: bool = True, deadband: float = 0.0,
):
    if stand_cmd:
        active = _standing_mask(env, command_name, command_threshold).float()
    else:
        cmd = env.command_manager.get_command(command_name)
        active = ((torch.linalg.vector_norm(cmd[:, :2], dim=1) + cmd[:, 2].abs()) < command_threshold).float()

    asset = env.scene[asset_cfg.name]
    e = (asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
         - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]).abs() - deadband
    return torch.clamp(e, min=0.0).square().sum(dim=-1) * active

# -- Joint deviation when standing
def joint_deviation_l1_when_standing_z(
    env,
    asset_cfg=SceneEntityCfg("robot"),
    command_name: str = "twist",
    command_threshold: float = 0.1,
    stand_cmd: bool = True,
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

    # if vel_threshold > 0.0 or angvel_threshold > 0.0:
    #     asset = env.scene[asset_cfg.name]
    #     if vel_threshold > 0.0:
    #         speed = torch.linalg.vector_norm(asset.data.root_lin_vel_b.torch[:, :2], dim=-1)
    #         active = active * (speed < vel_threshold).float()
    #     if angvel_threshold > 0.0:
    #         rate = torch.linalg.vector_norm(asset.data.root_ang_vel_b.torch, dim=-1)
    #         active = active * (rate < angvel_threshold).float()

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

def feet_air_time_turning_in_place(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "twist",
    command_threshold: float = 0.15,
    min_air_time: float = 0.08,
    air_cap: float = 0.30,
    minimum_support_time: float = 0.04,
    require_support: bool = True,
) -> torch.Tensor:
    """被要求原地转向时，每只脚落地就按其摆动时长给正奖励（鼓励抬脚换步）。

    单脚 credit = clamp(last_air_time, min_air_time, air_cap) / air_cap，越接近 air_cap 得分越高。
    ``require_support=True`` 时只统计“另一只脚还在支撑”的落地，避免跳过腾空惩罚动作。
    """
    if air_cap <= min_air_time:
        raise ValueError("air_cap must exceed min_air_time.")

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    assert sensor.data.last_air_time is not None, "contact sensor needs track_air_time=True"
    assert sensor.data.current_contact_time is not None

    ids = sensor_cfg.body_ids
    # IsaacLab returns a float 0/1 mask here, so cast before ``&``.
    first_contact = sensor.compute_first_contact(env.step_dt).torch[:, ids] > 0.5
    air_time = sensor.data.last_air_time.torch[:, ids]
    credit = torch.clamp(air_time, min=min_air_time, max=air_cap) / air_cap

    if require_support:
        support = torch.flip(sensor.data.current_contact_time.torch[:, ids], dims=(1,))
        landed = first_contact & (support > minimum_support_time)
    else:
        landed = first_contact

    reward = torch.sum(credit * landed.float(), dim=-1)

    is_commanded = _turning_mask(env, command_name, command_threshold)
    return reward * is_commanded.float()


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

def standing_lin_vel_xy_l2(
    env, command_name: str = "twist", command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """站立时惩罚根节点水平速度：被推时越快"被带走"，扣得越多。"""
    asset = env.scene[asset_cfg.name]
    standing = _standing_mask(env, command_name, command_threshold).float()
    return asset.data.root_lin_vel_b.torch[:, :2].square().sum(dim=-1) * standing


def standing_root_yaw_rate_l2(
    env, command_name: str = "twist", command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """站立时惩罚偏航角速度：防止被推得原地转。"""
    asset = env.scene[asset_cfg.name]
    standing = _standing_mask(env, command_name, command_threshold).float()
    return asset.data.root_ang_vel_b.torch[:, 2].square() * standing


# --------------------------------------------------------------------------- #
# Standing shaping -- ported from DroidUpE1_mjlab (src/tasks/amp/mdp/rewards.py)
# mjlab -> IsaacLab API 映射：
#   root_link_lin_vel_b -> root_lin_vel_b        body_link_pos_w  -> body_pos_w
#   body_link_quat_w    -> body_quat_w           body_link_ang_vel_w -> body_ang_vel_w
#   gravity_vec_w       -> data.GRAVITY_VEC_W    force_history    -> net_forces_w_history
#   门限：全部用"命令幅值 < command_threshold"，不是 command_mode
# --------------------------------------------------------------------------- #

def _cmd_magnitude(env, command_name: str = "twist") -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return torch.linalg.vector_norm(cmd[:, :2], dim=1) + cmd[:, 2].abs()

def moving_mask(env, command_name: str = "twist", threshold: float = 0.1) -> torch.Tensor:
    return _cmd_magnitude(env, command_name) > threshold

def _gravity_dir_b(asset, body_ids) -> torch.Tensor:
    """每帧 body 系单位重力方向, (N, K, 3)。"""
    quat = asset.data.body_quat_w.torch[:, body_ids, :]
    gravity_dir = torch.nn.functional.normalize(asset.data.GRAVITY_VEC_W.torch, dim=-1)
    return quat_apply_inverse(quat, gravity_dir.unsqueeze(1).expand(-1, quat.shape[1], -1))

def stand_torso_pitch_l2(
    env, asset_cfg: SceneEntityCfg, command_name: str = "twist",
    command_threshold: float = 0.1, pitch_offset: float = 0.0,
) -> torch.Tensor:
    """站立时躯干俯仰归零（pitch_offset 扣掉连杆安装角）。"""
    asset = env.scene[asset_cfg.name]
    g_b = _gravity_dir_b(asset, asset_cfg.body_ids)[:, 0, :]
    pitch = torch.asin(g_b[:, 0].clamp(-1.0, 1.0))
    still = (~moving_mask(env, command_name, command_threshold)).float()
    return torch.square(pitch - pitch_offset) * still

def zero_command_joint_l2(
    env, asset_cfg: SceneEntityCfg, command_name: str = "twist", threshold: float = 0.05,
) -> torch.Tensor:
    """站立时指定关节回默认角（L2）。"""
    asset = env.scene[asset_cfg.name]
    error = (asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
             - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(error), dim=-1) * (_cmd_magnitude(env, command_name) < threshold).float()

def stand_feet_heading_l2(
    env, feet_cfg: SceneEntityCfg, command_name: str = "twist", threshold: float = 0.05,
) -> torch.Tensor:
    """站立时双脚朝向对齐骨盆。"""
    asset = env.scene[feet_cfg.name]
    feet_quat_w = asset.data.body_quat_w.torch[:, feet_cfg.body_ids, :]
    fwd_b = torch.zeros((*feet_quat_w.shape[:-1], 3), device=feet_quat_w.device, dtype=feet_quat_w.dtype)
    fwd_b[..., 0] = 1.0
    fwd_w = math_utils.quat_apply(feet_quat_w, fwd_b)
    pelvis_quat_w = asset.data.root_quat_w.torch[:, None, :].expand_as(feet_quat_w)
    fwd_pelvis = quat_apply_inverse(pelvis_quat_w, fwd_w)
    heading = torch.atan2(fwd_pelvis[..., 1], fwd_pelvis[..., 0])
    return torch.sum(torch.square(heading), dim=-1) * (_cmd_magnitude(env, command_name) < threshold).float()

def _stable_stand_weight(
    env, sensor_cfg: SceneEntityCfg, torso_cfg: SceneEntityCfg,
    command_name: str = "twist", command_threshold: float = 0.1,
    planar_speed_scale: float = 0.2, torso_ang_vel_scale: float = 0.5,
    contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    """站立稳定度平滑权重：命令≈0 × 双脚接触 × 低速 × 躯干慢转。"""
    asset = env.scene[torso_cfg.name]
    planar_speed = torch.linalg.vector_norm(asset.data.root_lin_vel_b.torch[:, :2], dim=1)
    quat = asset.data.body_quat_w.torch[:, torso_cfg.body_ids, :].squeeze(1)
    ang_vel_w = asset.data.body_ang_vel_w.torch[:, torso_cfg.body_ids, :].squeeze(1)
    torso_ang_speed = torch.linalg.vector_norm(quat_apply_inverse(quat, ang_vel_w)[:, :2], dim=1)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids]
    peak = torch.linalg.vector_norm(force, dim=-1).amax(dim=1)         # (N, n_feet)
    double_support = (peak > contact_force_threshold).all(dim=-1)
    weight = torch.exp(-torch.square(planar_speed / planar_speed_scale))
    weight = weight * torch.exp(-torch.square(torso_ang_speed / torso_ang_vel_scale))
    return (_cmd_magnitude(env, command_name) < command_threshold).float() * double_support.float() * weight

def stable_stand_hip_roll_norm(
    env, asset_cfg: SceneEntityCfg, torso_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    command_name: str = "twist", command_threshold: float = 0.1,
    planar_speed_scale: float = 0.2, torso_ang_vel_scale: float = 0.5,
    contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    """站立时 hip roll 回默认（这正是你说的"hip 没收回去"）。"""
    asset = env.scene[asset_cfg.name]
    error = (asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
             - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids])
    weight = _stable_stand_weight(env, sensor_cfg, torso_cfg, command_name, command_threshold,
                                  planar_speed_scale, torso_ang_vel_scale, contact_force_threshold)
    return torch.linalg.vector_norm(error, dim=-1) * weight

def stable_stand_feet_fore_aft_l1(
    env, feet_cfg: SceneEntityCfg, torso_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    position_deadband: float = 0.02, command_name: str = "twist", command_threshold: float = 0.1,
    planar_speed_scale: float = 0.2, torso_ang_vel_scale: float = 0.5,
    contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    """站立时双脚前后位置归中（一脚在前一脚在后 -> 罚）。"""
    asset = env.scene[feet_cfg.name]
    feet = _feet_positions_b(asset, feet_cfg)
    error = torch.sum(torch.relu(torch.abs(feet[:, :, 0]) - position_deadband), dim=-1)
    weight = _stable_stand_weight(env, sensor_cfg, torso_cfg, command_name, command_threshold,
                                  planar_speed_scale, torso_ang_vel_scale, contact_force_threshold)
    return error * weight

def stable_stand_feet_lateral_center_l1(
    env, feet_cfg: SceneEntityCfg, torso_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    center_deadband: float = 0.01, command_name: str = "twist", command_threshold: float = 0.1,
    planar_speed_scale: float = 0.2, torso_ang_vel_scale: float = 0.5,
    contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    """站立时双脚横向中心归零（整体歪向一侧 -> 罚）。"""
    asset = env.scene[feet_cfg.name]
    feet = _feet_positions_b(asset, feet_cfg)
    center = 0.5 * (feet[:, 0, 1] + feet[:, 1, 1])
    error = torch.relu(torch.abs(center) - center_deadband)
    weight = _stable_stand_weight(env, sensor_cfg, torso_cfg, command_name, command_threshold,
                                  planar_speed_scale, torso_ang_vel_scale, contact_force_threshold)
    return error * weight

def _walk_style_mask(
    env,
    command_name: str = "twist",
    minimum_command: float = 0.1,
    maximum_forward_speed: float = 0.8,
    include_turning: bool = True,
) -> torch.Tensor:
    """挑出"行走风格"的移动命令：mixed 且不快跑，可选含纯转向。"""
    command = env.command_manager.get_command(command_name)
    mode = _command_mode(env, command_name)
    if mode is None:
        raise RuntimeError("walk-style foot rewards require AmpVelocityCommand.command_mode")
    walk = (mode == COMMAND_MODE_MIXED) & (command[:, 0] <= maximum_forward_speed)
    if include_turning:
        walk = walk | (mode == COMMAND_MODE_TURNING)
    return walk & (_cmd_magnitude(env, command_name) > minimum_command)

def walk_swing_foot_clearance(
    env,
    sensor_cfg: SceneEntityCfg,
    contact_height: float,
    target_clearance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    minimum_air_time: float = 0.16,
    maximum_air_time: float = 0.36,
    minimum_support_time: float = 0.04,
    minimum_speed: float = 0.1,
    maximum_forward_speed: float = 0.8,
    command_name: str = "twist",
) -> torch.Tensor:
    """有移动指令时，摆动脚抬不够高就罚（抬脚高度整形项）。

    框架适配：body/sensor 走 ``.torch``；指令走 ``command_manager``；
    地面高度用 ``env_origins``；``sensor_cfg`` 与 ``asset_cfg`` 都必须
    ``preserve_order=True``，否则左右脚索引对不上。
    """
    if target_clearance <= 0.0:
        raise ValueError("target_clearance must be positive.")
    if maximum_air_time <= minimum_air_time:
        raise ValueError("maximum_air_time must exceed minimum_air_time.")

    asset = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    assert sensor.data.current_air_time is not None, "contact sensor needs track_air_time=True"
    assert sensor.data.current_contact_time is not None
    if len(sensor_cfg.body_ids) != len(asset_cfg.body_ids):
        raise ValueError("sensor_cfg / asset_cfg must select the same feet (use preserve_order=True)")

    ground_z = env.scene.env_origins[:, 2]                                  # (N,)
    foot_height = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2] - ground_z.unsqueeze(-1)
    clearance = foot_height - contact_height
    clearance_deficit = torch.clamp(
        torch.relu(target_clearance - clearance) / target_clearance, max=1.0
    )

    air_time = sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    swing = (air_time > minimum_air_time) & (air_time < maximum_air_time)
    support = torch.flip(sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids], dims=(1,))
    supported_swing = swing & (support > minimum_support_time)

    command = env.command_manager.get_command(command_name)                 # (N, 3) = [vx, vy, wz]
    gait_mask = (_cmd_magnitude(env, command_name) > minimum_speed) \
                & (command[:, 0] <= maximum_forward_speed)

    return torch.mean(torch.square(clearance_deficit) * supported_swing.float(), dim=-1) * gait_mask.float()


def feet_crossing(env, minimum_distance: float, feet_cfg: SceneEntityCfg) -> torch.Tensor:
    """禁止两脚交叉：左脚 y 至少比右脚 y 大 minimum_distance。"""
    asset = env.scene[feet_cfg.name]
    feet = _feet_positions_b(asset, feet_cfg)
    signed = feet[:, 0, 1] - feet[:, 1, 1]
    return torch.relu(minimum_distance - signed)

def feet_spacing_non_lateral(
    env, minimum_distance: float, maximum_distance: float,
    lateral_command_threshold: float, feet_cfg: SceneEntityCfg, command_name: str = "twist",
) -> torch.Tensor:
    """非横移时强制两脚间距落在 [min, max]（横移时放开）。"""
    asset = env.scene[feet_cfg.name]
    feet = _feet_positions_b(asset, feet_cfg)
    distance = feet[:, 0, 1] - feet[:, 1, 1]
    outside = torch.relu(minimum_distance - distance) + torch.relu(distance - maximum_distance)
    non_lateral = env.command_manager.get_command(command_name)[:, 1].abs() < lateral_command_threshold
    return outside * non_lateral.float()

def feet_support_roll_l2(env, feet_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
                         moving_threshold: float = 0.1, contact_force_threshold: float = 5.0):
    """支撑脚的 roll（重力 y 分量）平方：防崴脚。"""
    asset = env.scene[feet_cfg.name]
    quat = asset.data.body_quat_w.torch[:, feet_cfg.body_ids, :]
    gravity = asset.data.GRAVITY_VEC_W.torch
    gravity = torch.nn.functional.normalize(gravity, dim=-1).unsqueeze(1).expand(-1, quat.shape[1], -1)
    roll_err = torch.square(quat_apply_inverse(quat, gravity)[:, :, 1])
    sensor: ContactSensor = env.sensors[sensor_cfg.name] if False else env.scene.sensors[sensor_cfg.name]
    peak = torch.linalg.vector_norm(
        sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids], dim=-1).amax(dim=1)
    support = (peak > contact_force_threshold).float()
    return torch.sum(roll_err * support, dim=1) * moving_mask(env).float()

def stable_stand_feet_spacing_l1(
    env, feet_cfg: SceneEntityCfg, torso_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    target_spacing: float = 0.22, spacing_deadband: float = 0.01,
    command_name: str = "twist", command_threshold: float = 0.1,
    planar_speed_scale: float = 0.4, torso_ang_vel_scale: float = 0.8,
    contact_force_threshold: float = 20.0,
) -> torch.Tensor:
    """站立时把两脚横向上间距拉回默认站姿（Z1 默认 ≈0.22 m）。"""
    asset = env.scene[feet_cfg.name]
    feet = _feet_positions_b(asset, feet_cfg)          # (N, 2, 3) base 系
    spacing = feet[:, 0, 1] - feet[:, 1, 1]            # left.y - right.y
    error = torch.relu((spacing - target_spacing).abs() - spacing_deadband)
    weight = _stable_stand_weight(env, sensor_cfg, torso_cfg, command_name, command_threshold,
                                  planar_speed_scale, torso_ang_vel_scale, contact_force_threshold)
    return error * weight

# --------------------------------------------------------------------------- #
# Upright gate -- 摔倒后把所有 shaping 项清零
# --------------------------------------------------------------------------- #

def upright_mask(
    env,
    min_root_height: float = 0.60,
    min_gravity_z: float = -0.60,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ground_z = env.scene.env_origins[:, 2]
    height = asset.data.root_pos_w.torch[:, 2] - ground_z
    gravity_z = asset.data.projected_gravity_b.torch[:, 2]
    return (height > min_root_height) & (gravity_z < min_gravity_z)


def upright_gated(func, min_root_height: float = 0.60, min_gravity_z: float = -0.60):
    """把任意 reward 函数包成"未摔倒才生效"（摔倒 → 该项恒 0）。"""
    @functools.wraps(func)
    def _wrapped(env, *args, **kwargs):
        value = func(env, *args, **kwargs)
        return value * upright_mask(env, min_root_height, min_gravity_z).float()
    return _wrapped