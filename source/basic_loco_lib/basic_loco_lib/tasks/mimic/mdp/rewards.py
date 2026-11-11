from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

import isaaclab.utils.math as math_utils

from basic_loco_lib.tasks.mimic.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = wp.to_torch(contact_sensor.compute_first_air(env.step_dt, env.physics_dt))[:, sensor_cfg.body_ids]
    last_contact_time = wp.to_torch(contact_sensor.data.last_contact_time)[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def _yaw_lateral_axis(command: MotionCommand) -> torch.Tensor:
    return math_utils.matrix_from_quat(math_utils.yaw_quat(command.robot_anchor_quat_w))[:, :, 1]


def foot_lateral_width_excess(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    foot_left_name: str = "leg_left_ankle_roll_link",
    foot_right_name: str = "leg_right_ankle_roll_link",
    deadband: float = 0.02,
    min_width: float | None = None,
    max_width: float | None = None,
    std: float = 0.05,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    left_index = command.cfg.body_names.index(foot_left_name)
    right_index = command.cfg.body_names.index(foot_right_name)

    y_axis = _yaw_lateral_axis(command)[:, :2]
    reference = command.body_pos_relative_w[:, [left_index, right_index], :2]
    robot = command.robot_body_pos_w[:, [left_index, right_index], :2]

    reference_width = torch.abs(torch.sum((reference[:, 0] - reference[:, 1]) * y_axis, dim=-1))
    robot_width = torch.abs(torch.sum((robot[:, 0] - robot[:, 1]) * y_axis, dim=-1))

    cost = torch.clamp(robot_width - reference_width - deadband, min=0.0)
    if min_width is not None:
        cost = cost + torch.clamp(min_width - robot_width, min=0.0)
    if max_width is not None:
        cost = cost + torch.clamp(robot_width - max_width, min=0.0)
    return cost / std


def foot_yaw_rel_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    sensor_name: str = "contact_forces",
    foot_names: tuple[str, ...] = ("leg_left_ankle_roll_link", "leg_right_ankle_roll_link"),
    contact_threshold: float = 5.0,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_ids = [command.cfg.body_names.index(name) for name in foot_names]

    reference_quat = command.body_quat_relative_w[:, body_ids, :]
    robot_quat = command.robot_body_quat_w[:, body_ids, :]
    num_feet = robot_quat.shape[1]

    reference_yaw = math_utils.yaw_quat(reference_quat.reshape(-1, 4))
    robot_yaw = math_utils.yaw_quat(robot_quat.reshape(-1, 4))
    relative = math_utils.quat_mul(math_utils.quat_conjugate(reference_yaw), robot_yaw)
    _, _, relative_yaw = math_utils.euler_xyz_from_quat(relative)

    footprint_ids = [command.robot.body_names.index(name) for name in foot_names]
    forces = env.scene.sensors[sensor_name].data.net_forces_w.torch[:, footprint_ids, 2]
    in_contact = forces > contact_threshold

    yaw_error = math_utils.wrap_to_pi(relative_yaw).reshape(-1, num_feet) * in_contact.float()
    return torch.sum(torch.square(yaw_error), dim=1)


def foot_heading_body_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    sensor_name: str = "contact_forces",
    foot_names: tuple[str, ...] = ("leg_left_ankle_roll_link", "leg_right_ankle_roll_link"),
    bias: float = 0.0,
    contact_threshold: float = 0.0,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_ids = [command.cfg.body_names.index(name) for name in foot_names]
    num_feet = len(body_ids)

    base_yaw = math_utils.yaw_quat(command.robot_anchor_quat_w)
    robot_yaw = math_utils.yaw_quat(command.robot_body_quat_w[:, body_ids, :].reshape(-1, 4))
    relative = math_utils.quat_mul(math_utils.quat_conjugate(base_yaw).repeat_interleave(num_feet, dim=0), robot_yaw)
    _, _, relative_yaw = math_utils.euler_xyz_from_quat(relative)
    error = math_utils.wrap_to_pi(relative_yaw).reshape(-1, num_feet) - bias

    if contact_threshold > 0.0:
        footprint_ids = [command.robot.body_names.index(name) for name in foot_names]
        forces = env.scene.sensors[sensor_name].data.net_forces_w.torch[:, footprint_ids, 2]
        error = error * (forces > contact_threshold).float()
    return torch.sum(torch.square(error), dim=1)


def arm_clear_body(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "motion",
    roll_offset: float = 0.05,
    roll_gain: float = 10.0,
    elbow_deadband: float = 0.0,
    elbow_vel_gain: float = 0.0,
    command_threshold: float = 0.0,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    assert len(asset_cfg.joint_ids) == 6, (
        "joint_ids must be [left roll, right roll, left yaw, right yaw, left elbow, right elbow]"
    )

    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    joint_pos = joint_pos - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]

    offset = torch.tensor([roll_offset, -roll_offset], device=joint_pos.device, dtype=joint_pos.dtype)
    roll = roll_gain * torch.sum(torch.abs(joint_pos[:, :2] - offset), dim=-1)
    yaw = torch.sum(torch.abs(joint_pos[:, 2:4]), dim=-1)
    if elbow_deadband > 0.0:
        elbow = torch.relu(torch.abs(joint_pos[:, 4:]) - elbow_deadband).sum(dim=-1)
    else:
        elbow = 0.0
    damping = elbow_vel_gain * torch.sum(joint_vel[:, 4:] ** 2, dim=-1)

    command: MotionCommand = env.command_manager.get_term(command_name)
    active = (torch.linalg.vector_norm(command.anchor_lin_vel_w[:, :2], dim=1) > command_threshold).float()
    return (roll + yaw + elbow + damping) * active

def motion_joint_pos_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    asset = env.scene[asset_cfg.name]
    reference = command.joint_pos[:, asset_cfg.joint_ids]
    current = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    error = torch.square(current - reference).mean(dim=-1)
    return torch.exp(-error / std**2)


def motion_joint_vel_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    asset = env.scene[asset_cfg.name]
    reference = command.joint_vel[:, asset_cfg.joint_ids]
    current = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    error = torch.square(current - reference).mean(dim=-1)
    return torch.exp(-error / std**2)

def arm_swing_natural(
    env: ManagerBasedRLEnv,
    sh_cfg: SceneEntityCfg,
    hip_cfg: SceneEntityCfg,
    swing_gain: float = 0.65,
    max_swing: float = 0.85,
    lead_time: float = 0.15,
    smooth_alpha: float = 0.25,
    arm_vel_gain: float = 0.02,
    elbow_cfg: SceneEntityCfg | None = None,
    elbow_ratio: float = 0.4,
    elbow_sign: float = 1.0,
    elbow_vel_gain: float = 0.02,
    swing_bias: float = 0.0,
    max_swing_back: float | None = None,
    max_swing_fwd: float | None = None,
    command_name: str = "motion",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    asset = env.scene[sh_cfg.name]
    assert len(sh_cfg.joint_ids) == 2, "sh_cfg must be [left shoulder_pitch, right shoulder_pitch]"
    assert len(hip_cfg.joint_ids) == 2, "hip_cfg must be [left hip_pitch, right hip_pitch]"

    default = asset.data.default_joint_pos.torch
    joint_pos = asset.data.joint_pos.torch
    joint_vel = asset.data.joint_vel.torch

    q = joint_pos[:, sh_cfg.joint_ids] - default[:, sh_cfg.joint_ids]
    qd = joint_vel[:, sh_cfg.joint_ids]
    h = joint_pos[:, hip_cfg.joint_ids] - default[:, hip_cfg.joint_ids]
    hd = joint_vel[:, hip_cfg.joint_ids]

    raw = -swing_gain * (h + lead_time * hd) + swing_bias
    low = -max_swing if max_swing_fwd is None else -abs(max_swing_fwd)
    high = max_swing if max_swing_back is None else abs(max_swing_back)
    target = torch.clamp(raw, low, high)

    command = env.command_manager.get_term(command_name)
    speed = torch.linalg.vector_norm(command.anchor_lin_vel_w[:, :2], dim=1)
    moving = torch.clamp(speed / max(2.0 * command_threshold, 1e-3) - 1.0, 0.0, 1.0).unsqueeze(-1)
    target = target * moving

    previous = getattr(env, "_arm_swing_target", None)
    if previous is None or previous.shape != target.shape:
        smoothed = target
    else:
        smoothed = (1.0 - smooth_alpha) * previous + smooth_alpha * target
    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is not None:
        smoothed = torch.where(reset_buf.unsqueeze(-1), target, smoothed)
    env._arm_swing_target = smoothed.detach().clone()

    loss = torch.sum(torch.abs(q - smoothed), dim=-1) + arm_vel_gain * torch.sum(qd**2, dim=-1)

    if elbow_cfg is not None:
        assert len(elbow_cfg.joint_ids) == 2, "elbow_cfg must be [left elbow, right elbow]"
        qe = joint_pos[:, elbow_cfg.joint_ids] - default[:, elbow_cfg.joint_ids]
        qde = joint_vel[:, elbow_cfg.joint_ids]
        elbow_target = elbow_sign * elbow_ratio * smoothed
        loss = (
            loss
            + torch.sum(torch.abs(qe - elbow_target), dim=-1)
            + elbow_vel_gain * torch.sum(qde**2, dim=-1)
        )

    return loss

def body_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    tolerance: float = 0.0,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids, :]
    num_envs, num_bodies = quat.shape[0], quat.shape[1]

    gravity_w = torch.tensor(env.cfg.sim.gravity, device=quat.device, dtype=quat.dtype)
    gravity_w = gravity_w / gravity_w.norm().clamp_min(1e-6)
    gravity_b = math_utils.quat_apply_inverse(
        quat.reshape(-1, 4), gravity_w.unsqueeze(0).expand(num_envs * num_bodies, -1)
    ).reshape(num_envs, num_bodies, 3)

    tilt = torch.relu(torch.abs(gravity_b[..., :2]) - tolerance)
    return torch.sum(torch.square(tilt), dim=-1).mean(dim=-1)