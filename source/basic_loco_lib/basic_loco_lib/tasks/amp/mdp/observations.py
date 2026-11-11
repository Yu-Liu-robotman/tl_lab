from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply_inverse,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_body_pos_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset = env.scene[anchor_cfg.name]

    anchor_pos_w = asset.data.body_pos_w.torch[:, anchor_cfg.body_ids[0]]
    anchor_quat_w = asset.data.body_quat_w.torch[:, anchor_cfg.body_ids[0]]

    body_pos_w = asset.data.body_pos_w.torch[:, body_cfg.body_ids]
    body_quat_w = asset.data.body_quat_w.torch[:, body_cfg.body_ids]

    num_bodies = body_pos_w.shape[1]
    pos_b, _ = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)


def robot_body_ori_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset = env.scene[anchor_cfg.name]

    anchor_pos_w = asset.data.body_pos_w.torch[:, anchor_cfg.body_ids[0]]
    anchor_quat_w = asset.data.body_quat_w.torch[:, anchor_cfg.body_ids[0]]

    body_pos_w = asset.data.body_pos_w.torch[:, body_cfg.body_ids]
    body_quat_w = asset.data.body_quat_w.torch[:, body_cfg.body_ids]

    num_bodies = body_pos_w.shape[1]
    _, ori_b = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_lin_vel_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset = env.scene[anchor_cfg.name]

    body_lin_vel_w = asset.data.body_link_lin_vel_w.torch[:, body_cfg.body_ids]
    body_quat_w = asset.data.body_quat_w.torch[:, body_cfg.body_ids]

    num_bodies = body_lin_vel_w.shape[1]
    body_lin_vel_b = quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        body_lin_vel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)

    return body_lin_vel_b.reshape(env.num_envs, -1)


def robot_body_ang_vel_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset = env.scene[anchor_cfg.name]

    body_ang_vel_w = asset.data.body_link_ang_vel_w.torch[:, body_cfg.body_ids]
    body_quat_w = asset.data.body_quat_w.torch[:, body_cfg.body_ids]

    num_bodies = body_ang_vel_w.shape[1]
    body_ang_vel_b = quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        body_ang_vel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)

    return body_ang_vel_b.reshape(env.num_envs, -1)


# --------------------------------------------------------------------------- #
# IMU mounting / zero bias (actor observations only)
# --------------------------------------------------------------------------- #
def imu_bias_state(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-episode IMU bias buffers, lazily created as identity / zero."""
    if not hasattr(env, "imu_R_bias"):
        env.imu_R_bias = torch.eye(3, device=env.device).repeat(env.num_envs, 1, 1)
        env.imu_angvel_bias = torch.zeros(env.num_envs, 3, device=env.device)
    return env.imu_R_bias, env.imu_angvel_bias


def projected_gravity_imu(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """``projected_gravity`` as seen through the (misaligned) IMU frame."""
    from basic_loco_lib.tasks.velocity import mdp

    gravity = mdp.projected_gravity(env, asset_cfg)
    rot_bias, _ = imu_bias_state(env)
    return torch.bmm(rot_bias, gravity.unsqueeze(-1)).squeeze(-1)


def base_ang_vel_imu(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """``base_ang_vel`` in the IMU frame plus the constant gyro zero bias."""
    from basic_loco_lib.tasks.velocity import mdp

    ang_vel = mdp.base_ang_vel(env, asset_cfg)
    rot_bias, angvel_bias = imu_bias_state(env)
    return torch.bmm(rot_bias, ang_vel.unsqueeze(-1)).squeeze(-1) + angvel_bias
