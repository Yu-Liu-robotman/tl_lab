from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils


def vel_forward_diff(data: torch.Tensor, dt: float) -> torch.Tensor:
    """Forward-difference velocity of ``data`` with shape ``(N, dim)``."""
    num_frames = data.shape[0]
    if num_frames < 2:
        raise RuntimeError(f"Input data has only {num_frames} frames, cannot compute velocity.")
    vel = torch.zeros_like(data)
    vel[:-1] = (data[1:] - data[:-1]) / dt
    vel[-1] = vel[-2]
    return vel


def ang_vel_from_quat_diff(quat: torch.Tensor, dt: float, in_frame: str = "body") -> torch.Tensor:
    """Angular velocity from consecutive quaternions ``(N, 4)`` in ``xyzw`` order."""
    if in_frame not in ("body", "world"):
        raise ValueError(f"Invalid in_frame value: {in_frame}. Must be 'body' or 'world'.")

    num_frames = quat.shape[0]
    if num_frames < 2:
        raise RuntimeError(f"Input quaternion has only {num_frames} frames, cannot compute angular velocity.")

    q1 = quat[:-1]
    q2 = quat[1:]
    diff_quat = math_utils.quat_mul(math_utils.quat_conjugate(q1), q2)
    diff_angle_axis = math_utils.axis_angle_from_quat(diff_quat)
    if in_frame == "world":
        diff_angle_axis = math_utils.quat_apply(q1, diff_angle_axis)

    ang_vel = torch.zeros((num_frames, 3), dtype=quat.dtype, device=quat.device)
    ang_vel[:-1] = diff_angle_axis / dt
    ang_vel[-1] = ang_vel[-2]
    return ang_vel


def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between ``q0`` and ``q1`` (``xyzw``)."""
    if q0.ndim >= 2:
        blend = blend.unsqueeze(-1)
    if q0.ndim >= 3:
        blend = blend.unsqueeze(-1)

    qx, qy, qz, qw = 0, 1, 2, 3
    cos_half_theta = (
        q0[..., qw] * q1[..., qw]
        + q0[..., qx] * q1[..., qx]
        + q0[..., qy] * q1[..., qy]
        + q0[..., qz] * q1[..., qz]
    )

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta.clamp(-1.0, 1.0))
    sin_half_theta = torch.sqrt(torch.clamp(1.0 - cos_half_theta * cos_half_theta, min=0.0))

    ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
    ratio_b = torch.sin(blend * half_theta) / sin_half_theta

    new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
    new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
    new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
    new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

    new_q = torch.cat([new_q_x, new_q_y, new_q_z, new_q_w], dim=len(new_q_w.shape) - 1)
    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
    return new_q


def linear_interpolate(x0: torch.Tensor, x1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """Linear interpolation between ``x0`` and ``x1`` with ``blend`` of shape ``(N, 1)``."""
    return x0 * (1 - blend) + x1 * blend


def calc_frame_blend(
    time: torch.Tensor, duration: torch.Tensor, num_frames: torch.Tensor, dt: torch.Tensor
):
    """Map motion times to neighbouring frame indices plus the blend coefficient."""
    phase = torch.clamp(time / duration, min=0.0, max=1.0)
    frame_idx0 = (phase * (num_frames - 1).float()).long()
    frame_idx1 = torch.minimum(frame_idx0 + 1, num_frames - 1)
    blend = (time - frame_idx0.float() * dt) / dt
    return frame_idx0, frame_idx1, blend
