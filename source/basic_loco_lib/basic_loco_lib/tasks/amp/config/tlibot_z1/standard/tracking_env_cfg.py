"""Tlibot Z1 (29 DoF) AMP locomotion task: commands, observations, rewards, terminations."""

from __future__ import annotations

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from basic_loco_lib.tasks.amp.config.tlibot_z1.standard.base_cfg import (
    EventCfg,
    PhysicsCfg,
    RobotEnvCfg as BaseRobotEnvCfg,
    RobotSceneCfg,
    Z1CollisionPipelineCfg,
)
from basic_loco_lib.tasks.amp.mdp import commands as amp_commands
from basic_loco_lib.tasks.amp.mdp import observations as amp_obs
from basic_loco_lib.tasks.amp.mdp import rewards as amp_rewards
from basic_loco_lib.tasks.velocity import mdp

__all__ = [
    "CommandsCfg",
    "EventCfg",
    "ObservationsCfg",
    "PhysicsCfg",
    "RewardsCfg",
    "RobotEnvCfg",
    "RobotSceneCfg",
    "TerminationsCfg",
    "Z1CollisionPipelineCfg",
]


##
# Commands
##


@configclass
class CommandsCfg:
    twist = amp_commands.AmpVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.20,
        rel_turning_envs=0.20,
        rel_lateral_envs=0.0,
        turning_deadband=0.2,
        lateral_deadband=0.1,
        rel_heading_envs=0.25,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        # Z1 is a ~0.9 m robot: roughly half of the G1 command envelope.
        ranges=amp_commands.AmpVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.70, 1.20),
            lin_vel_y=(-0.0, 0.0),
            ang_vel_z=(-0.8, 0.8),
            heading=(-math.pi / 2, math.pi / 2),
            pure_turn_ang_vel=(-0.6, 0.6),
            pure_lateral_lin_vel=(-0.3, 0.3),
        ),
    )


##
# Observations
##

_AMP_BODY_NAMES = (
    "base_link",
    "leg_left_hip_roll_link",
    "leg_left_knee_link",
    "leg_left_ankle_roll_link",
    "leg_right_hip_roll_link",
    "leg_right_knee_link",
    "leg_right_ankle_roll_link",
    "arm_left_shoulder_roll_link",
    "arm_left_elbow_link",
    "arm_left_wrist_yaw_link",
    "arm_right_shoulder_roll_link",
    "arm_right_elbow_link",
    "arm_right_wrist_yaw_link",
)
_ANCHOR_NAME = "torso_yaw_link"


def _amp_body_cfgs() -> dict:
    """Fresh SceneEntityCfg per term -- the manager resolves them in place."""
    return {
        "anchor_cfg": SceneEntityCfg("robot", body_names=_ANCHOR_NAME),
        "body_cfg": SceneEntityCfg("robot", body_names=list(_AMP_BODY_NAMES), preserve_order=True),
    }


@configclass
class ObservationsCfg:
    """Observation specifications for AMP."""

    @configclass
    class ActorCfg(ObsGroup):
        # term order mirrors the velocity task's policy group
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "twist"})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 4
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        # term order mirrors the velocity task's critic group
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "twist"})
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        body_pos_b = ObsTerm(func=amp_obs.robot_body_pos_b, params=_amp_body_cfgs())
        body_ori_b = ObsTerm(func=amp_obs.robot_body_ori_b, params=_amp_body_cfgs())

        def __post_init__(self):
            self.history_length = 4
            self.concatenate_terms = True

    @configclass
    class AmpCfg(ObsGroup):
        body_pos_b = ObsTerm(func=amp_obs.robot_body_pos_b, params=_amp_body_cfgs())
        body_ori_b = ObsTerm(func=amp_obs.robot_body_ori_b, params=_amp_body_cfgs())
        body_lin_vel_b = ObsTerm(func=amp_obs.robot_body_lin_vel_b, params=_amp_body_cfgs())
        body_ang_vel_b = ObsTerm(func=amp_obs.robot_body_ang_vel_b, params=_amp_body_cfgs())

        def __post_init__(self):
            self.concatenate_terms = True

    actor: ActorCfg = ActorCfg()
    critic: CriticCfg = CriticCfg()
    amp: AmpCfg = AmpCfg()


##
# Rewards
##


@configclass
class RewardsCfg:
    track_anchor_lin_vel = RewTerm(
        func=amp_rewards.track_anchor_linear_velocity,
        weight=1.0,
        params={
            "command_name": "twist",
            "std": 1.0,
            "mask_delay": True,
            "delay_env_rew_ratio": 0.0,
            "anchor_cfg": SceneEntityCfg("robot", body_names=_ANCHOR_NAME),
        },
    )
    track_anchor_ang_vel = RewTerm(
        func=amp_rewards.track_anchor_angular_velocity,
        weight=1.0,
        params={
            "command_name": "twist",
            "std": 1.0,
            "mask_delay": True,
            "delay_env_rew_ratio": 0.0,
            "anchor_cfg": SceneEntityCfg("robot", body_names=_ANCHOR_NAME),
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.84})

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    stand_to_action_commanded = RewTerm(
        func=amp_rewards.feet_not_lifted_when_commanded,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "command_name": "twist",
            "command_threshold": 0.3,
            "min_contact_time": 0.15,
            "velocity_ratio": 0.5,
        },
    )

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    applied_torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.01)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)


##
# Terminations
##


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(70.0)},
    )
    bad_root_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.5},
    )


##
# Environment configuration
##


@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    """Tlibot Z1 AMP locomotion environment with the task-specific MDP groups."""

    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        kept = list(self.action_joint_names)
        for group in (self.observations.actor, self.observations.critic):
            for term_name in ("joint_pos", "joint_vel"):
                term = getattr(group, term_name, None)
                if term is not None:
                    term.params = {
                        **(term.params or {}),
                        "asset_cfg": SceneEntityCfg("robot", joint_names=list(kept)),
                    }
