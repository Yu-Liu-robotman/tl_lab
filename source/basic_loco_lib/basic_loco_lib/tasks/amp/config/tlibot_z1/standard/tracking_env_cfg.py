"""Tlibot Z1 (29 DoF) AMP locomotion task: thin overrides on the shared task base."""

from __future__ import annotations

import math
from pathlib import Path

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from basic_loco_lib.robots.z1_29dof import Z129DOF as ROBOT_CFG
from basic_loco_lib.robots.z1_29dof import Z129DOF_ACTION_SCALE as ACTION_SCALE
from basic_loco_lib.tasks.amp.agents.rsl_rl_ppo_cfg import Z1_AMP_MOTION_DIR
from basic_loco_lib.tasks.amp.mdp import commands as amp_commands
from basic_loco_lib.tasks.amp.mdp import events as amp_events
from basic_loco_lib.tasks.amp.mdp import observations as amp_obs
from basic_loco_lib.tasks.amp.mdp import rewards as amp_rewards
from basic_loco_lib.tasks.amp.mdp.actions import DelayedJointPositionActionCfg
from basic_loco_lib.tasks.base_cfg.base_cfg import (
    ActionsCfg as BaseActionsCfg,
    EventCfg,
    PhysicsCfg,
    RobotEnvCfg as BaseRobotEnvCfg,
    RobotSceneCfg,
)
from basic_loco_lib.tasks.velocity import mdp
from basic_loco_lib.tasks.velocity.terrain import TLIBOT_Z1_ROUGH_TERRAINS_CFG as TERRAIN_CFG

##
# Actions
##
# @configclass
# class ActionsCfg(BaseActionsCfg):
#     """Joint-position action with a 0-2 control-step actuation delay."""

#     JointPositionAction = DelayedJointPositionActionCfg(
#         asset_name="robot",
#         joint_names=[".*"],
#         scale=1.0,
#         use_default_offset=True,
#         min_delay=0,
#         max_delay=2,
#     )


##
# Commands
##
@configclass
class CommandsCfg:
    twist = amp_commands.AmpVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.20, # 站立占比
        rel_turning_envs=0.00,   # 原地转向占比
        rel_lateral_envs=0.00,   # 横移占比
        rel_heading_envs=0.0,   # 朝向占比
        heading_command=False,
        debug_vis=False,
        ranges=amp_commands.AmpVelocityCommandCfg.Ranges(
            # forward only, and kept away from 0 so it is never a "fake stand" bucket
            lin_vel_x=(-0.7, 1.2),
            lin_vel_y=(-0.8, 0.8),
            ang_vel_z=(-0.8, 0.8),
            # heading=(-math.pi / 2, math.pi / 2),
            heading=(-0, 0),
            pure_turn_ang_vel=(-0.7, 0.7),    # 纯转向桶
            pure_lateral_lin_vel=(-0.5, 0.5), # 纯横移桶
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
        projected_gravity = ObsTerm(func=amp_obs.projected_gravity_imu, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=amp_obs.base_ang_vel_imu, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 10
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
        feet_contact = ObsTerm(
            func=mdp.feet_contact_mask,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=".*ankle_roll.*", preserve_order=True
                ),
                "threshold": 5.0,
            },
        )
        def __post_init__(self):
            self.history_length = 10
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
            "std": 0.5,
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
    # track_ang_vel_z_exp = RewTerm(func=amp_rewards.track_ang_vel_z_exp, weight=2.0, params={"std": 0.25})   
    base_height = RewTerm(
        func=amp_rewards.base_height_above_terrain_deadband,
        weight=-2.0,
        # no height scanner in this scene: falls back to the env origin height
        params={"target_height": 0.865, "deadband": 0.035},
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")},
    )
    feet_not_lifted_when_turning = RewTerm(
        func=amp_rewards.feet_not_lifted_when_turning, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*", preserve_order=True),
                "command_name": "twist", "command_threshold": 0.15,
                "min_contact_time": 0.10, "velocity_ratio": 0.5},
    )
    feet_air_time_turning = RewTerm(
        func=amp_rewards.feet_air_time_turning, weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*", preserve_order=True),
                "command_name": "twist", "command_threshold": 0.15,
                "min_air_time": 0.12, "air_cap": 0.2},
    )
    # ------------------------------------------------------------------------------------------
    # arm_clear_body = RewTerm(func=amp_rewards.arm_clear_body, weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", preserve_order=True, joint_names=[
    #             "arm_left_shoulder_roll_joint", "arm_right_shoulder_roll_joint",
    #             "arm_left_shoulder_yaw_joint", "arm_right_shoulder_yaw_joint",
    #             "arm_left_elbow_joint", "arm_right_elbow_joint"]),
    #         "roll_offset": 0.05, "roll_gain": 2.0,      # 0.05 → 0：不再额外外展
    #         "yaw_gain": 0.15,                          # 新增，见下面代码改动
    #         "elbow_deadband": 0.25, "elbow_vel_gain": 0.0,
    #         "command_threshold": 0,
    #     })
    # arm_swing_natural = RewTerm(
    #     func=amp_rewards.arm_swing_natural,
    #     weight=-1.0,
    #     params={
    #         "sh_cfg": SceneEntityCfg("robot", preserve_order=True,
    #             joint_names=["arm_left_shoulder_pitch_joint", "arm_right_shoulder_pitch_joint"]),
    #         "hip_cfg": SceneEntityCfg("robot", preserve_order=True,
    #             joint_names=["leg_left_hip_pitch_joint", "leg_right_hip_pitch_joint"]),
    #         "elbow_cfg": SceneEntityCfg("robot", preserve_order=True,
    #             joint_names=["arm_left_elbow_joint", "arm_right_elbow_joint"]),
    #         "swing_gain": 0.9,
    #         "max_swing": 0.30,
    #         "max_swing_back": 1.00, "max_swing_fwd": 1.00,
    #         "swing_bias": 0.05,
    #         "lead_time": 0.0,
    #         "smooth_alpha": 0.18,
    #         "arm_vel_gain": 0.03,
    #         "elbow_ratio": 0.35, "elbow_sign": 1.0,
    #         "elbow_vel_gain": 0.01,
    #         "center_alpha": 0.002,
    #         "huber": 0.05,
    #         "gate_width": 0.10,
    #         "elbow_lag": 0.20,
    #         "command_threshold": 0.10,
    #     },
    # )
    # base_ankle_torque = RewTerm(
    #     func=mdp.joint_torques_l2, weight=-1e-4,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint"])},
    # )
    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    base_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    base_joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    base_applied_torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.01)
    base_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    base_dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    base_energy = RewTerm(func=mdp.energy, weight=-2e-5)
    base_is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)
    base_flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    # body_link -> base_link 8°  |  base_link -> world_link   |  body_link -> world_link 8° 
    base_body_orientation_l2 = RewTerm(
        func=amp_rewards.body_flat_orientation_l2, weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_roll_link"]),
                "pitch_offset": 8 * math.pi / 180,      # +8.00°，扣掉 URDF 安装角
                "roll_offset": 0.0, "tolerance": 0.02,
                "relative_to_root": False,     # ★ 关键：相对骨盆测
                "only_when_standing": True}, # ★ 结构姿态行走时也成立，不用门控
    )
    base_stand_to_commanded = RewTerm(
        func=amp_rewards.feet_not_lifted_when_commanded, weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
                "command_name": "twist", "command_threshold": 0.1,
                "min_contact_time": 0.15, "velocity_ratio": 0.5})         
    base_stand_pose = RewTerm(
        func=amp_rewards.joint_deviation_l1_when_standing, weight=-1.0,
        params={"command_threshold": 0.1, "stand_cmd": True, "vel_threshold": 0.15,
                "angvel_threshold": 0.4, "require_contact": True, "min_contact_time": 0.15,
                "tolerance": 0.10,"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_.*"])})
    base_stand_foot_step_home = RewTerm(
        func=amp_rewards.foot_step_home_exp, weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*", preserve_order=True),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*", preserve_order=True),
                "default_foot_xy": ((0.014, 0.110), (0.008, -0.111)),
                "std": 0.04, "min_air_time": 0.10,
                "command_name": "twist", "command_threshold": 0.1})
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
    # actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum = None
    ignored_action_joints: tuple[str, ...] = (".*_wrist_.*",)
    # ignored_amp_joints: tuple[str, ...] = ("arm_.*",)
    ignored_amp_joints: tuple[str, ...] = ()
    def configure_robot(self):
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_generator = TERRAIN_CFG.copy()
        self.scene.terrain.max_init_terrain_level = TERRAIN_CFG.num_rows - 1
        self.actions.JointPositionAction.scale = ACTION_SCALE
        self.sim.use_newton_actuators = False

    def __post_init__(self):
        super().__post_init__()

        self.events.init_motion_loader = EventTerm(
            func=amp_events.init_motion_loader,
            mode="startup",
            params={
                "motion_dir": str(Z1_AMP_MOTION_DIR),
                "delay_reset_env_ratio": 0.0,
                "max_delay_steps": 250,
                "max_delay_joint_velocity": 50.0,
            },
        )
        self.events.reset_from_motion = EventTerm(
            func=amp_events.reset_from_motion_data,
            mode="reset",
            params={
                "motion_dir": str(Z1_AMP_MOTION_DIR),
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            },
        )
        self.events.push_robot.interval_range_s = (5.0, 10.0)
        self.events.push_robot.params = {"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}        # IMU mounting / zero bias, resampled every episode (actor observations only)
        self.events.imu_bias = EventTerm(
            func=amp_events.randomize_imu_bias,
            mode="reset",
            params={"rpy_max": (0.02, 0.02, 0.00), "angvel_bias_max": 0.015},
        )
        motion_dir = Path(Z1_AMP_MOTION_DIR)
        if not motion_dir.is_dir():
            print(
                f"[WARN] AMP motion directory not found: {motion_dir}\n"
                "[WARN] Motion-based reset and delayed recovery are disabled. Drop Z1-format\n"
                "[WARN] .npz clips there (joint_names/body_names must use the Z1 names) or pass\n"
                "[WARN] --motion_file <dir>; without expert clips the AMP style reward is meaningless."
            )
            self.events.init_motion_loader = None
            self.events.reset_from_motion = None

    def play_mode(self):
        super().play_mode()
        self.events.push_robot = None
