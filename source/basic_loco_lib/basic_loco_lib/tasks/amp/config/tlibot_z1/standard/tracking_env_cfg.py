"""Tlibot Z1 (29 DoF) AMP locomotion task: thin overrides on the shared task base."""

from __future__ import annotations

import math
from pathlib import Path

from basic_loco_lib.tasks.mimic import mdp_warp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from basic_loco_lib.tasks.base_cfg import mdp as base_mdp
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
    PushCurriculumCfg,
    RobotEnvCfg as BaseRobotEnvCfg,
    RobotSceneCfg,
)
from basic_loco_lib.tasks.amp.mdp.rewards import upright_gated as _UP   # 顶部
from basic_loco_lib.tasks.velocity import mdp
from basic_loco_lib.tasks.velocity.terrain import TLIBOT_Z1_ROUGH_TERRAINS_CFG as TERRAIN_CFG
"""
python scripts/rsl_rl/train.py --task Tlibot-Z1-AMP --viz none presets=newton
python scripts/rsl_rl/play.py --task Tlibot-Z1-AMP --num_envs 100 --viz viser presets=newton
python deploy_mujoco/task/velocity/deploy_mujoco.py tlibot_z1 --task z1_amp_locomotion
"""
##
# Actions
##
@configclass
class ActionsCfg(BaseActionsCfg):
    """Joint-position action with a 0-2 control-step actuation delay."""

    JointPositionAction = DelayedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
        min_delay=0,
        max_delay=2,
    )


##
# Commands
##
@configclass
class CommandsCfg:
    twist = amp_commands.AmpVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.2, # 站立占比
        rel_turning_envs=0.2,   # 原地转向占比
        rel_lateral_envs=0.2,   # 横移占比
        rel_heading_envs=0.0,   # 朝向占比
        heading_command=False,
        debug_vis=False,
        ranges=amp_commands.AmpVelocityCommandCfg.Ranges(
            # forward only, and kept away from 0 so it is never a "fake stand" bucket
            lin_vel_x=(-0.8, 1.0),
            lin_vel_y=(-0.8, 0.8),
            ang_vel_z=(-0.7, 0.7),
            # heading=(-math.pi / 2, math.pi / 2),
            heading=(-0, 0),
            pure_turn_ang_vel=(-0.6, 0.6),    # 纯转向桶
            pure_lateral_lin_vel=(-0.6, 0.6), # 纯横移桶
        ), 
    )

##
# Observations
##

_AMP_BODY_NAMES = (
    "base_link",
    "torso_roll_link",
    "leg_left_hip_roll_link",
    "leg_left_knee_link",
    "leg_left_ankle_roll_link",
    "leg_right_hip_roll_link",
    "leg_right_knee_link",
    "leg_right_ankle_roll_link",
    "arm_left_shoulder_roll_link",
    "arm_left_elbow_link",
    # "arm_left_wrist_yaw_link",
    "arm_right_shoulder_roll_link",
    "arm_right_elbow_link",
    # "arm_right_wrist_yaw_link",
)
_ANCHOR_NAME = "base_link"
_AMP_ANCHOR_NAME = "torso_roll_link"

def _amp_body_cfgs() -> dict:
    """Fresh SceneEntityCfg per term -- the manager resolves them in place."""
    return {
        "anchor_cfg": SceneEntityCfg("robot", body_names=_AMP_ANCHOR_NAME),
        "body_cfg": SceneEntityCfg("robot", body_names=list(_AMP_BODY_NAMES), preserve_order=True),
    }

@configclass
class ObservationsCfg:
    """Observation specifications for AMP."""

    @configclass
    class ActorCfg(ObsGroup):
        # term order mirrors the velocity task's policy group
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "twist"})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.06, n_max=0.06))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
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
        feet_contact = ObsTerm(func=mdp.feet_contact_mask,params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*", preserve_order=True),"threshold": 5.0})
        
        def __post_init__(self):
            self.history_length = 5
            self.concatenate_terms = True

    @configclass
    class AmpCfg(ObsGroup):
        # joint_pos = ObsTerm(func=mdp.joint_pos)  # N
        # joint_vel = ObsTerm(func=mdp.joint_vel)  # N
        body_pos_b = ObsTerm(func=amp_obs.robot_body_pos_b, params=_amp_body_cfgs()) # _AMP_BODY_NAMES * 3
        body_ori_b = ObsTerm(func=amp_obs.robot_body_ori_b, params=_amp_body_cfgs()) # _AMP_BODY_NAMES * 6
        body_lin_vel_b = ObsTerm(func=amp_obs.robot_body_lin_vel_b, params=_amp_body_cfgs())
        body_ang_vel_b = ObsTerm(func=amp_obs.robot_body_ang_vel_b, params=_amp_body_cfgs()) # _AMP_BODY_NAMES * 3
        # root_rot_tan_norm = ObsTerm(func=amp_obs.robot_root_local_rot_tan_norm,params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")})

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
        weight=1.5,
        params={
            "command_name": "twist",
            "std": 1.0,
            "mask_delay": False,
            "delay_env_rew_ratio": 0.0,
            "anchor_cfg": SceneEntityCfg("robot", body_names=_ANCHOR_NAME),
        },
    )
    track_anchor_ang_vel = RewTerm(
        func=amp_rewards.track_anchor_angular_velocity,
        weight=1.5,
        params={
            "command_name": "twist",
            "std": 1.0,
            "mask_delay": False,
            "delay_env_rew_ratio": 0.0,
            "anchor_cfg": SceneEntityCfg("robot", body_names=_ANCHOR_NAME),
        },
    )
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
    base_flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    # body_link -> base_link 8°  |  base_link -> world_link   |  body_link -> world_link 8° 
    base_body_orientation_l2 = RewTerm(
        func=amp_rewards.body_flat_orientation_l2, weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_roll_link"]),
                "pitch_offset": 8 * math.pi / 180,      # +8.00°，扣掉 URDF 安装角
                "roll_offset": 0.0, "tolerance": 0.5 * math.pi / 180,
                "relative_to_root": False,     
                "only_when_standing": True}, 
    )
    base_stand_to_commanded = RewTerm(
        func=_UP(amp_rewards.feet_not_lifted_when_commanded), weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
                "command_name": "twist", "command_threshold": 0.1,
                "min_contact_time": 0.15, "velocity_ratio": 0.5})  
    
    feet_not_lifted_when_turning = RewTerm(
        func=_UP(amp_rewards.feet_not_lifted_when_turning), weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*", preserve_order=True),
                "command_name": "twist", "command_threshold": 0.15,
                "min_contact_time": 0.10, "velocity_ratio": 0.5},
    )

    # ---- walk gait shaping (ported from DroidUpE1_mjlab) ----
    swing_foot_clearance = RewTerm(
        func=_UP(amp_rewards.walk_swing_foot_clearance), weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*", preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*", preserve_order=True),
            "contact_height": 0.057,
            "target_clearance": 0.035,
            "minimum_air_time": 0.04,
            "maximum_air_time": 0.45,
            "minimum_support_time": 0.02,
            "minimum_speed": 0.1,
            "maximum_forward_speed": 1.0,
            "command_name": "twist",
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")},
    )
    # ---- standing shaping (ported from DroidUpE1_mjlab) ----
    stand_hip_yaw = RewTerm(
        func=_UP(amp_rewards.zero_command_joint_l2), weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])})
    stand_torso_yaw = RewTerm(                                 # 参考的 waist_yaw
        func=_UP(amp_rewards.zero_command_joint_l2), weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["torso_yaw_joint"])})
    stand_feet_heading = RewTerm(
        func=_UP(amp_rewards.stand_feet_heading_l2), weight=-2.0,
        params={"feet_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"], preserve_order=True)})
    stable_stand_hip_roll = RewTerm(
        func=_UP(amp_rewards.stable_stand_hip_roll_norm), weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"]),
                "torso_cfg": SceneEntityCfg("robot", body_names=["torso_roll_link"]),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*",
                                             preserve_order=True),
                "planar_speed_scale": 0.4, "torso_ang_vel_scale": 0.8,
                "contact_force_threshold": 20.0})
    stable_stand_feet_fore_aft = RewTerm(
        func=_UP(amp_rewards.stable_stand_feet_fore_aft_l1), weight=-2.0,
        params={"feet_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"], preserve_order=True),
                "torso_cfg": SceneEntityCfg("robot", body_names=["torso_roll_link"]),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*",
                                             preserve_order=True),
                "planar_speed_scale": 0.4, "torso_ang_vel_scale": 0.8,
                "contact_force_threshold": 20.0})
    stable_stand_feet_lateral_center = RewTerm(
        func=_UP(amp_rewards.stable_stand_feet_lateral_center_l1), weight=-5.0,
        params={"feet_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"], preserve_order=True),
                "torso_cfg": SceneEntityCfg("robot", body_names=["torso_roll_link"]),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*",
                                             preserve_order=True),
                "planar_speed_scale": 0.2, "torso_ang_vel_scale": 0.8,
                "contact_force_threshold": 5.0})
    stable_stand_feet_spacing = RewTerm(
        func=_UP(amp_rewards.stable_stand_feet_spacing_l1), weight=-8.0,
        params={"feet_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"], preserve_order=True),
                "torso_cfg": SceneEntityCfg("robot", body_names=["torso_roll_link"]),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*", preserve_order=True),
                "target_spacing": 0.22, "spacing_deadband": 0.01,
                "planar_speed_scale": 0.4, "torso_ang_vel_scale": 0.8,
                "contact_force_threshold": 20.0})
    base_stand_lin_vel = RewTerm(func=_UP(amp_rewards.standing_lin_vel_xy_l2), weight=-2.0,
        params={"command_name": "twist", "command_threshold": 0.1})
    feet_crossing = RewTerm(
        func=_UP(amp_rewards.feet_crossing), weight=-2.0,
        params={"minimum_distance": 0.16,
                "feet_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"], preserve_order=True)})
    feet_spacing = RewTerm(
        func=_UP(amp_rewards.feet_spacing_non_lateral), weight=-1.0,
        params={"minimum_distance": 0.18, "maximum_distance": 0.28,
                "lateral_command_threshold": 0.1,
                "feet_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"], preserve_order=True)})
    feet_support_roll = RewTerm(
        func=_UP(amp_rewards.feet_support_roll_l2), weight=-2.0,
        params={"feet_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"], preserve_order=True),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*", preserve_order=True)})
##
# Terminations
##
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(60.0)},
    )
    bad_root_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.4},
    )
##
# Curriculum
##
# @configclass
# class CurriculumCfg(PushCurriculumCfg):
#     terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


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
    curriculum: PushCurriculumCfg = PushCurriculumCfg()
    ignored_action_joints: tuple[str, ...] = (".*_wrist_.*",)
    # ignored_amp_joints: tuple[str, ...] = ("arm_.*",)
    ignored_amp_joints: tuple[str, ...] = (".*_wrist_.*",)
    def configure_robot(self):
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.terrain.terrain_generator = TERRAIN_CFG.copy()
        # self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.actions.JointPositionAction.scale = ACTION_SCALE
        # self.actions.JointPositionAction.max_delay = 1
        # self.actions.JointPositionAction.min_delay = 0
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
                "motion_reset_ratio": 0.8,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            },
        )
        # ---------------地形课程-------------
        # self.scene.terrain.terrain_generator.curriculum = True

        # ---------------扰动-------------
        self.events.push_robot.interval_range_s = (7.0, 10.0)
        self.curriculum.push_velocity = CurrTerm(
                func=base_mdp.push_velocity_levels,
                params={
                    "event_name": "push_robot",
                    "nominal_range": {"x": 1.00, "y": 1.00},
                    "start_scale": 0.0,
                    "end_scale": 1.0,
                    "ramp_steps": 24 * 12000,
                },
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
