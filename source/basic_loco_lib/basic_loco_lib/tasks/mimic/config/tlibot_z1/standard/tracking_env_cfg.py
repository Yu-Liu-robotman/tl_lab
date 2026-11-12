"""Tlibot-Z1 mimic tracking configuration supporting PhysX and Newton backends."""

from __future__ import annotations
from basic_loco_lib import PROJECT_DIR

from basic_loco_lib.tasks.mimic import mdp_warp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg
from basic_loco_lib.tasks.base_cfg.base_cfg import (
    ActionsCfg as BaseActionsCfg,
    EventCfg,
    PhysicsCfg,
    RobotEnvCfg as BaseRobotEnvCfg,
    RobotSceneCfg,
)
import basic_loco_lib.tasks.mimic.mdp as mdp
import basic_loco_lib.tasks.mimic.mdp.rewards as mimic_rew
from basic_loco_lib.robots.z1_29dof import Z129DOF as ROBOT_CFG
from basic_loco_lib.robots.z1_29dof import Z129DOF_ACTION_SCALE as ACTION_SCALE
from basic_loco_lib.tasks.amp.agents.rsl_rl_ppo_cfg import Z1_AMP_MOTION_DIR
##

"""
python scripts/rsl_rl/train.py --task Tlibot-Z1-Mimic --viz none presets=newton 
python scripts/rsl_rl/play.py --task Tlibot-Z1-Mimic --num_envs 10 --viz viser presets=newton

python scripts/rsl_rl/play.py --task Tlibot-Z1-Mimic --num_envs 10 --viz None presets=newton \
    --record_csv  data/z1_amp/test/output/csv/z1_forward.csv \
    --record_steps 1000 \
    --skip_steps 50 \
    --load_run 2026-11-11_18-23-47-062 \

"""
VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}
MOTION_FILE = f"{PROJECT_DIR}/data/z1_amp/test/output/npz/output.npz"
ANCHOR_BODY_NAME = "base_link"
TRACK_BODY_NAMES = [
    "base_link",
    "leg_left_hip_pitch_link",
    "leg_left_knee_link",
    "leg_left_ankle_roll_link",
    "leg_right_hip_pitch_link",
    "leg_right_knee_link",
    "leg_right_ankle_roll_link",
    # "torso_pitch_link",
    # "arm_left_elbow_link",
    # "arm_right_elbow_link",
]

##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
            asset_name="robot",
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=False,
            motion_file=MOTION_FILE,
            anchor_body_name=ANCHOR_BODY_NAME,
            body_names=TRACK_BODY_NAMES,
            pose_range={},
            velocity_range={},
            joint_position_range=(-0.15, 0.15),
            adaptive_kernel_size=3,
            adaptive_lambda=0.8,
            adaptive_uniform_ratio=0.1,
            adaptive_alpha=0.001,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=ACTION_SCALE, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            # Enable multi-frame history for temporal information
            self.history_length = 4  # Use 4 frames of history

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # Enable multi-frame history for temporal information
            self.history_length = 4  # Use 4 frames of history

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Events supported by both Newton and PhysX backends."""

    add_joint_default_pos = None

    base_com = None
    physics_material = None
    push_robot = None


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- base
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-3e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # -- tracking
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    # ------
    foot_width = RewTerm(
        func=mimic_rew.foot_lateral_width_excess,
        weight=-2.0,
        params={"deadband": 0.02, "min_width": 0.13, "max_width": None, "std": 0.05},
    )
    foot_yaw_rel = RewTerm(func=mimic_rew.foot_yaw_rel_l2, weight=-0.2)
    foot_heading = RewTerm(
        func=mimic_rew.foot_heading_body_l2,
        weight=-0.5,
        params={"bias": 0.0, "contact_threshold": 0.0},
    )
    arm_clear_body = RewTerm(func=mimic_rew.arm_clear_body, weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", preserve_order=True, joint_names=[
                "arm_left_shoulder_roll_joint", "arm_right_shoulder_roll_joint",
                "arm_left_shoulder_yaw_joint", "arm_right_shoulder_yaw_joint",
                "arm_left_elbow_joint", "arm_right_elbow_joint"]),
            "roll_offset": 0.05, "roll_gain": 2.0,
            "elbow_deadband": 0.2, "elbow_vel_gain": 0.0,
            "command_threshold": 0,
        })
    arm_swing_natural = RewTerm(
        func=mimic_rew.arm_swing_natural,
        weight=-1.0,
        params={
            "sh_cfg": SceneEntityCfg("robot", preserve_order=True,
                joint_names=["arm_left_shoulder_pitch_joint", "arm_right_shoulder_pitch_joint"]),
            "hip_cfg": SceneEntityCfg("robot", preserve_order=True,
                joint_names=["leg_left_hip_pitch_joint", "leg_right_hip_pitch_joint"]),
            "elbow_cfg": SceneEntityCfg("robot", preserve_order=True,
                joint_names=["arm_left_elbow_joint", "arm_right_elbow_joint"]),
            "swing_gain": 0.55, "max_swing": 1.00, "max_swing_back": 0.30,
            "swing_bias": -0.05, "lead_time": 0.08, "smooth_alpha": 0.10,
            "arm_vel_gain": 0.02, "elbow_ratio": 0.22, "elbow_sign": 1.0,
            "elbow_vel_gain": 0.01, "command_threshold": 0,
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["torso_.*yaw.*_joint"])},
    )
    body_orientation_l2 = RewTerm(
        func=mimic_rew.body_orientation_l1,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_roll_link"),
            "tolerance": 0.05,
            "pitch_target": 0.08,
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-6.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-1.0)
    # ------
    motion_knee_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=0.25,
        params={
            "command_name": "motion",
            "std": 0.15,
            "body_names": ["leg_left_knee_link", "leg_right_knee_link"],
        },
    )
    motion_knee_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=0.25,
        params={
            "command_name": "motion",
            "std": 0.20,
            "body_names": ["leg_left_knee_link", "leg_right_knee_link"],
        },
    )
    motion_joint_pos = RewTerm(
        func=mdp.motion_joint_pos_error_exp,
        weight=0.8,
        params={
            "command_name": "motion",
            "std": 0.25,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["leg_.*_joint"]),
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    bad_motion_body_pos_z = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": ["leg_left_ankle_roll_link", "leg_right_ankle_roll_link"],
        },
    )

##


@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None
    ignored_action_joints: tuple[str, ...] = (".*_wrist_.*",)
    def configure_robot(self):
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.actions.JointPositionAction.scale = ACTION_SCALE
        self.sim.use_newton_actuators = False

    def __post_init__(self):
        super().__post_init__()         
        self.episode_length_s = 20.0     
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
