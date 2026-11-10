"""Tlibot Z1 (29 DoF) velocity locomotion configuration.

Ported from ``tasks/velocity/config/unitree_g1/standard/velocity_env_cfg.py``; the
MDP structure, observation layout (3 + 3 + 3 + 29 + 29 + 29 = 96 per frame) and
reward set are kept identical so the two tasks are directly comparable.

Robot-specific adaptations for Z1:

* joint naming: ``leg_left_hip_pitch_joint`` / ``torso_yaw_joint`` /
  ``arm_left_shoulder_pitch_joint`` (G1 uses ``left_hip_pitch_joint`` /
  ``waist_yaw_joint`` / ``left_shoulder_pitch_joint``);
* foot bodies: ``leg_{left,right}_ankle_roll_link`` -> regex ``.*ankle_roll.*``;
* per-joint action scale taken from ``Z129DOF_ACTION_SCALE`` instead of a single
  scalar (Z1 mixes RH85/RH63L/RH63/RH53 drives with very different torque limits).
"""

from __future__ import annotations

import math
import re

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg

from basic_loco_lib.robots.z1_29dof import Z129DOF as ROBOT_CFG
from basic_loco_lib.robots.z1_29dof import Z129DOF_ACTION_SCALE
from basic_loco_lib.tasks.velocity import mdp
from basic_loco_lib.tasks.velocity.terrain import COBBLESTONE_ROAD_CFG, TLIBOT_Z1_ROUGH_TERRAINS_CFG


# Standing height used by the base-height reward/termination. The Z1 spawns at
# z = 0.865 with slightly bent knees; retune if the robot settles noticeably
# higher or lower than this during the first few policy steps.
TARGET_BASE_HEIGHT = 0.82

# All actuated joints of the robot, in SDK/URDF order. Used to compute the
# complement of ``RobotEnvCfg.ignored_action_joints``.
ALL_JOINTS: list[str] = list(ROBOT_CFG.joint_sdk_names)


# @configclass
# class PhysicsCfg(PresetCfg):
#     physx: PhysxCfg = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
#     newton: NewtonCfg = NewtonCfg(
#         solver_cfg=MJWarpSolverCfg(
#             njmax=512,
#             nconmax=64,
#             cone="pyramidal",
#             impratio=1,
#             integrator="implicitfast",
#             ls_parallel=True,
#             ls_iterations=50,
#         ),
#         num_substeps=1,
#         debug_mode=False,
#     )
#     # Default backend for this task. Override with `presets=newton` on the CLI.
#     default = physx
@configclass
class Z1CollisionPipelineCfg(NewtonCollisionPipelineCfg):

    max_triangle_pairs: int = 4_000_000
    contact_reduction_hashtable_size_factor: float = 0.5

@configclass
class PhysicsCfg(PresetCfg):
    physx: PhysxCfg = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    newton: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=512,
            nconmax=64,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
            ls_iterations=50,
            use_mujoco_contacts=False,        # ① 改用 Newton 自己的 CollisionPipeline
        ),
        default_shape_cfg=NewtonShapeCfg(margin=0.01),   # ② mesh/heightfield 地形必须非零 margin
        collision_cfg=Z1CollisionPipelineCfg(        # ③ 管线参数
            broad_phase="explicit",
            reduce_contacts=True,
            max_triangle_pairs=1_000_000,
        ),
        num_substeps=1,
        debug_mode=False,
    )
    default = physx

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with the Z1 robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=TLIBOT_Z1_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=TLIBOT_Z1_ROUGH_TERRAINS_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    # Height observations are disabled, so no height scanner is required.
    height_scanner = None
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Event terms for the MDP, shared by PhysX and Newton backends."""

    # startup
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            # Z1 has no ``torso_link``: the trunk chain is
            # torso_yaw_link -> torso_pitch_link -> torso_roll_link (arms hang off the last one).
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_roll_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_roll_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.2, 0.2)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    Z1 uses per-joint scales (``Z129DOF_ACTION_SCALE``) because a single scalar
    would under-drive the strong RH85 hip/knee joints and over-drive the small
    wrist drives.
    """

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=Z129DOF_ACTION_SCALE, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    applied_torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.01)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "torso_.*_joint",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": TARGET_BASE_HEIGHT})

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Z1 locomotion velocity-tracking environment."""

    # Joints removed from the action space (regex, matched against the robot's joint names).
    # env.ignored_action_joints='[".*_wrist_.*", "torso_yaw_joint"]'
    ignored_action_joints: tuple[str, ...] = (".*_wrist_.*",)

    # Physics backend (preset-switchable)
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())
    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        # Drop the ignored joints from the action space. Their actuators keep the
        # default position target, so they behave as passive locks rather than
        # being driven by (and penalized through) the policy.
        patterns = tuple(self.ignored_action_joints or ())
        kept = [name for name in ALL_JOINTS if not any(re.fullmatch(p, name) for p in patterns)]
        dropped = [name for name in ALL_JOINTS if name not in kept]
        self.actions.JointPositionAction.joint_names = kept
        # The per-joint scale dict must be narrowed to the actuated subset too,
        # otherwise Isaac Lab raises "no match" for the dropped joints.
        scale_cfg = self.actions.JointPositionAction.scale
        if isinstance(scale_cfg, dict):
            self.actions.JointPositionAction.scale = {name: scale_cfg[name] for name in kept if name in scale_cfg}

        # Joint-state observations use the same subset: frozen joints stay at the
        # default pose, so their (near-constant) state would only add 12 dims per
        # frame without giving the policy any actionable information.
        # NOTE: a fresh SceneEntityCfg per term -- the manager resolves the cfg in
        # place (fills joint_ids), so sharing one instance across terms fails with
        # "Both 'joint_names' and 'joint_ids' are specified, and are not consistent".
        for group in (self.observations.policy, self.observations.critic):
            for term_name in ("joint_pos_rel", "joint_vel_rel"):
                term = getattr(group, term_name, None)
                if term is not None:
                    term.params = {
                        **(term.params or {}),
                        "asset_cfg": SceneEntityCfg("robot", joint_names=list(kept)),
                    }
        print(
            f"[INFO] action joints: {len(kept)}/{len(ALL_JOINTS)}"
            + (f" | frozen at default pose: {dropped}" if dropped else " (none ignored)")
        )

    def validate_config(self):
        if hasattr(self.scene, "camera") and self.scene.camera is not None:
            from isaaclab_newton.renderers import NewtonWarpRendererCfg

            if isinstance(self.scene.camera.renderer_cfg, NewtonWarpRendererCfg):
                allowed = {"rgb", "depth"}
                for data_type in self.scene.camera.data_types:
                    if data_type not in allowed:
                        raise ValueError(
                            f"Newton Warp renderer does not support '{data_type}'. Only rgb and depth are supported."
                        )


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotFlatEnvCfg(RobotEnvCfg):
    """Flat-terrain variant compatible with both PhysX and Newton."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None


@configclass
class RobotFlatPlayEnvCfg(RobotFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
