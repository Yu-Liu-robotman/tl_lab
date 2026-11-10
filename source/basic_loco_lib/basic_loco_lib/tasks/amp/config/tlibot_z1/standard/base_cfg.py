"""Shared Tlibot Z1 AMP environment scaffolding (scene, physics, events, actions).

Kept separate from ``tracking_env_cfg.py`` so task-specific command, observation,
reward, and termination groups can be swapped without touching the base.
"""

from __future__ import annotations

import re
from pathlib import Path

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR
from isaaclab_tasks.utils import PresetCfg

from basic_loco_lib.robots.z1_29dof import Z129DOF as ROBOT_CFG
from basic_loco_lib.robots.z1_29dof import Z129DOF_ACTION_SCALE
from basic_loco_lib.tasks.amp.agents.rsl_rl_ppo_cfg import Z1_AMP_MOTION_DIR
from basic_loco_lib.tasks.amp.mdp import events as amp_events
from basic_loco_lib.tasks.velocity import mdp
from basic_loco_lib.tasks.velocity.terrain import COBBLESTONE_ROAD_CFG, TLIBOT_Z1_ROUGH_TERRAINS_CFG

ALL_JOINTS: list[str] = list(ROBOT_CFG.joint_sdk_names)


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
            use_mujoco_contacts=False,
        ),
        default_shape_cfg=NewtonShapeCfg(margin=0.01),
        collision_cfg=Z1CollisionPipelineCfg(
            broad_phase="explicit",
            reduce_contacts=True,
            max_triangle_pairs=1_000_000,
        ),
        num_substeps=1,
        debug_mode=False,
    )
    default = physx


##
# Scene
##


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the AMP locomotion scene with the Z1 robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
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
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# Actions
##


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=Z129DOF_ACTION_SCALE,
        use_default_offset=True,
    )


##
# Events
##


@configclass
class EventCfg:
    init_motion_loader = EventTerm(
        func=amp_events.init_motion_loader,
        mode="startup",
        params={
            "motion_dir": str(Z1_AMP_MOTION_DIR),
            "delay_reset_env_ratio": 0.0,
            "max_delay_steps": 250,
            # Abort a delayed recovery early if physics enters a runaway state.
            "max_delay_joint_velocity": 50.0,
        },
    )
    reset_from_motion = EventTerm(
        func=amp_events.reset_from_motion_data,
        mode="reset",
        params={
            "motion_dir": str(Z1_AMP_MOTION_DIR),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        },
    )
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )
    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*"]),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    scale_joint_pd = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
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
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )


##
# Environment configuration
##


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Base Tlibot Z1 AMP environment; task groups are supplied by subclasses."""

    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg(), use_newton_actuators=False)
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    curriculum = None
    ignored_action_joints: tuple[str, ...] = (".*_wrist_.*",)

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        if getattr(self.scene, "contact_forces", None) is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        patterns = tuple(self.ignored_action_joints or ())
        kept = [name for name in ALL_JOINTS if not any(re.fullmatch(p, name) for p in patterns)]
        dropped = [name for name in ALL_JOINTS if name not in kept]
        self.actions.joint_pos.joint_names = kept
        scale_cfg = self.actions.joint_pos.scale
        if isinstance(scale_cfg, dict):
            self.actions.joint_pos.scale = {name: scale_cfg[name] for name in kept if name in scale_cfg}
        self.action_joint_names = kept
        print(
            f"[INFO] action joints: {len(kept)}/{len(ALL_JOINTS)}"
            + (f" | frozen at default pose: {dropped}" if dropped else " (none ignored)")
        )

        motion_dir = Path(self.events.init_motion_loader.params["motion_dir"])
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
