from __future__ import annotations

import re
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
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

from basic_loco_lib.tasks.velocity import mdp as velocity_mdp

BASE_BODY_NAME = "base_link"
FOOT_BODY_REGEX = ".*ankle_roll.*"


@configclass
class EventCfg:
    """Domain randomization shared by all tasks (startup + reset + interval)."""

    # -- startup: contact / mass / inertia --------------------------------
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.2),
            "dynamic_friction_range": (0.2, 1.2),
            "restitution_range": (0.05, 0.12),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
            "mass_distribution_params": (-2.0, 5.0),
            "operation": "add",
        },
    )
    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_BODY_NAME]),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},
        },
    )
    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*"]),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
    )

    # -- startup: actuator / joint properties -----------------------------
    scale_joint_pd = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.7, 1.3),  # +-30% Kp
            "damping_distribution_params": (0.7, 1.3),    # +-30% Kd
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    scale_joint_param = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "friction_distribution_params": (0.8, 1.2),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- reset -------------------------------------------------------------
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-1.57, 1.57)},
            # No initial velocity: nonzero values prevent convergence.
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

    # -- interval ----------------------------------------------------------
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CollisionPipelineCfg(NewtonCollisionPipelineCfg):
    """Newton collision pipeline sized for mesh / heightfield terrain."""

    max_triangle_pairs: int = 4_000_000
    contact_reduction_hashtable_size_factor: float = 0.5


# Backwards-compatible alias: this class used to be named after the Z1 task.
Z1CollisionPipelineCfg = CollisionPipelineCfg


@configclass
class PhysicsCfg(PresetCfg):
    """Physics backend presets shared by all tasks.

    Select one with ``presets=newton`` / ``presets=physx`` on the CLI.
    """

    physx: PhysxCfg = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    newton: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=512,
            nconmax=64,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
            ls_iterations=50,
            # Use Newton's own collision pipeline instead of MuJoCo contacts.
            use_mujoco_contacts=False,
        ),
        # mesh / heightfield terrain needs a non-zero collision margin.
        default_shape_cfg=NewtonShapeCfg(margin=0.01),
        collision_cfg=CollisionPipelineCfg(
            broad_phase="explicit",
            reduce_contacts=True,
            max_triangle_pairs=1_000_000,
        ),
        num_substeps=1,
        debug_mode=False,
    )
    default = newton


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Robot-agnostic terrain / sensor / light boilerplate.

    Subclasses must supply ``robot``; a terrain generator plus its initial level
    are usually filled in by overriding ``__post_init__``.
    """

    robot: ArticulationCfg = MISSING

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
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
    height_scanner = None
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class ActionsCfg:
    """Joint-position action boilerplate; subclasses override ``scale``."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Velocity-tracking observations: noisy policy group + privileged critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 10
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 10

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=velocity_mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(velocity_mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    ignored_action_joints: tuple[str, ...] = ()
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def configure_robot(self) -> None:
        """Bind ``scene.robot``, the terrain generator and the action scale."""

    def __post_init__(self):
        self.configure_robot()

        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        generator = self.scene.terrain.terrain_generator
        if generator is not None:
            curriculum = getattr(self, "curriculum", None)
            generator.curriculum = getattr(curriculum, "terrain_levels", None) is not None

        self._apply_ignored_action_joints()

    def _apply_ignored_action_joints(self) -> None:
        all_joints = list(getattr(self.scene.robot, "joint_sdk_names", None) or [])
        if not all_joints:
            return

        patterns = tuple(self.ignored_action_joints or ())
        kept = [name for name in all_joints if not any(re.fullmatch(p, name) for p in patterns)]
        dropped = [name for name in all_joints if name not in kept]
        self.actions.JointPositionAction.joint_names = kept

        scale_cfg = self.actions.JointPositionAction.scale
        if isinstance(scale_cfg, dict):
            self.actions.JointPositionAction.scale = {name: scale_cfg[name] for name in kept if name in scale_cfg}

        observations = getattr(self, "observations", None)
        if observations is None:
            return

        groups = {
            id(group): group
            for group in (
                getattr(observations, "policy", None),
                getattr(observations, "actor", None),
                getattr(observations, "critic", None),
            )
            if group is not None
        }
        for group in groups.values():
            for term_name in ("joint_pos_rel", "joint_vel_rel", "joint_pos", "joint_vel"):
                term = getattr(group, term_name, None)
                if term is not None:
                    term.params = {
                        **(term.params or {}),
                        "asset_cfg": SceneEntityCfg("robot", joint_names=list(kept)),
                    }

        print(
            f"[INFO] action joints: {len(kept)}/{len(all_joints)}"
            + (f" | frozen at default pose: {dropped}" if dropped else " (none ignored)")
        )

    def validate_config(self):
        camera = getattr(self.scene, "camera", None)
        if camera is None:
            return

        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        if isinstance(camera.renderer_cfg, NewtonWarpRendererCfg):
            allowed = {"rgb", "depth"}
            for data_type in camera.data_types:
                if data_type not in allowed:
                    raise ValueError(
                        f"Newton Warp renderer does not support '{data_type}'. Only rgb and depth are supported."
                    )
