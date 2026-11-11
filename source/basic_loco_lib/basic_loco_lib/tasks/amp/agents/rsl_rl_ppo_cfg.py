"""RL configuration for standard and Warp Unitree G1 AMP locomotion environments."""

from pathlib import Path

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from basic_loco_lib import PROJECT_DIR


@configclass
class RslRlAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Extended runner config with AMP-specific parameters (v5 rsl-rl-lib format)."""

    # Override algorithm to use AMPPPO
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        class_name="basic_loco_lib.algo.amp.algorithms.amp_ppo:AMPPPO",
    )

    # Actor model
    actor: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="scalar",
        ),
    )

    # Critic model
    critic: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )

    num_steps_per_env = 24
    max_iterations = 100001
    save_interval = 100
    experiment_name = "g1_amp_locomotion"
    obs_groups = {"actor": ["actor"], "critic": ["critic"]}

    # AMP-specific parameters
    amp_reward_coef: float = 0.02 * 10
    amp_motion_files: str = ""
    amp_num_preload_transitions: int = 200000
    amp_task_reward_lerp: float = 0.6
    amp_discr_hidden_dims: list = (1024, 512, 256)
    min_normalized_std: list = (0.05,) * 29
    amp_body_names: tuple = ()
    amp_anchor_name: str = ""
    amp_motion_labels: tuple = ()
    amp_motion_weights: tuple = ()
    amp_motion_velocity_threshold: float = 0.8
    amp_replay_buffer_size: int = 1000000
    check_for_nan: bool = False
    detailed_nan_check: bool = False


def g1_amp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
    """Create RL runner configuration for Unitree G1 AMP locomotion task."""
    return RslRlAmpRunnerCfg(
        amp_motion_files=str(PROJECT_DIR / "data" / "amp"),
        amp_body_names=(
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ),
        amp_anchor_name="torso_link",
    )


def g1_amp_warp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
    """Keep Warp runs separate from standard runs when selecting checkpoints."""
    cfg = g1_amp_ppo_runner_cfg()
    cfg.experiment_name = "g1_amp_locomotion_warp"
    return cfg


# Tlibot Z1 uses ``base_link`` as its root and ``torso_yaw_link`` as the trunk
# anchor (it has no ``torso_link``); legs/arms are prefixed ``leg_*`` / ``arm_*``.
Z1_AMP_BODY_NAMES: tuple = (
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

Z1_AMP_MOTION_DIR = PROJECT_DIR / "data" / "z1_amp" / "output" / "npz"
Z1_AMP_MOTION_FILES: tuple = tuple(sorted(str(path) for path in Path(Z1_AMP_MOTION_DIR).glob("*.npz")))
Z1_AMP_MOTION_SPEC_BY_FILE: dict[str, tuple[str, float]] = {
    "z1_forward_11102335_50.npz": ("forward", 1.0),
    "z1_forward2_backward.npz": ("backward", 0.0),
    "z1_23dof_forward_new_backward.npz": ("backward", 0.0),
    "z1_23dof_stand.npz": ("stand", 1.0),
    "z1_23dof_turn_left_segment_3649-3922.npz": ("turn_left", 0.0),
    "z1_23dof_turn_right_segment_3176-3449.npz": ("turn_right", 0.0),
}
_z1_file_names = tuple(Path(path).name for path in Z1_AMP_MOTION_FILES)
_z1_missing = tuple(name for name in _z1_file_names if name not in Z1_AMP_MOTION_SPEC_BY_FILE)
if _z1_missing:
    raise ValueError(f"Missing manual AMP motion specs for files: {_z1_missing}")
Z1_AMP_MOTION_LABELS: tuple = tuple(Z1_AMP_MOTION_SPEC_BY_FILE[name][0] for name in _z1_file_names)
Z1_AMP_MOTION_WEIGHTS: tuple = tuple(float(Z1_AMP_MOTION_SPEC_BY_FILE[name][1]) for name in _z1_file_names)

def z1_amp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
    """Create RL runner configuration for the Tlibot Z1 AMP locomotion task."""
    cfg = RslRlAmpRunnerCfg(
        amp_motion_files=str(Z1_AMP_MOTION_DIR),
        amp_body_names=Z1_AMP_BODY_NAMES,
        amp_anchor_name="torso_yaw_link",
        amp_motion_labels=Z1_AMP_MOTION_LABELS,
        amp_motion_weights=Z1_AMP_MOTION_WEIGHTS,
        amp_motion_velocity_threshold=1.0,
        min_normalized_std=(0.05,) * 23,
    )
    cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(
        use_data_augmentation=True,
        data_augmentation_func=(
            "basic_loco_lib.tasks.amp.mdp.symmetry:compute_symmetric_states_z1"
        ),
    )
    cfg.experiment_name = "z1_amp_locomotion"
    return cfg
