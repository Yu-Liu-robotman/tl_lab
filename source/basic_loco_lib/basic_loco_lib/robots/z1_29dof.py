import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from basic_loco_lib import ASSETS_DIR
@configclass
class TlibotArticulationCfg(ArticulationCfg):
    """Configuration for Tlibot articulations."""
    joint_sdk_names: list[str] | None = None

ARMATURE_RH85 = 0.044
ARMATURE_RH63L = 0.012 
ARMATURE_RH63 = 0.008 
ARMATURE_RH53 = 0.004374

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_RH85 = ARMATURE_RH85 * NATURAL_FREQ**2
STIFFNESS_RH63L = ARMATURE_RH63L * NATURAL_FREQ**2
STIFFNESS_RH63 = ARMATURE_RH63 * NATURAL_FREQ**2
STIFFNESS_RH53 = ARMATURE_RH53 * NATURAL_FREQ**2

DAMPING_RH85 = 2.0 * DAMPING_RATIO * ARMATURE_RH85 * NATURAL_FREQ
DAMPING_RH63L = 2.0 * DAMPING_RATIO * ARMATURE_RH63L * NATURAL_FREQ
DAMPING_RH63 = 2.0 * DAMPING_RATIO * ARMATURE_RH63 * NATURAL_FREQ
DAMPING_RH53 = 2.0 * DAMPING_RATIO * ARMATURE_RH53 * NATURAL_FREQ
# 173.52

Z129DOF = TlibotArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ASSETS_DIR}/z1_29dof/usd/z1_v1/z1_v1.usda",
        usd_path=f"{ASSETS_DIR}/z1_29dof/usd/29dof_add_shoe/z1_v1_add_shoe.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4 #, fix_root_link = True
        ),
    ),
    
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.865),
        joint_pos={
            "leg_left_hip_pitch_joint": -0.03,
            "leg_left_hip_roll_joint": 0.0,
            "leg_left_hip_yaw_joint": 0.0,
            "leg_left_knee_joint": 0.1,
            "leg_left_ankle_pitch_joint": -0.07,
            "leg_left_ankle_roll_joint": 0.0,
            "leg_right_hip_pitch_joint": -0.03,
            "leg_right_hip_roll_joint": 0.0,
            "leg_right_hip_yaw_joint": 0.0,
            "leg_right_knee_joint": 0.1,
            "leg_right_ankle_pitch_joint": -0.07,
            "leg_right_ankle_roll_joint": 0.0,

            "torso_yaw_joint": 0.0,
            "torso_pitch_joint": 0.0,
            "torso_roll_joint": 0.0,

            # "arm_left_shoulder_pitch_joint": 0.2,
            # "arm_left_shoulder_roll_joint": 0.2,
            # "arm_left_shoulder_yaw_joint": 0.0,
            # "arm_left_elbow_joint": 1.1,
            "arm_left_shoulder_pitch_joint": 0.188,
            "arm_left_shoulder_roll_joint": 0.06,
            "arm_left_shoulder_yaw_joint": 0.44,
            "arm_left_elbow_joint": 1.1,

            "arm_left_wrist_roll_joint": 0.0,
            "arm_left_wrist_pitch_joint": 0.0,
            "arm_left_wrist_yaw_joint": 0.0,

            # "arm_right_shoulder_pitch_joint": 0.2,
            # "arm_right_shoulder_roll_joint": -0.2,
            # "arm_right_shoulder_yaw_joint": 0.0,
            # "arm_right_elbow_joint": 1.1,

            "arm_right_shoulder_pitch_joint": 0.188,
            "arm_right_shoulder_roll_joint": -0.06,
            "arm_right_shoulder_yaw_joint": -0.44,
            "arm_right_elbow_joint": 1.1,

            "arm_right_wrist_roll_joint": 0.0,
            "arm_right_wrist_pitch_joint": 0.0,
            "arm_right_wrist_yaw_joint": 0.0,

        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "RH85": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=100.0,
            velocity_limit_sim=20.0,
            stiffness={
                ".*_hip_pitch_joint": 200,
                ".*_knee_joint": 200,
            },
            damping={
                ".*_hip_pitch_joint": 13,
                ".*_knee_joint": 13,
            },
            armature=0.044,
        ),
        "RH63L": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                "torso_yaw_joint",
            ],
            effort_limit_sim=75,
            velocity_limit_sim=20,
            stiffness={
                ".*_hip_roll_joint": 100,
                ".*_hip_yaw_joint": 100,
                "torso_yaw_joint": 200,
            },
            damping={
                ".*_hip_roll_joint": 6.5,
                ".*_hip_yaw_joint": 6.5,
                "torso_yaw_joint": 13,
            },
            armature={
                ".*_hip_roll_joint": 0.01,
                ".*_hip_yaw_joint": 0.08,
                "torso_yaw_joint": 0.01,
            },
        ),
        "RH63": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_ankle_pitch_joint",
                "torso_pitch_joint",
                "torso_roll_joint",
            ],
            effort_limit_sim=50,
            velocity_limit_sim=20,
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_RH63,
                ".*_shoulder_roll_joint": STIFFNESS_RH63,
                ".*_ankle_pitch_joint": 50,
                "torso_pitch_joint": 200,
                "torso_roll_joint": 1160,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_RH63,
                ".*_shoulder_roll_joint": DAMPING_RH63,
                ".*_ankle_pitch_joint": 3,
                "torso_pitch_joint": 13,
                "torso_roll_joint": 75,
            },
            armature={
                ".*_shoulder_pitch_joint": 0.01,
                ".*_shoulder_roll_joint": 0.01,
                ".*_ankle_pitch_joint": 0.08,
                "torso_pitch_joint": 0.01,
                "torso_roll_joint": 0.01,
            },
        ),
        "RH53": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim=25,
            velocity_limit_sim=30,
            stiffness={
                ".*_shoulder_yaw_joint": STIFFNESS_RH53,
                ".*_elbow_joint": STIFFNESS_RH53,
                ".*_ankle_roll_joint": 30,
            },
            damping={
                ".*_shoulder_yaw_joint": DAMPING_RH53,
                ".*_elbow_joint": DAMPING_RH53,
                ".*_ankle_roll_joint": 2,
            },
            armature={
                ".*_shoulder_yaw_joint": 0.01,
                ".*_elbow_joint": 0.01,
                ".*_ankle_roll_joint": 0.06,
            },
        ),
        "wrist_lock": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim=100.0,
            velocity_limit_sim=20.0,
            stiffness=1400.0,
            damping=15.0,
            armature=0.01,
        ),
    },
    joint_sdk_names=[
        'leg_left_hip_pitch_joint',
        'leg_left_hip_roll_joint',
        'leg_left_hip_yaw_joint',
        'leg_left_knee_joint',
        'leg_left_ankle_pitch_joint',
        'leg_left_ankle_roll_joint',
        'leg_right_hip_pitch_joint',
        'leg_right_hip_roll_joint',
        'leg_right_hip_yaw_joint',
        'leg_right_knee_joint',
        'leg_right_ankle_pitch_joint',
        'leg_right_ankle_roll_joint',

        'torso_yaw_joint',
        'torso_pitch_joint',
        'torso_roll_joint',

        "arm_left_shoulder_pitch_joint",
        "arm_left_shoulder_roll_joint",
        "arm_left_shoulder_yaw_joint",
        "arm_left_elbow_joint",

        "arm_left_wrist_roll_joint",
        "arm_left_wrist_pitch_joint",
        "arm_left_wrist_yaw_joint",

        "arm_right_shoulder_pitch_joint",
        "arm_right_shoulder_roll_joint",
        "arm_right_shoulder_yaw_joint",
        "arm_right_elbow_joint",

        "arm_right_wrist_roll_joint",
        "arm_right_wrist_pitch_joint",
        "arm_right_wrist_yaw_joint",        
    ],
)

# Z129DOF_ACTION_SCALE = {}
# for a in Z129DOF.actuators.values():
#     e = a.effort_limit_sim
#     s = a.stiffness
#     names = a.joint_names_expr
#     if not isinstance(e, dict):
#         e = dict.fromkeys(names, e)
#     if not isinstance(s, dict):
#         s = dict.fromkeys(names, s)
#     for n in names:
#         if n in e and n in s and s[n]:
#             Z129DOF_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

Z129DOF_ACTION_SCALE: dict[str, float] = {
    "leg_left_hip_pitch_joint": 0.5,
    "leg_left_hip_roll_joint": 0.2,
    "leg_left_hip_yaw_joint": 0.2,
    "leg_left_knee_joint": 0.3,

    "leg_left_ankle_pitch_joint": 0.25,
    "leg_left_ankle_roll_joint": 0.2,

    "leg_right_hip_pitch_joint": 0.5,
    "leg_right_hip_roll_joint": 0.2,
    "leg_right_hip_yaw_joint": 0.2,
    "leg_right_knee_joint": 0.3,

    "leg_right_ankle_pitch_joint": 0.25,
    "leg_right_ankle_roll_joint": 0.2,

    "torso_yaw_joint": 0.093,
    "torso_pitch_joint": 0.0625,
    "torso_roll_joint": 0.01077,
        
    "arm_left_shoulder_pitch_joint": 0.2,
    "arm_left_shoulder_roll_joint": 0.2,
    "arm_left_shoulder_yaw_joint": 0.05,
    "arm_left_elbow_joint": 0.2,
    
    "arm_left_wrist_roll_joint": 1.0,
    "arm_left_wrist_pitch_joint": 1.0,
    "arm_left_wrist_yaw_joint": 1.0,

    "arm_right_shoulder_pitch_joint": 0.2,
    "arm_right_shoulder_roll_joint": 0.2,
    "arm_right_shoulder_yaw_joint": 0.05,
    "arm_right_elbow_joint": 0.2,

    "arm_right_wrist_roll_joint": 1.0,
    "arm_right_wrist_pitch_joint": 1.0,
    "arm_right_wrist_yaw_joint": 1.0,
}

KPS = {}
KDS = {}
for a in Z129DOF.actuators.values():
    s = a.stiffness
    d = a.damping
    names = a.joint_names_expr
    if not isinstance(d, dict):
        d = dict.fromkeys(names, d)
    if not isinstance(s, dict):
        s = dict.fromkeys(names, s)
    for n in names:
        if n in d and n in s and s[n]:
            KPS[n] = s[n]
            KDS[n] = d[n]