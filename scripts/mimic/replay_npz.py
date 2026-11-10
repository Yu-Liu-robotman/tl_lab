"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_npz.py -f path_to_motion.npz
    
"""

"""Launch Isaac Sim Simulator first."""

"""
python3 scripts/mimic/replay_npz.py \
    -f /home/user/LY_TRA/basic_loco_lib-main/data/z1_amp/z1_23dof_forward_new.physx.npz
"""
import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--file", "-f", type=str, required=True)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# from basic_loco_lib.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG as ROBOT_CFG  # Currently only support G1-29dof
from basic_loco_lib.robots.z1_29dof import Z129DOF as ROBOT_CFG
from basic_loco_lib.tasks.mimic.mdp import MotionLoader

##
# Pre-defined configs
##


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    motion = MotionLoader(
        args_cli.file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_pose = robot.data.default_root_pose.torch.clone()
        root_vel = robot.data.default_root_vel.torch.clone()
        root_pose[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins
        root_pose[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_vel[:, :3] = motion.body_lin_vel_w[time_steps][:, 0]
        root_vel[:, 3:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_link_pose_to_sim_index(root_pose=root_pose)
        robot.write_root_com_velocity_to_sim_index(root_velocity=root_vel)
        robot.write_joint_position_to_sim_index(position=motion.joint_pos[time_steps])
        robot.write_joint_velocity_to_sim_index(velocity=motion.joint_vel[time_steps])
        simulation_app.update()  # 推动 Kit 主循环 -> 视口才会真正刷新（sim.render() 在有 visualizer 时才刷）
        scene.update(sim_dt)

        pos_lookat = root_pose[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
