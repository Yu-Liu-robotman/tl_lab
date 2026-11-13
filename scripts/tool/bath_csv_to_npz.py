"""
Batch convert motion csv files to npz recursively.

output_root/
 └── npz/      # AMP 可用：整机 29 关节 + 全部刚体 + joint_names/body_names/ordering

python3 scripts/tool/bath_csv_to_npz.py \
    --input_dir data/z1_amp/50 \
    --output_dir data/z1_amp/50/npz \
    --input_fps 50 \
    --output_fps 50 \
    --robot z1 \
    --backend newton \
    --overwrite \
    --viz kit \
    --pos_z_offset 0.02 \
    --viz kit \
    --frame_range 30 110

注意：npz 的关节/刚体列顺序由 --backend 决定（会写入 ordering 字段）；
AMP 的 motion reset 会按 joint_names 重排，所以请保证 --robot 与训练机器人一致。
"""

import argparse
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Batch replay motion csv files.")
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--recursive", action="store_true")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--input_fps", type=int, default=30)
parser.add_argument("--output_fps", type=int, default=50)
parser.add_argument("--robot", type=str, default="z1", choices=["z1"],
                    help="回放用的机器人（csv 列顺序需为该机器人的 joint_sdk_names 顺序，可缺腕部）")
parser.add_argument("--backend", type=str, default="physx", choices=["physx", "newton"],
                    help="回放所用物理后端。记录下来的关节/刚体顺序随后端变化，"
                         "生成的 npz 只能喂给相同后端的任务（会写入 ordering 字段）")
parser.add_argument("--pos_z_offset", type=float, default=0.0,help="base_link 的 z (m)") # 0.028
parser.add_argument("--csv_skip", nargs="*", default=["wrist"],help="csv 里不包含的关节名关键字（这些关节在 npz 里写 0）")
parser.add_argument("--frame_range", nargs=2, type=int, metavar=("START", "END"), default=None,
                    help="只转换这一段帧（从 1 开始、两端包含）")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

if args_cli.robot == "z1":
    from basic_loco_lib.robots.z1_29dof import Z129DOF as ROBOT_CFG

@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


class MotionLoader:
    def __init__(self, motion_file, input_fps, output_fps, device):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / input_fps
        self.output_dt = 1.0 / output_fps
        self.device = device
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=",")).float().to(self.device)
        if args_cli.frame_range is not None:
            s, e = int(args_cli.frame_range[0]), int(args_cli.frame_range[1])
            s = max(s, 1)
            e = min(e, motion.shape[0])
            if e - s + 1 < 2:
                raise ValueError(f"frame_range={args_cli.frame_range} 无效（该文件共 {motion.shape[0]} 帧）")
            motion = motion[s - 1:e]
            print(f"[CROP] frames {s}..{e} -> {motion.shape[0]} 帧")
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_dof_poss_input = motion[:, 7:]
        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"[LOAD] {self.motion_file}")

    def _compute_frame_blend(self, times):
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _lerp(self, a, b, blend):
        return a * (1 - blend) + b * blend

    def _slerp(self, a, b, blend):
        out = torch.zeros_like(a)
        for i in range(a.shape[0]):
            out[i] = quat_slerp(a[i], b[i], blend[i])
        return out

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device)
        self.output_frames = times.shape[0]

        index_0, index_1, blend = self._compute_frame_blend(times)

        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )

        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )

        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )

    def _so3_derivative(self, rotations, dt):
        q_prev = rotations[:-2]
        q_next = rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(
            self.motion_base_poss,
            spacing=self.output_dt,
            dim=0,
        )[0]

        self.motion_dof_vels = torch.gradient(
            self.motion_dof_poss,
            spacing=self.output_dt,
            dim=0,
        )[0]

        self.motion_base_ang_vels = self._so3_derivative(
            self.motion_base_rots,
            self.output_dt,
        )


def convert_file(csv_path, npz_path, sim, scene):
    print("=" * 80)
    print(f"[PROCESS] {csv_path}")

    motion = MotionLoader(
        motion_file=str(csv_path),
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
    )
    if args_cli.pos_z_offset != 0.0:
        motion.motion_base_poss[:, 2] += args_cli.pos_z_offset

    robot = scene["robot"]

    # robot_joint_indexes = robot.find_joints(
    #     scene.cfg.robot.joint_sdk_names,
    #     preserve_order=True,
    # )[0]
# ------------------------
    sdk_names = list(scene.cfg.robot.joint_sdk_names)
    robot_joint_indexes = robot.find_joints(sdk_names, preserve_order=True)[0]

    # csv 列名 = sdk_names 去掉 csv 不含的关节（默认腕关节）
    csv_names = [n for n in sdk_names if not any(k in n for k in args_cli.csv_skip)]
    n_csv = motion.motion_dof_poss.shape[1]
    print(f"[CHECK] sdk={len(sdk_names)} resolved={len(robot_joint_indexes)} "
          f"csv_names={len(csv_names)} csv_dof={n_csv} robot_joints={robot.num_joints}")

    if len(csv_names) == n_csv:
        csv_joint_ids = [robot_joint_indexes[sdk_names.index(n)] for n in csv_names]
    elif len(sdk_names) == n_csv:
        col_ids = [sdk_names.index(n) for n in csv_names]
        csv_joint_ids = [robot_joint_indexes[i] for i in col_ids]
        motion.motion_dof_poss = motion.motion_dof_poss[:, col_ids]
        motion.motion_dof_vels = motion.motion_dof_vels[:, col_ids]
        print(f"[FIX] csv 含全部 {n_csv} 个关节，已剔除 {args_cli.csv_skip}，保留 {len(col_ids)} 列")
    else:
        raise ValueError(
            f"csv_dof={n_csv}，sdk_names={len(sdk_names)}（去掉 {args_cli.csv_skip} 后 "
            f"{len(csv_names)}）对不上，无法建立映射"
        )
# ------------------------
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    for frame_idx in range(motion.output_frames):
        motion_base_pos = motion.motion_base_poss[frame_idx:frame_idx + 1]
        motion_base_rot = motion.motion_base_rots[frame_idx:frame_idx + 1]
        motion_base_lin_vel = motion.motion_base_lin_vels[frame_idx:frame_idx + 1]
        motion_base_ang_vel = motion.motion_base_ang_vels[frame_idx:frame_idx + 1]
        motion_dof_pos = motion.motion_dof_poss[frame_idx:frame_idx + 1]
        motion_dof_vel = motion.motion_dof_vels[frame_idx:frame_idx + 1]

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()

        # joint_pos[:, robot_joint_indexes] = motion_dof_pos
        # joint_vel[:, robot_joint_indexes] = motion_dof_vel
        joint_pos[:, robot_joint_indexes] = 0.0
        joint_vel[:, robot_joint_indexes] = 0.0

        joint_pos[:, csv_joint_ids] = motion_dof_pos
        joint_vel[:, csv_joint_ids] = motion_dof_vel

        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())

        # npz 记录整机全部关节（articulation 顺序）+ 全部刚体，供 AMP 使用
        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())

    for k in log.keys():
        if k != "fps":
            log[k] = np.stack(log[k], axis=0)

    # 自描述：写入顺序标签 + 关节/刚体名字。
    # AMP 的 reset 事件（tasks/amp/mdp/events.py::_load_dir）靠 joint_names 重排，
    # 缺了会直接 KeyError；所以这里必须写。
    np.savez(
        npz_path,
        ordering=np.array(args_cli.backend),
        joint_names=np.array(list(robot.joint_names)),
        body_names=np.array(list(robot.body_names)),
        **log,
    )
    print(f"[SAVE] {npz_path}")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    if args_cli.backend == "newton":
        from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

        # 与 mimic/amp 任务的 Newton 配置保持一致
        sim_cfg.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=128, nconmax=20, cone="pyramidal", impratio=1,
                integrator="implicitfast", ls_parallel=True, ls_iterations=15,
            ),
            num_substeps=1,
            debug_mode=False,
        )

    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(
        num_envs=1,
        env_spacing=2.0,
    )

    scene = InteractiveScene(scene_cfg)

    sim.reset()

    print("[INFO] Simulator initialized")

    input_dir = Path(args_cli.input_dir)

    if args_cli.recursive:
        csv_files = sorted(input_dir.rglob("*.csv"))
    else:
        csv_files = sorted(input_dir.glob("*.csv"))

    print(f"[INFO] Found {len(csv_files)} csv files")

    for csv_path in csv_files:
        rel_path = csv_path.relative_to(input_dir)

        if args_cli.output_dir is None:
            out_root = input_dir / "converted"
        else:
            out_root = Path(args_cli.output_dir)

        npz_path = (out_root / "npz" / rel_path).with_suffix(".npz")
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        if (npz_path.exists() and not args_cli.overwrite):
            print(f"[SKIP] {csv_path}")
            continue

        try:
            convert_file(
                csv_path=csv_path,
                npz_path=npz_path,
                sim=sim,
                scene=scene,
            )

        except Exception as e:
            print(f"[ERROR] {csv_path}")
            print(e)

    print("[DONE]")


if __name__ == "__main__":
    main()
    simulation_app.close()