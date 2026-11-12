#!/usr/bin/env python3
"""GMR csv 数据清洗：左右镜像（对称化） / 时间反演（前进 <-> 后退）。

csv 布局（**无表头**，逗号分隔）—— 权威来源 ``humanoid/scripts/tools/csv_to_npz.py``:
    col 0:3    root_pos   世界系 xyz (m)
    col 3:7    root_rot   xyzw（csv_to_npz.py:122 用 [3,0,1,2] 重排成 wxyz 再喂 IsaacLab）
    col 7:     dof_pos    关节顺序 = robot 配置的 ``joint_sdk_names``

镜像规则（矢状面 / 关于 xz 平面反射，S = diag(1,-1,1)，R_mirror = S R S）：
    root_pos : y 取反
    root_rot : xyzw 下 x、z 取反（wxyz 下 y、w 取反）
    dof_pos  : 每个关节取「左右对应关节」的值；roll / yaw 轴取反，pitch 与单轴不变
    —— 与仓库自带的对称模块一致：
       humanoid/envs/z1_29dof_amp_sym_flat/mdp/symmetry/z129dof_12dofamp.py
       （_ANTI_SYMMETRIC_JOINT_SUFFIXES = ("_roll_joint", "_yaw_joint")）

用法
----
    # 0) 先体检（不写文件）：确认左右配对关节的 ptp 接近，否则说明关节顺序不对
    python humanoid/scripts/tools/mirror_gmr_csv.py \
        humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/gmr/csv3/foward_segment_0122-0268.csv --check

    # 1) 左右镜像（默认 mode=left-right），输出 <stem>_mirrored.csv
    python scripts/tool/mirror_gmr_csv.py \
        /home/user/LY/turn_smplx_z1_v1.csv

    # 2) 整个目录批量
    python humanoid/scripts/tools/mirror_gmr_csv.py \
        humanoid/envs/z1_29dof_amp_sym_flat/datasets/23dof/gmr/csv3

    # 3) 时间反演：前进 -> 后退
    python scripts/tool/mirror_gmr_csv.py \
          /home/user/LY_TRA/basic_loco_lib-main/data/z1_amp/forward_30/z1_forward2.csv \
        --mode backward

    # 4) 换机器人变体（关节数不同时必须给对）
    python humanoid/scripts/tools/mirror_gmr_csv.py <file.csv> --dof-names my_20dof_joints.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# 关节顺序：权威来源 humanoid/assets/z1/z1.py 的 joint_sdk_names（23dof = 12 腿 + 3 躯干 + 8 臂）
# csv_to_npz.py 里就是用 robot.cfg.joint_sdk_names 给 csv 的 dof 列编号的，
# 所以这里的顺序必须与它逐项一致。
# --------------------------------------------------------------------------------------
Z1_23DOF_JOINT_NAMES = [
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
]

DOF_PRESETS: dict[str, list[str]] = {
    "z1_23dof": Z1_23DOF_JOINT_NAMES,
}

# 镜像取反的关节后缀（roll / yaw）；pitch 与单轴保持不变
_ANTI_SYMMETRIC_SUFFIXES = ("_roll_joint", "_yaw_joint")

ROOT_POS_SLICE = slice(0, 3)
ROOT_ROT_SLICE = slice(3, 7)
DOF_START = 7


# --------------------------------------------------------------------------------------
# 关节名 -> 左右对应 / 符号
# --------------------------------------------------------------------------------------
def counterpart(name: str) -> str:
    """返回左右对应关节名；没有 left/right 的关节返回自身。"""
    if "_left_" in name:
        return name.replace("_left_", "_right_")
    if "_right_" in name:
        return name.replace("_right_", "_left_")
    return name


def mirror_sign(name: str) -> float:
    """矢状面镜像下需要取反的关节轴（roll / yaw）。"""
    return -1.0 if name.endswith(_ANTI_SYMMETRIC_SUFFIXES) else 1.0


def build_mirror_maps(dof_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """返回 (src_index, sign)：``mirrored_dof[:, i] = sign[i] * dof[:, src_index[i]]``。"""
    index_of = {name: i for i, name in enumerate(dof_names)}
    if len(index_of) != len(dof_names):
        dup = sorted({n for n in dof_names if dof_names.count(n) > 1})
        raise SystemExit(f"关节名有重复：{dup}")

    src, sign = [], []
    for name in dof_names:
        partner = counterpart(name)
        if partner not in index_of:
            raise SystemExit(f"关节 '{name}' 的镜像对应 '{partner}' 不在关节列表里。")
        src.append(index_of[partner])
        sign.append(mirror_sign(name))
    return np.asarray(src, dtype=np.int64), np.asarray(sign, dtype=np.float64)


# --------------------------------------------------------------------------------------
# 读写
# --------------------------------------------------------------------------------------
def load_csv(path: Path) -> tuple[np.ndarray, np.dtype]:
    arr = np.loadtxt(path, delimiter=",", ndmin=2)
    if arr.size == 0:
        raise SystemExit(f"{path}: 读出来是空的。")
    if arr.ndim != 2:
        raise SystemExit(f"{path}: 期望二维数组，得到 {arr.shape}。")
    return arr, arr.dtype


def save_csv(path: Path, arr: np.ndarray, dtype: np.dtype) -> None:
    # %.9g 足够 float32 往返；float64 用 %.17g 保证 bit-exact 往返
    fmt = "%.9g" if np.issubdtype(dtype, np.floating) and dtype.itemsize <= 4 else "%.17g"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, arr, delimiter=",", fmt=fmt)


# --------------------------------------------------------------------------------------
# 变换
# --------------------------------------------------------------------------------------
def mirror_left_right(frame: np.ndarray, src: np.ndarray, sign: np.ndarray, quat_order: str) -> np.ndarray:
    """矢状面左右镜像。"""
    out = frame.copy()

    # root_pos: y 取反
    out[:, ROOT_POS_SLICE][:, 1] *= -1.0

    # root_rot: R_mirror = S R S，用四元数分量取反表达
    q = out[:, ROOT_ROT_SLICE]
    if quat_order == "xyzw":
        q[:, [0, 2]] *= -1.0  # x, z
    elif quat_order == "wxyz":
        q[:, [1, 3]] *= -1.0  # x, z
    else:
        raise SystemExit(f"未知的四元数顺序：{quat_order}")

    # dof_pos: 左右互换 + roll/yaw 取反（必须从**原始** frame 取，避免自引用）
    out[:, DOF_START:] = sign * frame[:, DOF_START:][:, src]
    return out


def reverse_time(frame: np.ndarray) -> np.ndarray:
    """时间反演（前进 -> 后退），并把新片段起点平移到原片段起点。"""
    out = frame[::-1].copy()
    out[:, ROOT_POS_SLICE] += frame[0, ROOT_POS_SLICE] - out[0, ROOT_POS_SLICE]
    return out


# --------------------------------------------------------------------------------------
# 诊断
# --------------------------------------------------------------------------------------
def diagnose(path: Path, frame: np.ndarray, dof_names: list[str], src: np.ndarray) -> bool:
    """打印体检信息；返回是否通过。"""
    n_dof = frame.shape[1] - DOF_START
    print(f"[check] {path.name}: frames={frame.shape[0]}, cols={frame.shape[1]} (dof={n_dof})")

    ok = True

    # 1) 有限性 / 四元数模长
    if not np.isfinite(frame).all():
        n_bad = int((~np.isfinite(frame)).any(axis=1).sum())
        print(f"  ✗ 有 {n_bad} 帧含 NaN/Inf")
        ok = False
    q = frame[:, ROOT_ROT_SLICE]
    q_err = float(np.abs(np.linalg.norm(q, axis=1) - 1.0).max())
    print(f"  |quat|-1 最大偏差 = {q_err:.2e}", "✓" if q_err < 1e-3 else "✗（不是单位四元数 / 顺序可能不对）")
    ok &= q_err < 1e-3

    # 2) 左右配对关节的 ptp 应接近 —— 差很多说明关节顺序不对
    dof = frame[:, DOF_START:]
    ptp = dof.max(axis=0) - dof.min(axis=0)
    print("  配对关节 ptp（左右应接近）:")
    worst = 0.0
    for i, name in enumerate(dof_names):
        j = int(src[i])
        if j <= i:
            continue
        a, b = ptp[i], ptp[j]
        lo = max(min(a, b), 1e-6)
        ratio = max(a, b) / lo
        worst = max(worst, ratio)
        flag = "" if ratio < 3.0 else "   <-- 差很多！"
        print(f"    [{i:2d}] {name:34s} ptp={a:.3f}  <->  [{j:2d}] {dof_names[j]:34s} ptp={b:.3f}{flag}")
    if worst >= 3.0:
        print(f"  ✗ 配对 ptp 最大比值 {worst:.1f} —— 关节顺序很可能不是该机器人变体的 joint_sdk_names")
        ok = False
    else:
        print(f"  ✓ 配对 ptp 最大比值 {worst:.2f}")

    # 3) root 是否真的在动
    span = frame[:, ROOT_POS_SLICE].ptp(axis=0)
    print(f"  root_pos 变化幅度 = {np.round(span, 3)}")
    return ok


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path, help="输入 .csv，或包含 .csv 的目录")
    parser.add_argument(
        "--mode",
        choices=("left-right", "backward"),
        default="left-right",
        help="left-right = 矢状面镜像（对称化）；backward = 时间反演（前进->后退）",
    )
    parser.add_argument("--suffix", default=None, help="输出后缀，默认 _mirrored / _backward")
    parser.add_argument("--out-dir", type=Path, default=None, help="输出目录，默认与输入同目录")
    parser.add_argument(
        "--dof-names",
        type=Path,
        default=None,
        help="覆盖关节顺序：一个纯文本文件，一行一个关节名",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(DOF_PRESETS),
        default="z1_23dof",
        help="内置关节顺序预设（--dof-names 优先）",
    )
    parser.add_argument(
        "--quat-order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="root_rot 的分量顺序，GMR csv 是 xyzw（默认）",
    )
    parser.add_argument("--check", action="store_true", help="只体检，不写文件")
    parser.add_argument("--inplace", action="store_true", help="原地覆盖输入文件（危险，建议先备份）")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def resolve_dof_names(args: argparse.Namespace) -> list[str]:
    if args.dof_names is not None:
        names = [
            line.strip()
            for line in args.dof_names.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not names:
            raise SystemExit(f"{args.dof_names}: 没解析到关节名。")
        return names
    return list(DOF_PRESETS[args.preset])


def iter_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(p for p in path.glob("*.csv") if p.is_file())
        if not files:
            raise SystemExit(f"{path}: 目录里没有 .csv")
        return files
    if not path.is_file():
        raise SystemExit(f"{path}: 不存在")
    return [path]


def main() -> None:
    args = parse_args()
    dof_names = resolve_dof_names(args)
    src, sign = build_mirror_maps(dof_names)
    suffix = args.suffix or ("_mirrored" if args.mode == "left-right" else "_backward")

    n_done = n_skip = 0
    for path in iter_inputs(args.input):
        frame, dtype = load_csv(path)
        expected = DOF_START + len(dof_names)
        if frame.shape[1] != expected:
            print(
                f"✗ {path.name}: {frame.shape[1]} 列，但 {len(dof_names)} 个关节需要 {expected} 列。"
                f"（用 --preset / --dof-names 指定正确的关节顺序）",
                file=sys.stderr,
            )
            n_skip += 1
            continue

        if args.check:
            diagnose(path, frame, dof_names, src)
            continue

        out = (
            mirror_left_right(frame, src, sign, args.quat_order)
            if args.mode == "left-right"
            else reverse_time(frame)
        )

        if args.inplace:
            out_path = path
        else:
            out_dir = args.out_dir or path.parent
            out_path = out_dir / f"{path.stem}{suffix}.csv"
        save_csv(out_path, out, dtype)
        n_done += 1
        if not args.quiet:
            print(f"[{args.mode}] {path.name} -> {out_path}")

    if args.check:
        print("[check] 只体检，未写文件")
    else:
        print(f"完成：写出 {n_done} 个文件" + (f"，跳过 {n_skip} 个" if n_skip else ""))


if __name__ == "__main__":
    main()
