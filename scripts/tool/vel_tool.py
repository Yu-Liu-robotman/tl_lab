#!/usr/bin/env python3
"""对 GMR 导出的 motion CSV 做「降速 / 变速」：按时间缩放生成新的 CSV，并顺便看速度。

原理：把动作按 s 倍速播放（s<1 变慢）。新时间轴第 j 帧对应原时间轴 j*s 帧，
      N_new = floor((N-1)/s) + 1，线性插值。速度是位置的一阶导，所以
      vx/vy/wz 会自动变成原来的 s 倍，不需要单独缩速度通道。

用法：
    python humanoid/scripts/vel_tool.py <目录|文件> --scale 0.6          # 放慢到 0.6 倍
    python humanoid/scripts/vel_tool.py <目录|文件> --target-vx 1.0      # 自动算 s，把 |vx| 降到 1.0
    python humanoid/scripts/vel_tool.py <目录|文件> --target-vx 1.0 --out-dir /tmp/slow
    python humanoid/scripts/vel_tool.py <文件> --amp-target-vx 0.30      # 空间缩放（做 AMP 低速数据用它）
    python humanoid/scripts/vel_tool.py <文件> --amp-scale 0.39          # 直接给幅度系数
    python humanoid/scripts/vel_tool.py <目录|文件>                      # 只看速度，不写文件
    python humanoid/scripts/vel_tool.py --csv-fps 60                    # CSV 帧率（默认 30）
    python humanoid/scripts/vel_tool.py --no-config                     # 不做配置对照

列布局（每帧 7 + N_JOINT 个数，GMR 导出格式）：
    root_pos(3, 世界系) | root_quat_xyzw(4) | dof_pos(N)
CSV 里没有速度列，vx/vy/wz 由根位置/偏航的数值差分得到，再转到机体系。
四元数顺序是 (x,y,z,w)，读取时转回 (w,x,y,z)；写出后仍是 xyzw。


python scripts/tool/vel_tool.py \
    data/z1_amp/50/z1_forward_v084.csv \
    --csv-fps 50 \
    --amp-target-vx  0.5
    --scale 0.6 
    --amp-target-vx  0.75

"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_HERE))
ENV_DIR = os.path.join(REPO, "humanoid", "envs", "z1_29dof_amp_sym_flat")
DEFAULT_DIR = os.path.join(ENV_DIR, "datasets", "23dof", "gmr", "csv3")
DEFAULT_CFG = os.path.join(ENV_DIR, "z1_config.py")


def load_csv(path):
    """读一个 GMR motion CSV -> (root_pos(N,3), root_quat_wxyz(N,4))。"""
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.ndim != 2 or data.shape[1] <= 7:
        raise ValueError(f"列数不足（root_pos 3 + root_quat 4 + dof），实际 shape={data.shape}")
    quat_wxyz = data[:, [6, 3, 4, 5]]          # csv 是 xyzw
    if not np.allclose(np.linalg.norm(quat_wxyz, axis=-1), 1.0, atol=1e-3):
        raise ValueError("root 四元数不是单位四元数，列布局可能不对")
    return data[:, :3], quat_wxyz


def analyse(path, csv_fps=30.0):
    bpos, bquat = load_csv(path)
    dt = 1.0 / float(csv_fps)
    n = bpos.shape[0]

    quat_norm = np.linalg.norm(bquat, axis=-1)
    w, x, y, z = bquat.T
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cy, sy = np.cos(yaw), np.sin(yaw)

    # 世界系差分 -> 机体系；偏航必须先 unwrap，否则跨 ±pi 会炸出尖峰
    dvx_w = np.gradient(bpos[:, 0], dt)
    dvy_w = np.gradient(bpos[:, 1], dt)
    vx_b = cy * dvx_w + sy * dvy_w
    vy_b = -sy * dvx_w + cy * dvy_w
    wz = np.gradient(np.unwrap(yaw), dt)

    dur = n * dt
    disp = float(np.hypot(*(bpos[-1, :2] - bpos[0, :2])))
    return {
        "name": os.path.basename(path),
        "src": path,
        "frames": n,
        "dt": dt,
        "dur": dur,
        "vx_mean": float(vx_b.mean()),
        "vx_min": float(vx_b.min()),
        "vx_max": float(vx_b.max()),
        "vy_mean": float(vy_b.mean()),
        "wz_mean": float(wz.mean()),
        "speed": disp / dur if dur > 0 else 0.0,
        "base_z": float(bpos[:, 2].mean()),
        "quat_ok": bool(abs(quat_norm.min() - 1.0) < 1e-3 and abs(quat_norm.max() - 1.0) < 1e-3),
    }


def retime(data, s):
    """时间缩放：新帧 j 采样原时间 s*j 处（s<1 变慢）。位置线性插值，四元数插值后归一化。"""
    n = data.shape[0]
    if n < 2:
        raise ValueError("帧数不足，无法缩放")
    n_new = int(np.floor((n - 1) / s)) + 1
    t = np.arange(n_new, dtype=np.float64) * s
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (t - i0)[:, None]
    out = data[i0] * (1.0 - w) + data[i1] * w
    q = out[:, 3:7]
    out[:, 3:7] = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return out


def amp_scale(data, alpha, min_amp=0.05):
    """空间缩放：把动作的「摆幅」缩到 alpha 倍，帧数与步频不变。

    与 retime 的关键差别：速度同为 alpha 倍，但抬脚高度与步幅**一起**缩小，
    得到真实的小步走；retime 是「原步幅 + 原抬脚 + 慢步频」= 原地高抬腿。

    q  = med + alpha * (q - med)    逐关节，med 为该关节中位数（保留姿态偏置）
    xy = xy0 + alpha * (xy - xy0)   绕首帧缩放平面位移
    z / 四元数 / 帧数不动

    峰峰值 < min_amp 的关节视为常量（躯干、腕部小抖动），不缩放。
    """
    out = np.array(data, dtype=np.float64, copy=True)
    if out.ndim != 2 or out.shape[0] < 2:
        raise ValueError("帧数不足，无法缩放")
    if not 0.0 < alpha <= 3.0:
        raise ValueError(f"alpha={alpha:g} 超出合理范围 (0, 3]")

    q = out[:, 7:]
    med = np.median(q, axis=0)
    flat = (q.max(axis=0) - q.min(axis=0)) < min_amp
    q_new = med + alpha * (q - med)
    q_new[:, flat] = q[:, flat]
    out[:, 7:] = q_new

    out[:, :2] = out[0, :2] + alpha * (out[:, :2] - out[0, :2])
    return out


def write_csv(path, data):
    np.savetxt(path, data, delimiter=",", fmt="%.9g")


def parse_config(cfg_path):
    """粗略解析 z1_config.py：速度常量、启用的 (label, 文件)、amp_motion_cmd_ranges 原文。"""
    if not os.path.exists(cfg_path):
        return None
    src = open(cfg_path).read()

    consts = {
        k: float(v)
        for k, v in re.findall(
            r"^(WALK_VX_BACK|WALK_VX|WALK_VY|WALK_WZ|COMMAND_THRESHOLD)\s*=\s*(-?[\d.]+)", src, re.M
        )
    }

    used, in_files = [], False
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("amp_motion_files"):
            in_files = True
            continue
        if in_files and s.startswith("]"):
            in_files = False
            continue
        if in_files and s and not s.startswith("#") and not s.startswith("["):
            m = re.search(r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)', s)
            if m:
                used.append((m.group(1), m.group(2)))

    m = re.search(r"amp_motion_cmd_ranges\s*=\s*\{(.*?)\n\s*\}", src, re.S)
    ranges = m.group(1) if m else ""
    return consts, used, ranges


def main():
    ap = argparse.ArgumentParser(description="查看 AMP 数据集各段动作的速度")
    ap.add_argument("paths", nargs="*", help="目录或文件；留空则用默认 CSV 目录")
    ap.add_argument("--config", default=DEFAULT_CFG, help="用于对照的 z1_config.py")
    ap.add_argument("--no-config", action="store_true", help="不解析配置文件")
    ap.add_argument("--sort", default="name", choices=["name", "vx", "vy", "wz", "dur"], help="排序字段")
    ap.add_argument("--csv-fps", type=float, default=30.0, help="CSV 帧率（GMR 导出为 30fps）")
    ap.add_argument("--scale", type=float, default=None, help="时间缩放系数：s<1 放慢（vx 变成 s 倍）")
    ap.add_argument("--target-vx", type=float, default=None, help="目标 |vx|，自动算 s = target/|vx|")
    ap.add_argument("--amp-scale", type=float, default=None, help="空间缩放系数 alpha：摆幅×alpha，帧数不变")
    ap.add_argument("--amp-target-vx", type=float, default=None, help="空间缩放版 target：alpha = target/|vx|")
    ap.add_argument("--out-dir", default=None, help="输出目录（默认与原文件同目录）")
    ap.add_argument("--slow-only", action="store_true", help="只降速：|vx| 已经不高于目标时跳过")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的输出文件")
    args = ap.parse_args()

    files = []
    for t in (args.paths or [DEFAULT_DIR]):
        if os.path.isdir(t):
            files += sorted(glob.glob(os.path.join(t, "*.csv")))
        else:
            files.append(t)
    if not files:
        print("没找到任何 .csv", file=sys.stderr)
        return 1

    rows, bad = [], []
    for f in files:
        try:
            rows.append(analyse(f, args.csv_fps))
        except Exception as e:  # noqa: BLE001
            bad.append((os.path.basename(f), str(e)))

    key = {
        "name": lambda r: r["name"],
        "vx": lambda r: -r["vx_mean"],
        "vy": lambda r: -abs(r["vy_mean"]),
        "wz": lambda r: -abs(r["wz_mean"]),
        "dur": lambda r: -r["dur"],
    }[args.sort]
    rows.sort(key=key)

    hdr = (
        f"{'文件':40s}{'帧':>4s}{'时长':>5s}{'vx':>7s}{'vx范围':>12s}"
        f"{'vy':>7s}{'wz':>7s}{'位移 v':>7s}{'z':>6s}"
    )
    print(hdr)
    print("-" * 95)
    for r in rows:
        print(
            f"{r['name']:40s}{r['frames']:4d}{r['dur']:5.1f}"
            f"{r['vx_mean']:+7.2f}"
            f" {r['vx_min']:+5.2f}~{r['vx_max']:+5.2f}"
            f"{r['vy_mean']:+7.2f}{r['wz_mean']:+7.2f}{r['speed']:7.2f}{r['base_z']:6.2f}"
            f"{'' if r['quat_ok'] else '  ⚠四元数'}"
        )
    for name, err in bad:
        print(f"[跳过] {name}: {err}", file=sys.stderr)

    # ---------------- 降速：写出时间缩放后的新 CSV ----------------
    if args.scale is not None or args.target_vx is not None:
        print(f"\n{'文件':40s}{'s':>6s}{'vx 前':>8s}{'vx 后':>8s}{'帧':>6s}  输出")
        print("-" * 100)
        for r in rows:
            if args.target_vx is not None:
                v = abs(r["vx_mean"])
                if v < 1e-3:
                    print(f"[跳过] {r['name']}: |vx|≈0，无法按目标 vx 缩放", file=sys.stderr)
                    continue
                s = float(args.target_vx) / v
                if args.slow_only and s > 1.0:
                    print(f"[跳过] {r['name']}: |vx|={v:.2f} 不高于目标 {args.target_vx:g}", file=sys.stderr)
                    continue
            else:
                s = float(args.scale)
            if not 0.05 <= s <= 20.0:
                print(f"[跳过] {r['name']}: 缩放系数 s={s:.3f} 不合理", file=sys.stderr)
                continue

            new = retime(np.loadtxt(r["src"], delimiter=",", ndmin=2), s)
            out_dir = args.out_dir or os.path.dirname(r["src"])
            stem = os.path.splitext(r["name"])[0]
            dst = os.path.join(out_dir, f"{stem}_s{s:.2f}.csv")
            if os.path.exists(dst) and not args.force:
                print(f"[跳过] {os.path.basename(dst)} 已存在（--force 覆盖）", file=sys.stderr)
                continue
            os.makedirs(out_dir, exist_ok=True)
            write_csv(dst, new)
            after = analyse(dst, args.csv_fps)["vx_mean"]
            print(
                f"{r['name']:40s}{s:6.2f}{r['vx_mean']:+8.2f}{after:+8.2f}"
                f"{len(new):6d}  {dst}"
            )

    # ---------------- 空间缩放：写出小步幅 / 低抬脚的新 CSV ----------------
    if args.amp_scale is not None or args.amp_target_vx is not None:
        print(f"\n{'文件':40s}{'α':>6s}{'vx 前':>8s}{'vx 后':>8s}{'帧':>6s}  输出")
        print("-" * 100)
        for r in rows:
            if args.amp_target_vx is not None:
                v = abs(r["vx_mean"])
                if v < 1e-3:
                    print(f"[跳过] {r['name']}: |vx|≈0，无法按目标 vx 缩放", file=sys.stderr)
                    continue
                a = float(args.amp_target_vx) / v
            else:
                a = float(args.amp_scale)
            if not 0.05 <= a <= 3.0:
                print(f"[跳过] {r['name']}: α={a:.3f} 不合理（0.05~3.0）", file=sys.stderr)
                continue

            new = amp_scale(np.loadtxt(r["src"], delimiter=",", ndmin=2), a)
            out_dir = args.out_dir or os.path.dirname(r["src"])
            stem = os.path.splitext(r["name"])[0]
            dst = os.path.join(out_dir, f"{stem}_a{a:.2f}.csv")
            if os.path.exists(dst) and not args.force:
                print(f"[跳过] {os.path.basename(dst)} 已存在（--force 覆盖）", file=sys.stderr)
                continue
            os.makedirs(out_dir, exist_ok=True)
            write_csv(dst, new)
            after = analyse(dst, args.csv_fps)["vx_mean"]
            print(f"{r['name']:40s}{a:6.2f}{r['vx_mean']:+8.2f}{after:+8.2f}{len(new):6d}  {dst}")

    if args.no_config:
        return 0
    parsed = parse_config(args.config)
    if not parsed:
        return 0
    consts, used, ranges = parsed
    by_name = {r["name"]: r for r in rows}

    print(f"\n===== z1_config.py 里当前启用的数据（{len(used)} 条）=====")
    print(f"{'label':14s}{'文件':40s}{'vx':>7s}{'vx范围':>12s}")
    print("-" * 73)
    for label, path in used:
        r = by_name.get(os.path.basename(path))
        if r is None:
            print(f"{label:14s}{os.path.basename(path):40s}   (不在本次扫描范围)")
        else:
            print(
                f"{label:14s}{r['name']:40s}{r['vx_mean']:+7.2f}"
                f" {r['vx_min']:+5.2f}~{r['vx_max']:+5.2f}"
            )

    if consts:
        print("\n速度常量：" + "  ".join(f"{k}={v:g}" for k, v in consts.items()))
    if ranges:
        print("amp_motion_cmd_ranges：")
        for line in ranges.strip().splitlines():
            print("   " + line.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
