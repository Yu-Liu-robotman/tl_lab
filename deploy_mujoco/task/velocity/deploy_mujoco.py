"""Sim2sim (MuJoCo) rollout for basic_loco_lib velocity policies (unitree_g1).

Robot-specific paths, PD parameters, joint orders and camera settings live in
``<robot>/<robot>.yaml``. Relative yaml paths resolve against that directory.

Usage
    python deploy_mujoco.py unitree_g1
    python deploy_mujoco.py tlibot_z1 --task z1_amp_locomotion --load_run 2026-11-09_16-05-29
    python deploy_mujoco.py tlibot_z1 --task tlibot_z1_velocity          # 该 task 最新一次导出
    python deploy_mujoco.py tlibot_z1 --policy /path/to/policy.onnx
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.velocity_command_panel import VelocityCommandPanel

SCRIPT_DIR = Path(__file__).resolve().parent


def quat_to_mat(q_wxyz):
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.ascontiguousarray(q_wxyz))
    return rot.reshape(3, 3)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def viewer_setup(viewer, model, cam, track_body_name):
    """Apply the yaml 'viewer:' camera block; track the root body."""
    viewer.cam.distance = cam["distance"]
    viewer.cam.azimuth = cam["azimuth"]
    viewer.cam.elevation = cam["elevation"]
    viewer.cam.lookat[:] = np.asarray(cam["lookat"], dtype=float)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, track_body_name)


def find_robots():
    """Robot dirs shipping a <name>/<name>.yaml next to this script."""
    return sorted(p.parent.name for p in SCRIPT_DIR.glob("*/*.yaml") if p.parent.name == p.stem)


def main():
    parser = argparse.ArgumentParser(description="Deploy basic_loco_lib velocity policy in MuJoCo (sim2sim)")
    parser.add_argument("robot", help=f"robot dir, e.g. {find_robots()} -> loads <robot>/<robot>.yaml")
    parser.add_argument("--policy", default=None, help="policy.onnx to load (default: auto-detect newest export).")
    parser.add_argument(
        "--task",
        default=None,
        help=(
            "experiment name under logs/rsl_rl/ (e.g. z1_amp_locomotion, tlibot_z1_velocity). "
            "Combined with --load_run it selects that run; alone it selects the newest export of that task."
        ),
    )
    parser.add_argument(
        "--load_run",
        default=None,
        help=(
            "run dir name to load (e.g. 2026-11-09_16-05-29); searched under the --task experiment, "
            "or under every experiment when --task is omitted. A path to the run dir / its exported "
            "dir / an .onnx file is also accepted."
        ),
    )
    parser.add_argument("--params", default=None, help="params.yaml to merge (default: next to the policy).")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="stop after this many seconds of simulated time (default: run until the window is closed).",
    )
    args = parser.parse_args()

    config_path = SCRIPT_DIR / args.robot / f"{args.robot}.yaml"
    if not config_path.is_file():
        raise SystemExit(f"[ERROR] no config for robot '{args.robot}' (expected {config_path});"
                         f" known: {find_robots()}")
    config_dir = config_path.parent
    with open(config_path) as f:
        config = yaml.safe_load(f)

    _AUTO_KEYS = (
        "mj_joint_names", "joint_orders", "backend", "obs_history_length",
        "action_joint_names", "action_mj_indices", "obs_joint_names",
        "obs_joint_mj_indices", "command_ranges", "ang_vel_scale",
        "joint_vel_scale", "stiffness", "damping", "default_joint_pos",
        "action_scale",
    )

    # ---- locate the policy -------------------------------------------------
    # Priority: --policy > (--task / --load_run) > policy.onnx beside this yaml
    # > newest exported bundle under logs/rsl_rl/ (optionally narrowed by the
    # yaml's ``experiment`` key).
    REPO_ROOT = SCRIPT_DIR.parents[2]
    log_root = REPO_ROOT / "logs" / "rsl_rl"

    def _newest_exported(root: Path):
        if not root.is_dir():
            return None
        # rsl-rl layout is logs/rsl_rl/<experiment>/<run>/exported/policy.onnx;
        # rglob keeps this depth-agnostic (experiment-narrowed or repo-wide).
        found = sorted(root.rglob("exported/policy.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
        return found[0] if found else None

    def _list_runs(root: Path) -> list[str]:
        """Available run names (with their experiment when searching everything)."""
        if root == log_root:
            runs = sorted(
                {f"{p.parent.parent.name} [{p.parent.parent.parent.name}]"
                 for p in root.glob("*/*/exported/policy.onnx")}
            )
        else:
            runs = sorted({p.parent.parent.name for p in root.glob("*/exported/policy.onnx")})
        return runs[-10:]

    def _resolve_task(task: str) -> Path:
        root = log_root / task
        if not root.is_dir():
            tasks = sorted(p.name for p in log_root.iterdir() if p.is_dir())
            raise SystemExit(
                f"[ERROR] --task '{task}' not found under {log_root}\n"
                f"        available tasks: {tasks}"
            )
        return root

    def _resolve_load_run(load_run: str, root: Path) -> Path:
        """Resolve ``--load_run``: run dir name, run dir, exported dir, or policy file."""
        run = Path(load_run).expanduser()
        if run.is_file():
            return run
        if run.is_dir():
            for cand in (run / "exported" / "policy.onnx", run / "policy.onnx"):
                if cand.is_file():
                    return cand
            hits = sorted(run.rglob("policy.onnx"))
            if hits:
                return hits[0]
            raise SystemExit(f"[ERROR] --load_run: no policy.onnx under {run}")
        # Bare run dir name.
        cands = [root / load_run / "exported" / "policy.onnx", root / load_run / "policy.onnx"]
        cands += sorted(root.glob(f"*/{load_run}/exported/policy.onnx"))
        cands += sorted(root.glob(f"*/{load_run}/policy.onnx"))
        hits = [c for c in cands if c.is_file()]
        if not hits:
            raise SystemExit(
                f"[ERROR] --load_run '{load_run}': no exported policy found under {root}\n"
                f"        available runs: {_list_runs(root)}"
            )
        if len(hits) > 1:
            print(f"[WARN] --load_run '{load_run}' matched {len(hits)} runs; using {hits[0]}")
        return hits[0]

    if args.policy:
        policy_path = Path(args.policy).expanduser().resolve()
        if not policy_path.is_file():
            raise SystemExit(f"[ERROR] --policy not found: {policy_path}")
    elif args.task or args.load_run:
        task_root = _resolve_task(args.task) if args.task else log_root
        if args.load_run:
            policy_path = _resolve_load_run(args.load_run, task_root).expanduser().resolve()
            print(f"[INFO] task={args.task or '(auto)'} run={args.load_run}")
        else:
            policy_path = _newest_exported(task_root)
            if policy_path is None:
                raise SystemExit(
                    f"[ERROR] --task '{args.task}': no exported/policy.onnx under {task_root}\n"
                    f"        available runs: {_list_runs(task_root)}"
                )
            print(f"[INFO] task={args.task} -> newest export of run {policy_path.parent.parent.name}")
    else:
        local_policy = config_dir / str(config.get("policy", "policy.onnx"))
        if local_policy.is_file():
            policy_path = local_policy
        else:
            experiment = config.get("experiment")
            policy_path = None
            if experiment:
                policy_path = _newest_exported(log_root / str(experiment))
            if policy_path is None:
                policy_path = _newest_exported(log_root)
            if policy_path is None:
                raise SystemExit(
                    "[ERROR] no policy found. Do one of:\n"
                    f"  * train first, so {log_root}/<experiment>/<run>/exported/policy.onnx exists;\n"
                    f"  * copy policy.onnx + params.yaml next to {config_path.name};\n"
                    "  * pass --task <experiment> [--load_run <run>] to pick a specific run;\n"
                    "  * pass --policy /path/to/policy.onnx"
                )
    print(f"[INFO] policy: {policy_path}")

    # ---- merge auto-exported params ----------------------------------------
    params_path = Path(args.params).expanduser().resolve() if args.params else policy_path.parent / "params.yaml"
    if params_path.is_file():
        with open(params_path) as f:
            auto_params = yaml.safe_load(f) or {}
        merged = [k for k in _AUTO_KEYS if k in auto_params]
        for key in merged:
            config[key] = auto_params[key]
        print(f"[INFO] merged auto-exported params from {params_path}: {', '.join(merged)}")
    else:
        missing = [key for key in _AUTO_KEYS if key not in config]
        if missing:
            raise SystemExit(
                f"[ERROR] {params_path} not found and {config_path.name} is missing {missing}.\n"
                "        Train first (or run play.py --export_only) to generate params.yaml,\n"
                "        or pass --params /path/to/params.yaml"
            )
        print(f"[WARN] no params.yaml at {params_path}; using the hand-written blocks in {config_path.name}")

    xml_path = config_dir / config["xml_path"]
    sim_dt = config["simulation_dt"]
    decimation = int(config["control_decimation"])
    obs_history_length = int(config.get("obs_history_length", 1))
    # Viewer refresh rate. Syncing the viewer every physics step (200 Hz) is the
    # dominant cost of this loop and made the rollout run in slow motion.
    render_fps = float(config.get("render_fps", 60.0) or 60.0)
    render_every = max(1, int(round(1.0 / (sim_dt * render_fps))))
    realtime = bool(config.get("realtime", True))
    duration = args.duration
    command = np.zeros(3, dtype=np.float32)

    mj_joint_names = [str(n) for n in config["mj_joint_names"]]
    root_body_name = str(config["root_body_name"])
    num_joints = len(mj_joint_names)
    backend = config["backend"]
    print(f"Policy backend (joint order): {backend}")
    MJ2LAB = np.asarray(config["joint_orders"][backend], dtype=int)
    LAB2MJ = np.argsort(MJ2LAB)

    # The policy may output fewer dims than there are joints when the training
    # config freezes some (``ignored_action_joints``). ``action_mj_indices`` maps
    # each policy output to its mj joint; default = full lab order (backward
    # compatible with params.yaml that predates this key).
    action_mj_indices = np.asarray(config.get("action_mj_indices", LAB2MJ), dtype=int)
    action_dim = int(action_mj_indices.size)

    # Joint-state observations may cover a different subset than the actions
    # (e.g. joints frozen at the default pose are excluded from both when the
    # training config sets ``ignored_action_joints``).
    obs_mj_indices = np.asarray(config.get("obs_joint_mj_indices", MJ2LAB), dtype=int)
    num_obs_joints = int(obs_mj_indices.size)

    kps = np.array(config["stiffness"], dtype=float)
    kds = np.array(config["damping"], dtype=float)
    default_dof = np.array(config["default_joint_pos"], dtype=float)
    action_scale = np.array(config["action_scale"], dtype=float)

    # One obs frame: command(3) | projected_gravity(3) | base_ang_vel(3)
    # | joint_pos_rel(n_obs) | joint_vel_rel(n_obs) | last_action(action_dim).
    frame_dim = 9 + 2 * num_obs_joints + action_dim
    obs_dim = obs_history_length * frame_dim

    print(f"Loading MuJoCo model: {xml_path}")
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    m.opt.timestep = sim_dt
    d = mujoco.MjData(m)

    print(f"Loading ONNX policy: {policy_path}")
    ort_session = onnxruntime.InferenceSession(str(policy_path))

    # Spawn at the velocity task's default pose, without reset noise.
    d.qpos[0:3] = config["initial_pos"]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7 : 7 + num_joints] = default_dof
    mujoco.mj_forward(m, d)

    target_dof_pos = default_dof.copy()
    action = np.zeros(action_dim, dtype=np.float32)
    obs_buffer = deque(maxlen=obs_history_length)
    print(f"policy obs: {obs_dim} dims (history_length={obs_history_length}); command={command.tolist()}")
    if render_every > 1:
        print(f"viewer: {render_fps:.0f} Hz (sync every {render_every} physics steps), realtime={realtime}")
    timestep = 0
    sim_elapsed = 0.0
    slow_warned = False

    with VelocityCommandPanel(command, config["command_ranges"]) as command_panel, \
            mujoco.viewer.launch_passive(m, d) as viewer:
        viewer_setup(viewer, m, config["viewer"], root_body_name)
        start = time.time()
        while viewer.is_running() and command_panel.running:
            if timestep % decimation == 0:
                command[:] = command_panel.read()
                if not command_panel.running:
                    break
                joint_pos = d.qpos[7 : 7 + num_joints].copy()
                joint_vel = d.qvel[6 : 6 + num_joints].copy()
                projected_gravity = quat_to_mat(d.qpos[3:7]).T @ np.array([0.0, 0.0, -1.0])
                obs = np.concatenate(
                    [
                        command,
                        projected_gravity,
                        d.qvel[3:6] * config["ang_vel_scale"],
                        (joint_pos - default_dof)[obs_mj_indices],
                        joint_vel[obs_mj_indices] * config["joint_vel_scale"],
                        action,
                    ]
                )
                if not obs_buffer:
                    for _ in range(obs_history_length):
                        obs_buffer.append(obs)
                else:
                    obs_buffer.append(obs)

                policy_input = np.concatenate(obs_buffer).astype(np.float32)
                outputs = ort_session.run(
                    output_names=["actions"],
                    input_feed={"obs_history": policy_input.reshape(1, -1)},
                )
                action = outputs[0][0]
                target_dof_pos = default_dof.copy()
                target_dof_pos[action_mj_indices] = (
                    action * action_scale[action_mj_indices] + default_dof[action_mj_indices]
                )

            # PD control at physics rate.
            tau = pd_control(
                target_dof_pos, d.qpos[7:], kps,
                np.zeros_like(d.qvel[6:]), d.qvel[6:], kds,
            )
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            timestep += 1
            sim_elapsed += sim_dt

            # Refresh the viewer at ``render_fps`` (not at the physics rate).
            if timestep % render_every == 0:
                viewer.sync()

            # Pace against the wall clock. The previous per-step sleep used
            # ``sim_dt - (now - step_start)``, which assumes a step costs nothing:
            # once a step took longer than ``sim_dt`` it never slept again, the
            # loop could not catch up and the rollout silently ran slow.
            if realtime:
                ahead = sim_elapsed - (time.time() - start)
                if ahead > 0.0005:
                    time.sleep(ahead)

            if timestep % 500 == 0:
                wall = time.time() - start
                factor = sim_elapsed / max(wall, 1e-9)
                print(f"t={sim_elapsed:6.1f}s  wall={wall:6.1f}s  realtime={factor:.2f}x")
                if factor < 0.9 and not slow_warned:
                    slow_warned = True
                    print(
                        "[WARN] rollout is slower than real time; lower 'render_fps' / window size "
                        "in the task yaml (rendering is usually the bottleneck)."
                    )

            if duration is not None and sim_elapsed >= duration:
                break

    print(f"\nSim2sim rollout complete: {timestep} physics steps, {d.time:.3f} s")


if __name__ == "__main__":
    main()
