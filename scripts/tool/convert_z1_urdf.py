"""把 URDF 转成 USD，并且打开自碰撞（self_collision=True）。"""

import argparse
import os

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.physics import PhysicsCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.version import standalone_importers_available

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="输入 .urdf")
parser.add_argument("--output", required=True, help="输出目录（不是文件！）")
add_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.require_kit = not standalone_importers_available()
args_cli.physics = "isaacsim_physx" if args_cli.require_kit else "newton_mjwarp"

cfg = UrdfConverterCfg(
    asset_path=os.path.abspath(args_cli.input),
    usd_dir=os.path.abspath(args_cli.output),
    force_usd_conversion=True,      # 强制覆盖旧产物
    fix_base=False,
    merge_fixed_joints=True,        # 你原来 --merge-joints
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="none",         # 你原来 --joint-target-type none
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
    ),
    self_collision=True,            # ★ 就这一个字是关键
)

with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli):
    converter = UrdfConverter(cfg)
    print("生成成功:", converter.usd_path)