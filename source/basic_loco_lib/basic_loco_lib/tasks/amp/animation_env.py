"""AMP animation environment: hosts the motion-data and animation managers.

Ported from ``tlibot_legged_lab`` (``legged_lab/envs/manager_based_animation_env.py``) and
adapted to this repo's env stack (xyzw quaternions, Newton backend, existing ``AmpEnv``).
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

from basic_loco_lib.algo.amp.managers import AnimationManager, MotionDataManager

from .amp_env import AmpEnv


class AmpAnimationEnv(AmpEnv):
    """``AmpEnv`` plus a motion-data / animation manager pair per motion set.

    Discovery mirrors the reference framework: every ``motion_data_<type>`` attribute on the env
    cfg is paired with ``animation_<type>``. Pairs with an empty dataset directory are skipped so
    the same cfg can run with or without reference motion.
    """

    def load_managers(self):
        self.motion_data_managers: dict[str, MotionDataManager] = {}
        self.animation_managers: dict[str, AnimationManager] = {}

        cfg = self.cfg
        for attr_name in dir(cfg):
            if not attr_name.startswith("motion_data_"):
                continue
            motion_type = attr_name[len("motion_data_"):]
            animation_cfg_name = f"animation_{motion_type}"
            if not hasattr(cfg, animation_cfg_name):
                print(f"[WARN] {animation_cfg_name} not found, skip motion set '{motion_type}'")
                continue
            motion_cfg = getattr(cfg, attr_name)
            dataset_cfg = getattr(motion_cfg, "motion_dataset", None)
            if dataset_cfg is None or not getattr(dataset_cfg, "motion_data_dir", ""):
                continue

            motion_manager = MotionDataManager(motion_cfg, self)
            self.motion_data_managers[motion_type] = motion_manager
            setattr(self, f"motion_data_manager_{motion_type}", motion_manager)

            animation_manager = AnimationManager(getattr(cfg, animation_cfg_name), self)
            self.animation_managers[motion_type] = animation_manager
            setattr(self, f"animation_manager_{motion_type}", animation_manager)

            print(f"[INFO] MotionDataManager '{motion_type}': {motion_manager}")
            print(f"[INFO] AnimationManager '{motion_type}': {animation_manager}")

        if not self.animation_managers:
            raise RuntimeError(
                "No motion/animation manager was created. Check that the env cfg declares "
                "motion_data_<type> / animation_<type> with a non-empty motion_data_dir."
            )

        super().load_managers()

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        for animation_manager in self.animation_managers.values():
            animation_manager.reset(env_ids)
        self.sim.forward()

    def step(self, action: torch.Tensor):
        obs, rewards, terminated, time_outs, extras = super().step(action)
        for animation_manager in self.animation_managers.values():
            animation_manager.update(dt=self.step_dt)
        return obs, rewards, terminated, time_outs, extras
