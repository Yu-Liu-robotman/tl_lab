from __future__ import annotations

import re

from gym import Env
import torch
from prettytable import PrettyTable
from typing import TYPE_CHECKING
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerBase, ManagerTermBase
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from .animation_manager_cfg import AnimationTermCfg
from .motion_data_manager import MotionDataTerm

if TYPE_CHECKING:
    from legged_lab.envs import ManagerBasedAnimationEnv

#: 指令类型的判定条件："vx>0.1" / "abs(vx)<=0.1" / "vx<-0.1"
#: vx/vy = base_velocity 指令的线速度 x/y，w = 角速度 z
_COND_RE = re.compile(
    r"^(?:(?P<abs>abs)\(\s*)?(?P<var>vx|vy|w)\s*\)?\s*(?P<op>>=|<=|>|<)\s*"
    r"(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$"
)
_COND_OPS = {">": torch.gt, ">=": torch.ge, "<": torch.lt, "<=": torch.le}
#: 条件里的变量名 -> base_velocity 指令分量下标 (command = [vx, vy, wz])
_CMD_COLUMN = {"vx": 0, "vy": 1, "w": 2}

class AnimationTerm(ManagerTermBase):
    cfg: AnimationTermCfg
    _env: ManagerBasedAnimationEnv
    
    def __init__(self, cfg: AnimationTermCfg, env: ManagerBasedAnimationEnv):
        super().__init__(cfg, env)
        
        if cfg.num_steps_to_use > 0:
            self.step_indices = torch.arange(0, cfg.num_steps_to_use, dtype=torch.long, device=env.device)
        elif cfg.num_steps_to_use < 0:
            self.step_indices = torch.arange(cfg.num_steps_to_use + 1, 1, dtype=torch.long, device=env.device)
        else:
            raise ValueError("num_steps_to_use cannot be zero.")
        
        # Get motion data term
        # self.motion_data_term: MotionDataTerm = env.motion_data_manager.get_term(cfg.motion_data_term)

        # self.motion_data_term_up: MotionDataTerm = env.motion_data_manager_up.get_term(cfg.motion_data_term)
        # self.motion_data_term_walk: MotionDataTerm = env.motion_data_manager_walk.get_term(cfg.motion_data_term)
        self.motion_data_terms :dict[str, MotionDataTerm] = {}
        self.motion_num_dofs = {}
        self.num_key_bodies = {}
        
        for motion_type, manager in env.motion_data_managers.items():
            term = manager.get_term(cfg.motion_data_term)

            self.motion_data_terms[motion_type] = term
            self.motion_num_dofs[motion_type] = term.num_dofs
            self.num_key_bodies[motion_type] = term.num_key_bodies

        # create up buffers 
        self.num_steps = len(self.step_indices)
        for component in cfg.motion_data_components:
            for motion_name, motion_term in self.motion_data_terms.items():
                buffer_shape = (self.num_envs, self.num_steps)
                if component in ["root_pos_w", "root_vel_w", "root_vel_b", "root_ang_vel_w", "root_ang_vel_b"]:
                    buffer_shape += (3,)
                elif component == "root_quat":
                    buffer_shape += (4,)
                elif component in ["dof_pos", "dof_vel"]:
                    num_dofs = motion_term.num_dofs
                    buffer_shape += (num_dofs,)
                elif component == "key_body_pos_b":
                    num_key_bodies = motion_term.num_key_bodies
                    buffer_shape += (num_key_bodies, 3)
                else:
                    raise ValueError(f"Unknown motion data component: {component}")
                setattr(self,f"{component}_buffer_{motion_name}",torch.zeros(buffer_shape,device=env.device,dtype=torch.float32))

        # motion ids for each env
        for name in self.motion_data_terms.keys():
            setattr(self,f"motion_ids_{name}",torch.zeros(self.num_envs,device=env.device,dtype=torch.long))
            setattr(self,f"motion_fetch_time_{name}",torch.zeros((self.num_envs, self.num_steps),device=env.device,dtype=torch.float32))
            setattr(self,f"motion_durations_{name}",torch.zeros(self.num_envs,device=env.device,dtype=torch.float32))
            # last command / motion type per env, used by the skill-based switching
            setattr(self,f"cmd_cache_{name}",torch.zeros((self.num_envs, 3),device=env.device,dtype=torch.float32))
            setattr(self,f"motion_type_cache_{name}",torch.zeros(self.num_envs,device=env.device,dtype=torch.long))

        assert not (cfg.skill and not cfg.type_rules), "skill=True requires cfg.type_rules"

        ids = torch.arange(self.num_envs, device=env.device)
        for name, motion_term in self.motion_data_terms.items():
            self._set_motions(name, motion_term, ids, None)

        if self.cfg.enable_visualization:
            self.vis_root_offset = torch.tensor(self.cfg.vis_root_offset, device=env.device, dtype=torch.float32).unsqueeze(0)  # (1, 3)
            
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/KeyBodyVisualizerFromTerm",
                markers={
                    "red_sphere": sim_utils.SphereCfg(
                        radius=0.03, 
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
                    ),
                }
            )
            self.key_body_marker: VisualizationMarkers = VisualizationMarkers(marker_cfg)

    def _current_cmd(self):
        if not hasattr(self._env, "command_manager"):
            return None
        try:
            return self._env.command_manager.get_command(self.cfg.command_name)
        except KeyError:
            return None

    def _command_types(self, motion_term, env_ids, cmd):
        """Command -> motion type index (0 = no match). A type without clip raises."""
        assert cmd is not None, (
            f"skill=True requires a '{self.cfg.command_name}' command term"
        )
        cmd = cmd[env_ids]
        types = torch.zeros(len(env_ids), dtype=torch.long, device=self._env.device)
        for i, rule in enumerate(self.cfg.type_rules, start=1):
            if rule[0] not in motion_term.motion_type:
                raise KeyError(f"motion_type '{rule[0]}' missing, available: {motion_term.motion_type}")
            match = torch.ones(len(env_ids), dtype=torch.bool, device=self._env.device)
            for cond in rule[1:]:
                match &= self._cond_mask(cond, cmd)
            types[match] = i
        return types

    def _cond_mask(self, cond: str, cmd: torch.Tensor) -> torch.Tensor:
        """One condition string (e.g. 'vx>0.1', 'abs(vy)<=0.1') -> boolean mask.

        ``cmd`` is the (N, 3) base_velocity command; ``vx``/``vy``/``w`` select its x/y linear
        and z angular components.
        """
        m = _COND_RE.match(str(cond).strip())
        if m is None:
            raise ValueError(
                f"bad type_rules condition {cond!r}; use e.g. 'vx>0.1', 'abs(vx)<=0.1', "
                f"'vy>0.1', 'abs(vy)<=0.1', 'w<-0.1'"
            )
        col = _CMD_COLUMN[m["var"]]
        if col >= cmd.shape[1]:
            raise ValueError(f"command has only {cmd.shape[1]} components, cannot evaluate {cond!r}")
        value = cmd[:, col]
        return _COND_OPS[m["op"]](value.abs() if m["abs"] else value, float(m["val"]))

    def _set_motions(self, name, motion_term, env_ids, types=None):
        """Sample motion ids/times and fetch data for env_ids; types=None -> random over all clips."""
        motion_ids = getattr(self, f"motion_ids_{name}")
        motion_fetch_time = getattr(self, f"motion_fetch_time_{name}")

        if types is None:
            motion_ids[env_ids] = motion_term.sample_motions(len(env_ids))
        else:
            for i, rule in enumerate(self.cfg.type_rules, start=1):
                sel = env_ids[types == i]
                if len(sel) > 0:
                    motion_ids[sel] = motion_term.sample_motions_type(len(sel), motion_type=rule[0])
            # type 0 == no type_rule matched. The command ranges can produce combos the rules
            # don't cover (e.g. |vx|>0.1 AND |vy|>0.1 at once), and falling through would keep a
            # stale motion_id, so sample the whole library instead and warn once.
            unmatched = env_ids[types == 0]
            if len(unmatched) > 0:
                motion_ids[unmatched] = motion_term.sample_motions(len(unmatched))
                if not getattr(self, "_warned_unmatched_type", False):
                    self._warned_unmatched_type = True
                    print(
                        f"[WARN] {len(unmatched)}/{len(env_ids)} envs got a command matching NONE of the "
                        f"{len(self.cfg.type_rules)} type_rules -> sampled a random clip for them. "
                        f"Check that type_rules covers the whole command range."
                    )

        getattr(self, f"motion_durations_{name}")[env_ids] = motion_term.get_motion_durations(motion_ids[env_ids])
        truncate_time = self.num_steps * self._env.step_dt
        if not self.cfg.random_initialize:
            motion_fetch_time[env_ids, 0] = 0.0
        elif self.cfg.num_steps_to_use > 0:
            motion_fetch_time[env_ids, 0] = motion_term.sample_times(motion_ids[env_ids], truncate_time_end=truncate_time)
        else:
            motion_fetch_time[env_ids, 0] = motion_term.sample_times(motion_ids[env_ids], truncate_time_start=truncate_time)
        if self.num_steps > 1:
            motion_fetch_time[env_ids, 1:] = motion_fetch_time[env_ids, 0:1] + self.step_indices[1:].float() * self._env.step_dt
        self._fetch_motion_data(motion_name=name, env_ids=env_ids)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            return
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self._env.device)[env_ids]
        ids = torch.as_tensor(env_ids, device=self._env.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        cmd = self._current_cmd()
        for name, motion_term in self.motion_data_terms.items():
            # skill=True: pick the clip from the command (already resampled upstream)
            types = self._command_types(motion_term, ids, cmd) if self.cfg.skill else None
            if types is not None:
                getattr(self, f"motion_type_cache_{name}")[ids] = types
                getattr(self, f"cmd_cache_{name}")[ids] = cmd[ids]
            self._set_motions(name, motion_term, ids, types)

    def update(self, dt: float):
        cmd = self._current_cmd()
        for name, motion_term in self.motion_data_terms.items():
            motion_ids = getattr(self, f"motion_ids_{name}")
            motion_fetch_time = getattr(self, f"motion_fetch_time_{name}")

            # skill: switch the reference clip when the command *type* changes
            if self.cfg.skill and cmd is not None:
                cmd_cache = getattr(self, f"cmd_cache_{name}")
                type_cache = getattr(self, f"motion_type_cache_{name}")
                changed = (cmd != cmd_cache).any(dim=1).nonzero().flatten()
                if changed.numel() > 0:
                    cmd_cache[changed] = cmd[changed]
                    types = self._command_types(motion_term, changed, cmd)
                    switch = types != type_cache[changed]
                    if switch.any():
                        env_ids, types = changed[switch], types[switch]
                        type_cache[env_ids] = types
                        self._set_motions(name, motion_term, env_ids, types)

            if self.cfg.random_fetch:
                if self.cfg.num_steps_to_use > 0:
                    motion_fetch_time[:, 0] = motion_term.sample_times(motion_ids,truncate_time_end=self.num_steps * dt)
                else:
                    motion_fetch_time[:, 0] = motion_term.sample_times(motion_ids,truncate_time_start=self.num_steps * dt)
                if self.num_steps > 1:
                    motion_fetch_time[:, 1:] = (motion_fetch_time[:, 0:1]+ self.step_indices[1:].float() * dt)
            # fetch motion data
            self._fetch_motion_data(name)
        # Advance time
        if not self.cfg.random_fetch:
            for name in self.motion_data_terms.keys():
                motion_fetch_time = getattr(self,f"motion_fetch_time_{name}")
                motion_fetch_time += dt
        if self.cfg.enable_visualization:
            self._visualize()


    def _fetch_motion_data(self, motion_name: str, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs,device=self._env.device)

        motion_term = self.motion_data_terms[motion_name]

        motion_ids = getattr(self, f"motion_ids_{motion_name}")
        motion_fetch_time = getattr(self, f"motion_fetch_time_{motion_name}")

        # Flatten for batch processing
        num_queries = len(env_ids) * self.num_steps
        motion_times_flat = motion_fetch_time[env_ids].reshape(-1)
        motion_ids_flat = motion_ids[env_ids].repeat_interleave(self.num_steps)

        # Sanity check
        assert motion_times_flat.shape[0] == num_queries
        assert motion_ids_flat.shape[0] == num_queries

        # Fetch motion data
        motion_data_dict = motion_term.get_motion_state(
            motion_ids=motion_ids_flat,
            motion_times=motion_times_flat,
        )

        # Store buffers
        for component in self.cfg.motion_data_components:
            if component in motion_data_dict:
                buffer_name = f"{component}_buffer_{motion_name}"
                data = motion_data_dict[component]
                data_reshaped = data.view(len(env_ids),self.num_steps,*data.shape[1:])
                getattr(self, buffer_name)[env_ids, :, :] = data_reshaped      
                     
    def _visualize(self):
        # try to get the 'robot_anim' articulation
        robot_anim: Articulation = self._env.scene["robot_anim"]
        if robot_anim is None:
            print("[WARNING] AnimationTerm visualization: 'robot_anim' not found in the scene.")
            return
        
        # check if some necessary buffers are available
        if not hasattr(self, "root_pos_w_buffer") or not hasattr(self, "root_quat_buffer") or not hasattr(self, "dof_pos_buffer"):
            print("[WARNING] AnimationTerm visualization: 'root_pos_w_buffer' or 'root_quat_buffer' or 'dof_pos_buffer' not found.")
            return
        
        root_pos_w = self.root_pos_w_buffer[:, 0, :]  # (num_envs, 3)
        root_quat = self.root_quat_buffer[:, 0, :]    # (num_envs, 4)
        dof_pos = self.dof_pos_buffer[:, 0, :]        # (num_envs, num_dofs)
        
        root_states = robot_anim.data.default_root_state.clone()
        root_states[:, :3] = root_pos_w + self._env.scene.env_origins[:, :3] + self.vis_root_offset
        root_states[:, 3:7] = root_quat
        root_states[:, 7:10] = 0.0  # zero linear velocity
        root_states[:, 10:13] = 0.0  # zero angular velocity
        robot_anim.write_root_pose_to_sim_index(root_pose=root_states[:, :7])
        robot_anim.write_root_velocity_to_sim_index(root_velocity=root_states[:, 7:13])
        
        joint_pos = robot_anim.data.default_joint_pos.clone()
        joint_pos[:, :] = dof_pos
        joint_vel = torch.zeros_like(robot_anim.data.default_joint_vel)
        robot_anim.write_joint_position_to_sim_index(position=joint_pos)
        robot_anim.write_joint_velocity_to_sim_index(velocity=joint_vel)
            
        key_body_pos_b = self.key_body_pos_b_buffer[:, 0, :, :]  # (num_envs, num_key_bodies, 3)
        num_key_bodies = key_body_pos_b.shape[1]
        key_body_pos_w = root_states[:, :3].unsqueeze(1) + math_utils.quat_apply(
            root_quat.unsqueeze(1).expand(-1, num_key_bodies, -1),
            key_body_pos_b
        )
        self.key_body_marker.visualize(
            translations=key_body_pos_w.reshape(-1, 3)
        )
            
    # Some helper functions
    def _get_motion_buffer(self,component: str,env_ids: Sequence[int] = None,motion_type: str = None,) -> torch.Tensor:
        if motion_type is None:
            raise ValueError("motion_type must be specified")
        buffer_name = f"{component}_buffer_{motion_type}"
        if not hasattr(self, buffer_name):
            raise AttributeError(f"{buffer_name} not found in AnimationTerm.")
        buffer = getattr(self, buffer_name)
        if env_ids is None:
            return buffer
        return buffer[env_ids, :, :]
    
    def get_root_pos_w(self,env_ids: Sequence[int] = None,motion_type: str = None):

        return self._get_motion_buffer("root_pos_w",env_ids,motion_type)
    
    def get_root_quat(self,env_ids: Sequence[int] = None,motion_type: str = None):

        return self._get_motion_buffer("root_quat",env_ids,motion_type)

    def get_dof_pos(self,env_ids: Sequence[int] = None,motion_type: str = None):

        return self._get_motion_buffer("dof_pos",env_ids,motion_type)
    
    def get_dof_vel(self,env_ids: Sequence[int] = None,motion_type: str = None):

        return self._get_motion_buffer("dof_vel",env_ids,motion_type)

    def get_key_body_pos_b(self,env_ids: Sequence[int] = None,motion_type: str = None):

        return self._get_motion_buffer("key_body_pos_b",env_ids,motion_type)
    
    def get_root_vel_w(self,env_ids: Sequence[int] = None,motion_type: str = None):

        return self._get_motion_buffer("root_vel_w",env_ids,motion_type)

    def get_root_ang_vel_w(self,env_ids: Sequence[int] = None,motion_type: str = None):

        return self._get_motion_buffer("root_ang_vel_w",env_ids,motion_type)

    def get_root_ang_vel_b(self,env_ids: Sequence[int] = None,motion_type: str = None):

        return self._get_motion_buffer("root_ang_vel_b",env_ids,motion_type)

    def get_root_vel_b(self,env_ids: Sequence[int] = None,motion_type: str = None):

        return self._get_motion_buffer("root_vel_b",env_ids,motion_type)

        
            
class AnimationManager(ManagerBase):
    
    def __init__(self, cfg: object, env: ManagerBasedAnimationEnv):
        if cfg is None:
            raise ValueError("AnimationManager configuration is required.")
        
        self._terms: dict[str, AnimationTerm] = {}
        self._term_cfgs: dict[str, AnimationTermCfg] = {}
        
        super().__init__(cfg, env)
        
    def __str__(self) -> str:
        """Returns: A string representation for animation data manager."""
        msg = f"<AnimationManager> contains {len(self._terms)} active terms.\n"
        
        # create table for term infomation
        table = PrettyTable()
        table.title = "Animation Manager Terms"
        table.field_names = ["Index", "Term Name", "Motion Data Term", "Num Steps to Use", "Random Init", "Random Fetch"]
        for index, (term_name, term_cfg) in enumerate(self._term_cfgs.items()):
            table.add_row([
                index, 
                term_name,
                term_cfg.motion_data_term,
                term_cfg.num_steps_to_use,
                term_cfg.random_initialize,
                term_cfg.random_fetch
            ])
        msg += table.get_string()
        msg += "\n"
        
        return msg

    def update(self, dt: float):
        """Update all animation terms.

        Args:
            dt: The time step to update the animation terms.
        """
        for term in self._terms.values():
            term.update(dt)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            return {}
        # reset all terms
        for term in self._terms.values():
            term.reset(env_ids)
        return {}
        
    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active command terms."""
        return list(self._terms.keys())
    
    def get_term(self, term_name: str) -> AnimationTerm:
        """Get the animation data term by name."""
        if term_name not in self._terms:
            raise KeyError(f"Animation data term '{term_name}' not found.")
        return self._terms[term_name]

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms
        for term_name, term_cfg in cfg_items:
            # check for non config
            if term_cfg is None:
                continue
            # check for valid config type
            if not isinstance(term_cfg, AnimationTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type AnimationTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = AnimationTerm(term_cfg, self._env)
            # add class to dict
            self._terms[term_name] = term
            self._term_cfgs[term_name] = term_cfg





