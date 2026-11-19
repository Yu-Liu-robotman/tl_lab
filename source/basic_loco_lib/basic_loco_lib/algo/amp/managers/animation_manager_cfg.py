from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

@configclass
class AnimationTermCfg:
    """"Configureation for an animation"""
    
    motion_data_term: str = MISSING
    """The motion data term to use for this animation term."""
    
    motion_data_components: list[str] = MISSING
    """The components of the motion data to use for this animation term."""
    
    num_steps_to_use: int = 1
    """Number of steps of motion data to extract from the motion data term.
        If positive, extracts current and future steps.
        If negative, extracts current and past steps.
        1 and -1 both extract only the current step.
        0 is invalid.
    """
    
    random_initialize: bool = False
    """Whether to randomly initialize the starting point in the motion data term."""
    
    random_fetch: bool = False
    """Whether to randomly fetch the motion data at each step."""

    skill: bool = False
    """Whether to pick the reference motion by the velocity command.

    True  -> 按指令用 type_rules 分类选参考动作（缺该类型数据时直接报错）
    False -> 忽略指令，全库随机采样
    """

    command_name: str = "twist"
    """Command term the ``type_rules`` conditions are evaluated against."""

    type_rules: tuple = ()
    """``((motion_type, *conditions), ...)`` used when ``skill=True``: all conditions of a row must
    hold (AND) to pick that motion type, e.g. ``("walk", "vx>0.1")``,
    ``("turn_left", "w>0.1", "abs(vx)<=0.1")``. Conditions are ``vx``/``w`` with
    ``> >= < <=`` (``abs(...)`` allowed). Later rows override earlier ones; index + 1 is the
    internal motion type id."""

    enable_visualization: bool = True
    """Whether to enable visualization for this animation term."""
    
    vis_root_offset: list[float] = (0.0, 0.0, 0.0)
    """Root position offset for visualization (x, y, z)."""