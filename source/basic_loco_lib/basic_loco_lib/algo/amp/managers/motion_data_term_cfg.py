from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

MotionWeight = tuple[float, str]

@configclass 
class MotionDataTermCfg:
    """
    Configuration for the motion data term in the motion data manager.
    """
    
    weight: float = 1.0
    """Weight of this term in the motion data manager."""
    
    motion_data_dir: str = MISSING
    """Directory containing motion data files.
    
    Only supports reading .pkl files from this directory.
    """

    motion_data_weights: dict[str, MotionWeight | float] = MISSING
    """Weights for the motion data in this term."""

    root_body_name: str = "base_link"
    """Root body whose pose is treated as the motion root (used when reading .npz clips)."""

    key_body_names: tuple[str, ...] = ()
    """Key bodies tracked as reference end-effectors (used when reading .npz clips)."""

    joint_names: tuple[str, ...] = ()
    """Joint ordering to reindex to; empty keeps the order stored in the clip."""
    