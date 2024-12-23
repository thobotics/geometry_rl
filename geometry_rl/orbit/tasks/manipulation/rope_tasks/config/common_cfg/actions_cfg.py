from __future__ import annotations
from omni.isaac.orbit.utils import configclass

from .grippers_cfg import ClosingGrippers
from ... import mdp


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    def __post_init__(self):
        for i in range(ClosingGrippers.N_GRIPPERS):
            self.__dict__[f"joint_pos_{i}"] = mdp.CubeActionTermCfg(asset_name=f"cube_{i}")
