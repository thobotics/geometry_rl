from __future__ import annotations
from omni.isaac.orbit.utils import configclass

from .grippers_cfg import Grippers, TwoSuctionGrippers
from ... import mdp


@configclass
class LinearWithYawRotationNoZActionsCfg:
    """Action specifications for the MDP."""

    def __post_init__(self):
        for i in range(Grippers.N_GRIPPERS):
            self.__dict__[f"joint_pos_{i}"] = mdp.CubeActionTermCfg(
                asset_name=f"cube_{i}", rotation_axis=2, z_action=False
            )


@configclass
class LinearWithYawRotationZActionsCfg:
    """Action specifications for the MDP."""

    def __post_init__(self):
        for i in range(Grippers.N_GRIPPERS):
            self.__dict__[f"joint_pos_{i}"] = mdp.CubeActionTermCfg(
                asset_name=f"cube_{i}", rotation_axis=2, z_action=True
            )


@configclass
class OnlyLinearActionsNoZCfg:
    """Action specifications for the MDP."""

    def __post_init__(self):
        for i in range(Grippers.N_GRIPPERS):
            self.__dict__[f"joint_pos_{i}"] = mdp.CubeActionLinearTermCfg(asset_name=f"cube_{i}", z_action=False)


@configclass
class OnlyLinearActionsZCfg:
    """Action specifications for the MDP."""

    def __post_init__(self):
        for i in range(TwoSuctionGrippers.N_GRIPPERS):
            self.__dict__[f"joint_pos_{i}"] = mdp.CubeActionLinearTermCfg(asset_name=f"cube_{i}", z_action=True)
