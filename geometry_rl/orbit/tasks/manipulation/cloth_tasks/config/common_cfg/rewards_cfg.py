from __future__ import annotations
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils import configclass

from ... import mdp
from .aux_data import aux_data


@configclass
class ICLRHangingRewardsCfg:
    """Reward terms for the MDP."""

    hole_hanger_tracking = RewTerm(
        func=mdp.hole_centroid_hanger_distance,
        params={
            "aux_data": aux_data,
            "asset_cfg": SceneEntityCfg("cloth"),
            "hanger_cfg": SceneEntityCfg("hanger"),
            "n_last_steps": 2,
            "weight_n_last_steps": 5.0,
            "orientation_weight": 0.1,
        },
        weight=-0.8,
    )

    points_velocity = RewTerm(
        func=mdp.points_velocity,
        params={"asset_cfg": SceneEntityCfg("cloth")},
        weight=-0.2,
    )

    points_distorion = RewTerm(
        func=mdp.points_distortion,
        params={"aux_data": aux_data, "asset_cfg": SceneEntityCfg("cloth")},
        weight=-1.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-2e-3)


@configclass
class HangingRewardsCfg:
    """Reward terms for the MDP."""

    hole_hanger_tracking = RewTerm(
        func=mdp.hole_centroid_hanger_distance,
        params={
            "aux_data": aux_data,
            "asset_cfg": SceneEntityCfg("cloth"),
            "hanger_cfg": SceneEntityCfg("hanger"),
            "n_last_steps": 2,
            "weight_n_last_steps": 5.0,
            "orientation_weight": 0.1,
        },
        weight=-0.8,
    )

    points_velocity = RewTerm(
        func=mdp.points_velocity,
        params={"asset_cfg": SceneEntityCfg("cloth")},
        weight=-0.2,
    )

    points_distorion = RewTerm(
        func=mdp.points_distortion,
        params={
            "aux_data": aux_data,
            "asset_cfg": SceneEntityCfg("cloth"),
            "area_based": True,
            "edge_based": False,
        },
        weight=-1.1,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-2e-3)
