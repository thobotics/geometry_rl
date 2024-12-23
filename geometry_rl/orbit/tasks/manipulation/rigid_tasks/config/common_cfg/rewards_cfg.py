from __future__ import annotations
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils import configclass

from ... import mdp


@configclass
class SlidingRewardsCfg:
    """Reward terms for the MDP."""

    object_target_tracking = RewTerm(
        func=mdp.object_target_distance,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target"),
            "n_last_steps": 2,
            "weight_n_last_steps": 5.0,
            "orientation_weight": 0.5,
        },
        weight=-0.8,
    )

    object_velocity = RewTerm(
        func=mdp.object_velocity,
        params={"asset_cfg": SceneEntityCfg("object")},
        weight=-0.1,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-2e-3)


@configclass
class InsertionRewardsCfg:
    """Reward terms for the MDP."""

    object_target_tracking = RewTerm(
        func=mdp.object_insertion,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target"),
            "n_last_steps": 2,
            "weight_n_last_steps": 5.0,
            "orientation_weight": 0.5,
            "weight_orientation_n_last_steps": 5.0,
            "z_weight": 0.5,
        },
        weight=-0.8,
    )


@configclass
class InsertionTwoAgentsRewardsTwoCfg:
    """Reward terms for the MDP."""

    object_target_tracking = RewTerm(
        func=mdp.object_insertion,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target"),
            "n_last_steps": 2,
            "weight_n_last_steps": 5.0,
            "orientation_weight": 0.1,
            "weight_orientation_n_last_steps": 7.5,
            "z_weight": 0.0,
        },
        weight=-0.8,
    )


@configclass
class PushingRewardsCfg:
    """Reward terms for the MDP."""

    object_target_tracking = RewTerm(
        func=mdp.object_target_distance,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target"),
            "n_last_steps": 5,
            "weight_n_last_steps": 10.0,
            "orientation_weight": 0.1,
        },
        weight=-0.8,
    )

    object_ee_tracking = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "cube_cfg": SceneEntityCfg("cube_0"),
        },
        weight=-0.2,
    )
