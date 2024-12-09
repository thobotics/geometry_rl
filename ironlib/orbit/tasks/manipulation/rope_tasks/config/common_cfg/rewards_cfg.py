from __future__ import annotations
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils import configclass

from ... import mdp
from .aux_data import aux_data


@configclass
class ClosingRewardsCfg:
    """Reward terms for the MDP."""

    rope_closing = RewTerm(
        func=mdp.rope_closing,
        params={
            "cube_0_cfg": SceneEntityCfg("cube_0"),
            "cube_1_cfg": SceneEntityCfg("cube_1"),
            "n_last_steps": 20,
        },
        weight=-2.0,
    )

    rope_wrapping = RewTerm(
        func=mdp.rope_wrapping,
        params={
            "asset_cfg": SceneEntityCfg("rope"),
            "hanger_cfg": SceneEntityCfg("hanger"),
        },
        weight=-8e-1,
    )

    links_velocity = RewTerm(
        func=mdp.links_velocity,
        params={"asset_cfg": SceneEntityCfg("rope")},
        weight=-1e-2,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)


@configclass
class ShapingRewardsCfg:
    """Reward terms for the MDP."""

    rope_target_tracking = RewTerm(
        func=mdp.rope_target_distances,
        params={
            "aux_data": aux_data,
            "asset_cfg": SceneEntityCfg("rope"),
            "n_last_steps": 10,
            "weight_n_last_steps": 5.0,
        },
        weight=-1.0,
    )

    # points_velocity = RewTerm(
    #     func=mdp.links_velocity,
    #     params={
    #         "asset_cfg": SceneEntityCfg("rope"),
    #         "n_last_steps": 5,
    #         "weight_n_last_steps": 2.0,
    #     },
    #     weight=-1e-3,
    # )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
