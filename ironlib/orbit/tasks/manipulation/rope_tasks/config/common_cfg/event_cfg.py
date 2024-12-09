from __future__ import annotations
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils import configclass

from math import pi

from ... import mdp
from .grippers_cfg import ClosingGrippers, ShapingGrippers


@configclass
class ClosingRandomizationCfg:
    """Configuration for randomization."""

    def __post_init__(self):
        """Post initialization."""
        self.__dict__[f"reset_cubes"] = RandTerm(
            func=mdp.reset_cubes_uniform_rotation_around_z,
            mode="reset",
            params={
                "angle_range": (-pi / 4, pi / 4),
                "velocity_range": {},
                "n_cubes": ClosingGrippers.N_GRIPPERS,
            },
        )

        self.__dict__["rotate_around_target"] = RandTerm(
            func=mdp.reset_cubes_uniform_around_origin,
            mode="reset",
            params={
                "origin_name": "hanger",
                "origin_position_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                },
                "angle_ranges": {
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (-pi, pi),
                    # "yaw": (0.0, 0.0),
                },
                "velocity_range": {},
                "n_cubes": ClosingGrippers.N_GRIPPERS,
            },
        )


@configclass
class ShapingRandomizationCfg:
    """Configuration for randomization."""

    def __post_init__(self):
        """Post initialization."""

        self.__dict__[f"reset_cubes"] = RandTerm(
            func=mdp.reset_cubes_uniform_rotation_around_z_with_target_shape,
            mode="reset",
            params={
                "angle_range": (-pi / 2, -pi / 4, pi / 4, pi / 2),
                "velocity_range": {},
                "n_cubes": ShapingGrippers.N_GRIPPERS,
                "target_cfg": SceneEntityCfg("target"),
                "target_angle_range": (-pi / 2, pi / 2),
            },
        )
