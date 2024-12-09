from __future__ import annotations
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.utils import configclass

from math import pi

from ... import mdp
from .grippers_cfg import HangingGrippers

from ironlib.orbit.tasks.common.world_frame_randomization import reset_cubes_uniform_around_origin


@configclass
class HangingRandomizationCfg:
    """Configuration for randomization."""

    def __post_init__(self):
        """Post initialization."""

        self.__dict__[f"reset_cubes"] = RandTerm(
            func=mdp.reset_cubes_uniform_rotation_around_y,
            mode="reset",
            params={
                "angle_range": (-pi, pi),
                "velocity_range": {},
                "n_cubes": HangingGrippers.N_GRIPPERS,
            },
        )

        self.__dict__["rotate_around_target"] = RandTerm(
            func=reset_cubes_uniform_around_origin,
            mode="reset",
            params={
                "origin_name": "hanger",
                "origin_position_range": {
                    "x": (-0.5, 0.5),
                    # "y": (-0.5, 0.5),
                    "z": (-0.5, 0.5),
                },
                "angle_ranges": {
                    "roll": (-pi / 4, pi / 2),
                    "pitch": (-pi / 2, pi / 2),
                    "yaw": (-pi, pi),
                    # One rotation
                    # "roll": (0, 0),
                    # "pitch": (0, 0),
                    # "yaw": (0, 0),
                    # Only roll
                    # "roll": (-pi / 4, pi / 2),
                    # "pitch": (0, 0),
                    # "yaw": (0, 0),
                    # Quarter half yaw
                    # "roll": (0, 0),
                    # "pitch": (0, 0),
                    # "yaw": (-pi / 8, pi / 8),
                    # Quarter yaw
                    # "roll": (0, 0),
                    # "pitch": (0, 0),
                    # "yaw": (-pi / 4, pi / 4),
                    # Half yaw
                    # "roll": (0, 0),
                    # "pitch": (0, 0),
                    # "yaw": (-pi / 2, pi / 2),
                    # Full yaw
                    # "roll": (0, 0),
                    # "pitch": (0, 0),
                    # "yaw": (-pi, pi),
                },  # upper hemisphere and a bit lower hemisphere (-pi / 4)
                "velocity_range": {},
                "n_cubes": HangingGrippers.N_GRIPPERS,
            },
        )
