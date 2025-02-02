from __future__ import annotations
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.utils import configclass
from geometry_rl.orbit.tasks.common.world_frame_randomization import (
    reset_objects_uniform_around_origin,
)

from math import pi

from ... import mdp


@configclass
class SlidingRandomizationCfg:
    """Configuration for randomization."""

    def __post_init__(self):
        """Post initialization."""

        """ 
        1. Randomly translate the object cube, keeping the target in place.
        2. Gobaly rotate the object cube and the target in z-axis.
        3. Randomly rotate the object cube and the target in x-axis.
        """

        self.__dict__["only_translate_object_cube"] = RandTerm(
            func=mdp.reset_joint_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (-1.0, -1.0),
                    "pitch": (-pi / 2, -pi / 2),
                },
                "velocity_range": {},
                "asset_names": ["object", "cube_0"],
                "use_default_root_state_for_translation": True,
                "use_default_root_state_for_rotation": True,
            },
        )

        self.__dict__["only_translate_target"] = RandTerm(
            func=mdp.reset_joint_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "z": (-1.0, -1.0),
                    "pitch": (-pi / 2, -pi / 2),
                },
                "velocity_range": {},
                "asset_names": ["target"],
                "use_default_root_state_for_translation": True,
                "use_default_root_state_for_rotation": True,
            },
        )

        self.__dict__["only_rotate_object_cube"] = RandTerm(
            func=mdp.reset_joint_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "roll": (-pi, pi),
                },
                "velocity_range": {},
                "asset_names": ["object", "cube_0"],
                "use_default_root_state_for_translation": False,
                "use_default_root_state_for_rotation": False,
            },
        )

        self.__dict__["only_rotate_target"] = RandTerm(
            func=mdp.reset_joint_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "roll": (-pi, pi),
                },
                "velocity_range": {},
                "asset_names": ["target"],
                "use_default_root_state_for_translation": False,
                "use_default_root_state_for_rotation": False,
            },
        )


@configclass
class InsertionRandomizationCfg:
    """Configuration for randomization."""

    def __post_init__(self):
        """Post initialization."""

        """ 
        1. Randomly translate the object cube, keeping the target in place.
        2. Gobaly rotate the object cube and the target in z-axis.
        3. Randomly rotate the object cube and the target in x-axis.
        """

        self.__dict__["only_translate_object_cube"] = RandTerm(
            func=mdp.reset_joint_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (0.0, 0.5),
                    "pitch": (-pi / 2, -pi / 2),
                },
                "velocity_range": {},
                "asset_names": ["object", "cube_0"],
                "use_default_root_state_for_translation": True,
                "use_default_root_state_for_rotation": True,
            },
        )

        self.__dict__["only_translate_target"] = RandTerm(
            func=mdp.reset_joint_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "z": (-1.0, -1.0),
                    "pitch": (-pi / 2, -pi / 2),
                },
                "velocity_range": {},
                "asset_names": ["target", "target_hole"],
                "use_default_root_state_for_translation": True,
                "use_default_root_state_for_rotation": True,
            },
        )

        self.__dict__["only_rotate_object_cube"] = RandTerm(
            func=mdp.reset_joint_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "roll": (-pi, pi),
                },
                "velocity_range": {},
                "asset_names": ["object", "cube_0"],
                "use_default_root_state_for_translation": False,
                "use_default_root_state_for_rotation": False,
            },
        )

        self.__dict__["only_rotate_target"] = RandTerm(
            func=mdp.reset_joint_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "roll": (-pi, pi),
                },
                "velocity_range": {},
                "asset_names": ["target", "target_hole"],
                "use_default_root_state_for_translation": False,
                "use_default_root_state_for_rotation": False,
            },
        )


@configclass
class InsertionTwoAgentsRandomizationCfg:
    """Configuration for randomization."""

    def __post_init__(self):
        """Post initialization."""

        self.__dict__["only_rotate_object_cube"] = RandTerm(
            func=mdp.reset_joint_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.25, 0.75),
                    "y": (-0.75, 0.75),
                    "z": (0.5, 1.25),
                    "roll": (-pi, pi),
                },
                "velocity_range": {},
                "asset_names": ["object", "cube_0", "cube_1"],
                "use_default_root_state_for_translation": True,
                "use_default_root_state_for_rotation": True,
            },
        )

        self.__dict__["only_rotate_target"] = RandTerm(
            func=mdp.reset_joint_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "roll": (-pi, pi),
                },
                "velocity_range": {},
                "asset_names": ["target", "target_hole"],
                "use_default_root_state_for_translation": True,
                "use_default_root_state_for_rotation": True,
            },
        )

        self.__dict__["rotate_around_target"] = RandTerm(
            func=reset_objects_uniform_around_origin,
            mode="reset",
            params={
                "angle_ranges": {
                    # Stand placement
                    "roll": (0, 0),
                    "pitch": (-pi / 2, 0),
                    "yaw": (-pi, pi),
                },
                "asset_names": ["object", "cube_0", "cube_1"],
                "origin_names": ["target", "target_hole"],
                "use_default_root_state_for_translation": False,
                "use_default_root_state_for_rotation": True,
            },
        )


@configclass
class PushingRandomizationCfg:
    """Configuration for randomization."""

    def __post_init__(self):
        """Post initialization."""

        """ 
        1. Randomly translate the object cube, keeping the target in place.
        2. Gobaly rotate the object cube and the target in z-axis.
        3. Randomly rotate the object cube and the target in x-axis.
        """

        self.__dict__["only_translate_object_cube"] = RandTerm(
            func=mdp.reset_joint_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-1.0, -1.0),
                    "pitch": (-pi / 2, -pi / 2),
                },
                "velocity_range": {},
                "asset_names": ["object", "cube_0"],
                "use_default_root_state_for_translation": True,
                "use_default_root_state_for_rotation": True,
            },
        )

        self.__dict__["only_translate_target"] = RandTerm(
            func=mdp.reset_joint_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "z": (-1.0, -1.0),
                    "pitch": (-pi / 2, -pi / 2),
                },
                "velocity_range": {},
                "asset_names": ["target"],
                "use_default_root_state_for_translation": True,
                "use_default_root_state_for_rotation": True,
            },
        )

        self.__dict__["only_rotate_object_cube"] = RandTerm(
            func=mdp.reset_joint_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "roll": (-pi, pi),
                },
                "velocity_range": {},
                "asset_names": ["object", "cube_0"],
                "use_default_root_state_for_translation": False,
                "use_default_root_state_for_rotation": False,
            },
        )

        self.__dict__["only_rotate_target"] = RandTerm(
            func=mdp.reset_joint_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "roll": (-pi, pi),
                },
                "velocity_range": {},
                "asset_names": ["target"],
                "use_default_root_state_for_translation": False,
                "use_default_root_state_for_rotation": False,
            },
        )
