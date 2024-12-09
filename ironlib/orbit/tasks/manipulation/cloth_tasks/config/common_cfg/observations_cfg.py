from __future__ import annotations

from dataclasses import MISSING, dataclass
from typing import List

from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.noise.noise_cfg import AdditiveGaussianNoiseCfg

from ... import mdp
from .aux_data import aux_data
from .grippers_cfg import HangingGrippers


class FullSceneObservation:
    @classmethod
    def grippers_positions(cls):
        return ObsTerm(
            func=mdp.object_positions,
            params={
                "names": [f"cube_{i}" for i in range(HangingGrippers.N_GRIPPERS)],
            },
        )

    @classmethod
    def grippers_velocities(cls):
        return ObsTerm(
            func=mdp.object_velocities,
            params={
                "names": [f"cube_{i}" for i in range(HangingGrippers.N_GRIPPERS)],
            },
        )

    @classmethod
    def grippers_init_positions(cls):
        return ObsTerm(
            func=mdp.object_init_positions,
            params={
                "names": [f"cube_{i}" for i in range(HangingGrippers.N_GRIPPERS)],
            },
        )

    @classmethod
    def target_hook_positions(cls):
        return ObsTerm(
            func=mdp.object_positions,
            params={
                "names": ["hanger"],
            },
        )

    @classmethod
    def target_hook_orientation(cls):
        return ObsTerm(
            func=mdp.object_orientations,
            params={
                "names": ["hanger"],
            },
        )

    @classmethod
    def target_hook_geometry(cls):
        return ObsTerm(
            func=mdp.object_geometry,
            params={"name": "hanger", "aux_data": aux_data},
        )

    @classmethod
    def init_points_positions(cls):
        return ObsTerm(
            func=mdp.cloth_geometry_positions,
            params={
                "asset_cfg": SceneEntityCfg("cloth"),
                "aux_data": aux_data,
            },
        )

    @classmethod
    def cloth_edges(cls):
        return ObsTerm(
            func=mdp.cloth_edges,
            params={
                "asset_cfg": SceneEntityCfg("cloth"),
                "aux_data": aux_data,
            },
        )

    @classmethod
    def points_positions(cls):
        return ObsTerm(
            func=mdp.points_positions,
            params={
                "asset_cfg": SceneEntityCfg("cloth"),
            },
        )

    @classmethod
    def points_velocities(cls):
        return ObsTerm(
            func=mdp.points_velocities,
            params={
                "asset_cfg": SceneEntityCfg("cloth"),
            },
        )

    @classmethod
    def hole_boundary_positions(cls):
        return ObsTerm(
            func=mdp.hole_boundary_positions,
            params={
                "asset_cfg": SceneEntityCfg("cloth"),
                "aux_data": aux_data,
            },
        )

    @classmethod
    def hole_boundary_target_distances(cls):
        return ObsTerm(
            func=mdp.hole_boundary_target_distances,
            params={
                "asset_cfg": SceneEntityCfg("cloth"),
                "hanger_cfg": SceneEntityCfg("hanger"),
                "aux_data": aux_data,
            },
        )

    @classmethod
    def points_distortion(cls):
        return ObsTerm(
            func=mdp.points_distortion_obs,
            params={
                "asset_cfg": SceneEntityCfg("cloth"),
                "aux_data": aux_data,
            },
        )

    @classmethod
    def hole_boundary_indices(cls):
        return ObsTerm(
            func=mdp.hole_boundary_indices,
            params={
                "asset_cfg": SceneEntityCfg("cloth"),
                "aux_data": aux_data,
            },
        )


@configclass
class FullObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ScalarCfg(ObsGroup):
        hole_target_distances = FullSceneObservation.hole_boundary_target_distances()
        cloth_edges_length = FullSceneObservation.cloth_edges()

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PositionVectorsCfg(ObsGroup):
        grippers = FullSceneObservation.grippers_positions()
        particles = FullSceneObservation.points_positions()
        init_particles = FullSceneObservation.init_points_positions()
        hole_boundary = FullSceneObservation.hole_boundary_positions()
        target_hook = FullSceneObservation.target_hook_positions()

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class VelocityVectorsCfg(ObsGroup):
        grippers = FullSceneObservation.grippers_velocities()
        particles = FullSceneObservation.points_velocities()

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class InfoCfg(ObsGroup):
        hole_boundary_indices = FullSceneObservation.hole_boundary_indices()

    # observation groups
    scalars: ScalarCfg = ScalarCfg()
    position_vectors: PositionVectorsCfg = PositionVectorsCfg()
    velocity_vectors: VelocityVectorsCfg = VelocityVectorsCfg()
    infos: InfoCfg = InfoCfg()
