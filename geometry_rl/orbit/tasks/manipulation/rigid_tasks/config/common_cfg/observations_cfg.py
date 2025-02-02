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
from .grippers_cfg import Grippers


class FullSceneObservation:
    @classmethod
    def grippers_positions(cls):
        return ObsTerm(
            func=mdp.object_positions,
            params={
                "names": [f"cube_{i}" for i in range(Grippers.N_GRIPPERS)],
            },
        )

    @classmethod
    def grippers_velocities(cls):
        return ObsTerm(
            func=mdp.object_velocities,
            params={
                "names": [f"cube_{i}" for i in range(Grippers.N_GRIPPERS)],
            },
        )

    @classmethod
    def grippers_angular_velocities(cls):
        return ObsTerm(
            func=mdp.object_angular_velocities,
            params={
                "names": [f"cube_{i}" for i in range(Grippers.N_GRIPPERS)],
            },
        )

    @classmethod
    def grippers_init_positions(cls):
        return ObsTerm(
            func=mdp.object_init_positions,
            params={
                "names": [f"cube_{i}" for i in range(Grippers.N_GRIPPERS)],
            },
        )

    @classmethod
    def target_positions(cls):
        return ObsTerm(
            func=mdp.object_positions,
            params={
                "names": ["target"],
            },
        )

    @classmethod
    def target_geometry(cls):
        return ObsTerm(
            func=mdp.object_geometry_positions,
            params={"name": "target", "aux_data": aux_data},
        )

    @classmethod
    def object_geometry_positions(cls):
        return ObsTerm(
            func=mdp.object_geometry_positions,
            params={"name": "object", "aux_data": aux_data},
            noise=AdditiveGaussianNoiseCfg(
                mean=0.0,
                std=0.05,
            ),
        )

    @classmethod
    def object_num_points(cls):
        return ObsTerm(
            func=mdp.object_num_points,
            params={"name": "object", "aux_data": aux_data},
        )

    @classmethod
    def object_geometry_edges(cls):
        return ObsTerm(
            func=mdp.object_geometry_edges,
            params={"name": "object", "aux_data": aux_data},
        )

    @classmethod
    def object_num_edges(cls):
        return ObsTerm(
            func=mdp.object_num_edges,
            params={"name": "object", "aux_data": aux_data},
        )

    @classmethod
    def object_positions(cls):
        return ObsTerm(
            func=mdp.object_positions,
            params={
                "names": ["object"],
            },
        )

    @classmethod
    def object_velocities(cls):
        return ObsTerm(
            func=mdp.object_velocities,
            params={
                "names": ["object"],
            },
            noise=AdditiveGaussianNoiseCfg(
                mean=0.0,
                std=0.05,
            ),
        )

    @classmethod
    def object_angular_velocities(cls):
        return ObsTerm(
            func=mdp.object_angular_velocities,
            params={
                "names": ["object"],
            },
        )

    @classmethod
    def object_target_distance_obs(cls):
        return ObsTerm(
            func=mdp.object_target_distance_obs,
            params={
                "aux_data": aux_data,
                "asset_cfg": SceneEntityCfg("object"),
                "target_cfg": SceneEntityCfg("target"),
            },
        )


@configclass
class FullObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ScalarCfg(ObsGroup):
        object_target_distances = FullSceneObservation.object_target_distance_obs()

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PositionVectorsCfg(ObsGroup):
        grippers = FullSceneObservation.grippers_positions()
        object_geometry = FullSceneObservation.object_geometry_positions()
        target_geometry = FullSceneObservation.target_geometry()

        def __post_init__(self):
            # self.enable_corruption = True
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class VelocityVectorsCfg(ObsGroup):
        grippers = FullSceneObservation.grippers_velocities()
        grippers_angular = FullSceneObservation.grippers_angular_velocities()
        object_geometry = FullSceneObservation.object_velocities()
        object_geometry_angular = FullSceneObservation.object_angular_velocities()

        def __post_init__(self):
            # self.enable_corruption = True
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class InfoCfg(ObsGroup):
        object_num_points = FullSceneObservation.object_num_points()
        object_geometry_edges = FullSceneObservation.object_geometry_edges()
        object_num_edges = FullSceneObservation.object_num_edges()

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    scalars: ScalarCfg = ScalarCfg()
    position_vectors: PositionVectorsCfg = PositionVectorsCfg()
    velocity_vectors: VelocityVectorsCfg = VelocityVectorsCfg()
    infos: InfoCfg = InfoCfg()


@configclass
class NoObjectVelObservationCfg(FullObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class VelocityVectorsCfg(ObsGroup):
        grippers = FullSceneObservation.grippers_velocities()
        grippers_angular = FullSceneObservation.grippers_angular_velocities()

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    velocity_vectors: VelocityVectorsCfg = VelocityVectorsCfg()


class FullSceneObservationTwoAgents(FullSceneObservation):
    @classmethod
    def grippers_positions(cls):
        return ObsTerm(
            func=mdp.object_positions,
            params={
                "names": [f"cube_{i}" for i in range(2)],
            },
        )

    @classmethod
    def grippers_velocities(cls):
        return ObsTerm(
            func=mdp.object_velocities,
            params={
                "names": [f"cube_{i}" for i in range(2)],
            },
        )

    @classmethod
    def grippers_angular_velocities(cls):
        return ObsTerm(
            func=mdp.object_angular_velocities,
            params={
                "names": [f"cube_{i}" for i in range(2)],
            },
        )

    @classmethod
    def grippers_init_positions(cls):
        return ObsTerm(
            func=mdp.object_init_positions,
            params={
                "names": [f"cube_{i}" for i in range(2)],
            },
        )


@configclass
class FullSceneObservationTwoAgentsCfg(FullObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PositionVectorsCfg(ObsGroup):
        grippers = FullSceneObservationTwoAgents.grippers_positions()
        object_geometry = FullSceneObservationTwoAgents.object_geometry_positions()
        target_geometry = FullSceneObservationTwoAgents.target_geometry()

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class VelocityVectorsCfg(ObsGroup):
        grippers = FullSceneObservationTwoAgents.grippers_velocities()
        # object_geometry = FullSceneObservationTwo.object_velocities()

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    position_vectors: PositionVectorsCfg = PositionVectorsCfg()
    velocity_vectors: VelocityVectorsCfg = VelocityVectorsCfg()
