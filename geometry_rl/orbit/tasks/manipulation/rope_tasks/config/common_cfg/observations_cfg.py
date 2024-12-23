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
from .grippers_cfg import ClosingGrippers


class FullSceneObservation:
    @classmethod
    def grippers_positions(cls):
        return ObsTerm(
            func=mdp.object_positions,
            params={
                "names": [f"cube_{i}" for i in range(ClosingGrippers.N_GRIPPERS)],
            },
        )

    @classmethod
    def grippers_velocities(cls):
        return ObsTerm(
            func=mdp.object_velocities,
            params={
                "names": [f"cube_{i}" for i in range(ClosingGrippers.N_GRIPPERS)],
            },
        )

    @classmethod
    def grippers_init_positions(cls):
        return ObsTerm(
            func=mdp.object_init_positions,
            params={
                "names": [f"cube_{i}" for i in range(ClosingGrippers.N_GRIPPERS)],
            },
        )

    @classmethod
    def target_geometry_positions(cls):
        return ObsTerm(
            func=mdp.target_geometry_positions,
            params={
                "asset_cfg": SceneEntityCfg("target"),
                "aux_data": aux_data,
            },
        )

    @classmethod
    def target_hanger_positions(cls):
        return ObsTerm(
            func=mdp.target_positions,
            params={
                "asset_cfg": SceneEntityCfg("hanger"),
            },
        )

    @classmethod
    def links_positions(cls):
        return ObsTerm(
            func=mdp.links_positions,
            params={
                "asset_cfg": SceneEntityCfg("rope"),
            },
        )

    @classmethod
    def links_velocities(cls):
        return ObsTerm(
            func=mdp.links_velocities,
            params={
                "asset_cfg": SceneEntityCfg("rope"),
            },
        )

    @classmethod
    def rope_target_distances_obs(cls):
        return ObsTerm(
            func=mdp.rope_target_distances_obs,
            params={
                "asset_cfg": SceneEntityCfg("rope"),
                "aux_data": aux_data,
            },
        )


@configclass
class ClosingObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ScalarCfg(ObsGroup):
        rope_target_distances = FullSceneObservation.rope_target_distances_obs()

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PositionVectorsCfg(ObsGroup):
        grippers = FullSceneObservation.grippers_positions()
        links = FullSceneObservation.links_positions()
        target_geometry = FullSceneObservation.target_hanger_positions()

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class VelocityVectorsCfg(ObsGroup):
        grippers = FullSceneObservation.grippers_velocities()
        links = FullSceneObservation.links_velocities()

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    scalars: ScalarCfg = ScalarCfg()
    position_vectors: PositionVectorsCfg = PositionVectorsCfg()
    velocity_vectors: VelocityVectorsCfg = VelocityVectorsCfg()


@configclass
class ShapingObservationsCfg(ClosingObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PositionVectorsCfg(ObsGroup):
        grippers = FullSceneObservation.grippers_positions()
        links = FullSceneObservation.links_positions()
        target_geometry = FullSceneObservation.target_geometry_positions()

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    position_vectors: PositionVectorsCfg = PositionVectorsCfg()
