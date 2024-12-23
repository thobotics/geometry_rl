from __future__ import annotations

import os
import random
import omni.isaac.orbit.sim as sim_utils

from dataclasses import MISSING
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.assets import (
    RigidObjectCfg,
    RigidObjectWithAttachmentCfg,
    JointAttachmentCfg,
)
from geometry_rl.orbit.tasks.common.sim_utils import MultiAssetCfg, spawn_fixed_number_of_multi_object_sdf
from .env_cfg import RigidInsertionTwoAgentsEnvCfg
from ..common_cfg.grippers_cfg import TwoSuctionGrippers
from ..common_cfg.scene_cfg import RigidSceneCfg


random.seed(2)

obj_usd_files = [
    "plus_low.usda",
    "pentagon_low.usda",
    "star_low.usda",
    "T_low.usda",
    "diamond_low.usda",
    "heart_low.usda",
    "hexagon_low.usda",
    "triangle_low.usda",
    # "-------------------",
    # "plus_hq.usda",
    # "pentagon_hq.usda",
    # "star_hq.usda",
    # "T_hq.usda",
    # "diamond_hq.usda",
    # "heart_hq.usda",
    # "hexagon_hq.usda",
    # "triangle_hq.usda",
]

target_usd_files = [
    "plus_low_placement.usda",
    "pentagon_low_placement.usda",
    "star_low_placement.usda",
    "T_low_placement.usda",
    "diamond_low_placement.usda",
    "heart_low_placement.usda",
    "hexagon_low_placement.usda",
    "triangle_low_placement.usda",
]


init_pos = [(0.0, 0.0, 1.0)] * len(obj_usd_files)  # Stand placement
target_init_pos = [(-0.5, 0.0, 1.25)] * len(obj_usd_files)  # Stand placement
target_scales = [(0.025, 0.0125, 0.0125)] * len(obj_usd_files)  # Stand placement
colors = [(0.02745, 0.156862, 0.20392)] * len(obj_usd_files)
scales = [(0.0125, 0.0125, 0.0125)] * len(obj_usd_files)


class MultiGeometryScene:

    num_geometries = len(obj_usd_files)

    @classmethod
    def rigid_object(cls):
        mesh_list = []
        for i in range(cls.num_geometries):
            mesh = sim_utils.UsdFileCfg(
                usd_path=os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "..",
                    "..",
                    "..",
                    "assets",
                    "insertion_kitting",
                    obj_usd_files[i],
                ),
                scale=scales[i],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                    linear_damping=10.0,  # high damping to reduce oscillations
                    angular_damping=10.0,  # high damping to reduce oscillations
                    max_angular_velocity=1000.0,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                    # locked_rot_axis=3,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colors[i]),
            )
            mesh_list.append(mesh)
        return mesh_list

    @classmethod
    def target_object(cls):
        """Virtual target object for the insertion task. Note that the collision is disabled."""
        mesh_list = []
        for i in range(cls.num_geometries):
            mesh = sim_utils.UsdFileCfg(
                usd_path=os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "..",
                    "..",
                    "..",
                    "assets",
                    "insertion_kitting",
                    obj_usd_files[i],
                ),
                scale=scales[i],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                    disable_gravity=True,
                    locked_pos_axis=7,
                    locked_rot_axis=7,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.294117, 0.270588, 0.0), opacity=1.0),
            )
            mesh_list.append(mesh)
        return mesh_list

    @classmethod
    def target_hole_object(cls):
        """Real hole object for the insertion task"""
        mesh_list = []
        for i in range(cls.num_geometries):
            mesh = sim_utils.UsdFileCfg(
                usd_path=os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "..",
                    "..",
                    "..",
                    "assets",
                    "insertion_kitting",
                    target_usd_files[i],
                ),
                scale=target_scales[i],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                    disable_gravity=True,
                    locked_pos_axis=7,
                    locked_rot_axis=7,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.043137, 0.019607, 0.0), opacity=1.0),
            )
            mesh_list.append(mesh)
        return mesh_list


@configclass
class MultiAssetInitialStatesCfg:
    """Configuration for initial states of multiple assets."""

    pos: list[tuple[float, float, float]] = MISSING
    """Initial positions of the assets."""

    rot: list[tuple[float, float, float, float]] = [
        (1.0, 0.0, 0.0, 0.0) for _ in range(MultiGeometryScene.num_geometries)
    ]
    """Initial rotations of the assets."""

    lin_vel: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0) for _ in range(MultiGeometryScene.num_geometries)]
    """Initial linear velocities of the assets."""

    ang_vel: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0) for _ in range(MultiGeometryScene.num_geometries)]
    """Initial angular velocities of the assets."""


class MultiAssetsScene:
    """Example scene configuration.

    The scene comprises of a light source and a deformable mesh controlled by to two attached cubes.
    """

    @classmethod
    def rigid_object(cls):
        return RigidObjectWithAttachmentCfg(
            prim_path="{ENV_REGEX_NS}/object",
            attachments=[
                JointAttachmentCfg(
                    joint_type="fixed",
                    prim_path="{ENV_REGEX_NS}/" + f"cube_{i}" + TwoSuctionGrippers.default_attachment_path(),
                    attached_link_idx=None,
                    local_pos0=TwoSuctionGrippers.local_pos0[i],
                    local_pos1=TwoSuctionGrippers.local_pos1[i],
                    name=f"attachment_{i}",
                )
                for i in range(TwoSuctionGrippers.N_GRIPPERS)
            ],
            spawn=MultiAssetCfg(
                func=spawn_fixed_number_of_multi_object_sdf,
                assets_cfg=MultiGeometryScene.rigid_object(),
            ),
            init_state=MultiAssetInitialStatesCfg(
                pos=init_pos,
            ),
        )

    @classmethod
    def target_object(cls):
        return RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/target",
            spawn=MultiAssetCfg(
                func=spawn_fixed_number_of_multi_object_sdf,
                assets_cfg=MultiGeometryScene.target_object(),
            ),
            init_state=MultiAssetInitialStatesCfg(
                pos=target_init_pos,
            ),
        )

    @classmethod
    def target_hole_object(cls):
        return RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/target_hole",
            spawn=MultiAssetCfg(
                func=spawn_fixed_number_of_multi_object_sdf,
                assets_cfg=MultiGeometryScene.target_hole_object(),
            ),
            init_state=MultiAssetInitialStatesCfg(
                pos=target_init_pos,
            ),
        )

    @classmethod
    def cube_params(cls, index: int, rigid: bool = True):
        return {
            "prim_path": "{ENV_REGEX_NS}/" + f"cube_{index}",
            "spawn": MultiAssetCfg(
                func=spawn_fixed_number_of_multi_object_sdf,
                assets_cfg=[
                    TwoSuctionGrippers.default_cfg(rigid=True, size=(0.025, 0.025, 0.025))
                    for _ in range(MultiGeometryScene.num_geometries)
                ],
            ),
            "init_state": MultiAssetInitialStatesCfg(
                pos=[TwoSuctionGrippers.grippers_init_state[index] for _ in range(MultiGeometryScene.num_geometries)],
            ),
        }


@configclass
class MultiAssetsRigidSceneCfg(RigidSceneCfg):

    def __post_init__(self):
        """Post initialization."""
        for i in range(TwoSuctionGrippers.N_GRIPPERS):
            self.__dict__[f"cube_{i}"] = RigidObjectCfg(**MultiAssetsScene.cube_params(i, rigid=True))

        # This make sure grippers are added before cloth
        self.__dict__["object"] = MultiAssetsScene.rigid_object()
        self.__dict__["target"] = MultiAssetsScene.target_object()
        self.__dict__["target_hole"] = MultiAssetsScene.target_hole_object()


@configclass
class RigidInsertionTwoAgentsMultiEnvCfg(RigidInsertionTwoAgentsEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: MultiAssetsRigidSceneCfg = MultiAssetsRigidSceneCfg(num_envs=40, env_spacing=4.0, replicate_physics=False)

    def __post_init__(self):
        """Post initialization."""

        super().__post_init__()

        # general settings
        self.decimation = 4
        self.warmup_steps = 0
        self.episode_length_s = 4.0  # = decimation * dt * 100 steps
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
