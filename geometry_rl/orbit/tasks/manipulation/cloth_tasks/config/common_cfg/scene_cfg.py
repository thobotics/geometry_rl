from __future__ import annotations

from dataclasses import MISSING, dataclass
from typing import List

import os
import random
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.sim.schemas as schemas
from omni.isaac.orbit.assets import (
    AssetBaseCfg,
    ClothObjectCfg,
    RigidObjectCfg,
    BodyAttachmentCfg,
)
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.utils import configclass

from .grippers_cfg import HangingGrippers
from geometry_rl.orbit.tasks.common.sim_utils import MultiAssetCfg, spawn_fixed_number_of_multi_object_sdf

random.seed(2)


##
# Scene definition
##

""" Cloth properties """
num_particles_per_row = 15
cloth_size = (num_particles_per_row, num_particles_per_row)
cloth_holes = [
    (
        num_particles_per_row / 2,
        num_particles_per_row / 2,  # 10
        num_particles_per_row / 15,
    )
]

radius = 0.5 * 1 / (num_particles_per_row + 1)
restOffset = radius
contactOffset = restOffset * 1.5


@configclass
class HangingSceneCfg(InteractiveSceneCfg):
    """Example scene configuration.

    The scene comprises of a light source and a deformable mesh controlled by to two attached cubes.
    """

    hanger = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/hanger",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..",
                "..",
                "..",
                "assets",
                "cylinder.usd",
            ),
            scale=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=True,
                locked_pos_axis=7,
                locked_rot_axis=7,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.294117, 0.270588, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -1.5, 4.5)),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.SphereLightCfg(color=(0.75, 0.75, 0.75), intensity=10000.0, radius=5.0),
    )

    def __post_init__(self):
        """Post initialization."""
        spawn_gripper = HangingGrippers.default_cfg(rigid=True)
        for i in range(HangingGrippers.N_GRIPPERS):
            self.__dict__[f"cube_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"cube_{i}",
                spawn=spawn_gripper,
                init_state=RigidObjectCfg.InitialStateCfg(pos=HangingGrippers.grippers_init_state[i]),
            )

        # This make sure grippers are added before cloth
        self.__dict__["cloth"] = ClothObjectCfg(
            prim_path="{ENV_REGEX_NS}/plain_cloth",
            attachments=[
                BodyAttachmentCfg(
                    prim_path="{ENV_REGEX_NS}/" + f"cube_{i}" + HangingGrippers.default_attachment_path(),
                    name=f"attachment_{i}",
                )
                for i in range(HangingGrippers.N_GRIPPERS)
            ],
            spawn=sim_utils.SquareClothWithHoles(
                size=cloth_size,
                holes=cloth_holes,
                cloth_props=schemas.ClothPropertiesCfg(
                    spring_stretch_stiffness=2e6,
                    spring_bend_stiffness=1.0,
                    spring_shear_stiffness=100.0,
                    spring_damping=0.02,
                    cloth_path="mesh",
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                particle_material=sim_utils.ParticleMaterialCfg(drag=0.1, friction=0.2),
                particle_system_props=schemas.ParticleSystemPropertiesCfg(
                    rest_offset=restOffset,
                    contact_offset=contactOffset,
                    solid_rest_offset=restOffset,
                    fluid_rest_offset=restOffset,
                    particle_contact_offset=contactOffset,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02745, 0.156862, 0.20392), metallic=0.2),
            ),
            init_state=ClothObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 4.5),
                rot=(0.707, 0.707, 0.0, 0.0),
            ),
        )


### Multi-Environment Configuration ###


def split_combinations(combinations, train_size=50, test_size=20, randomize=True, seed=2):
    # Ensure the seed is set for reproducible shuffling
    if seed is not None:
        random.seed(seed)

    # Shuffle the list of combinations
    if randomize:
        random.shuffle(combinations)

    # Split into training and testing sets
    train_set = combinations[:train_size]
    test_set = combinations[train_size : train_size + test_size]

    return train_set, test_set


all_combinations = set()

center_x = round(cloth_holes[0][0])
center_y = round(cloth_holes[0][1])
max_offset = 3
radius = cloth_holes[0][2] * 1.1

num_holes = 40
while len(all_combinations) < num_holes:
    x = random.randint(center_x - max_offset, center_x + max_offset)
    y = random.randint(center_y - max_offset, center_y + max_offset)
    all_combinations.add((x, y, radius))

all_combinations = list(all_combinations)

# TRAIN_SIZE = 7
# TEST_SIZE = 3

TRAIN_SIZE = 20
TEST_SIZE = 20

# Split the combinations into training and testing sets
train_set, test_set = split_combinations(all_combinations, train_size=TRAIN_SIZE, test_size=TEST_SIZE, randomize=False)
geom_set = train_set


class MultiGeometryScene:

    num_geometries = TRAIN_SIZE

    @classmethod
    def cloth(cls):
        mesh_list = []
        for i in range(cls.num_geometries):
            mesh = sim_utils.SquareClothWithHoles(
                size=cloth_size,
                holes=[(geom_set[i][0], geom_set[i][1], geom_set[i][2])],
                cloth_props=schemas.ClothPropertiesCfg(
                    spring_stretch_stiffness=2e6,
                    spring_bend_stiffness=1.0,
                    spring_shear_stiffness=100.0,
                    spring_damping=0.02,
                    cloth_path="mesh",
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                particle_material=sim_utils.ParticleMaterialCfg(drag=0.1, friction=0.2),
                particle_system_props=schemas.ParticleSystemPropertiesCfg(
                    rest_offset=restOffset,
                    contact_offset=contactOffset,
                    solid_rest_offset=restOffset,
                    fluid_rest_offset=restOffset,
                    particle_contact_offset=contactOffset,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02745, 0.156862, 0.20392), metallic=0.2),
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
    def cloth(cls):
        return ClothObjectCfg(
            prim_path="{ENV_REGEX_NS}/plain_cloth",
            attachments=[
                BodyAttachmentCfg(
                    prim_path="{ENV_REGEX_NS}/" + f"cube_{i}" + HangingGrippers.default_attachment_path(),
                    name=f"attachment_{i}",
                )
                for i in range(HangingGrippers.N_GRIPPERS)
            ],
            spawn=MultiAssetCfg(
                func=spawn_fixed_number_of_multi_object_sdf,
                assets_cfg=MultiGeometryScene.cloth(),
            ),
            init_state=MultiAssetInitialStatesCfg(
                pos=[(0.0, 0.0, 4.5) for _ in range(MultiGeometryScene.num_geometries)],
                rot=[
                    (0.707, 0.0, 0.0, 0.707)  # BE CAREFUL THIS IS X,Y,Z,W
                    for _ in range(MultiGeometryScene.num_geometries)
                ],
            ),
        )

    @classmethod
    def cube_params(cls, index: int, rigid: bool = True):
        return {
            "prim_path": "{ENV_REGEX_NS}/" + f"cube_{index}",
            "spawn": MultiAssetCfg(
                func=spawn_fixed_number_of_multi_object_sdf,
                assets_cfg=[HangingGrippers.default_cfg(rigid=True) for _ in range(MultiGeometryScene.num_geometries)],
            ),
            "init_state": MultiAssetInitialStatesCfg(
                pos=[HangingGrippers.grippers_init_state[index] for _ in range(MultiGeometryScene.num_geometries)],
            ),
        }


@configclass
class MultiAssetsClothSceneCfg(HangingSceneCfg):

    def __post_init__(self):
        """Post initialization."""
        for i in range(HangingGrippers.N_GRIPPERS):
            self.__dict__[f"cube_{i}"] = RigidObjectCfg(**MultiAssetsScene.cube_params(i, rigid=True))

        # This make sure grippers are added before cloth
        self.__dict__["cloth"] = MultiAssetsScene.cloth()
