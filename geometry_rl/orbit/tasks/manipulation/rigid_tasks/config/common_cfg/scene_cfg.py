from __future__ import annotations

from dataclasses import MISSING, dataclass
from typing import List

import os
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import (
    AssetBaseCfg,
    RigidObjectCfg,
    RigidObjectWithAttachmentCfg,
    JointAttachmentCfg,
)
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.utils import configclass

from .grippers_cfg import Grippers


@configclass
class RigidSceneCfg(InteractiveSceneCfg):
    """Example scene configuration.

    The scene comprises of a light source and a deformable mesh controlled by to two attached cubes.
    """

    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/target",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..",
                "..",
                "..",
                "assets",
                "L_3.usda",
            ),
            scale=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=True,
                locked_pos_axis=7,
                locked_rot_axis=7,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.0), opacity=0.05),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(1e4, 1e4), color=(0.0, 0.0, 0.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.SphereLightCfg(color=(0.75, 0.75, 0.75), intensity=10000.0, radius=5.0),
    )

    def __post_init__(self):
        """Post initialization."""
        spawn_gripper = Grippers.default_cfg(rigid=True)
        for i in range(Grippers.N_GRIPPERS):
            self.__dict__[f"cube_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"cube_{i}",
                spawn=spawn_gripper,
                init_state=RigidObjectCfg.InitialStateCfg(pos=Grippers.grippers_init_state[i]),
            )

        # This make sure grippers are added before cloth
        self.__dict__["object"] = RigidObjectWithAttachmentCfg(
            prim_path="{ENV_REGEX_NS}/object",
            attachments=[
                JointAttachmentCfg(
                    joint_type="fixed",
                    prim_path="{ENV_REGEX_NS}/" + f"cube_{i}" + Grippers.default_attachment_path(),
                    attached_link_idx=None,
                    local_pos0=Grippers.local_pos0[i],
                    local_pos1=Grippers.local_pos1[i],
                    name=f"attachment_{i}",
                )
                for i in range(Grippers.N_GRIPPERS)
            ],
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "..",
                    "..",
                    "..",
                    "assets",
                    "L_3.usda",
                ),
                scale=(0.05, 0.05, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                    linear_damping=10.0,  # high damping to reduce oscillations
                    angular_damping=10.0,
                    max_angular_velocity=1000.0,
                    locked_pos_axis=4,
                    locked_rot_axis=3,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.3)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )
