from __future__ import annotations

from dataclasses import MISSING, dataclass
from typing import List

import os
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import (
    AssetBaseCfg,
    RigidObjectCfg,
    RopeCfg,
    JointAttachmentCfg,
)
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.utils import configclass

from .grippers_cfg import ClosingGrippers, ShapingGrippers


##
# Scene definition
##


@configclass
class ClosingSceneCfg(InteractiveSceneCfg):
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
            # scale=(0.5, 0.5, 0.5),
            scale=(0.8, 0.8, 0.8),
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
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.0, 3.0, 0.0),
            rot=(0.707, 0.707, 0.0, 0.0),
        ),
    )

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(1e4, 1e4), color=(0.0, 0.0, 0.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5)),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.SphereLightCfg(color=(0.75, 0.75, 0.75), intensity=10000.0, radius=5.0),
    )

    def __post_init__(self):
        """Post initialization."""
        spawn_gripper = ClosingGrippers.default_cfg(rigid=True)
        for i in range(ClosingGrippers.N_GRIPPERS):
            self.__dict__[f"cube_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"cube_{i}",
                spawn=spawn_gripper,
                init_state=RigidObjectCfg.InitialStateCfg(pos=ClosingGrippers.grippers_init_state[i]),
            )

        # This make sure grippers are added before cloth
        self.__dict__["rope"] = RopeCfg(
            prim_path="{ENV_REGEX_NS}/rope",
            attachments=[
                JointAttachmentCfg(
                    joint_type="fixed",
                    prim_path="{ENV_REGEX_NS}/" + f"cube_{i}" + ClosingGrippers.default_attachment_path(),
                    attached_link_idx=ClosingGrippers.grippers_link_indices[i],
                    local_pos0=ClosingGrippers.local_pos0[i],
                    local_pos1=ClosingGrippers.local_pos1[i],
                    name=f"attachment_{i}",
                )
                for i in range(ClosingGrippers.N_GRIPPERS)
            ],
            spawn=sim_utils.RopeShapeCfg(
                num_links=ClosingGrippers.ROPE_NUM_LINKS,
                length=ClosingGrippers.ROPE_LENGTH,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02745, 0.156862, 0.20392), metallic=0.2),
            ),
            init_state=RopeCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )


@configclass
class ShapingSceneCfg(InteractiveSceneCfg):
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
                "cylinder.usd",
            ),
            scale=(0.4, 0.4, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=True,
                locked_pos_axis=7,
                locked_rot_axis=7,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.294117, 0.270588, 0.0), opacity=0.25),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0, 0.0, -1.0)),
    )

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(1e4, 1e4), color=(0.0, 0.0, 0.0), visible=False),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5)),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0, exposure=3.0),
    )

    def __post_init__(self):
        """Post initialization."""
        spawn_gripper = ShapingGrippers.default_cfg(rigid=True)
        for i in range(ShapingGrippers.N_GRIPPERS):
            self.__dict__[f"cube_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"cube_{i}",
                spawn=spawn_gripper,
                init_state=RigidObjectCfg.InitialStateCfg(pos=ShapingGrippers.grippers_init_state[i]),
            )

        # This make sure grippers are added before cloth
        self.__dict__["rope"] = RopeCfg(
            prim_path="{ENV_REGEX_NS}/rope",
            attachments=[
                JointAttachmentCfg(
                    joint_type="fixed",
                    prim_path="{ENV_REGEX_NS}/" + f"cube_{i}" + ShapingGrippers.default_attachment_path(),
                    attached_link_idx=ShapingGrippers.grippers_link_indices[i],
                    local_pos0=ShapingGrippers.local_pos0[i],
                    local_pos1=ShapingGrippers.local_pos1[i],
                    name=f"attachment_{i}",
                )
                for i in range(ShapingGrippers.N_GRIPPERS)
            ],
            spawn=sim_utils.RopeShapeCfg(
                num_links=ShapingGrippers.ROPE_NUM_LINKS,
                length=ShapingGrippers.ROPE_LENGTH,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02745, 0.156862, 0.20392), metallic=0.2),
            ),
            init_state=RopeCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )
