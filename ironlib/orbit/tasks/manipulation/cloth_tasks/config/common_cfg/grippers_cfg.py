from __future__ import annotations

import enum
import os
import omni.isaac.orbit.sim as sim_utils


class GRIPPER_TYPE(enum.Enum):
    """Gripper type."""

    CUBOID = enum.auto()
    PANDA_USD = enum.auto()


GripperType = GRIPPER_TYPE.CUBOID


class HangingGrippers:
    N_GRIPPERS = 4

    # add cube
    grippers_init_state = [
        (-0.5, 0.0, 5.0),
        (0.5, 0.0, 5.0),
        (-0.5, 0.0, 4.0),
        (0.5, 0.0, 4.0),
    ]

    @classmethod
    def default_cfg(cls, rigid: bool = True, **kwargs):
        if GripperType == GRIPPER_TYPE.PANDA_USD:
            return cls.usd_cfg(rigid=rigid, **kwargs)
        else:
            return cls.cuboid_cfg(rigid=rigid, **kwargs)

    @classmethod
    def default_attachment_path(cls, rigid: bool = True):
        if GripperType == GRIPPER_TYPE.PANDA_USD:
            return "/panda_hand/collisions"
        else:
            return ""

    @classmethod
    def build_physics_props(cls, rigid: bool = True):
        if rigid:
            return {
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                    disable_gravity=True,
                    linear_damping=10.0,  # high damping to reduce oscillations
                    locked_rot_axis=7,
                ),
                "collision_props": sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            }
        else:
            return {
                "collision_props": sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            }

    @classmethod
    def usd_cfg(cls, rigid: bool = True, scale: float = 10.0):
        physics_props = cls.build_physics_props(rigid)
        usd_file = "panda_hand_finger_instanceable_fixed.usda" if not rigid else "panda_hand_finger_instanceable.usda"
        return sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..",
                "..",
                "..",
                "assets",
                "franka",
                usd_file,
            ),
            scale=(scale, scale, scale),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            **physics_props,
        )

    @classmethod
    def cuboid_cfg(cls, rigid: bool = True, size: tuple[float, float, float] = (0.1, 0.1, 0.1)):
        physics_props = cls.build_physics_props(rigid)
        return sim_utils.CuboidCfg(
            size=size,
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
            **physics_props,
        )
