from __future__ import annotations

import enum
import os
import omni.isaac.orbit.sim as sim_utils


class GRIPPER_TYPE(enum.Enum):
    """Gripper type."""

    CUBOID = enum.auto()
    PANDA_USD = enum.auto()


GripperType = GRIPPER_TYPE.CUBOID


class Grippers:
    N_GRIPPERS = 1

    # add cube
    grippers_init_state = [
        (0.0, 0.0, 0.0),
    ]

    local_pos0 = [
        (0.0, 0.0, -0.25),  # lower-end from the z-scale of the gripper
    ]

    local_pos1 = [
        (0.0, 0.0, 1.0),  # upper-end from the z-scale of the object
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
    def build_physics_props(cls, rigid: bool = True, locked_rot_axis: int = 3, disable_gravity: bool = True):
        if rigid:
            return {
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                    disable_gravity=disable_gravity,
                    linear_damping=10.0,  # high damping to reduce oscillations
                    angular_damping=10.0,
                    max_angular_velocity=1000.0,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                    locked_rot_axis=locked_rot_axis,
                    max_depenetration_velocity=1.0,
                ),
                "collision_props": sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                "mass_props": sim_utils.MassPropertiesCfg(mass=10.0),
            }
        else:
            return {
                "collision_props": sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            }

    @classmethod
    def usd_cfg(cls, rigid: bool = True, scale: float = 10.0, **kwargs):
        physics_props = cls.build_physics_props(rigid, **kwargs)
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
    def cuboid_cfg(cls, rigid: bool = True, size: tuple[float, float, float] = (0.025, 0.025, 0.5), **kwargs):
        physics_props = cls.build_physics_props(rigid, **kwargs)
        return sim_utils.CuboidCfg(
            size=size,
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.0, 0.0)),
            **physics_props,
        )


class PushingGrippers(Grippers):

    grippers_init_state = [(-0.9, 0.0, 1.25) for _ in range(Grippers.N_GRIPPERS)]

    local_pos0 = [
        (-0.25, 0.0, 0.0),  # lower-end from the z-scale of the gripper
    ]

    local_pos1 = [
        (1.0, 0.0, 0.0),  # upper-end from the z-scale of the object
    ]

    @classmethod
    def build_physics_props(cls, rigid: bool = True, locked_rot_axis: int = 3, disable_gravity: bool = True):
        if rigid:
            return {
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                    disable_gravity=disable_gravity,
                    linear_damping=10.0,  # high damping to reduce oscillations
                    angular_damping=10.0,
                    max_angular_velocity=1000.0,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                    locked_rot_axis=locked_rot_axis,
                    max_depenetration_velocity=1.0,
                ),
                "collision_props": sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                "mass_props": sim_utils.MassPropertiesCfg(mass=10.0),
            }
        else:
            return {
                "collision_props": sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            }


class SuctionGrippers(Grippers):

    grippers_init_state = [(0.0, 0.0, 1.0) for _ in range(Grippers.N_GRIPPERS)]

    local_pos0 = [
        (-0.25, 0.0, 0.0),  # lower-end from the z-scale of the gripper
    ]

    local_pos1 = [
        (1.0, 0.0, 0.0),  # upper-end from the z-scale of the object
    ]


class TwoSuctionGrippers(Grippers):

    N_GRIPPERS = 2

    grippers_init_state = [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)]

    local_pos0 = [
        (-0.075, 0.0, 0.0),  # lower-end from the z-scale of the gripper
        (-0.075, 0.0, 0.0),  # lower-end from the z-scale of the gripper
    ]

    local_pos1 = [
        (0.0, 4.0, 0.0),  # upper-end from the z-scale of the object
        (0.0, -4.0, 0.0),  # upper-end from the z-scale of the object
    ]

    @classmethod
    def build_physics_props(cls, rigid: bool = True, **kwargs):
        if rigid:
            return {
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                    # disable_gravity=True,
                    disable_gravity=False,
                    linear_damping=10.0,  # high damping to reduce oscillations
                    max_angular_velocity=1000.0,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                    locked_rot_axis=7,
                ),
                "collision_props": sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                "mass_props": sim_utils.MassPropertiesCfg(mass=1.0),
            }
        else:
            return {
                "collision_props": sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            }
