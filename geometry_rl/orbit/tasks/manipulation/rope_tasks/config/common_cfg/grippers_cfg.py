from __future__ import annotations

import enum
import omni.isaac.orbit.sim as sim_utils


class GRIPPER_TYPE(enum.Enum):
    """Gripper type."""

    CUBOID = enum.auto()


GripperType = GRIPPER_TYPE.CUBOID


class ClosingGrippers:

    ROPE_NUM_LINKS = 40
    ROPE_LENGTH = 5.0

    # N_GRIPPERS = 3
    N_GRIPPERS = 2

    # add cube
    grippers_init_state = [
        (0.0, 0.0, 0.1),
        (3.9, 0.0, 0.1),  # =(ROPE_LENGTH / ROPE_NUM_LINKS - radius) * (ROPE_NUM_LINKS - 1)
        # (7.9, 0.0, 0.1),  # =(ROPE_LENGTH / ROPE_NUM_LINKS - radius) * (ROPE_NUM_LINKS - 1)
        # (1.95, 0.0, 0.1),  # =(ROPE_LENGTH / ROPE_NUM_LINKS - radius) * (ROPE_NUM_LINKS - 1) / 2
    ]

    local_pos0 = [
        (0.0, 0.0, 0.0),
        (-0.05, 0.0, 0.0),
        # (0.0, -0.1, 0.0),
    ]

    local_pos1 = [
        (-0.05, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        # (0.0, 0.0, 0.0),
    ]

    # grippers_link_indices = [0, ROPE_NUM_LINKS - 1, ROPE_NUM_LINKS // 2]
    grippers_link_indices = [0, ROPE_NUM_LINKS - 1]

    @classmethod
    def default_cfg(cls, rigid: bool = True, **kwargs):
        return cls.cuboid_cfg(rigid=rigid, **kwargs)

    @classmethod
    def default_attachment_path(cls, rigid: bool = True):
        return "/geometry/mesh"

    @classmethod
    def build_physics_props(cls, rigid: bool = True):
        if rigid:
            return {
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                    disable_gravity=True,
                    linear_damping=10.0,  # high damping to reduce oscillations
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                    locked_pos_axis=4,  # lock z-axis
                    locked_rot_axis=7,
                ),
                "collision_props": sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            }
        else:
            return {
                "collision_props": sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            }

    @classmethod
    def cuboid_cfg(cls, rigid: bool = True, size: tuple[float, float, float] = (0.1, 0.1, 0.1)):
        physics_props = cls.build_physics_props(rigid)
        return sim_utils.CuboidCfg(
            size=size,
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
            **physics_props,
        )


class ShapingGrippers(ClosingGrippers):

    ROPE_NUM_LINKS = 80
    ROPE_LENGTH = 10.0

    # add cube
    grippers_init_state = [
        (0.0, 0.0, 0.1),
        # (3.9, 0.0, 0.1),  # =(ROPE_LENGTH / ROPE_NUM_LINKS - radius) * (ROPE_NUM_LINKS - 1)
        # (1.95, 0.0, 0.1),  # =(ROPE_LENGTH / ROPE_NUM_LINKS - radius) * (ROPE_NUM_LINKS - 1) / 2
        (7.9, 0.0, 0.1),  # =(ROPE_LENGTH / ROPE_NUM_LINKS - radius) * (ROPE_NUM_LINKS - 1)
        # (3.95, 0.0, 0.1),  # =(ROPE_LENGTH / ROPE_NUM_LINKS - radius) * (ROPE_NUM_LINKS - 1) / 2
    ]

    local_pos0 = [
        (0.0, 0.0, 0.0),
        (-0.05, 0.0, 0.0),
        # (0.0, -0.1, 0.0),
    ]

    local_pos1 = [
        (-0.05, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        # (0.0, 0.0, 0.0),
    ]

    # grippers_link_indices = [0, ROPE_NUM_LINKS - 1, ROPE_NUM_LINKS // 2]
    grippers_link_indices = [0, ROPE_NUM_LINKS - 1]
