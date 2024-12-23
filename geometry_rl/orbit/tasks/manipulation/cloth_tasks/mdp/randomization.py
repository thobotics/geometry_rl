from omni.isaac.orbit.assets import Articulation, RigidObject, ClothObject
from omni.isaac.orbit.envs import BaseEnv
from omni.isaac.orbit.managers import RandomizationTermCfg, SceneEntityCfg
from omni.isaac.orbit.managers.manager_base import ManagerTermBase
from omni.isaac.orbit.utils.math import (
    sample_uniform,
)

import torch


def build_rotation_matrix_around_y_torch(angles, use_degrees=False):
    # Convert angles to radians
    angles_rad = torch.deg2rad(angles) if use_degrees else angles
    cos_angles = torch.cos(angles_rad)
    sin_angles = torch.sin(angles_rad)

    # Create rotation matrices for each angle
    zeros = torch.zeros_like(angles)
    ones = torch.ones_like(angles)
    rotation_matrices = torch.stack(
        [
            cos_angles,
            zeros,
            sin_angles,
            zeros,
            ones,
            zeros,
            -sin_angles,
            zeros,
            cos_angles,
        ],
        dim=-1,
    ).view(-1, 3, 3)

    return rotation_matrices


def rotate_positions_around_y_torch(positions, rotation_matrices):
    # Rotate each position
    rotated_positions = torch.bmm(rotation_matrices, positions.unsqueeze(-1)).squeeze(-1)

    return rotated_positions


def reset_cubes_uniform_rotation_around_y(
    env: BaseEnv,
    env_ids: torch.Tensor,
    n_cubes: int,
    angle_range: tuple[float, float],
    velocity_range: dict[str, tuple[float, float]],
    use_degrees: bool = False,
):
    angle = torch.rand(env_ids.shape[0], device=env.device) * (angle_range[1] - angle_range[0]) + angle_range[0]

    rotation_matrices = build_rotation_matrix_around_y_torch(angle, use_degrees)

    # get cloth center
    cloth: ClothObject = env.scene["cloth"]
    cloth_center = cloth.data.default_root_state_w[env_ids, 0:3].clone()

    for i in range(n_cubes):
        cube_cfg = SceneEntityCfg(f"cube_{i}")
        cube: RigidObject = env.scene[cube_cfg.name]
        root_states = cube.data.default_root_state[env_ids].clone()

        rotated_positions = rotate_positions_around_y_torch(root_states[:, 0:3] - cloth_center, rotation_matrices)
        rotated_positions += cloth_center
        rotated_positions += env.scene.env_origins[env_ids]
        orientations = root_states[:, 3:7]

        # velocities
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=cube.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=cube.device)

        velocities = root_states[:, 7:13] + rand_samples

        # set into the physics simulation
        cube.write_root_pose_to_sim(torch.cat([rotated_positions, orientations], dim=-1), env_ids=env_ids)
        cube.write_root_velocity_to_sim(velocities, env_ids=env_ids)
