from omni.isaac.orbit.assets import Articulation, RigidObject, ClothObject
from omni.isaac.orbit.envs import BaseEnv
from omni.isaac.orbit.managers import RandomizationTermCfg, SceneEntityCfg
from omni.isaac.orbit.managers.manager_base import ManagerTermBase
from omni.isaac.orbit.utils.math import (
    sample_uniform,
    quat_from_euler_xyz,
    quat_mul,
    yaw_quat,
)

import torch


def build_rotation_matrix_torch(angles, use_degrees=True):
    # Convert angles to radians
    angles_rad = torch.deg2rad(angles) if use_degrees else angles
    cos_angles = torch.cos(angles_rad)
    sin_angles = torch.sin(angles_rad)

    # Rotation matrices around x, y, and z axes
    rotation_matrix_x = torch.stack(
        [
            torch.ones_like(angles[:, 0]),
            torch.zeros_like(angles[:, 0]),
            torch.zeros_like(angles[:, 0]),
            torch.zeros_like(angles[:, 0]),
            cos_angles[:, 0],
            -sin_angles[:, 0],
            torch.zeros_like(angles[:, 0]),
            sin_angles[:, 0],
            cos_angles[:, 0],
        ],
        dim=-1,
    ).view(-1, 3, 3)

    rotation_matrix_y = torch.stack(
        [
            cos_angles[:, 1],
            torch.zeros_like(angles[:, 1]),
            sin_angles[:, 1],
            torch.zeros_like(angles[:, 1]),
            torch.ones_like(angles[:, 1]),
            torch.zeros_like(angles[:, 1]),
            -sin_angles[:, 1],
            torch.zeros_like(angles[:, 1]),
            cos_angles[:, 1],
        ],
        dim=-1,
    ).view(-1, 3, 3)

    rotation_matrix_z = torch.stack(
        [
            cos_angles[:, 2],
            -sin_angles[:, 2],
            torch.zeros_like(angles[:, 2]),
            sin_angles[:, 2],
            cos_angles[:, 2],
            torch.zeros_like(angles[:, 2]),
            torch.zeros_like(angles[:, 2]),
            torch.zeros_like(angles[:, 2]),
            torch.ones_like(angles[:, 2]),
        ],
        dim=-1,
    ).view(-1, 3, 3)

    # Combine rotation matrices
    rotation_matrices = torch.bmm(rotation_matrix_z, torch.bmm(rotation_matrix_y, rotation_matrix_x))

    return rotation_matrices


def rotate_positions_torch(positions, rotation_matrices):
    rotated_positions = torch.bmm(rotation_matrices, positions.unsqueeze(-1)).squeeze(-1)

    return rotated_positions


def reset_cubes_uniform_around_origin(
    env: BaseEnv,
    env_ids: torch.Tensor,
    n_cubes: int,
    origin_name: str,
    origin_position_range: dict[str, tuple[float, float]],
    angle_ranges: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    use_degrees: bool = False,
):
    # Generate random Euler angles for rotation
    angles = torch.stack(
        [
            torch.rand(env_ids.shape[0], device=env.device) * (angle_ranges["roll"][1] - angle_ranges["roll"][0])
            + angle_ranges["roll"][0],
            torch.rand(env_ids.shape[0], device=env.device) * (angle_ranges["pitch"][1] - angle_ranges["pitch"][0])
            + angle_ranges["pitch"][0],
            torch.rand(env_ids.shape[0], device=env.device) * (angle_ranges["yaw"][1] - angle_ranges["yaw"][0])
            + angle_ranges["yaw"][0],
        ],
        dim=-1,
    )

    rotation_matrices = build_rotation_matrix_torch(angles, use_degrees=use_degrees)

    env_center: torch.Tensor = env.scene.env_origins[env_ids]

    # Get origin center and rotate the cubes around it
    origin_object: RigidObject = env.scene[origin_name]
    origin_root_states = origin_object.data.default_root_state[env_ids].clone()
    origin_center = origin_root_states[:, 0:3]

    # Randomize origin position
    range_list = [origin_position_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=env.device)
    origin_positions = origin_center + env_center + rand_samples

    # Randomize origin orientation
    angles_rad = torch.deg2rad(angles) if use_degrees else angles
    origin_orientations = origin_root_states[:, 3:7]
    addition_orientations = quat_from_euler_xyz(angles_rad[:, 0], angles_rad[:, 1], angles_rad[:, 2])
    new_origin_orientations = quat_mul(origin_orientations, addition_orientations)

    # Set into the physics simulation
    origin_object.write_root_pose_to_sim(
        torch.cat([origin_positions, new_origin_orientations], dim=-1), env_ids=env_ids
    )

    # Rotate cubes
    for i in range(n_cubes):
        cube_cfg = SceneEntityCfg(f"cube_{i}")
        cube: RigidObject = env.scene[cube_cfg.name]
        root_states = cube.data.root_state_w[env_ids].clone()
        root_states[:, 0:3] -= env_center

        rotated_positions = rotate_positions_torch(
            root_states[:, 0:3] - origin_center,
            rotation_matrices,
        )
        rotated_positions += origin_center
        rotated_positions += env_center

        orientations = root_states[:, 3:7]

        # Velocities
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=cube.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=cube.device)

        velocities = root_states[:, 7:13] + rand_samples

        # Set into the physics simulation
        cube.write_root_pose_to_sim(torch.cat([rotated_positions, orientations], dim=-1), env_ids=env_ids)
        cube.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_objects_uniform_around_origin(
    env: BaseEnv,
    env_ids: torch.Tensor,
    angle_ranges: dict[str, tuple[float, float]],
    asset_names: list[str],
    origin_names: list[str],
    use_degrees: bool = False,
    use_default_root_state_for_translation: bool = False,
    use_default_root_state_for_rotation: bool = False,
):
    # Generate random Euler angles for rotation
    angles = torch.stack(
        [
            torch.rand(env_ids.shape[0], device=env.device) * (angle_ranges["roll"][1] - angle_ranges["roll"][0])
            + angle_ranges["roll"][0],
            torch.rand(env_ids.shape[0], device=env.device) * (angle_ranges["pitch"][1] - angle_ranges["pitch"][0])
            + angle_ranges["pitch"][0],
            torch.rand(env_ids.shape[0], device=env.device) * (angle_ranges["yaw"][1] - angle_ranges["yaw"][0])
            + angle_ranges["yaw"][0],
        ],
        dim=-1,
    )

    rotation_matrices = build_rotation_matrix_torch(angles, use_degrees=use_degrees)

    # Get origin center and rotate the cubes around it
    for origin_name in origin_names:
        origin_object = env.scene[origin_name]
        origin_default_root_states = origin_object.data.default_root_state[env_ids].clone()
        origin_root_states = origin_object.data.root_state_w[env_ids].clone()

        if use_default_root_state_for_translation:
            origin_root_states[:, 0:3] = origin_default_root_states[:, 0:3] + env.scene.env_origins[env_ids]

        if use_default_root_state_for_rotation:
            origin_root_states[:, 3:7] = origin_default_root_states[:, 3:7]

        origin_center = origin_root_states[:, 0:3]
        origin_positions = origin_center

        # Randomize origin orientation
        angles_rad = torch.deg2rad(angles) if use_degrees else angles
        origin_orientations = origin_root_states[:, 3:7]
        addition_orientations = quat_from_euler_xyz(angles_rad[:, 0], angles_rad[:, 1], angles_rad[:, 2])
        new_origin_orientations = quat_mul(origin_orientations, addition_orientations)

        # Set into the physics simulation
        origin_object.write_root_pose_to_sim(
            torch.cat([origin_positions, new_origin_orientations], dim=-1),
            env_ids=env_ids,
        )

    # Rotate cubes
    for name in asset_names:
        asset: RigidObject | Articulation = env.scene[name]

        default_root_states = asset.data.default_root_state[env_ids].clone()
        root_states = asset.data.root_state_w[env_ids].clone()

        if use_default_root_state_for_translation:
            root_states[:, 0:3] = default_root_states[:, 0:3] + env.scene.env_origins[env_ids]

        if use_default_root_state_for_rotation:
            root_states[:, 3:7] = default_root_states[:, 3:7]

        rotated_positions = rotate_positions_torch(
            root_states[:, 0:3] - origin_center,
            rotation_matrices,
        )
        rotated_positions += origin_center
        orientations = quat_mul(root_states[:, 3:7], addition_orientations)
        # orientations = root_states[:, 3:7]

        # Set into the physics simulation
        asset.write_root_pose_to_sim(torch.cat([rotated_positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros_like(root_states[:, 7:]), env_ids=env_ids)
