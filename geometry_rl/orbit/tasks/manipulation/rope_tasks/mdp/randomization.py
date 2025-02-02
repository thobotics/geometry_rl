from omni.isaac.orbit.assets import Articulation, RigidObject, Rope
from omni.isaac.orbit.envs import BaseEnv
from omni.isaac.orbit.managers import RandomizationTermCfg, SceneEntityCfg
from omni.isaac.orbit.managers.manager_base import ManagerTermBase
from omni.isaac.orbit.utils.math import sample_uniform, quat_from_euler_xyz, quat_mul
from omni.isaac.orbit.utils.math import (
    sample_uniform,
)

import torch


def build_rotation_matrix_around_z_torch(angles, use_degrees=False):
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
            -sin_angles,
            zeros,
            sin_angles,
            cos_angles,
            zeros,
            zeros,
            zeros,
            ones,
        ],
        dim=-1,
    ).view(-1, 3, 3)

    return rotation_matrices


def rotate_positions_around_z_torch(positions, rotation_matrices):
    # Rotate each position
    rotated_positions = torch.bmm(rotation_matrices, positions.unsqueeze(-1)).squeeze(-1)

    return rotated_positions


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


def reset_cubes_uniform_rotation_around_z(
    env: BaseEnv,
    env_ids: torch.Tensor,
    n_cubes: int,
    angle_range: tuple[float, float],
    velocity_range: dict[str, tuple[float, float]],
    use_degrees: bool = False,
    n_repeat: int = 5,
):
    angle = torch.rand(env_ids.shape[0], device=env.device) * (angle_range[1] - angle_range[0]) + angle_range[0]

    rotation_matrices = build_rotation_matrix_around_z_torch(angle, use_degrees=use_degrees)

    # get cloth center
    rope: Rope = env.scene["rope"]
    rope_link_pos = rope.data.default_link_pos[env_ids].clone() - env.scene.env_origins[env_ids].unsqueeze(1)
    rope_center = rope_link_pos.mean(dim=1)

    # rotate cubes
    for i in range(n_cubes):
        cube_cfg = SceneEntityCfg(f"cube_{i}")
        cube: RigidObject = env.scene[cube_cfg.name]
        root_states = cube.data.default_root_state[env_ids].clone()

        rotated_positions = rotate_positions_around_z_torch(root_states[:, 0:3] - rope_center, rotation_matrices)
        rotated_positions += rope_center
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

    # rotate rope
    num_links = rope_link_pos.shape[1]
    rope_center_repeat = rope_center.unsqueeze(1).repeat(1, num_links, 1)
    env_origin_repeat = env.scene.env_origins[env_ids].unsqueeze(1).repeat(1, num_links, 1)

    rope_difference = rope_link_pos.view(-1, 3) - rope_center_repeat.view(-1, 3)
    rotated_positions = rotate_positions_around_z_torch(
        rope_difference,
        rotation_matrices.unsqueeze(1).repeat(1, num_links, 1, 1).reshape(-1, 3, 3),
    )
    rotated_positions += rope_center_repeat.reshape(-1, 3)
    rotated_positions += env_origin_repeat.reshape(-1, 3)

    rotated_link_state = torch.cat(
        [rotated_positions, rope.data.default_link_rot[env_ids].clone().reshape(-1, 4)],
        dim=-1,
    )
    for _ in range(n_repeat):
        rope.write_link_pose_to_sim(rotated_link_state, env_ids=env_ids)
        rope.write_link_velocity_to_sim(
            torch.zeros(rotated_positions.shape[0], 6, device=env.device),
            env_ids=env_ids,
        )


def reset_cubes_uniform_around_origin(
    env: BaseEnv,
    env_ids: torch.Tensor,
    n_cubes: int,
    origin_name: str,
    origin_position_range: dict[str, tuple[float, float]],
    angle_ranges: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    use_degrees: bool = False,
    n_repeat: int = 5,
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
    # Note: We don't rotate the origin object, only translate it
    origin_object.write_root_pose_to_sim(torch.cat([origin_positions, origin_orientations], dim=-1), env_ids=env_ids)

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

    rope: Rope = env.scene["rope"]
    rope_link_pos = rope.data.default_link_pos[env_ids].clone() - env.scene.env_origins[env_ids].unsqueeze(1)

    # rotate rope
    num_links = rope_link_pos.shape[1]
    origin_center_repeat = origin_center.unsqueeze(1).repeat(1, num_links, 1)
    env_origin_repeat = env.scene.env_origins[env_ids].unsqueeze(1).repeat(1, num_links, 1)

    rope_difference = rope_link_pos.view(-1, 3) - origin_center_repeat.view(-1, 3)
    rotated_positions = rotate_positions_around_z_torch(
        rope_difference,
        rotation_matrices.unsqueeze(1).repeat(1, num_links, 1, 1).reshape(-1, 3, 3),
    )
    rotated_positions += origin_center_repeat.reshape(-1, 3)
    rotated_positions += env_origin_repeat.reshape(-1, 3)

    rotated_link_state = torch.cat(
        [rotated_positions, rope.data.default_link_rot[env_ids].clone().reshape(-1, 4)],
        dim=-1,
    )
    for _ in range(n_repeat):
        rope.write_link_pose_to_sim(rotated_link_state, env_ids=env_ids)
        rope.write_link_velocity_to_sim(
            torch.zeros(rotated_positions.shape[0], 6, device=env.device),
            env_ids=env_ids,
        )


def reset_cubes_uniform_rotation_around_z_with_target_shape(
    env: BaseEnv,
    env_ids: torch.Tensor,
    n_cubes: int,
    angle_range: tuple[float, float] | tuple[float, float, float, float],
    velocity_range: dict[str, tuple[float, float]],
    target_cfg: SceneEntityCfg = None,
    target_angle_range: tuple[float, float] | tuple[float, float, float, float] = (
        -0.0,
        0.0,
    ),
    use_degrees: bool = False,
    n_repeat: int = 5,
):
    # rotate target
    if target_cfg is not None:
        if len(target_angle_range) == 2:
            target_angle = (
                torch.rand(env_ids.shape[0], device=env.device) * (target_angle_range[1] - target_angle_range[0])
                + target_angle_range[0]
            )
        else:
            first_range = target_angle_range[0:2]
            second_range = target_angle_range[2:4]

            # Generate random boolean to decide which range to sample from for each element
            choose_second_range = torch.rand(env_ids.shape[0], device=env.device) > 0.5

            # Sample from the first or second range
            target_angle = torch.where(
                choose_second_range,
                torch.rand(env_ids.shape[0], device=env.device) * (second_range[1] - second_range[0]) + second_range[0],
                torch.rand(env_ids.shape[0], device=env.device) * (first_range[1] - first_range[0]) + first_range[0],
            )
        target: RigidObject = env.scene[target_cfg.name]
        root_states = target.data.default_root_state[env_ids].clone()
        root_states[:, 0:3] += env.scene.env_origins[env_ids]

        angles_rad = torch.deg2rad(target_angle) if use_degrees else target_angle
        origin_orientations = root_states[:, 3:7]
        addition_orientations = quat_from_euler_xyz(
            torch.zeros_like(angles_rad), torch.zeros_like(angles_rad), angles_rad
        )
        new_origin_orientations = quat_mul(origin_orientations, addition_orientations)

        # set into the physics simulation
        target.write_root_pose_to_sim(
            torch.cat([root_states[:, 0:3], new_origin_orientations], dim=-1),
            env_ids=env_ids,
        )

    if len(angle_range) == 2:
        angle = torch.rand(env_ids.shape[0], device=env.device) * (angle_range[1] - angle_range[0]) + angle_range[0]
    else:
        first_range = angle_range[0:2]
        second_range = angle_range[2:4]

        # Generate random boolean to decide which range to sample from for each element
        choose_second_range = torch.rand(env_ids.shape[0], device=env.device) > 0.5

        # Sample from the first or second range
        angle = torch.where(
            choose_second_range,
            torch.rand(env_ids.shape[0], device=env.device) * (second_range[1] - second_range[0]) + second_range[0],
            torch.rand(env_ids.shape[0], device=env.device) * (first_range[1] - first_range[0]) + first_range[0],
        )

    angle += target_angle if target_cfg is not None else 0.0

    rotation_matrices = build_rotation_matrix_around_z_torch(angle, use_degrees=use_degrees)

    # get rope center
    rope: Rope = env.scene["rope"]
    rope_link_pos = rope.data.default_link_pos[env_ids].clone() - env.scene.env_origins[env_ids].unsqueeze(1)
    rope_center = rope_link_pos.mean(dim=1)

    # rotate cubes
    for i in range(n_cubes):
        cube_cfg = SceneEntityCfg(f"cube_{i}")
        cube: RigidObject = env.scene[cube_cfg.name]
        root_states = cube.data.default_root_state[env_ids].clone()

        rotated_positions = rotate_positions_around_z_torch(root_states[:, 0:3] - rope_center, rotation_matrices)
        rotated_positions += rope_center
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

    # rotate rope
    num_links = rope_link_pos.shape[1]
    rope_center_repeat = rope_center.unsqueeze(1).repeat(1, num_links, 1)
    env_origin_repeat = env.scene.env_origins[env_ids].unsqueeze(1).repeat(1, num_links, 1)

    rope_difference = rope_link_pos.view(-1, 3) - rope_center_repeat.view(-1, 3)
    rotated_positions = rotate_positions_around_z_torch(
        rope_difference,
        rotation_matrices.unsqueeze(1).repeat(1, num_links, 1, 1).reshape(-1, 3, 3),
    )
    rotated_positions += rope_center_repeat.reshape(-1, 3)
    rotated_positions += env_origin_repeat.reshape(-1, 3)

    rotated_link_state = torch.cat(
        [rotated_positions, rope.data.default_link_rot[env_ids].clone().reshape(-1, 4)],
        dim=-1,
    )
    for _ in range(n_repeat):
        rope.write_link_pose_to_sim(rotated_link_state, env_ids=env_ids)
        rope.write_link_velocity_to_sim(
            torch.zeros(rotated_positions.shape[0], 6, device=env.device),
            env_ids=env_ids,
        )
