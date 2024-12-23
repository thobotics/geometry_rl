from omni.isaac.orbit.assets import Articulation, RigidObject, ClothObject
from omni.isaac.orbit.envs import BaseEnv
from omni.isaac.orbit.managers import RandomizationTermCfg, SceneEntityCfg
from omni.isaac.orbit.managers.manager_base import ManagerTermBase
from omni.isaac.orbit.utils.math import (
    quat_mul,
    sample_uniform,
    quat_from_euler_xyz,
)


import torch


def build_rotation_matrix_around_y_torch(angles):
    # Convert angles to radians
    angles_rad = torch.deg2rad(angles)
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
):
    angle = torch.rand(env_ids.shape[0], device=env.device) * (angle_range[1] - angle_range[0]) + angle_range[0]

    rotation_matrices = build_rotation_matrix_around_y_torch(angle)

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


def reset_joint_root_state_uniform(
    env: BaseEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]] | dict[str, tuple[float, float, float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_names: list[str],
    use_default_root_state_for_translation: bool = True,
    use_default_root_state_for_rotation: bool = True,
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """

    device = env.device

    # Check if pose_range contains tuples of length 4 for sampling from two different ranges
    if all(len(pose_range.get(key, (0.0, 0.0, 0.0, 0.0))) == 4 for key in ["x", "y", "z", "roll", "pitch", "yaw"]):
        pos_rand_samples = torch.empty((len(env_ids), 6), device=device)
        for i, key in enumerate(["x", "y", "z", "roll", "pitch", "yaw"]):
            min1, max1, min2, max2 = pose_range.get(key, (0.0, 0.0, 0.0, 0.0))
            rand_choice = torch.randint(0, 2, (len(env_ids),), device=device)
            pos_rand_samples[:, i] = torch.where(
                rand_choice == 0,
                sample_uniform(min1, max1, (len(env_ids),), device=device),
                sample_uniform(min2, max2, (len(env_ids),), device=device),
            )
    else:
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=device)
        pos_rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=device)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=device)
    vel_rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=device)

    for name in asset_names:
        # extract the used quantities (to enable type-hinting)
        asset: RigidObject | Articulation = env.scene[name]
        # get default root state
        default_root_states = asset.data.default_root_state[env_ids].clone()
        root_states = asset.data.root_state_w[env_ids].clone()

        if use_default_root_state_for_translation:
            root_states[:, 0:3] = default_root_states[:, 0:3] + env.scene.env_origins[env_ids]
        if use_default_root_state_for_rotation:
            root_states[:, 3:7] = default_root_states[:, 3:7]

        positions = root_states[:, 0:3] + pos_rand_samples[:, 0:3]
        orientations = root_states[:, 3:7]
        addition_orientations = quat_from_euler_xyz(
            pos_rand_samples[:, 3], pos_rand_samples[:, 4], pos_rand_samples[:, 5]
        )
        new_orientations = quat_mul(orientations, addition_orientations)

        velocities = root_states[:, 7:13] + vel_rand_samples

        # set into the physics simulation
        asset.write_root_pose_to_sim(torch.cat([positions, new_orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
