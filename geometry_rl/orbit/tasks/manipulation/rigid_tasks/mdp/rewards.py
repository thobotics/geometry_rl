# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
import omni.isaac.orbit.utils.math as math_utils
from .observations import object_geometry_positions

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv
    from ..config.common_cfg.aux_data import AuxiliaryData


def compute_centroids(positions, k_neighbors_indices):
    """
    Compute the centroids of the nearest neighbors for each batch.

    Args:
        positions (Tensor): Tensor of shape (batch_size, n_points, 3) containing 3D coordinates of nodes.
        k_neighbors_indices (Tensor): Tensor of shape (batch_size, k) containing indices of the k-nearest neighbors for each batch.

    Returns:
        Tensor: Tensor of shape (batch_size, 3) containing the centroids of the k-neighbors for each batch.
    """
    device = positions.device
    batch_size, k = k_neighbors_indices.shape
    centroids = torch.zeros(batch_size, 3, dtype=torch.float32, device=device)

    for i in range(batch_size):
        # Gather the positions of the k-nearest neighbors for the current batch
        neighbor_positions = positions[i, k_neighbors_indices[i], :]
        # Compute the centroid by averaging along the dimension 0 (k neighbors)
        centroid = torch.mean(neighbor_positions, dim=0)
        centroids[i] = centroid

    return centroids


def object_ee_distance(
    env: RLTaskEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""

    object_pos = env.scene[asset_cfg.name].data.root_pos_w - env.scene.env_origins
    cube_pos = env.scene[cube_cfg.name].data.root_pos_w - env.scene.env_origins

    distance_vector = object_pos - cube_pos
    distance = torch.norm(distance_vector, dim=-1)

    return distance


def object_target_distance(
    env: RLTaskEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    weight_n_last_steps: float = 1.0,
    n_last_steps: int = 0,
    orientation_weight: float = 0.1,
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""

    object_pos = env.scene[asset_cfg.name].data.root_pos_w - env.scene.env_origins
    target_pos = env.scene[target_cfg.name].data.root_pos_w - env.scene.env_origins

    distance_vector = object_pos - target_pos
    distance = torch.norm(distance_vector, dim=-1)

    object_ori = env.scene[asset_cfg.name].data.root_quat_w
    target_ori = env.scene[target_cfg.name].data.root_quat_w
    ori_distance = math_utils.quat_error_magnitude(object_ori, target_ori)

    if n_last_steps > 0:
        # penalize the distance only in the last n_last_steps steps
        distance = torch.where(
            env.episode_length_buf >= env.max_episode_length - n_last_steps,
            weight_n_last_steps * distance,
            distance,
        )
        ori_distance = torch.where(
            env.episode_length_buf >= env.max_episode_length - n_last_steps,
            weight_n_last_steps * ori_distance,
            ori_distance,
        )

    return distance + orientation_weight * ori_distance


def object_insertion(
    env: RLTaskEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    weight_n_last_steps: float = 1.0,
    n_last_steps: int = 0,
    orientation_weight: float = 0.25,
    weight_orientation_n_last_steps: float = 1.0,
    z_weight: float = 1.0,
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""

    object_pos = env.scene[asset_cfg.name].data.root_pos_w - env.scene.env_origins
    target_pos = env.scene[target_cfg.name].data.root_pos_w - env.scene.env_origins

    distance_vector = object_pos - target_pos
    distance = torch.norm(distance_vector, dim=-1)

    # Penalize the distance in the z-direction
    z_distance = torch.abs(distance_vector[:, 2])

    object_ori = env.scene[asset_cfg.name].data.root_quat_w
    target_ori = env.scene[target_cfg.name].data.root_quat_w
    ori_distance = math_utils.quat_error_magnitude(object_ori, target_ori)

    if n_last_steps > 0:
        # penalize the distance only in the last n_last_steps steps
        distance = torch.where(
            env.episode_length_buf >= env.max_episode_length - n_last_steps,
            weight_n_last_steps * distance,
            distance,
        )
        ori_distance = torch.where(
            env.episode_length_buf >= env.max_episode_length - n_last_steps,
            weight_orientation_n_last_steps * ori_distance,
            ori_distance,
        )

    return distance + orientation_weight * ori_distance + z_weight * z_distance


def object_velocity(
    env: RLTaskEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for minimizing the velocity of the deformable object."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w.norm(dim=-1) + asset.data.root_ang_vel_w.norm(dim=-1)
