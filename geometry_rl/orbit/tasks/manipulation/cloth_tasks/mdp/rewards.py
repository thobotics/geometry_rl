# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import ClothObject
from omni.isaac.orbit.managers import SceneEntityCfg

from ....common.utils import (
    get_k_boundary_nodes,
    get_k_boundary_nodes_in_multi_asset,
    extract_distance_matrix_from_two_sets,
)
from .observations import (
    hole_boundary_positions,
    points_positions,
    object_geometry,
    cloth_geometry_positions,
)

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


def hole_centroid_hanger_distance(
    env: RLTaskEnv,
    aux_data: AuxiliaryData,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cloth"),
    hanger_cfg: SceneEntityCfg = SceneEntityCfg("hanger"),
    weight_n_last_steps: float = 1.0,
    n_last_steps: int = 0,
    orientation_weight: float = 0.1,
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""

    if aux_data.holes_boundary_nodes_indices is None:
        aux_data.holes_boundary_nodes_indices = get_k_boundary_nodes_in_multi_asset(
            env,
            asset_cfg,
        )
    holes_boundary_nodes_indices = aux_data.holes_boundary_nodes_indices
    points_position = env.scene[asset_cfg.name].data.points_pos_w.clone()
    points_position -= env.scene.env_origins.unsqueeze(1)

    centroids = compute_centroids(points_position, holes_boundary_nodes_indices)
    hanger_pos = env.scene[hanger_cfg.name].data.root_pos_w - env.scene.env_origins

    distance_vector = centroids - hanger_pos
    distance = torch.norm(distance_vector, dim=-1)
    unit_distance_vector = distance_vector / distance.unsqueeze(-1)

    hanger_ori = env.scene[hanger_cfg.name].data.root_quat_w
    local_forward = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0).repeat(env.num_envs, 1).to(hanger_ori.device)
    cos_angle = torch.sum(unit_distance_vector * local_forward, dim=-1).clamp(-1.0, 1.0)
    ori_distance = torch.abs(cos_angle - 1.0)

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


def points_velocity(
    env: RLTaskEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cloth"),
) -> torch.Tensor:
    """Reward the agent for minimizing the velocity of the deformable object."""
    asset: ClothObject = env.scene[asset_cfg.name]
    return asset.data.points_vel_w.norm(dim=-1).mean(dim=-1)


# Calculate the edge lengths given the points and edge connections
def calculate_edge_lengths(points, edges):
    lengths = torch.norm(points[edges[:, 0]] - points[edges[:, 1]], dim=1)
    return lengths


def calculate_areas_vectorized_torch(points, triangles):
    # Extract the points corresponding to each triangle
    p1 = points[triangles[:, 0], :]
    p2 = points[triangles[:, 1], :]
    p3 = points[triangles[:, 2], :]

    # Calculate the vectors for the sides of the triangles
    v1 = p2 - p1
    v2 = p3 - p1

    # Calculate the cross product of the vectors for each triangle (3D cross product)
    cross_product = torch.cross(v1, v2, dim=1)

    # Calculate the area for each triangle
    areas = 0.5 * torch.norm(cross_product, dim=1)

    return areas


def points_distortion(
    env: RLTaskEnv,
    aux_data: AuxiliaryData,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cloth"),
    area_based: bool = False,
    edge_based: bool = True,
) -> torch.Tensor:
    """Reward the agent for minimizing the distortion of the deformable object."""
    asset: ClothObject = env.scene[asset_cfg.name]
    points_positions = asset.data.points_pos_w.clone() - env.scene.env_origins.unsqueeze(1)
    geom_positions = cloth_geometry_positions(env, asset_cfg, aux_data).reshape(env.num_envs, -1, 3)

    # Compute areas
    triangles = aux_data.cloth_triangles
    edges = aux_data.cloth_edges

    area_deviation = []
    edge_length_deviation = []
    for i in range(env.num_envs):

        if edge_based:
            initial_edge_lengths = calculate_edge_lengths(geom_positions[i, :], edges[i])
            deformed_edge_lengths = calculate_edge_lengths(points_positions[i, :], edges[i])

            edge_length_deviation.append(
                torch.abs((deformed_edge_lengths - initial_edge_lengths) / initial_edge_lengths)
            )

        if area_based:
            initial_areas = calculate_areas_vectorized_torch(geom_positions[i, :], triangles[i])
            deformed_areas = calculate_areas_vectorized_torch(points_positions[i, :], triangles[i])

            area_deviation.append(torch.abs((deformed_areas - initial_areas) / initial_areas))

    distance = torch.zeros(env.num_envs, device=points_positions.device)
    if area_based:
        distance += torch.stack(area_deviation).mean(dim=-1)
    if edge_based:
        distance += torch.stack(edge_length_deviation).mean(dim=-1)

    return distance


def points_distortion_obs(
    env: RLTaskEnv,
    aux_data: AuxiliaryData,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cloth"),
    area_based: bool = False,
    edge_based: bool = True,
) -> torch.Tensor:
    asset: ClothObject = env.scene[asset_cfg.name]
    points_positions = asset.data.points_pos_w.clone() - env.scene.env_origins.unsqueeze(1)
    geom_positions = cloth_geometry_positions(env, asset_cfg, aux_data).reshape(env.num_envs, -1, 3)

    # Compute areas
    edges = aux_data.cloth_edges

    edge_length_deviation = []
    for i in range(env.num_envs):

        initial_edge_lengths = calculate_edge_lengths(geom_positions[i, :], edges[i])
        deformed_edge_lengths = calculate_edge_lengths(points_positions[i, :], edges[i])

        edge_length_deviation.append(torch.abs((deformed_edge_lengths - initial_edge_lengths) / initial_edge_lengths))

    distance = torch.stack(edge_length_deviation, dim=0)
    return distance
