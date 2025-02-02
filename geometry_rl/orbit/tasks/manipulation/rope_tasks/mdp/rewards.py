# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Rope, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg

from .observations import (
    generate_positions_UV,
)

from omni.isaac.orbit.utils.math import quat_apply_yaw


if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv
    from ..config.common_cfg.aux_data import AuxiliaryData


def points_inside_polygon_batch(polygon, points):
    """
    Batched version of point in polygon algorithm (Jordan theorem).

    :param polygon: Tensor of shape [batch_size, num_vertices, 3]
    :param points: Tensor of shape [batch_size, 3]
    :return: Tensor of shape [batch_size, 1] with boolean values indicating if points are inside the polygon
    """
    batch_size = polygon.size(0)
    num_vertices = polygon.size(1)

    x = points[:, 0]
    y = points[:, 1]

    inside = torch.zeros(batch_size, dtype=torch.bool, device=polygon.device)

    p1x = polygon[:, 0, 0]
    p1y = polygon[:, 0, 1]
    for i in range(num_vertices + 1):
        p2x = polygon[:, i % num_vertices, 0]
        p2y = polygon[:, i % num_vertices, 1]
        condition1 = y > torch.min(p1y, p2y)
        condition2 = y <= torch.max(p1y, p2y)
        condition3 = x <= torch.max(p1x, p2x)
        condition4 = p1y != p2y
        x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
        condition5 = (p1x == p2x) | (x <= x_inters)
        toggle = condition1 & condition2 & condition3 & condition4 & condition5
        inside = inside ^ toggle
        p1x, p1y = p2x, p2y

    centers = torch.zeros(batch_size, 3, device=polygon.device)
    centers[:, 0] = polygon[:, :, 0].mean(dim=1)
    centers[:, 1] = polygon[:, :, 1].mean(dim=1)
    centers[:, 2] = polygon[:, :, 2].mean(dim=1)

    return inside, centers


def rope_wrapping(
    env: RLTaskEnv,
    asset_cfg: SceneEntityCfg,
    hanger_cfg: SceneEntityCfg,
) -> torch.Tensor:

    # target = aux_data.target_geometry_positions
    asset: Rope = env.scene[asset_cfg.name]
    points_pos = asset.data.link_pos_w.clone()
    points_pos = points_pos.reshape(env.num_envs, -1, 3)
    points_pos -= env.scene.env_origins.unsqueeze(1)

    hanger_pos = env.scene[hanger_cfg.name].data.root_pos_w - env.scene.env_origins
    inside, centers = points_inside_polygon_batch(points_pos, hanger_pos)

    distence_centers_and_hanger = torch.norm(hanger_pos[..., :2] - centers[..., :2], dim=-1)

    return distence_centers_and_hanger


def rope_combined(
    env: RLTaskEnv,
    asset_cfg: SceneEntityCfg,
    hanger_cfg: SceneEntityCfg,
    cube_0_cfg: SceneEntityCfg,
    cube_1_cfg: SceneEntityCfg,
    distance_threshold: float = 0.01,
    gripper_distance_threshold: float = 0.02,
    weight_gripper_close: float = 5.0,
) -> torch.Tensor:

    asset: Rope = env.scene[asset_cfg.name]
    points_pos = asset.data.link_pos_w.clone()
    points_pos = points_pos.reshape(env.num_envs, -1, 3)
    points_pos -= env.scene.env_origins.unsqueeze(1)

    hanger_pos = env.scene[hanger_cfg.name].data.root_pos_w - env.scene.env_origins
    inside, centers = points_inside_polygon_batch(points_pos, hanger_pos)

    distance_centers_and_hanger = torch.norm(hanger_pos[..., :2] - centers[..., :2], dim=-1)

    cube_0_pos = env.scene[cube_0_cfg.name].data.root_pos_w - env.scene.env_origins
    cube_1_pos = env.scene[cube_1_cfg.name].data.root_pos_w - env.scene.env_origins

    grippers_distance = torch.norm(cube_0_pos[..., :2] - cube_1_pos[..., :2], dim=-1)

    rewards = torch.where(
        distance_centers_and_hanger <= distance_threshold,
        distance_centers_and_hanger + (grippers_distance * weight_gripper_close),
        distance_centers_and_hanger,
    )

    return rewards


def rope_closing(
    env: RLTaskEnv,
    cube_0_cfg: SceneEntityCfg,
    cube_1_cfg: SceneEntityCfg,
    n_last_steps: int = 0,
) -> torch.Tensor:
    cube_0_pos = env.scene[cube_0_cfg.name].data.root_pos_w - env.scene.env_origins
    cube_1_pos = env.scene[cube_1_cfg.name].data.root_pos_w - env.scene.env_origins

    grippers_distance = torch.norm(cube_0_pos[..., :2] - cube_1_pos[..., :2], dim=-1)

    if n_last_steps > 0:
        # penalize the distance only in the last n_last_steps steps
        grippers_distance = torch.where(
            env.episode_length_buf >= env.max_episode_length - n_last_steps,
            grippers_distance,
            grippers_distance * 0.0,
        )

    return grippers_distance


def shape_descriptor(positions):
    # positions is of shape [N, 3] where N is the number of points
    vectors = positions[1:] - positions[:-1]  # Compute vectors between adjacent points
    norms = torch.norm(vectors, dim=1, keepdim=True)  # Compute norms of vectors

    # Normalize vectors
    unit_vectors = vectors / (norms + 1e-6)

    # Compute angles between consecutive vectors
    cos_angles = (unit_vectors[:-1] * unit_vectors[1:]).sum(dim=1)
    cos_angles = torch.clamp(cos_angles, -1.0, 1.0)  # Ensure values are within valid range for arccos
    angles_between_segments = torch.acos(cos_angles)

    # Compute global direction vector (from first to last point)
    global_direction_vector = positions[-1] - positions[0]
    global_direction_norm = torch.norm(global_direction_vector, keepdim=True)
    global_unit_vector = global_direction_vector / global_direction_norm

    # Compute angles between each segment vector and global direction vector
    cos_global_angles = (unit_vectors * global_unit_vector).sum(dim=1)
    cos_global_angles = torch.clamp(cos_global_angles, -1.0, 1.0)
    angles_with_global_direction = torch.acos(cos_global_angles)

    # Compute relative positions (distances from the midpoint)
    midpoint = (positions[0] + positions[-1]) / 2
    relative_vectors = positions - midpoint
    relative_distances = torch.norm(relative_vectors, dim=1)

    # Concatenate all features to form the shape descriptor
    relative_vectors = relative_vectors.flatten()
    shape_descriptor_tensor = torch.cat(
        [
            angles_between_segments,
            angles_with_global_direction,
            relative_vectors,
            relative_distances,
        ],
        dim=0,
    )
    return shape_descriptor_tensor


def rope_target_distances(
    env: RLTaskEnv,
    asset_cfg: SceneEntityCfg,
    aux_data: AuxiliaryData,
    weight_n_last_steps: float = 1.0,
    n_last_steps: int = 0,
) -> torch.Tensor:

    if aux_data.target_geometry_positions is None:
        aux_data.target_geometry_positions = generate_positions_UV().to(env.device)

    target_object: RigidObject = env.scene["target"]
    target_orientation = target_object.data.root_quat_w

    target = aux_data.target_geometry_positions
    target = quat_apply_yaw(
        target_orientation.unsqueeze(1).repeat(1, target.shape[0], 1),
        target.unsqueeze(0).repeat(env.num_envs, 1, 1),
    )
    asset: Rope = env.scene[asset_cfg.name]
    points_pos = asset.data.link_pos_w.clone()
    points_pos = points_pos.reshape(env.num_envs, -1, 3)
    points_pos -= env.scene.env_origins.unsqueeze(1)

    current_descriptor = torch.vmap(shape_descriptor)(points_pos[..., :2])
    target_descriptor = torch.vmap(shape_descriptor)(target[..., :2])
    distance = (current_descriptor - target_descriptor).pow(2).mean(dim=-1)

    if n_last_steps > 0 and torch.all(env.episode_length_buf >= env.max_episode_length - n_last_steps):
        distance = distance * weight_n_last_steps

    return distance


def links_velocity(
    env: RLTaskEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """Reward the agent for minimizing the velocity of the deformable object."""
    asset: Rope = env.scene[asset_cfg.name]
    return asset.data.link_vel_w.norm(dim=-1).mean(dim=-1)


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
