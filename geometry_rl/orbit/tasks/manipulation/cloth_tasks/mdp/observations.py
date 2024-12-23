# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import RigidObject, ClothObject, FixedObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import transform_points
from ....common.utils import (
    get_k_boundary_nodes,
    get_k_boundary_nodes_in_multi_asset,
    extract_cloth_geometry_positions,
    get_geometry_from_rigid_object,
    extract_distance_matrix_from_two_sets,
)

from scipy.spatial import Delaunay


if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv
    from ..config.common_cfg.aux_data import AuxiliaryData


def base_position(env: RLTaskEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root positions in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | FixedObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def object_positions(env: RLTaskEnv, names: list) -> torch.Tensor:
    """Root positions in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)

    positions = []
    for name in names:
        asset: RigidObject | FixedObject = env.scene[name]
        positions.append(asset.data.root_pos_w - env.scene.env_origins)

    return torch.cat(positions, dim=-1)


def object_orientations(env: RLTaskEnv, names: list) -> torch.Tensor:
    """Root orientations in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)

    orientations = []
    for name in names:
        asset: RigidObject | FixedObject = env.scene[name]
        orientations.append(asset.data.root_quat_w)

    return torch.cat(orientations, dim=-1)


def object_velocities(env: RLTaskEnv, names: list) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)

    velocities = []
    for name in names:
        asset: RigidObject | FixedObject = env.scene[name]
        if hasattr(asset.data, "root_lin_vel_w"):
            velocities.append(asset.data.root_lin_vel_w)
        else:
            velocities.append(torch.zeros_like(asset.data.root_pos_w))

    return torch.cat(velocities, dim=-1)


def object_geometry(env: RLTaskEnv, name: str, aux_data: AuxiliaryData) -> torch.Tensor:
    """Root geometry in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)

    if aux_data.target_hook_geometry_positions is None:
        aux_data.target_hook_geometry_positions = get_geometry_from_rigid_object(env, name)
    return aux_data.target_hook_geometry_positions.reshape(env.num_envs, -1)


def object_init_positions(env: RLTaskEnv, names: list) -> torch.Tensor:
    """Root init position in the asset's root frame."""

    positions = []
    for name in names:
        asset: RigidObject | FixedObject = env.scene[name]
        positions.append(asset._data.default_root_state[:, :3].clone())

    positions = torch.cat(positions, dim=-1).reshape(-1, len(positions), 3)
    return positions.reshape(env.num_envs, -1)


def points_positions(env: RLTaskEnv, asset_cfg: SceneEntityCfg, indices: torch.Tensor = None) -> torch.Tensor:
    """Points positions in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: ClothObject = env.scene[asset_cfg.name]
    points_pos = asset.data.points_pos_w.clone()

    if indices is not None:
        indices = indices.unsqueeze(-1).expand(-1, -1, 3).to(points_pos.device)
        points_pos = torch.gather(points_pos, 1, indices)

    points_pos -= env.scene.env_origins.unsqueeze(1)
    return points_pos.view(env.num_envs, -1)


def points_velocities(env: RLTaskEnv, asset_cfg: SceneEntityCfg, indices: torch.Tensor = None) -> torch.Tensor:
    """Points linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: ClothObject = env.scene[asset_cfg.name]
    points_vel = asset.data.points_vel_w.clone()
    if indices is not None:
        indices = indices.unsqueeze(-1).expand(-1, -1, 3).to(points_vel.device)
        points_vel = torch.gather(points_vel, 1, indices)
    return points_vel.view(env.num_envs, -1)


def cloth_geometry_positions(env: RLTaskEnv, asset_cfg: SceneEntityCfg, aux_data: AuxiliaryData) -> torch.Tensor:
    if aux_data.cloth_geometry_positions is None:
        cloth_object = env.scene[asset_cfg.name]
        cloth_geometry_positions = extract_cloth_geometry_positions(cloth_object, torch.arange(env.num_envs))
        # Transform to make sure the orientation is correct
        translation, orientation = cloth_object.root_view.get_world_poses()
        cloth_geometry_positions = transform_points(cloth_geometry_positions, translation, orientation)
        # Transform back to the local frame
        aux_data.cloth_geometry_positions = cloth_geometry_positions - env.scene.env_origins.unsqueeze(1)

        initial_points = aux_data.cloth_geometry_positions
        initial_points = initial_points.cpu().numpy()

        cloth_triangles = []
        cloth_edges = []
        for i in range(env.num_envs):
            cloth_plane = initial_points[i][..., [0, 2]]
            cloth_triangle = Delaunay(cloth_plane).simplices
            cloth_triangle = torch.tensor(cloth_triangle, device=env.device)
            cloth_triangles.append(cloth_triangle)

            edges1 = cloth_triangle[:, [0, 1]]
            edges2 = cloth_triangle[:, [1, 2]]
            edges3 = cloth_triangle[:, [2, 0]]
            edges = torch.cat((edges1, edges2, edges3), dim=0)

            # Sort each edge to ensure unique edges (order does not matter)
            edges = torch.sort(edges, dim=1)[0]

            # Remove duplicate edges
            unique_edges = torch.unique(edges, dim=0)

            # if aux_data.holes_boundary_nodes_indices is None:
            #     hole_boundary_positions(env, asset_cfg, aux_data)
            # idx = aux_data.holes_boundary_nodes_indices[i].to(env.device)
            # mask = ~(torch.isin(unique_edges[:, 0], idx) | torch.isin(unique_edges[:, 1], idx))
            # unique_edges = unique_edges[mask]

            cloth_edges.append(unique_edges)

        aux_data.cloth_triangles = cloth_triangles
        aux_data.cloth_edges = cloth_edges

    return aux_data.cloth_geometry_positions.reshape(env.num_envs, -1)


def cloth_edges(env: RLTaskEnv, asset_cfg: SceneEntityCfg, aux_data: AuxiliaryData) -> torch.Tensor:
    if aux_data.cloth_edges is None:
        cloth_geometry_positions(env, asset_cfg, aux_data)

    return torch.cat(aux_data.cloth_edges, dim=0).reshape(env.num_envs, -1)


def hole_boundary_positions(env: RLTaskEnv, asset_cfg: SceneEntityCfg, aux_data: AuxiliaryData) -> torch.Tensor:
    if aux_data.holes_boundary_nodes_indices is None:
        aux_data.holes_boundary_nodes_indices = get_k_boundary_nodes_in_multi_asset(
            env,
            asset_cfg,
        )
    return points_positions(
        env,
        asset_cfg,
        aux_data.holes_boundary_nodes_indices,
    )


def hole_boundary_indices(env: RLTaskEnv, asset_cfg: SceneEntityCfg, aux_data: AuxiliaryData) -> torch.Tensor:
    if aux_data.holes_boundary_nodes_indices is None:
        aux_data.holes_boundary_nodes_indices = get_k_boundary_nodes_in_multi_asset(
            env,
            asset_cfg,
        )

    return aux_data.holes_boundary_nodes_indices


def hole_boundary_target_distances(
    env: RLTaskEnv,
    asset_cfg: SceneEntityCfg,
    hanger_cfg: SceneEntityCfg,
    aux_data: AuxiliaryData,
) -> torch.Tensor:
    holes_positions = hole_boundary_positions(env, asset_cfg, aux_data).reshape(env.num_envs, -1, 3)
    target = object_positions(env, [hanger_cfg.name]).unsqueeze(1)

    distances = torch.norm(holes_positions - target, dim=-1)
    # distances = holes_positions - target
    return distances.reshape(env.num_envs, -1)
