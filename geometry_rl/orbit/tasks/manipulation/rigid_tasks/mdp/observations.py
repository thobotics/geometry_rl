# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import RigidObject, FixedObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import transform_points
from ....common.utils import get_geometry_from_rigid_object


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
        # velocities.append(torch.zeros_like(asset.data.root_pos_w))

    return torch.cat(velocities, dim=-1)


def object_angular_velocities(env: RLTaskEnv, names: list) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)

    velocities = []
    for name in names:
        asset: RigidObject | FixedObject = env.scene[name]
        if hasattr(asset.data, "root_ang_vel_w"):
            velocities.append(asset.data.root_ang_vel_w)
        else:
            velocities.append(torch.zeros_like(asset.data.root_pos_w))
        # velocities.append(torch.zeros_like(asset.data.root_pos_w))

    return torch.cat(velocities, dim=-1)


def object_geometry(env: RLTaskEnv, name: str, aux_data: AuxiliaryData) -> torch.Tensor:
    """Root geometry in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)

    if aux_data.object_geometry_positions is None:
        (
            aux_data.object_geometry_positions,
            aux_data.num_points,
            aux_data.object_geometry_edges,
            aux_data.num_edges,
        ) = get_geometry_from_rigid_object(env, name, return_num=True, return_edges=True)
    return aux_data.object_geometry_positions


def object_geometry_positions(env: RLTaskEnv, name: str, aux_data: AuxiliaryData) -> torch.Tensor:
    """Root geometry in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)

    object_geom = object_geometry(env, name, aux_data)

    object_pos = object_positions(env, [name])
    object_ori = object_orientations(env, [name])
    object_point = transform_points(object_geom, object_pos, object_ori)

    return object_point.reshape(env.num_envs, -1)


def object_num_points(env: RLTaskEnv, name: str, aux_data: AuxiliaryData) -> int:
    """Number of points in the object geometry."""
    # extract the used quantities (to enable type-hinting)

    if aux_data.num_points is None:
        _ = object_geometry(env, name, aux_data)
    return aux_data.num_points


def object_geometry_edges(env: RLTaskEnv, name: str, aux_data: AuxiliaryData) -> torch.Tensor:
    """Root geometry in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)

    if aux_data.object_geometry_edges is None:
        _ = object_geometry(env, name, aux_data)
    return aux_data.object_geometry_edges


def object_num_edges(env: RLTaskEnv, name: str, aux_data: AuxiliaryData) -> int:
    """Number of points in the object geometry."""
    # extract the used quantities (to enable type-hinting)

    if aux_data.num_edges is None:
        _ = object_geometry(env, name, aux_data)
    return aux_data.num_edges


def object_init_positions(env: RLTaskEnv, names: list) -> torch.Tensor:
    """Root init position in the asset's root frame."""

    positions = []
    for name in names:
        asset: RigidObject | FixedObject = env.scene[name]
        positions.append(asset._data.default_root_state[:, :3].clone())

    positions = torch.cat(positions, dim=-1).reshape(-1, len(positions), 3)
    return positions.reshape(env.num_envs, -1)


def object_target_distance_obs(
    env: RLTaskEnv,
    asset_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    aux_data: AuxiliaryData,
) -> torch.Tensor:
    object_pos = object_geometry_positions(env, asset_cfg.name, aux_data)
    target = object_geometry_positions(env, target_cfg.name, aux_data)

    distances = torch.norm(object_pos - target, dim=-1)
    return distances.reshape(env.num_envs, -1)
