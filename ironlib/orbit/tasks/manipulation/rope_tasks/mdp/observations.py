# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import RigidObject, Rope, FixedObject
from omni.isaac.orbit.managers import SceneEntityCfg
from ....common.utils import (
    get_geometry_from_rigid_object,
)

from omni.isaac.orbit.utils.math import quat_apply_yaw
from scipy.interpolate import interp1d

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv
    from ..config.common_cfg.aux_data import AuxiliaryData


def generate_positions_UV(num_links=80, rope_length=10.0, radius=0.025):
    import pyLasaDataset as lasa

    using_data = lasa.DataSet.WShape
    # using_data = lasa.DataSet.PShape

    demos = using_data.demos
    demo_0 = demos[0]
    pos = demo_0.pos

    # Create an array of evenly spaced points for interpolation
    interp_points = np.linspace(0, pos.shape[1] - 1, num_links)

    # Interpolate the x and y positions separately
    x_interp_func = interp1d(np.arange(pos.shape[1]), pos[0])
    y_interp_func = interp1d(np.arange(pos.shape[1]), pos[1])

    x_interp = x_interp_func(interp_points)
    y_interp = y_interp_func(interp_points)
    pos_interp = np.vstack((x_interp, y_interp))

    # Calculate the total length of the interpolated path
    distances = np.sqrt(np.diff(pos_interp[0]) ** 2 + np.diff(pos_interp[1]) ** 2)
    total_length = np.sum(distances)

    # Calculate the scaling factor to match the desired rope length
    scale_factor = rope_length / total_length

    # Scale the interpolated positions
    pos_interp_scaled = pos_interp * scale_factor

    target_shape = torch.tensor(pos_interp_scaled.T, dtype=torch.float32)
    target_shape = torch.cat((target_shape, torch.ones_like(target_shape[:, :1]) * 0.1), dim=1)

    return target_shape


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


def target_positions(env: RLTaskEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root positions in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)

    positions = []
    asset: RigidObject | FixedObject = env.scene[asset_cfg.name]
    positions = asset.data.root_pos_w - env.scene.env_origins

    rope = env.scene["rope"]
    num_links = rope.data._max_links
    positions = positions.unsqueeze(1).expand(-1, num_links, -1)

    return positions.reshape(env.num_envs, -1)


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

    if aux_data.target_geometry_positions is None:
        aux_data.target_geometry_positions = get_geometry_from_rigid_object(env, name)
    return aux_data.target_geometry_positions.reshape(env.num_envs, -1)


def object_init_positions(env: RLTaskEnv, names: list) -> torch.Tensor:
    """Root init position in the asset's root frame."""

    positions = []
    for name in names:
        asset: RigidObject | FixedObject = env.scene[name]
        positions.append(asset._data.default_root_state[:, :3].clone())

    positions = torch.cat(positions, dim=-1).reshape(-1, len(positions), 3)
    return positions.reshape(env.num_envs, -1)


def links_positions(env: RLTaskEnv, asset_cfg: SceneEntityCfg, indices: torch.Tensor = None) -> torch.Tensor:
    """Points positions in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Rope = env.scene[asset_cfg.name]
    points_pos = asset.data.link_pos_w.clone()
    points_pos = points_pos.reshape(env.num_envs, -1, 3)

    if indices is not None:
        indices = indices.unsqueeze(-1).expand(-1, -1, 3).to(points_pos.device)
        points_pos = torch.gather(points_pos, 1, indices)

    points_pos -= env.scene.env_origins.unsqueeze(1)
    return points_pos.view(env.num_envs, -1)


def links_velocities(env: RLTaskEnv, asset_cfg: SceneEntityCfg, indices: torch.Tensor = None) -> torch.Tensor:
    """Points linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Rope = env.scene[asset_cfg.name]
    points_vel = asset.data.link_vel_w.clone()
    if indices is not None:
        indices = indices.unsqueeze(-1).expand(-1, -1, 3).to(points_vel.device)
        points_vel = torch.gather(points_vel, 1, indices)
    return points_vel.view(env.num_envs, -1)


def target_geometry_positions(env: RLTaskEnv, asset_cfg: SceneEntityCfg, aux_data: AuxiliaryData) -> torch.Tensor:
    """Root geometry in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)

    if aux_data.target_geometry_positions is None:
        aux_data.target_geometry_positions = generate_positions_UV().to(env.device)

    target_object: RigidObject = env.scene[asset_cfg.name]
    target_orientation = target_object.data.root_quat_w

    target_geometry_positions = aux_data.target_geometry_positions
    target_geometry_positions = quat_apply_yaw(
        target_orientation.unsqueeze(1).repeat(1, target_geometry_positions.shape[0], 1),
        target_geometry_positions.unsqueeze(0).repeat(env.num_envs, 1, 1),
    )

    return target_geometry_positions.reshape(env.num_envs, -1)


def rope_target_distances_obs(
    env: RLTaskEnv,
    asset_cfg: SceneEntityCfg,
    aux_data: AuxiliaryData,
) -> torch.Tensor:
    positions = links_positions(env, asset_cfg).reshape(env.num_envs, -1, 3)

    distances = torch.zeros((env.num_envs, 1), device=env.device)

    # distances = torch.norm(links_positions - target, dim=-1)
    return distances.reshape(env.num_envs, -1)
