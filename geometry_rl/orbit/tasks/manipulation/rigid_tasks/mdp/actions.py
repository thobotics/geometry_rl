from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import FixedObject, RigidObject
from omni.isaac.orbit.managers import ActionTerm, ActionTermCfg
from omni.isaac.orbit.utils import configclass
import omni.isaac.orbit.utils.math as math_utils

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


@torch.jit.script
def calculate_angular_velocity(v, r):
    r_norm_squared = torch.sum(r * r, dim=1, keepdim=True)
    v_dot_r = torch.sum(v * r, dim=1, keepdim=True)
    v_parallel = (v_dot_r / r_norm_squared) * r
    v_tangential = v - v_parallel
    omega = torch.cross(r, v_tangential, dim=1) / r_norm_squared

    return v_parallel, v_tangential, omega


class CubeActionTerm(ActionTerm):
    """Simple action term that apply a velocity command to the cube."""

    _asset: FixedObject | RigidObject
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: CubeActionTermCfg, env: RLTaskEnv):
        # call super constructor
        super().__init__(cfg, env)
        # create buffers
        self._raw_actions = torch.zeros(env.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros(env.num_envs, 6, device=self.device)
        self._rotation_axis = cfg.rotation_axis
        self._z_action = cfg.z_action

    """
    Properties.
    """

    @property
    def action_linear_scale(self) -> float:
        return 1.0

    @property
    def action_angular_scale(self) -> float:
        return 20.0

    @property
    def action_max(self) -> float:
        return 1.0

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # no-processing of actions
        self._processed_actions[:] = self._raw_actions[:]

    def apply_actions(self):
        vel_command = self._processed_actions
        vel_command = torch.clamp(
            vel_command,
            -self.action_max,
            self.action_max,
        )

        # Decompose the velocity command into linear and angular components
        current_pose = self._asset.data.root_state_w[:, :3] - self._env.scene.env_origins
        _, _, omega = calculate_angular_velocity(vel_command[:, 3:], current_pose)

        v_parallel = vel_command[:, :3] * self.action_linear_scale
        omega = omega * self.action_angular_scale

        vel_command = torch.cat([v_parallel, omega], dim=1)
        if not self._z_action:
            vel_command[:, 2] = 0.0
        if self._rotation_axis > -1:
            mask = torch.zeros(3, device=self.device)
            mask[self._rotation_axis] = 1.0
            vel_command[:, 3:] *= mask

        if isinstance(self._asset, RigidObject):
            self._asset.write_root_velocity_to_sim(vel_command)
        else:
            vel_command *= self._env.physics_dt
            current_pose = self._asset.data.root_state_w
            current_pose[:, :3] = current_pose[:, :3] + vel_command
            self._asset.write_root_pose_to_sim(current_pose)


class CubeActionLinearTerm(CubeActionTerm):
    """Simple action term that apply a velocity command to the cube."""

    def __init__(self, cfg: CubeActionLinearTermCfg, env: RLTaskEnv):
        # call super constructor
        super(CubeActionTerm, self).__init__(cfg, env)
        # create buffers
        self._raw_actions = torch.zeros(env.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros(env.num_envs, 3, device=self.device)
        self._z_action = cfg.z_action

    """
    Properties.
    """

    def apply_actions(self):
        vel_command = self._processed_actions
        vel_command = torch.clamp(
            vel_command,
            -self.action_max,
            self.action_max,
        )

        vel_command = torch.cat([vel_command, torch.zeros_like(vel_command)], dim=1)
        if not self._z_action:
            vel_command[:, 2] = 0.0

        if isinstance(self._asset, RigidObject):
            self._asset.write_root_velocity_to_sim(vel_command)
        else:
            vel_command *= self._env.physics_dt
            current_pose = self._asset.data.root_state_w
            current_pose[:, :3] = current_pose[:, :3] + vel_command
            self._asset.write_root_pose_to_sim(current_pose)


@configclass
class CubeActionTermCfg(ActionTermCfg):
    """Configuration for the cube action term."""

    class_type: type = CubeActionTerm
    """The class corresponding to the action term."""

    rotation_axis: int = -1  # -1: all axes, 0: x-axis, 1: y-axis, 2: z-axis
    z_action: bool = False


@configclass
class CubeActionLinearTermCfg(ActionTermCfg):
    """Configuration for the cube action term."""

    class_type: type = CubeActionLinearTerm
    """The class corresponding to the action term."""

    z_action: bool = False
    """Whether to allow z-axis movement."""
