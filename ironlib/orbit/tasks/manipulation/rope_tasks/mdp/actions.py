from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import FixedObject, RigidObject
from omni.isaac.orbit.managers import ActionTerm, ActionTermCfg
from omni.isaac.orbit.utils import configclass

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


class CubeActionTerm(ActionTerm):
    """Simple action term that apply a velocity command to the cube."""

    _asset: FixedObject | RigidObject
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: CubeActionTermCfg, env: RLTaskEnv):
        # call super constructor
        super().__init__(cfg, env)
        # create buffers
        self._raw_actions = torch.zeros(env.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros(env.num_envs, 3, device=self.device)

    """
    Properties.
    """

    @property
    def action_scale(self) -> float:
        return 5.0

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
        vel_command = vel_command * self.action_scale
        vel_command[:, 2] = 0.0  # no z velocity

        if isinstance(self._asset, RigidObject):
            # add zero angular velocity
            self._asset.write_root_velocity_to_sim(torch.cat([vel_command, torch.zeros_like(vel_command)], dim=-1))
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
