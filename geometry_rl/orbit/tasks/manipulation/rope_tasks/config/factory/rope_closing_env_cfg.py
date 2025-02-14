from __future__ import annotations

from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.utils import configclass

from ..common_cfg.actions_cfg import ActionsCfg
from ..common_cfg.observations_cfg import ClosingObservationsCfg
from ..common_cfg.rewards_cfg import ClosingRewardsCfg
from ..common_cfg.terminations_cfg import TerminationsCfg
from ..common_cfg.event_cfg import ClosingRandomizationCfg
from ..common_cfg.scene_cfg import ClosingSceneCfg
from omni.isaac.orbit.envs import ViewerCfg


##
# Environment configuration
##


@configclass
class RopeClosingEnvCfg(RLTaskEnvCfg):
    """Configuration for the lifting environment."""

    viewer: ViewerCfg = ViewerCfg(resolution=(2560, 1440))  # 2K

    # Scene settings
    scene: ClosingSceneCfg = ClosingSceneCfg(num_envs=40, env_spacing=40.0, replicate_physics=False)

    # Basic settings
    observations: ClosingObservationsCfg = ClosingObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    # MDP settings
    rewards: ClosingRewardsCfg = ClosingRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    randomization: ClosingRandomizationCfg = ClosingRandomizationCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.warmup_steps = 0
        self.episode_length_s = 4.0  # = decimation * dt * 200 steps
        # simulation settings
        self.sim.dt = 0.01  # 100Hz

        # Note: the following settings are copied from OIGE FrankaDeformable
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.025
        self.sim.physx.gpu_max_rigid_contact_count = 128 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 12 * 1024 * 1024
        self.sim.physx.gpu_found_lost_pairs_capacity = 128 * 1024
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physx.gpu_max_soft_body_contacts = 1 * 256 * 1024
        self.sim.physx.gpu_max_particle_contacts = 256 * 1024
        self.sim.physx.gpu_heap_capacity = 4 * 1024 * 1024
        self.sim.physx.gpu_temp_buffer_capacity = 4 * 1024 * 1024
        self.sim.physx.gpu_max_num_partitions = 2
