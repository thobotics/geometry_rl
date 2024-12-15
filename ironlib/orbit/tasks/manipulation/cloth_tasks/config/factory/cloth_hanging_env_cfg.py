from __future__ import annotations

from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.utils import configclass

from ..common_cfg.actions_cfg import ActionsCfg
from ..common_cfg.observations_cfg import FullObservationsCfg
from ..common_cfg.rewards_cfg import HangingRewardsCfg, ICLRHangingRewardsCfg
from ..common_cfg.terminations_cfg import TerminationsCfg
from ..common_cfg.event_cfg import HangingRandomizationCfg
from ..common_cfg.scene_cfg import HangingSceneCfg, MultiAssetsClothSceneCfg

##
# Environment configuration
##


@configclass
class ClothHangingEnvCfg(RLTaskEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: HangingSceneCfg = HangingSceneCfg(num_envs=40, env_spacing=5.0, replicate_physics=False)

    # Basic settings
    observations: FullObservationsCfg = FullObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    # MDP settings
    rewards: ICLRHangingRewardsCfg = ICLRHangingRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    randomization: HangingRandomizationCfg = HangingRandomizationCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.warmup_steps = 0
        self.episode_length_s = 2.0  # = decimation * dt * 100 steps
        # simulation settings
        self.sim.dt = 0.01  # 100Hz

        # Note: the following settings are copied from OIGE FrankaDeformable
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.025
        self.sim.physx.gpu_max_rigid_contact_count = 128 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 8 * 1024 * 1024
        self.sim.physx.gpu_found_lost_pairs_capacity = 128 * 1024
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 128 * 1024
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 128 * 1024
        self.sim.physx.gpu_max_soft_body_contacts = 1 * 1024 * 1024
        self.sim.physx.gpu_max_particle_contacts = 128 * 1024
        self.sim.physx.gpu_heap_capacity = 8 * 1024 * 1024
        self.sim.physx.gpu_temp_buffer_capacity = 4 * 1024 * 1024
        self.sim.physx.gpu_max_num_partitions = 4


### Multi Environment Configuration ###
@configclass
class ClothHangingMultiEnvCfg(ClothHangingEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: MultiAssetsClothSceneCfg = MultiAssetsClothSceneCfg(num_envs=40, env_spacing=5.0, replicate_physics=False)
