from __future__ import annotations

from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.assets import AssetBaseCfg
from ..common_cfg.scene_cfg import RigidSceneCfg
from ..common_cfg.observations_cfg import (
    FullObservationsCfg,
    NoObjectVelObservationCfg,
    FullSceneObservationTwoAgentsCfg,
)
from ..common_cfg.actions_cfg import (
    LinearWithYawRotationNoZActionsCfg,
    LinearWithYawRotationZActionsCfg,
    OnlyLinearActionsNoZCfg,
    OnlyLinearActionsZCfg,
)
from ..common_cfg.rewards_cfg import (
    SlidingRewardsCfg,
    InsertionRewardsCfg,
    InsertionTwoAgentsRewardsCfg,
    PushingRewardsCfg,
)
from ..common_cfg.terminations_cfg import TerminationsCfg
from ..common_cfg.event_cfg import (
    SlidingRandomizationCfg,
    InsertionRandomizationCfg,
    InsertionTwoAgentsRandomizationCfg,
    PushingRandomizationCfg,
)
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.envs import ViewerCfg


@configclass
class RigidSlidingEnvCfg(RLTaskEnvCfg):
    """Configuration for the lifting environment."""

    viewer: ViewerCfg = ViewerCfg(resolution=(2560, 1440))  # 2K

    # Scene settings
    scene: RigidSceneCfg = RigidSceneCfg(num_envs=40, env_spacing=2.0, replicate_physics=False)

    # Basic settings
    observations: NoObjectVelObservationCfg = NoObjectVelObservationCfg()
    actions: LinearWithYawRotationNoZActionsCfg = LinearWithYawRotationNoZActionsCfg()

    # MDP settings
    rewards: SlidingRewardsCfg = SlidingRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    randomization: SlidingRandomizationCfg = SlidingRandomizationCfg()

    def __post_init__(self):
        """Post initialization."""

        self.scene.__dict__["ground"] = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(size=(1e4, 1e4), color=(0.0, 0.0, 0.0)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )

        # general settings
        self.decimation = 4
        self.warmup_steps = 0
        self.episode_length_s = 4.0  # = decimation * dt * 100 steps
        # simulation settings
        self.sim.dt = 0.01  # 100Hz

        # Note: the following settings are copied from OIGE FrankaDeformable
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.025
        self.sim.physx.gpu_max_rigid_contact_count = 2**20
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_found_lost_pairs_capacity = 128 * 1024
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 128 * 1024
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 128 * 1024
        self.sim.physx.gpu_max_soft_body_contacts = 1 * 1024 * 1024
        self.sim.physx.gpu_max_particle_contacts = 128 * 1024
        self.sim.physx.gpu_heap_capacity = 8 * 1024 * 1024
        self.sim.physx.gpu_temp_buffer_capacity = 4 * 1024 * 1024
        self.sim.physx.gpu_max_num_partitions = 4


@configclass
class RigidInsertionEnvCfg(RigidSlidingEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: RigidSceneCfg = RigidSceneCfg(num_envs=40, env_spacing=5.0, replicate_physics=False)

    # Basic settings
    actions: LinearWithYawRotationZActionsCfg = LinearWithYawRotationZActionsCfg()

    # MDP settings
    rewards: InsertionRewardsCfg = InsertionRewardsCfg()
    randomization: InsertionRandomizationCfg = InsertionRandomizationCfg()


@configclass
class RigidInsertionTwoAgentsEnvCfg(RigidInsertionEnvCfg):
    """Configuration for the lifting environment."""

    # Basic settings
    observations: FullSceneObservationTwoAgentsCfg = FullSceneObservationTwoAgentsCfg()
    actions: OnlyLinearActionsZCfg = OnlyLinearActionsZCfg()

    # MDP settings
    rewards: InsertionTwoAgentsRewardsCfg = InsertionTwoAgentsRewardsCfg()
    randomization: InsertionTwoAgentsRandomizationCfg = InsertionTwoAgentsRandomizationCfg()


@configclass
class RigidPushingEnvCfg(RLTaskEnvCfg):
    """Configuration for the lifting environment."""

    viewer: ViewerCfg = ViewerCfg(resolution=(2560, 1440))  # 2K

    # Scene settings
    scene: RigidSceneCfg = RigidSceneCfg(num_envs=40, env_spacing=2.0, replicate_physics=False)

    # Basic settings
    observations: FullObservationsCfg = FullObservationsCfg()
    actions: OnlyLinearActionsNoZCfg = OnlyLinearActionsNoZCfg()

    # MDP settings
    rewards: PushingRewardsCfg = PushingRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    randomization: PushingRandomizationCfg = PushingRandomizationCfg()

    def __post_init__(self):
        """Post initialization."""

        self.scene.__dict__["ground"] = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(size=(1e4, 1e4), color=(0.0, 0.0, 0.0)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )

        # general settings
        self.decimation = 4
        self.warmup_steps = 0
        self.episode_length_s = 4.0  # = decimation * dt * 100 steps
        # simulation settings
        self.sim.dt = 0.01  # 100Hz

        # Note: the following settings are copied from OIGE FrankaDeformable
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.025
        self.sim.physx.gpu_max_rigid_contact_count = 2**20
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_found_lost_pairs_capacity = 128 * 1024
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 128 * 1024
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 128 * 1024
        self.sim.physx.gpu_max_soft_body_contacts = 1 * 1024 * 1024
        self.sim.physx.gpu_max_particle_contacts = 128 * 1024
        self.sim.physx.gpu_heap_capacity = 8 * 1024 * 1024
        self.sim.physx.gpu_temp_buffer_capacity = 4 * 1024 * 1024
        self.sim.physx.gpu_max_num_partitions = 4
