import gymnasium as gym
from .rigid_sliding_multi_env_cfg import RigidSlidingMultiEnvCfg
from .rigid_insertion_multi_env_cfg import RigidInsertionMultiEnvCfg
from .rigid_insertion_two_agents_multi_env_cfg import RigidInsertionTwoAgentsMultiEnvCfg
from .rigid_pushing_multi_env_cfg import RigidPushingMultiEnvCfg

##
# Register Gym environments.
##
# Environment configuration mappings
env_configurations = [
    (
        "Isaac-Rigid-Sliding-Multi-v0",
        RigidSlidingMultiEnvCfg,
        None,
    ),
    (
        "Isaac-Rigid-Pushing-Multi-v0",
        RigidPushingMultiEnvCfg,
        None,
    ),
    (
        "Isaac-Rigid-Insertion-Multi-v0",
        RigidInsertionMultiEnvCfg,
        None,
    ),
    (
        "Isaac-Rigid-Insertion-Two-Agents-Multi-v0",
        RigidInsertionTwoAgentsMultiEnvCfg,
        None,
    ),
]

# Register Gym environments
for env_id, cfg, runner_cfg in env_configurations:
    kwargs = {"env_cfg_entry_point": cfg}
    if runner_cfg:
        kwargs["rsl_rl_cfg_entry_point"] = runner_cfg

    gym.register(
        id=env_id,
        entry_point="omni.isaac.orbit.envs:RLTaskEnv",
        kwargs=kwargs,
        disable_env_checker=True,
    )
