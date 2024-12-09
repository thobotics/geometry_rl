# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym
from .rope_closing_env_cfg import RopeClosingEnvCfg
from .rope_shaping_env_cfg import RopeShapingEnvCfg

##
# Register Gym environments.
##
# Environment configuration mappings
env_configurations = [
    (
        "Isaac-Rope-Closing-v0",
        RopeClosingEnvCfg,
        None,
    ),
    (
        "Isaac-Rope-Shaping-v0",
        RopeShapingEnvCfg,
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
