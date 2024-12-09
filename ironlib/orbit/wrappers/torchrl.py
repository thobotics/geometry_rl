from __future__ import annotations

import importlib.util

import itertools
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch

from tensordict import TensorDictBase
from torchrl.envs.libs.gym import (
    GymWrapper,
    _AsyncMeta,
)
from torchrl.envs.utils import _classproperty, make_composite_from_td
from ironlib.orbit.utils.tensordict import remap_dict_key

_has_orbit = importlib.util.find_spec("omni.isaac.orbit") is not None


class _SyncMeta(_AsyncMeta):
    def __call__(cls, *args, **kwargs):
        """This class overrides the __call__ method of the AsyncMeta class,
        for now since we do not explicitly use any gym API, final_observation
        is ignored.
        """
        instance: GymWrapper = super(_AsyncMeta, cls).__call__(*args, **kwargs)
        return instance


class OrbitTorchRLWrapper(GymWrapper, metaclass=_SyncMeta):
    """Wrapper for Orbit environments.

    The original library can be found `here <https://github.com/NVIDIA-Omniverse/orbit>`_
    and is based on IsaacGym which can be downloaded `through NVIDIA's webpage.

    .. note:: Orbit environments cannot be executed consecutively, ie. instantiating one
        environment after another (even if it has been cleared) will cause
        CUDA memory issues. We recommend creating one environment per process only.
        If you need more than one environment, the best way to achieve that is
        to spawn them across processes.

    .. note:: Orbit works on CUDA devices by essence. Make sure your machine
        has GPUs available and the required setup for Orbit (eg, Ubuntu 20.04).

    """

    _envs = None
    _warmup_steps: int = 0
    libname = "gymnasium"

    @property
    def lib(self):
        import omni.isaac.orbit  # noqa

        return omni.isaac.orbit

    def __init__(
        self,
        env: "omni.isaac.orbit.envs.RLTaskEnv",
        device: str,
        **kwargs,  # noqa: F821
    ):
        super().__init__(env, False, device=torch.device(device), **kwargs)
        if not hasattr(self, "task"):
            # by convention in IsaacGymEnvs
            self.task = env.__name__

    def _make_specs(self, env: "gym.Env") -> None:  # noqa: F821
        super()._make_specs(env)

        # Further processing
        data = self.rollout(3).get("next")[..., 0]
        del data[self.reward_key]
        for done_key in self.done_keys:
            try:
                del data[done_key]
            except KeyError:
                continue
        specs = make_composite_from_td(data)

        obs_spec = self.observation_spec
        obs_spec.unlock_()
        obs_spec.update(specs)
        obs_spec.lock_()
        self.__dict__["full_observation_spec"] = obs_spec

    @classmethod
    def _make_envs(cls, *, task, num_envs, device, seed=None, config=None):
        from omni.isaac.orbit.envs import RLTaskEnv  # noqa
        import omni.isaac.contrib_tasks  # noqa: F401
        import omni.isaac.orbit_tasks  # noqa: F401
        import ironlib.orbit.tasks  # noqa: F401
        from omni.isaac.orbit.envs import RLTaskEnvCfg
        from omni.isaac.orbit_tasks.utils import parse_env_cfg

        if cls.libname == "gymnasium":
            import gymnasium as gym
        else:
            import gym

        use_gpu = device.startswith("cuda")
        env_cfg: RLTaskEnvCfg = parse_env_cfg(task, use_gpu=use_gpu, num_envs=num_envs)

        cls._warmup_steps = config.pop("warmup_steps", 0)
        env_cfg.warmup_steps = cls._warmup_steps
        env_cfg.episode_length_s += (
            cls._warmup_steps * env_cfg.decimation * env_cfg.sim.dt
        )

        if cls._envs is None:
            envs = gym.make(
                task,
                cfg=env_cfg,
                render_mode="rgb_array" if config["video"] else None,
                disable_env_checker=True,
            )
            cls._envs = envs
        else:
            envs = cls._envs

        if seed is not None:
            # make sure that all torch code is also reproductible
            envs.seed(seed)

        return envs

    @property
    def _is_batched(self):
        return True

    def _set_seed(self, seed: int) -> int:
        self._envs.seed(seed)
        return seed

    def read_action(self, action):
        """Reads the action obtained from the input TensorDict and transforms it in the format expected by the contained environment.

        Args:
            action (Tensor or TensorDict): an action to be taken in the environment

        Returns: an action in a format compatible with the contained environment.

        """
        return action

    def read_done(
        self,
        terminated: bool = None,
        truncated: bool | None = None,
        done: bool | None = None,
    ) -> Tuple[bool, bool, bool]:
        if terminated is not None:
            if not isinstance(terminated, torch.Tensor):
                terminated = torch.tensor(
                    terminated, dtype=torch.bool, device=self.device
                )
            terminated = terminated.bool()

        if truncated is not None:
            if not isinstance(truncated, torch.Tensor):
                truncated = torch.tensor(
                    truncated, dtype=torch.bool, device=self.device
                )

            truncated = truncated.bool()
        if done is not None:
            done = done.bool()
        return terminated, truncated, done, done.any()

    def read_reward(self, total_reward, step_reward=None):
        """Reads a reward and the total reward so far (in the frame skip loop) and returns a sum of the two.

        Args:
            total_reward (torch.Tensor or TensorDict): total reward so far in the step
            step_reward (reward in the format provided by the inner env): reward of this particular step

        """
        if step_reward is None:
            step_reward = 0.0
        if isinstance(total_reward, torch.Tensor):
            total_reward = total_reward.reshape(self.reward_spec.shape)
        return total_reward + step_reward

    def read_obs(
        self, observations: Union[Dict[str, Any], torch.Tensor, np.ndarray]
    ) -> Dict[str, Any]:
        """Reads an observation from the environment and returns an observation compatible with the output TensorDict.

        Args:
            observations (observation under a format dictated by the inner env): observation to be read.

        """
        if not isinstance(observations, (TensorDictBase, dict)):
            (key,) = itertools.islice(self.observation_spec.keys(True, True), 1)
            observations = {key: observations}
        return observations

    def _reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        td_out = super()._reset(tensordict)

        # Dummy actions before getting the first observation
        obs = None
        action = torch.zeros(self.env.unwrapped.action_space.shape, device=self.device)
        for _ in range(self._warmup_steps):
            obs = self.env.unwrapped.step(action)[0]

        if obs is not None and isinstance(obs, dict):
            for key in obs:
                td_out[key] = obs[key]

        self.env.unwrapped.render()

        return td_out

    def dump_env_cfg(self, path: str) -> None:
        """Dumps the environment configuration to a file.

        Args:
            path (str): the path to the file where the configuration will be dumped.

        """
        from omni.isaac.orbit.utils.io import dump_yaml

        dump_yaml(path, self.env.unwrapped.cfg)


class OrbitTorchRLEnv(OrbitTorchRLWrapper):
    """A TorchRL Env interface for IsaacGym environments.

    See :class:`~.OrbitTorchRLWrapper` for more information.

    Examples:
        >>> env = OrbitTorchRLEnv(task="Isaac-Ant-v0", num_envs=2000, device="cuda:0")
        >>> rollout = env.rollout(3)
        >>> assert env.batch_size == (2000,)

    """

    @_classproperty
    def available_envs(cls):
        if not _has_orbit:
            return

        if cls.libname == "gymnasium":
            import gymnasium as gym
        else:
            import gym

        orbit_envs = [
            task for task in gym.envs.registry.keys() if task.startswith("Isaac")
        ]
        yield orbit_envs

    def __init__(self, task=None, *, env=None, num_envs, device, config=None):
        if env is not None and task is not None:
            raise RuntimeError("Cannot provide both `task` and `env` arguments.")
        elif env is not None:
            task = env

        seed = config.pop("seed", None)
        envs = self._make_envs(
            task=task,
            num_envs=num_envs,
            device=device,
            seed=seed,
            config=config,
        )
        self.task = task
        super().__init__(envs, device=device)
