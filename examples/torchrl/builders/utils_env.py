from typing import Any, Dict, List, Optional
import hydra
from tensordict import TensorDictBase
import torch

from torchrl.envs import TransformedEnv
from ironlib.orbit.wrappers.torchrl import OrbitTorchRLEnv

import ast
from omegaconf import OmegaConf


def tuples_resolver(s: str) -> list:
    # Convert the string to a list of tuples using `ast.literal_eval`
    try:
        tuples = ast.literal_eval(s)
        if isinstance(tuples, list) and all(isinstance(item, tuple) for item in tuples):
            return tuples
        else:
            raise ValueError("Input string does not represent a list of tuples")
    except:
        raise ValueError("Invalid input format for tuples resolver")


# Register the resolver with Hydra
OmegaConf.register_new_resolver("tuples", tuples_resolver)


class orbit_info_dict_reader:
    def __init__(
        self,
        root_key: str,
        prefix_key_name: Optional[str] = None,
    ):
        self.root_key = root_key
        self.prefix_key_name = prefix_key_name
        self._info_spec = dict()

    def __call__(self, info_dict: Dict[str, Any], tensordict: TensorDictBase) -> TensorDictBase:
        for key, value in info_dict.items():
            if key == self.root_key:
                for k, v in value.items():
                    k_str = f"{self.root_key}/{k}"
                    if k_str not in self._info_spec:
                        self._info_spec[k_str] = v

                    if self.prefix_key_name and k.startswith(self.prefix_key_name):
                        tensordict[k_str] = v
        return tensordict

    def reset(self):
        self._info_spec = dict()

    @property
    def info_spec(self):
        return self._info_spec


def make_orbit_env(
    env_name="Isaac-Cartpole-v0",
    device="cuda",
    batch_size=1,
    env_config=None,
):
    transform_config = env_config.pop("transform", None)

    orbit_env_cls = OrbitTorchRLEnv

    env = orbit_env_cls(
        env_name,
        device=device,
        num_envs=batch_size,
        config=env_config,
    )

    if env_config["env_info"]:
        env.set_info_dict_reader(
            orbit_info_dict_reader(
                root_key=env_config["env_info"]["root_key"],
                prefix_key_name=env_config["env_info"]["prefix_key_name"],
            )
        )

    if transform_config is not None:
        env = TransformedEnv(env)

        for transform_cfg in transform_config:
            transform_obj = hydra.utils.instantiate(transform_cfg)

            keys = ["in_keys", "out_keys", "in_keys_inv", "out_keys_inv"]

            for key in keys:
                if hasattr(transform_obj, key):
                    key_dict = getattr(transform_obj, key)
                    for i, k in enumerate(key_dict):
                        try:
                            key_dict[i] = ast.literal_eval(k)
                        except:
                            continue

            env.append_transform(transform_obj)

    return env
