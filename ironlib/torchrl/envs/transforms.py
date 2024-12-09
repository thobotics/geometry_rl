from __future__ import annotations

import collections
import multiprocessing as mp
from copy import copy
from typing import Any, Dict, List, Optional, OrderedDict, Sequence, Tuple, Union
from torchrl.envs.transforms import Transform, VecNorm, Compose, ObservationNorm
from torchrl.envs.transforms.transforms import (
    _apply_to_composite,
    _set_missing_tolerance,
)
from torchrl.envs.common import _EnvPostInit, EnvBase, make_tensordict
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.utils import expand_as_right, NestedKey

import numpy as np

import torch

from torchrl.data.tensor_specs import (
    BoundedTensorSpec,
    TensorSpec,
    CompositeSpec,
)


def reshape_tensor_shape(original_shape, new_shape):
    # Compute the total number of elements in the original shape
    total_elements = 1
    for dim in original_shape:
        total_elements *= dim

    # Calculate the product of the dimensions in the new shape, excluding -1
    product_of_new_dims = 1
    for dim in new_shape:
        if dim != -1:
            product_of_new_dims *= dim

    # Replace -1 in the new shape with the appropriate value
    reshaped_shape = []
    for dim in new_shape:
        if dim == -1:
            # Calculate the missing dimension
            missing_dim = total_elements // product_of_new_dims
            reshaped_shape.append(missing_dim)
        else:
            reshaped_shape.append(dim)

    return reshaped_shape


def _max_left(val, dest):
    while val.ndimension() > dest.ndimension():
        val = val.max(0).values
    return val


def _min_left(val, dest):
    while val.ndimension() > dest.ndimension():
        val = val.min(0).values
    return val


def _count_left(val, dest):
    count = 1  # Initialize count
    while val.ndimension() > dest.ndimension():
        count *= val.size(0)
        val = val[0]
    return count


class ReshapeTransform(Transform):
    """A transform to reshape a tensor.

    Args:
        in_keys (list of NestedKeys): input entries (read)
        out_keys (list of NestedKeys): input entries (write)
        in_keys_inv (list of NestedKeys): input entries (read) during :meth:`~.inv` calls.
        out_keys_inv (list of NestedKeys): input entries (write) during :meth:`~.inv` calls.

    Keyword Args:
        shape (tuple): shape to reshape to. Note that the batch dimension is not included.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> base_env = GymEnv("Pendulum-v1")
        >>> env = TransformedEnv(base_env, ReshapeTransform(in_keys=['observation'], shape=(-1, 3))
        >>> r = env.rollout(100)
        >>> assert (r["observation"] <= 0.1).all()
    """

    def __init__(
        self,
        in_keys=None,
        out_keys=None,
        in_keys_inv=None,
        out_keys_inv=None,
        *,
        out_shape: tuple = None,
    ):
        if in_keys is None:
            in_keys = []
        if out_keys is None:
            out_keys = copy(in_keys)
        if in_keys_inv is None:
            in_keys_inv = []
        if out_keys_inv is None:
            out_keys_inv = copy(in_keys_inv)
        super().__init__(in_keys, out_keys, in_keys_inv, out_keys_inv)
        if out_shape is None:
            raise ValueError("shape must be specified")
        self.out_shape = out_shape
        self._original_shape = None

    def _apply_transform(self, obs: torch.Tensor) -> None:
        self._original_shape = obs.shape
        return obs.reshape([obs.shape[0]] + list(self.out_shape))

    def _inv_apply_transform(self, state: torch.Tensor) -> torch.Tensor:
        return state.reshape(self._original_shape)

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        observation_spec.shape = reshape_tensor_shape(
            observation_spec.shape, [observation_spec.shape[0]] + list(self.out_shape)
        )
        return observation_spec

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset


class NDVecNorm(VecNorm):
    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.lock is not None:
            self.lock.acquire()

        for key, out_key in zip(self.in_keys, self.out_keys):
            key_str = self._key_str(key)
            if key not in tensordict.keys(include_nested=True):
                continue
            self._init(tensordict, key)
            # update and standardize
            new_val = self._update(
                key,
                tensordict.get(key),
                N=max(1, _count_left(tensordict.get(key), self._td.get(key_str + "_sum"))),
            )

            tensordict.set(out_key, new_val)

        if self.lock is not None:
            self.lock.release()

        return tensordict

    def transform_output_spec(self, output_spec: CompositeSpec) -> CompositeSpec:
        for observation_key in self.parent.full_observation_spec.keys(True):
            if observation_key in self.in_keys:
                for i, out_key in enumerate(self.out_keys):  # noqa: B007
                    if self.in_keys[i] == observation_key:
                        break
                else:
                    # unreachable
                    raise RuntimeError
                output_spec["full_observation_spec"][out_key] = output_spec["full_observation_spec"][
                    observation_key
                ].clone()
        return output_spec


class MinMaxNorm(Transform):
    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        shared_td: Optional[TensorDictBase] = None,
        lock: mp.Lock = None,
        shapes: List[torch.Size] = None,
        min: float = -1.0,
        max: float = 1.0,
    ) -> None:
        if lock is None:
            lock = mp.Lock()
        if in_keys is None:
            in_keys = ["observation", "reward"]
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self._td = shared_td
        if shared_td is not None and not (shared_td.is_shared() or shared_td.is_memmap()):
            raise RuntimeError("shared_td must be either in shared memory or a memmap " "tensordict.")
        if shared_td is not None:
            for key in in_keys:
                if (
                    (key + "_max" not in shared_td.keys())
                    or (key + "_min" not in shared_td.keys())
                    or (key + "_count" not in shared_td.keys())
                ):
                    raise KeyError(f"key {key} not present in the shared tensordict " f"with keys {shared_td.keys()}")

        self.lock = lock
        self.shapes = shapes
        self.min_range = min
        self.max_range = max

    def _key_str(self, key):
        if not isinstance(key, str):
            key = "_".join(key)
        return key

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.lock is not None:
            self.lock.acquire()

        for key in self.in_keys:
            if key not in tensordict.keys(include_nested=True):
                continue
            self._init(tensordict, key)
            # update and standardize
            new_val = self._update(
                key,
                tensordict.get(key),
                N=max(1, _count_left(tensordict.get(key), self._td.get(key + "_max"))),
            )

            tensordict.set(key, new_val)

        if self.lock is not None:
            self.lock.release()

        return tensordict

    forward = _call

    def _init(self, tensordict: TensorDictBase, key: str) -> None:
        key_str = self._key_str(key)
        if self._td is None or key_str + "_max" not in self._td.keys():
            if key is not key_str and key_str in tensordict.keys():
                raise RuntimeError(f"Conflicting key names: {key_str} from VecNorm and input tensordict keys.")
            if self.shapes is None:
                td_view = tensordict.view(-1)
                td_select = td_view[0]
                item = td_select.get(key)
                d = {key_str + "_max": torch.zeros_like(item)}
                d.update({key_str + "_min": torch.zeros_like(item)})
            else:
                idx = 0
                for in_key in self.in_keys:
                    if in_key != key:
                        idx += 1
                    else:
                        break
                shape = self.shapes[idx]
                item = tensordict.get(key)
                d = {key_str + "_max": torch.zeros(shape, device=item.device, dtype=item.dtype)}
                d.update({key_str + "_min": torch.zeros(shape, device=item.device, dtype=item.dtype)})

            d.update({key_str + "_count": torch.zeros(1, device=item.device, dtype=torch.float)})
            if self._td is None:
                self._td = TensorDict(d, batch_size=[])
            else:
                self._td.update(d)
        else:
            pass

    def _update(self, key, value, N) -> torch.Tensor:
        key = self._key_str(key)
        _max = self._td.get(key + "_max")
        _min = self._td.get(key + "_min")
        _count = self._td.get(key + "_count")

        _max = self._td.get(key + "_max")
        value_max = _max_left(value, _max)
        _max = torch.max(_max, value_max)
        self._td.set_(
            key + "_max",
            _max,
        )

        _min = self._td.get(key + "_min")
        value_min = _min_left(value, _min)
        _min = torch.min(_min, value_min)
        self._td.set_(
            key + "_min",
            _min,
        )

        _count = self._td.get(key + "_count")
        _count += N
        self._td.set_(
            key + "_count",
            _count,
        )

        scale = (self.max_range - self.min_range) / torch.max(_max - _min)
        return (value - _min) * scale + self.min_range

    def to_observation_norm(self) -> Union[Compose, ObservationNorm]:
        """Converts VecNorm into an ObservationNorm class that can be used at inference time."""
        out = []
        for key in self.in_keys:
            _max = self._td.get(key + "_max")
            _min = self._td.get(key + "_min")
            _count = self._td.get(key + "_count")

            scale = (self.max_range - self.min_range) / torch.max(_max - _min)
            loc = -_min * scale + self.min_range

            # obs = obs * scale + loc
            _out = ObservationNorm(
                loc=loc,
                scale=scale,
                standard_normal=False,
                in_keys=self.in_keys,
            )
            if len(self.in_keys) == 1:
                return _out
            else:
                out += ObservationNorm
        return Compose(*out)

    @staticmethod
    def build_td_for_shared_vecnorm(
        env: EnvBase,
        keys: Optional[Sequence[str]] = None,
        memmap: bool = False,
    ) -> TensorDictBase:
        raise NotImplementedError("this feature is currently put on hold.")

    def get_extra_state(self) -> OrderedDict:
        return collections.OrderedDict({"lock": self.lock, "td": self._td})

    def set_extra_state(self, state: OrderedDict) -> None:
        lock = state["lock"]
        if lock is not None:
            """
            since locks can't be serialized, we have use cases for stripping them
            for example in ParallelEnv, in which case keep the lock we already have
            to avoid an updated tensor dict being sent between processes to erase locks
            """
            self.lock = lock
        td = state["td"]
        if td is not None and not td.is_shared():
            raise RuntimeError("Only shared tensordicts can be set in VecNorm transforms")
        self._td = td

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"keys={self.in_keys})"

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        _lock = state.pop("lock", None)
        if _lock is not None:
            state["lock_placeholder"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]):
        if "lock_placeholder" in state:
            state.pop("lock_placeholder")
            _lock = mp.Lock()
            state["lock"] = _lock
        self.__dict__.update(state)
