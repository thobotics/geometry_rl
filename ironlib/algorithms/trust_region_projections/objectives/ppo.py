import contextlib

import math
import warnings
from dataclasses import dataclass
from typing import Tuple

import torch
from tensordict.nn import dispatch, ProbabilisticTensorDictSequential, TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.utils import NestedKey
from torch import distributions as d
from torchrl.objectives.ppo import PPOLoss, ClipPPOLoss
from torchrl.objectives.utils import distance_loss

from .utils import _clip_value_loss


class ClipPPOLoss2(ClipPPOLoss):

    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential | None = None,
        critic_network: TensorDictModule | None = None,
        *,
        clip_epsilon: float = 0.2,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        loss_critic_type: str = "smooth_l1",
        normalize_advantage: bool = True,
        gamma: float = None,
        separate_losses: bool = False,
        clip_value: float = None,
        **kwargs,
    ):
        super().__init__(
            actor_network=actor_network,
            critic_network=critic_network,
            clip_epsilon=clip_epsilon,
            entropy_bonus=entropy_bonus,
            samples_mc_entropy=samples_mc_entropy,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            loss_critic_type=loss_critic_type,
            normalize_advantage=normalize_advantage,
            gamma=gamma,
            separate_losses=separate_losses,
            **kwargs,
        )

        if clip_value is not None:
            if isinstance(clip_value, float):
                clip_value = torch.tensor(clip_value)
            elif isinstance(clip_value, torch.Tensor):
                if clip_value.numel() != 1:
                    raise ValueError(
                        f"clip_value must be a float or a scalar tensor, got {clip_value}."
                    )
            else:
                raise ValueError(
                    f"clip_value must be a float or a scalar tensor, got {clip_value}."
                )
        self.register_buffer("clip_value", clip_value)

    def loss_critic(self, tensordict: TensorDictBase) -> torch.Tensor:
        # TODO: if the advantage is gathered by forward, this introduces an
        # overhead that we could easily reduce.
        if self.separate_losses:
            tensordict = tensordict.detach()
        try:
            target_return = tensordict.get(self.tensor_keys.value_target)
        except KeyError:
            raise KeyError(
                f"the key {self.tensor_keys.value_target} was not found in the input tensordict. "
                f"Make sure you provided the right key and the value_target (i.e. the target "
                f"return) has been retrieved accordingly. Advantage classes such as GAE, "
                f"TDLambdaEstimate and TDEstimate all return a 'value_target' entry that "
                f"can be used for the value loss."
            )

        if self.clip_value:
            try:
                old_state_value = tensordict.get(self.tensor_keys.value)
            except KeyError:
                raise KeyError(
                    f"clip_value is set to {self.clip_value}, but "
                    f"the key {self.tensor_keys.value} was not found in the input tensordict. "
                    f"Make sure that the value_key passed to PPO exists in the input tensordict."
                )

        with (
            self.critic_network_params.to_module(self.critic_network)
            if self.functional
            else contextlib.nullcontext()
        ):
            state_value_td = self.critic_network(tensordict)

        try:
            state_value = state_value_td.get(self.tensor_keys.value)
        except KeyError:
            raise KeyError(
                f"the key {self.tensor_keys.value} was not found in the input tensordict. "
                f"Make sure that the value_key passed to PPO is accurate."
            )

        loss_value = distance_loss(
            target_return,
            state_value,
            loss_function=self.loss_critic_type,
        )

        clip_fraction = None
        if self.clip_value:
            loss_value, clip_fraction = _clip_value_loss(
                old_state_value,
                state_value,
                self.clip_value.to(state_value.device),
                target_return,
                loss_value,
                self.loss_critic_type,
            )
        return self.critic_coef * loss_value
