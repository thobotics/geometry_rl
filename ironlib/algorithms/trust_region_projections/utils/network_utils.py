from copy import deepcopy
from typing import Iterable, Sequence, Union

import gymnasium as gym
import numpy as np
import torch as ch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.vec_env import VecEnvWrapper, VecNormalize
from torch.optim.optimizer import Optimizer


def polyak_update(source: nn.Module, target: nn.Module, tau: float):
    """
    polyak weight update of target network.
    Args:
        source: source network to copy the weights from
        target: target network to copy the weights to
        tau: polyak weighting parameter

    Returns:

    """
    assert 0 <= tau <= 1, f"Tau has value {tau}, but needs to be 0 <= tau <= 1."
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def fanin_init(tensor, scale=1 / 3):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = np.sqrt(3 * scale / fan_in)
    return tensor.data.uniform_(-bound, bound)


def initialize_weights(mod, initialization_type, gain: float = 0.01, scale=1 / 3, init_w=3e-3):
    """
    Weight initializer for the models.
    Inputs: A model, Returns: none, initializes the parameters
    """
    for p in mod.parameters():
        if initialization_type == "normal":
            if len(p.data.shape) >= 2:
                p.data.normal_(init_w)  # 0.01
            else:
                p.data.zero_()
        elif initialization_type == "uniform":
            if len(p.data.shape) >= 2:
                p.data.uniform_(-init_w, init_w)
            else:
                p.data.zero_()
        elif initialization_type == "fanin":
            if len(p.data.shape) >= 2:
                fanin_init(p, scale)
            else:
                p.data.zero_()
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                nn.init.orthogonal_(p.data, gain=gain)
            else:
                p.data.zero_()
        else:
            raise ValueError("Need a valid initialization key")


def get_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    kernel_init: str = "orthogonal",
    activation_type: str = "tanh",
    layer_norm: bool = False,
    use_bias: bool = True,
):
    """
    create the hidden part of an MLP
    Args:
        input_dim: dimensionality of previous layer/input layer
        hidden_sizes: iterable of hidden unit sizes
        kernel_init: kernel initializer
        activation_type:
        layer_norm: use layer_norm with tanh
        use_bias: use bias of dense layer

    Returns: call on last hidden layer

    """

    activation = get_activation(activation_type)

    affine_layers = nn.ModuleList()

    prev = input_dim
    x = nn.Linear(prev, hidden_sizes[0], bias=use_bias)
    initialize_weights(x, kernel_init)
    affine_layers.append(x)
    prev = hidden_sizes[0]

    if layer_norm:
        x = nn.LayerNorm(prev)
        affine_layers.extend([x, ch.nn.Tanh()])
    else:
        affine_layers.append(activation)

    for i, l in enumerate(hidden_sizes[1:]):
        x = nn.Linear(prev, l, bias=use_bias)
        initialize_weights(x, kernel_init)
        affine_layers.extend([x, activation])
        prev = l

    return affine_layers


def get_activation(activation_type: str):
    if activation_type.lower() == "tanh":
        return nn.Tanh()
    elif activation_type.lower() == "relu":
        return nn.ReLU()
    elif activation_type.lower() == "leaky_relu":
        return nn.LeakyReLU()
    elif activation_type.lower() == "elu":
        return nn.ELU()
    elif activation_type.lower() == "prelu":
        return nn.PReLU()
    elif activation_type.lower() == "celu":
        return nn.CELU()
    else:
        raise ValueError(f"Optimizer {activation_type} is not supported.")


def get_optimizer(
    optimizer_type: str,
    model_parameters: Union[Iterable[ch.Tensor], Iterable[dict]],
    learning_rate: float,
    **kwargs,
):
    """
    Get optimizer instance for given model parameters
    Args:
        model_parameters:
        optimizer_type:
        learning_rate:
        **kwargs:

    Returns:

    """
    if optimizer_type.lower() == "sgd":
        return optim.SGD(model_parameters, learning_rate, **kwargs)
    elif optimizer_type.lower() == "sgd_momentum":
        momentum = kwargs.pop("momentum") if kwargs.get("momentum") else 0.9
        return optim.SGD(model_parameters, learning_rate, momentum=momentum, **kwargs)
    elif optimizer_type.lower() == "adam":
        return optim.Adam(model_parameters, learning_rate, **kwargs)
    elif optimizer_type.lower() == "adamw":
        return optim.AdamW(model_parameters, learning_rate, **kwargs)
    elif optimizer_type.lower() == "adagrad":
        return optim.adagrad.Adagrad(model_parameters, learning_rate, **kwargs)
    else:
        ValueError(f"Optimizer {optimizer_type} is not supported.")


def get_lr_schedule(
    schedule_type: str, optimizer: Optimizer, total_iters: int, **kwargs
) -> Union[optim.lr_scheduler._LRScheduler, None]:
    if not schedule_type or schedule_type.isspace():
        return None

    elif schedule_type.lower() == "linear":
        return optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_iters)

    elif schedule_type.lower() == "exp":
        # This scales the exp decay always to the same slope for different steps
        n_digits = len(str(total_iters)) - 1
        scaling = 5 + 5 - (5 / (total_iters / 10**n_digits))
        gamma = np.sum(9 * 10.0 ** -np.arange(1, n_digits)) + scaling * 10**-n_digits
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    elif schedule_type.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters)

    elif schedule_type.lower() == "papi":
        # Multiply learning rate with 0.8 every time the backtracking fails
        return optim.lr_scheduler.MultiplicativeLR(optimizer, lambda n_calls: 0.8)

    # elif schedule_type.lower() == "performance":
    elif "performance" in schedule_type.lower():
        decay = float(schedule_type.lower().split("_")[-1])
        return optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: decay), optim.lr_scheduler.MultiplicativeLR(
            optimizer, lambda epoch: 1.01
        )

    else:
        raise ValueError(
            f"Learning rate schedule {schedule_type} is not supported. Select one of [None, linear, papi, performance]."
        )


def sync_envs_normalization(env: gym.Env, eval_env: gym.Env) -> None:
    """
    https://github.com/DLR-RM/stable-baselines3/blob/852d635742e97495d200b104ab8e09f128211e26/stable_baselines3/common/vec_env/__init__.py#L58
    Sync eval env and train env when using VecNormalize
    :param env:
    :param eval_env:
    """
    env_tmp, eval_env_tmp = env, eval_env
    while isinstance(env_tmp, VecEnvWrapper):
        if isinstance(env_tmp, VecNormalize):
            # Only synchronize if observation normalization exists
            if hasattr(env_tmp, "obs_rms"):
                eval_env_tmp.obs_rms = deepcopy(env_tmp.obs_rms)
            eval_env_tmp.ret_rms = deepcopy(env_tmp.ret_rms)
        env_tmp = env_tmp.venv
        eval_env_tmp = eval_env_tmp.venv
