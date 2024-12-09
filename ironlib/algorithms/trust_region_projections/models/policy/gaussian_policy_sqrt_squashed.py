import torch as ch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .gaussian_policy_sqrt import GaussianPolicySqrt
from ..value.vf_net import VFNet
from ...utils.network_utils import initialize_weights


class GaussianPolicySqrtSquashed(GaussianPolicySqrt):
    """
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and a cholesky matrix, which parameterize a full gaussian distribution.
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        init,
        hidden_sizes=(64, 64),
        activation: str = "tanh",
        layer_norm: bool = False,
        contextual_std: bool = False,
        trainable_std: bool = True,
        init_std: float = 1.0,
        share_weights=False,
        vf_model: VFNet = None,
        minimal_std=1e-5,
        scale: float = 1e-4,
        gain: float = 0.01,
    ):
        super().__init__(
            obs_dim,
            action_dim,
            init,
            hidden_sizes,
            activation,
            layer_norm,
            contextual_std,
            trainable_std,
            init_std,
            share_weights,
            vf_model,
            minimal_std,
            scale,
            gain,
        )

        self.squash_fun = nn.Tanh()

    def _get_mean(self, action_dim, prev_size=None, init=None, gain=0.01, scale=1e-4):
        """initialize according to SAC paper/code"""
        mean = nn.Linear(prev_size, action_dim)
        initialize_weights(mean, "uniform", init_w=1e-3)
        return mean

    def _get_std_layer(
        self, prev_size: int, action_dim: int, init: str, gain=0.01, scale=1e-4
    ):
        """initialize according to SAC paper/code

        Args:
            gain:
            scale:
        """
        chol_shape = action_dim * (action_dim + 1) // 2
        std = nn.Linear(prev_size, chol_shape)
        initialize_weights(std, "uniform", init_w=1e-3)
        # reinitialize bias because the default from above assumes fixed value
        std.bias.data.uniform_(-1e-3, 1e-3)
        return std

    def log_probability(self, p: Tuple[ch.Tensor, ch.Tensor], x: ch.Tensor, **kwargs):
        """
        Adapted from
        https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L73

        This corrects the Gaussian log prob by computing log(1 - tanh(x)^2).

        log(1 - tanh(x)^2)
         = log(sech(x)^2)
         = 2 * log(sech(x))
         = 2 * log(2e^-x / (e^-2x + 1))
         = 2 * (log(2) - x - log(e^-2x + 1))
         = 2 * (log(2) - x - softplus(-2x))
        Args:
            p: distribution
            x: values
            **kwargs: optional pre_squash_x = arctanh(x)

        Returns: Corrected Gaussian log prob

        """
        pre_squash_x = kwargs.get("pre_squash_x")
        if pre_squash_x is None:
            pre_squash_x = ch.log((1 + x) / (1 - x)) / 2

        nll = super().log_probability(p, pre_squash_x)
        adjustment = -2.0 * (
            nll.new_tensor(2.0).log() - pre_squash_x - F.softplus(-2.0 * pre_squash_x)
        ).sum(dim=1)
        return nll + adjustment

    def squash(self, x) -> ch.Tensor:
        return self.squash_fun(x)

    def get_value(self, x, train=True):
        return x.new([-1.0])
