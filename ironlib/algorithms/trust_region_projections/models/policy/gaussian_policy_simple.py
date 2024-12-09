import numpy as np
import torch as ch
import torch.nn as nn
from typing import Tuple

from .abstract_gaussian_policy import AbstractGaussianPolicy
from ..value.vf_net import VFNet
from ...utils.network_utils import initialize_weights


class GaussianPolicySimple(AbstractGaussianPolicy):
    """
    A continuous policy using a Gaussian.
    The parameterizing tensor is a mean and std vector, which parameterize a diagonal gaussian distribution.
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        init,
        hidden_sizes=(64, 64),
        activation: str = "tanh",
        contextual_std: bool = False,
        init_std: float = 1.0,
        share_weights=False,
        vf_model: VFNet = None,
        minimal_std=1e-5,
    ):
        super().__init__(
            obs_dim,
            action_dim,
            init,
            hidden_sizes,
            activation,
            contextual_std,
            init_std,
            init_std=share_weights,
            share_weights=vf_model,
            vf_model=minimal_std,
        )

        # set layers to none to avoid confusion with other NN policies.
        self._affine_layers = None
        # initalize mean and std with correct size
        self._mean = (
            self._get_mean_layer(obs_dim, action_dim, init)
            if self.contextual_std
            else self._get_mean(action_dim, action_dim, init, scale=0.01)
        )
        self._pre_std = self._get_std(contextual_std, action_dim, action_dim, init)

    def _get_std_parameter(self, action_dim: int):
        chol_shape = action_dim * (action_dim + 1) // 2
        flat_chol = ch.normal(0, 0.01, (chol_shape,))
        return nn.Parameter(flat_chol)

    def _get_std_layer(
        self, prev_size: int, action_dim: int, init: str, gain=0.01, scale=1e-4
    ):
        chol_shape = action_dim * (action_dim + 1) // 2
        flat_chol = nn.Linear(prev_size, chol_shape)
        initialize_weights(flat_chol, init, gain=gain, scale=scale)
        return flat_chol

    def _get_mean(self, action_dim, prev_size=None, init=None, gain=0.01, scale=1e-4):
        # TODO maybe make this more configurable
        mean = ch.normal(0, scale, (action_dim,))
        return nn.Parameter(mean)

    def _get_mean_layer(self, prev_size, action_dim, init, gain=0.01, scale=1e-4):
        mean = nn.Linear(prev_size, action_dim)
        initialize_weights(mean, init, gain=gain, scale=scale)
        return mean

    def forward(self, x, train=True):
        self.train(train)

        std = self._pre_std(x) if self.contextual_std else self._pre_std
        std = self.diag_activation(std + self._pre_activation_shift) + self.minimal_std
        std = std.diag_embed().expand(x.shape[0], -1, -1)

        return self._mean(x), std

    def sample(self, p: Tuple[ch.Tensor, ch.Tensor]):
        means, std = p
        std = std.diagonal(dim1=-2, dim2=-1)
        return (means + ch.randn_like(means) * std).detach()

    def rsample(self, p: Tuple[ch.Tensor, ch.Tensor]):
        means, std = p
        std = std.diagonal(dim1=-2, dim2=-1)
        return means + ch.randn_like(means) * std

    def log_probability(self, p: Tuple[ch.Tensor, ch.Tensor], x: ch.Tensor, **kwargs):
        mean, std = p
        k = x.shape[-1]

        maha_part = self.maha(x, mean, std)
        const = np.log(2.0 * np.pi) * k
        logdet = self.log_determinant(std)

        nll = -0.5 * (maha_part + const + logdet)
        return nll

    def entropy(self, p: Tuple[ch.Tensor, ch.Tensor]):
        _, std = p
        logdet = self.log_determinant(std)
        k = std.shape[-1]
        return 0.5 * (k * np.log(2 * np.e * np.pi) + logdet)

    def log_determinant(self, std: ch.Tensor):
        """
        Returns the log determinant of a diagonal matrix
        Inputs:
        - mat, a diagonal matrix
        Returns:
        - The determinant of mat, aka product of the diagonal
        """
        std = std.diagonal(dim1=-2, dim2=-1)
        return 2 * std.log().sum(-1)

    def maha(self, mean: ch.Tensor, mean_other: ch.Tensor, std: ch.Tensor):
        std = std.diagonal(dim1=-2, dim2=-1)
        diff = mean - mean_other
        return (diff / std).pow(2).sum(-1)

    def precision(self, std: ch.Tensor):
        return (1 / self.covariance(std).diagonal(dim1=-2, dim2=-1)).diag_embed()

    def covariance(self, std: ch.Tensor):
        return std.pow(2)

    def set_std(self, std: ch.Tensor) -> None:
        assert not self.contextual_std
        self._pre_std.data = (
            self.diag_activation_inv(std.diagonal() - self.minimal_std)
            - self._pre_activation_shift
        )

    @property
    def is_diag(self):
        return True
