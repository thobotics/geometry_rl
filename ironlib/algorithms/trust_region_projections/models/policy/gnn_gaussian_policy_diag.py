from typing import Tuple, Dict

import numpy as np
import torch as ch
import torch.nn as nn

from .abstract_gnn_gaussian_policy import (
    AbstractGNNGaussianPolicy,
)
from ...utils.network_utils import initialize_weights


class GNNGaussianPolicyDiag(AbstractGNNGaussianPolicy):
    """
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and std vector, which parameterize a diagonal gaussian distribution.
    """

    def _get_std_parameter(self, action_dim, scale=0.01):
        std = ch.normal(0, scale, (action_dim,))
        return nn.Parameter(std)

    def _get_std_layer(self, prev_size, action_dim, init, gain=0.01, scale=1e-4):
        std = nn.Linear(prev_size, action_dim)
        initialize_weights(std, init, gain=gain, scale=scale)
        return std

    def forward(
        self,
        *args,
        train=True,
    ):
        self.train(train)

        batch_size = args[0].shape[0]

        a_out = self.mgn_forward(
            *args,
            train=train,
        )

        if isinstance(self.action_dim, list):
            means = []
            stds = []
            a_outs = a_out.view(batch_size, -1, a_out.shape[1])
            for i in range(self.num_actuators):
                std = self._pre_std[i](a_outs[:, i]) if self.contextual_std else self._pre_std[i]
                std = self.diag_activation(std + self._pre_activation_shift[i]) + self.minimal_std
                # std = std.clamp(max=STD_MAX)
                if not self.contextual_std:
                    std = std.tile((batch_size, 1))
                std = std.reshape(batch_size, -1)

                mean = self._mean[i](a_outs[:, i])

                if self.use_tanh_mean:
                    mean = nn.functional.tanh(mean)

                means.append(mean)
                stds.append(std)

            mean = ch.cat(means, dim=-1)

            std = ch.cat(stds, dim=-1)
            std = std.diag_embed().expand((batch_size, -1, -1))
        else:
            if self.post_fc:
                hidden = a_out
            else:
                mean, hidden = a_out

            std = self._pre_std(hidden) if self.contextual_std else self._pre_std
            std = self.diag_activation(std + self._pre_activation_shift) + self.minimal_std
            # std = std.clamp(max=STD_MAX)
            if not self.contextual_std:
                std = std.tile((hidden.shape[0], 1))

            std = std.reshape(batch_size, -1)
            std = std.diag_embed().expand((batch_size, -1, -1))

            if self.post_fc:
                mean = self._mean(hidden)

                if self.use_tanh_mean:
                    mean = nn.functional.tanh(mean)

        mean = mean.reshape(batch_size, -1)

        return mean, std**2

    def sample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        return self.rsample(p, n).detach()

    def rsample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        means, std = p
        std = std.diagonal(dim1=-2, dim2=-1)
        eps = ch.randn((n,) + means.shape, dtype=std.dtype, device=std.device)
        samples = means + eps * std
        # squeeze when n == 1
        return samples.squeeze(0)

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
        Args:
            std: a diagonal matrix
        Returns:
            The log determinant of std, aka log sum the diagonal
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
        # avoid 0 for diagonal elements -->softplus^-1 fails
        shifted_min = self.minimal_std + ch.finfo(std.dtype).eps
        std_min = std.diagonal().clamp(min=shifted_min.to(std.device)) - self.minimal_std.to(std.device)
        self._pre_std.data = self.diag_activation_inv(std_min) - self._pre_activation_shift

    @property
    def is_diag(self):
        return True
