import torch as ch
import torch.nn as nn

from .gaussian_policy_full import GaussianPolicyFull
from ..value.vf_net import VFNet


class GaussianPolicySqrt(GaussianPolicyFull):
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
            init_std=init_std,
            share_weights=share_weights,
            vf_model=vf_model,
            minimal_std=minimal_std,
            scale=scale,
            gain=gain,
        )

        self.diag_activation = nn.Softplus()

    def forward(self, x: ch.Tensor, train: bool = True):
        mean, chol = super(GaussianPolicySqrt, self).forward(x, train)
        sqrt = chol @ chol.transpose(-2, -1)

        return mean, sqrt

    def log_determinant(self, std):
        """
        Returns the log determinant of a sqrt matrix
        Args:
            std: sqrt matrix
        Returns:
            The log determinant of std, aka log sum the diagonal

        """
        return 4 * ch.cholesky(std).diagonal(dim1=-2, dim2=-1).log().sum(-1)

    def maha(self, mean, mean_other, std):
        diff = (mean - mean_other)[..., None]
        return (ch.linalg.solve(std, diff) ** 2).sum([-2, -1])

    def precision(self, std):
        cov = self.covariance(std)
        return ch.linalg.solve(
            cov, ch.eye(cov.shape[-1], dtype=std.dtype, device=std.device)
        )

    def covariance(self, std: ch.Tensor):
        return std @ std

    def _get_preactivation_shift(self, init_std, minimal_std):
        return self.diag_activation_inv(ch.sqrt(init_std) - ch.sqrt(minimal_std))

    def set_std(self, std: ch.Tensor) -> None:
        std = ch.cholesky(std, upper=False)
        super(GaussianPolicySqrt, self).set_std(std)

    @property
    def is_root(self):
        return True
