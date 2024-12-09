import torch as ch
import torch.nn as nn
import torch.nn.functional as F

from .gaussian_policy_diag import GaussianPolicyDiag
from ..value.vf_net import VFNet
from ...utils.network_utils import initialize_weights


class GaussianPolicyDiagSquashed(GaussianPolicyDiag):
    """
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and std vector, which parameterize a diagonal gaussian distribution.
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
        use_tanh_mean=False,
    ):
        super().__init__(
            obs_dim,
            action_dim,
            init,
            hidden_sizes,
            activation,
            layer_norm,
            contextual_std,
            trainable_std=trainable_std,
            init_std=init_std,
            share_weights=share_weights,
            vf_model=vf_model,
            minimal_std=minimal_std,
            scale=scale,
            gain=gain,
            use_tanh_mean=False,
        )

        self.squash_fun = nn.Tanh()

    def _get_mean(self, action_dim, prev_size=None, init=None, gain=0.01, scale=1e-4):
        """initialize according to SAC paper/code"""
        mean = nn.Linear(prev_size, action_dim)
        initialize_weights(mean, "uniform", init_w=1e-5)
        return mean

    def _get_std_layer(self, prev_size, action_dim, init, gain=0.01, scale=1e-4):
        """initialize according to SAC paper/code

        Args:
            gain:
            scale:
        """
        std = nn.Linear(prev_size, action_dim)
        initialize_weights(std, "uniform", init_w=1e-3)
        # reinitialize bias because the default from above assumes fixed value
        std.bias.data.uniform_(-1e-3, 1e-3)
        return std

    def log_probability(self, p, x, **kwargs):
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
            eps = ch.finfo(x.dtype).eps
            x = x.clamp(min=-1.0 + eps, max=1.0 - eps)
            # atanh
            pre_squash_x = 0.5 * (x.log1p() - (-x).log1p())

        nll = super().log_probability(p, pre_squash_x)
        adjustment = -2.0 * (
            nll.new_tensor(2.0).log() - pre_squash_x - F.softplus(-2.0 * pre_squash_x)
        ).sum(dim=-1)
        return nll + adjustment

    def squash(self, x) -> ch.Tensor:
        return self.squash_fun(x)

    def get_value(self, x, train=True):
        return x.new([-1.0])
