from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch as ch
import torch.nn as nn

from ..value.gnn_vf_net import GNNVFNet
from ...utils.network_utils import get_mlp, initialize_weights
from ...utils.torch_utils import inverse_softplus


class AbstractGaussianPolicy(nn.Module, ABC):
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
        use_tanh_mean: bool = False,
        share_weights=False,
        vf_model: GNNVFNet = None,
        minimal_std: float = 1e-5,
        scale: float = 1e-4,
        gain: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.contextual_std = contextual_std
        self.share_weights = share_weights
        self.minimal_std = ch.tensor(minimal_std)
        self.init_std = ch.tensor(init_std)
        self.use_tanh_mean = use_tanh_mean

        self._affine_layers = get_mlp(
            obs_dim, hidden_sizes, init, activation, layer_norm, True
        )

        prev_size = hidden_sizes[-1]

        self.diag_activation = nn.Softplus()
        self.diag_activation_inv = inverse_softplus

        # This shift is applied to the Parameter/cov NN output before applying the transformation
        # and gives hence the wanted initial cov
        self._pre_activation_shift = self._get_preactivation_shift(
            self.init_std, self.minimal_std
        )
        self._mean = self._get_mean(action_dim, prev_size, init, gain, scale)
        self._pre_std = self._get_std(
            contextual_std, action_dim, prev_size, init, gain, scale
        )
        if not trainable_std:
            assert (
                not self.contextual_std
            ), "Cannot freeze std while using a contextual std."
            self._pre_std.requires_grad_(False)

        self.vf_model = vf_model

        if share_weights:
            self.final_value = nn.Linear(prev_size, 1)
            initialize_weights(self.final_value, init, gain=1.0)

    @abstractmethod
    def forward(self, x, train=True, **kwargs):
        pass

    def get_value(self, x, train=True, **kwargs):
        if self.share_weights:
            self.train(train)
            for affine in self._affine_layers:
                x = affine(x)
            value = self.final_value(x).squeeze(-1)
        elif self.vf_model:
            value = self.vf_model(x, train)
        else:
            raise ValueError(
                "Must be sharing weights or use joint training to use get_value."
            )

        return value

    def squash(self, x):
        return x

    def _get_mean(self, action_dim, prev_size=None, init=None, gain=0.01, scale=1e-4):
        """
        Constructor method for mean prediction.
        Args:
            action_dim: action dimension for output shape
            prev_size: previous layer's output size
            init: initialization type of layer.
            scale

        Returns:

        """
        mean = nn.Linear(prev_size, action_dim)
        initialize_weights(mean, init, gain=gain, scale=scale)
        return mean

    # @final
    def _get_std(
        self,
        contextual_std: bool,
        action_dim,
        prev_size=None,
        init=None,
        gain=0.01,
        scale=1e-4,
    ) -> Union[nn.Parameter, nn.Module]:
        """
        Constructor method for std prediction. Do not overwrite.
        Args:
            contextual_std: whether to make the std context dependent or not
            action_dim: action dimension for output shape
            prev_size: previous layer's output size
            init: initialization type of layer.

        Returns:

        """
        if contextual_std:
            return self._get_std_layer(prev_size, action_dim, init, gain, scale)
        else:
            return self._get_std_parameter(action_dim)

    def _get_preactivation_shift(self, init_std, minimal_std):
        """
        Compute the prediction shift to enforce an initial covariance value for contextual and non contextual policies.
        Args:
            init_std: value to initalize the covariance output with.
            minimal_std: lower bound on the covariance.

        Returns:
            preactivation shift to enforce minimal and initial covariance.
        """
        return self.diag_activation_inv(init_std - minimal_std)

    @abstractmethod
    def _get_std_parameter(self, action_dim) -> nn.Parameter:
        """
        Creates a trainiable variable for predicting the std for a non contextual policy.
        Args:
            action_dim: action dimension for output shape

        Returns:
            torch trainable variable for covariance prediction.
        """
        pass

    @abstractmethod
    def _get_std_layer(
        self, prev_size, action_dim, init, gain=0.01, scale=1e-4
    ) -> nn.Module:
        """
        Creates a layer for predicting the std for a contextual policy.
        Args:
            gain:
            scale:
            prev_size: previous layer's output size
            action_dim: action dimension for output shape
            init: initialization type of layer.

        Returns:
            torch layer for covariance prediction.
        """
        pass

    @abstractmethod
    def sample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        """
        Given prob dist p=(mean, var),
         Args:
            p: tuple (means, var). means (batch_size, action_space), var (action_space,).
                p are batched probability distributions you're sampling from
            n: number of samples

        Returns:
            actions sampled from p_i (batch_size, action_dim)
        """
        pass

    @abstractmethod
    def rsample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        """
        Given prob dist p=(mean, var),
         Args:
            p: tuple (means, var). means (batch_size, action_space), var (action_space,).
                p are batched probability distributions you're sampling from
                This version applies the reparametrization trick and allows to backpropagate through it.
            n: number of samples
        Returns:
            actions sampled from p_i (batch_size, action_dim)
        """
        pass

    @abstractmethod
    def log_probability(
        self, p: Tuple[ch.Tensor, ch.Tensor], x: ch.Tensor, **kwargs
    ) -> ch.Tensor:
        """
        Computes the log probability of x given a batched distributions p (mean, std)
        Args:
            p: tuple (means, var). means (batch_size, action_space), var (action_space,).
            x:
            **kwargs:

        Returns:

        """
        pass

    @abstractmethod
    def entropy(self, p: Tuple[ch.Tensor, ch.Tensor]) -> ch.Tensor:
        """
        Get entropies over the probability distributions given by p = (mean, var).
        mean shape (batch_size, action_space), var shape (action_space,)
        """
        pass

    @abstractmethod
    def log_determinant(self, std: ch.Tensor) -> ch.Tensor:
        """
        Returns the log determinant of the std matrix
        Args:
            std: either a diagonal, cholesky, or sqrt matrix depending on the policy
        Returns:
            The log determinant of std, aka log sum the diagonal
        """
        pass

    @abstractmethod
    def maha(self, mean, mean_other, std) -> ch.Tensor:
        """
        Compute the mahalanbis distance between two means. std is the scaling matrix.
        Args:
            mean: left mean
            mean_other: right mean
            std: scaling matrix

        Returns:
            mahalanobis distance between mean and mean_other
        """
        pass

    @abstractmethod
    def precision(self, std: ch.Tensor) -> ch.Tensor:
        """
        Compute precision matrix given the std.
        Args:
            std: std matrix

        Returns:
            precision matrix
        """
        pass

    @abstractmethod
    def covariance(self, std) -> ch.Tensor:
        """
        Compute the full covariance matrix given the std.
        Args:
            std:

        Returns:

        """
        pass

    @abstractmethod
    def set_std(self, std: ch.Tensor) -> None:
        """
        For the NON-contextual case we do not need to regress the std, we can simply set it
        Args:
            std: projected std

        Returns:

        """
        pass

    def get_last_layer(self):
        """
        Returns last layer of network. Only required for the PAPI projection.
        Returns:

        """
        return self._affine_layers[-1].weight.data

    def papi_weight_update(self, eta: ch.Tensor, A: ch.Tensor):
        """
        Update the last layer alpha according to papi paper [Akrour et al., 2019]
        Args:
            eta: alpha
            A: intermediate policy alpha matrix

        Returns:

        """
        self._affine_layers[-1].weight.data *= eta
        self._affine_layers[-1].weight.data += (1 - eta) * A

    @property
    def is_root(self):
        """
        Whether policy is returning a full sqrt matrix as std.
        Returns:

        """
        return False

    @property
    def is_diag(self):
        """
        Whether the policy is returning a diagonal matrix as std.
        Returns:

        """
        return False
