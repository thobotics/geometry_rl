from abc import abstractmethod
from typing import Dict

import torch
import torch as ch

from ..value.gnn_vf_net import GNNVFNet

from .abstract_gaussian_policy import AbstractGaussianPolicy
from ...utils.torch_utils import inverse_softplus

from ironlib.modules.pyg_models.gnn.base_gnn import BaseGNN
from ironlib.modules.pyg_data.base_data import BaseData


class AbstractGNNGaussianPolicy(AbstractGaussianPolicy):
    def __init__(
        self,
        gnn: BaseGNN,
        hyper_data: BaseData,
        action_dim,
        num_actuators,
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
        share_action_dim: bool = True,
        post_fc: bool = True,
        **kwargs,
    ):
        super(AbstractGaussianPolicy, self).__init__()

        self.action_dim = action_dim
        self.contextual_std = contextual_std
        self.share_weights = share_weights
        self.minimal_std = torch.tensor(minimal_std)
        self.init_std = torch.tensor(init_std)
        self.use_tanh_mean = use_tanh_mean
        self.post_fc = post_fc

        prev_size = hidden_sizes[-1]

        self.diag_activation = torch.nn.Softplus()
        self.diag_activation_inv = inverse_softplus

        device = gnn.device

        if isinstance(action_dim, list):
            self.action_dim = action_dim
            prev_size = hidden_sizes[-1]

            self._pre_activation_shift = []
            self._mean = []
            self._pre_std = []
            for a_dim in action_dim:
                _pre_activation_shift = self._get_preactivation_shift(self.init_std, self.minimal_std).to(device)
                _mean = self._get_mean(a_dim, prev_size, init, gain, scale).to(device)
                _pre_std = self._get_std(contextual_std, a_dim, prev_size, init, gain, scale).to(device)
                if not trainable_std:
                    assert not self.contextual_std, "Cannot freeze std while using a contextual std."
                    _pre_std.requires_grad_(False)

                self._pre_activation_shift.append(_pre_activation_shift)
                self._mean.append(_mean)
                self._pre_std.append(_pre_std)
        else:
            if share_action_dim:
                action_dim_shared = action_dim // num_actuators
            else:
                action_dim_shared = action_dim
            self._pre_activation_shift = self._get_preactivation_shift(self.init_std, self.minimal_std)
            self._mean = self._get_mean(action_dim_shared, prev_size, init, gain, scale)
            self._pre_std = self._get_std(contextual_std, action_dim_shared, prev_size, init, gain, scale)
            if not trainable_std:
                assert not self.contextual_std, "Cannot freeze std while using a contextual std."
                self._pre_std.requires_grad_(False)

        self.num_actuators = num_actuators
        self.hyper_data = hyper_data
        self.gnn = gnn

    @abstractmethod
    def forward(self, x, infos: Dict, train=True):
        pass

    def gnn_forward(
        self,
        *args,
        train=True,
    ):
        # Create the graph
        data, input_vector = self.hyper_data.build_data(
            *args,
            train=train,
        )

        return self.gnn.one_step(
            data,
            input_vector,
        )

    def get_value(
        self,
        observation: ch.Tensor,
        face_indices: ch.Tensor,
        key_point_indices: ch.Tensor,
        mesh_velocities: ch.Tensor = None,
        fixed_bodies: ch.Tensor = None,
        actuator_velocities: ch.Tensor = None,
        train=True,
    ):
        pass
