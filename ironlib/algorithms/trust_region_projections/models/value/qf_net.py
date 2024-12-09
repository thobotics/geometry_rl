import torch as ch
import torch.nn as nn

from .vf_net import VFNet
from ...utils.network_utils import initialize_weights, get_mlp


class QFNet(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=1,
        init="fanin",
        hidden_sizes=(256, 256),
        activation: str = "relu",
        layer_norm: bool = False,
    ):
        """
        Initializes the value network.
        Args:
            input_dim: the input dimension of the network (i.e dimension of state)
            output_dim: number of output nodes
            init: initialization of layers
            hidden_sizes: an iterable of integers, each of which represents the size
                    of a hidden layer in the neural network.
            activation: activation of hidden layers
            layer_norm: use layer normalization with tanh after first layer
        Returns: Initialized Value network

        """
        super().__init__()
        # self.activation = get_activation(activation)
        self._affine_layers = get_mlp(
            input_dim, hidden_sizes, init, activation, layer_norm, True
        )

        self.final = self.get_final(hidden_sizes[-1], output_dim, init)

    def get_final(self, prev_size, output_dim, init, gain=1.0, scale=1 / 3):
        final = nn.Linear(prev_size, output_dim)
        # initialize_weights(final, "uniform", init_w=3e-3)
        return final

    def forward(self, x, train=True):
        self.train(train)
        x = ch.cat(x, dim=-1)

        for affine in self._affine_layers:
            x = affine(x)
        return self.final(x).squeeze(-1)
