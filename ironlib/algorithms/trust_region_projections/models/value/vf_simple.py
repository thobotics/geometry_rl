import torch.nn as nn
import torch as ch


class VFSimple(nn.Module):
    """
    An example value network, with support for arbitrarily many
    fully connected hidden layers (by default 2 * 64-neuron layers),
    maps a state of size (state_dim) -> a scalar value.
    """

    def __init__(self, output_dim=1, init_std=1e-3):
        """
        Initializes the value network.
        Args:
            input_dim: the input dimension of the network (i.e dimension of state)
            output_dim: number of output nodes
            init: initialization of layers
            hidden_sizes: an iterable of integers, each of which represents the size
                    of a hidden layer in the neural network.
            activation: activation of hidden layers
        Returns: Initialized Value network

        """
        """

        """
        super().__init__()
        self.final = ch.normal(0, init_std, (output_dim,), requires_grad=True)

    def forward(self, x, train=True):
        return self.final.expand(x.shape[0])

    def get_value(self, x):
        return self(x)
