import torch.nn as nn
import torch as ch

from ...utils.network_utils import get_activation, get_mlp, initialize_weights


class VFNet(nn.Module):
    """
    An example value network, with support for arbitrarily many
    fully connected hidden layers (by default 2 * 64-neuron layers),
    maps a state of size (state_dim) -> a scalar value.
    """

    def __init__(
        self,
        input_dim,
        init="orthogonal",
        hidden_sizes=(64, 64),
        activation: str = "tanh",
        layer_norm: bool = False,
        n_atoms: int = -1,
        vmin: float = -1,
        vmax: float = 1,
    ):
        """
        Initializes the value network.
        Args:
            input_dim: the input dimension of the network (i.e dimension of state)
            init: initialization of layers
            hidden_sizes: an iterable of integers, each of which represents the size
                    of a hidden layer in the neural network.
            activation: activation of hidden layers
            layer_norm: use layer normalization with tanh after first layer
            n_atoms: when >0 distributional critic with this many bins is used
        Returns: Initialized Value network

        """

        super().__init__()
        # self.activation = get_activation(activation)
        self._affine_layers = get_mlp(input_dim, hidden_sizes, init, activation, layer_norm, True)

        if n_atoms > 0:
            self.n_atoms = n_atoms
            self.vmin = vmin
            self.vmax = vmax
            self.atoms = ch.linspace(vmin, vmax, self.n_atoms)

        self.final = self.get_final(hidden_sizes[-1], n_atoms if n_atoms > 0 else 1, init)

    def get_final(self, prev_size, output_dim, init, gain=1.0, scale=1e-4):
        final = nn.Linear(prev_size, output_dim)
        initialize_weights(final, init, scale=1.0)
        return final

    def forward(self, *x: ch.Tensor, train=True, **kwargs):
        """
        Performs inference using the value network.
        Inputs:
        - x, the state passed in from the agent
        Returns:
        - The scalar (float) value of that state, as estimated by the net
        """

        self.train(train)

        if len(x) > 1:
            x = ch.cat([*x], -1)

        if len(x.shape) == 3:
            batch_size, n_rollouts, _ = x.shape
            x = x.reshape(batch_size * n_rollouts, -1)
        else:
            batch_size, _ = x.shape
            n_rollouts = 1

        for affine in self._affine_layers:
            x = affine(x)

        values = self.final(x)

        # if self.n_atoms > 0:
        #     return ch.softmax(values, dim=-1)

        if n_rollouts > 1:
            values = values.reshape(batch_size, n_rollouts, 1)

        return values

    def get_value(self, x):
        return self(x)
