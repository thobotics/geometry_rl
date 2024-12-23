import torch
import torch.nn as nn

from geometry_rl.modules.pyg_models.gnn.base_gnn import BaseGNN
from geometry_rl.modules.pyg_data.base_data import BaseData


class GNNVFNet(nn.Module):
    """
    An example value network, with support for arbitrarily many
    fully connected hidden layers (by default 2 * 64-neuron layers),
    maps a state of size (state_dim) -> a scalar value.
    """

    def __init__(
        self,
        gnn: BaseGNN,
        hyper_data: BaseData,
        init="orthogonal",
        hidden_sizes=(64, 64),
        activation: str = "tanh",
        layer_norm: bool = False,
        mesh_pos_obs: bool = False,
        actuator_vel_obs: bool = False,
        **kwargs,
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

        self.mesh_pos_obs = mesh_pos_obs
        self.actuator_vel_obs = actuator_vel_obs

        self.hyper_data = hyper_data
        self.gnn = gnn
        self.final = nn.Linear(hidden_sizes[-1], 1)

    def forward(
        self,
        *args,
        train=True,
    ):
        """
        Performs inference using the value network.
        Inputs:
        - x, the state passed in from the agent
        Returns:
        - The scalar (float) value of that state, as estimated by the net
        """

        self.train(train)

        if len(args[0].shape) == 3:
            batch_size, n_rollouts, _ = args[0].shape
        else:
            batch_size, _ = args[0].shape
            n_rollouts = 1
            args = [arg.unsqueeze(1) for arg in args]

        c_outs = []
        for i in range(n_rollouts):
            c_out = self.gnn_forward(
                *[arg[:, i] for arg in args],
                train=train,
            )
            c_outs.append(c_out)

        c_outs = torch.stack(c_outs, dim=1)
        values = self.final(c_outs)

        if n_rollouts == 1:
            values = values.squeeze(1)

        return values

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
