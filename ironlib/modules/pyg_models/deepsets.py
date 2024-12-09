from typing import Dict, List

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP


class DeepSets(nn.Module):
    def __init__(
        self,
        input_dim_node,
        output_dim,
        hidden_dim=64,
        norm: List[str] = [None, None],
        **ignored,
    ):
        super(DeepSets, self).__init__()
        self.input_dim = input_dim_node
        self._device = None

        self.mlp_inner = MLP([input_dim_node, hidden_dim, hidden_dim], norm=norm[0])
        self.mlp_outer = MLP([hidden_dim, hidden_dim, output_dim], norm=norm[1])

    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    def forward(self, data, input_vector, **kwargs):
        self.one_step(data, input_vector, **kwargs)

    def one_step(
        self,
        graph: HeteroData,
        u_dict: Dict[str, torch.Tensor],
        **ignored,
    ):
        with torch.no_grad():
            x_list = []

            batch_size = len(graph)
            for node_type in graph.node_types:
                u = u_dict[node_type]
                x_list.append(u.reshape(batch_size, -1, u.shape[-1]))

            x = torch.cat(x_list, dim=1).to(self.device)

        x = self.mlp_inner(x)
        x = x.sum(dim=1)  # Sum pooling over the sequence length dimension
        x = self.mlp_outer(x)
        return x
