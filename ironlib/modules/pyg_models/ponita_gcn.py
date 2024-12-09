from typing import List, Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU


import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData

from .ponita.ponita import Ponita
from .ponita.utils.to_from_sphere import scalar_to_sphere, vec_to_sphere


class PonitaGCN(torch.nn.Module):
    def __init__(
        self,
        input_dim_node,
        output_dim,
        output_dim_vec,
        num_layers=2,
        hidden_dim=64,
        dropout=0.1,
        num_ori=20,
        degree=2,
        widening_factor=4,
        attention=False,
        ponita_dim=3,
        only_upper_hemisphere=False,
        **ignored,
    ):
        super(PonitaGCN, self).__init__()
        self.input_dim = input_dim_node
        self._device = None
        self._homogeneous_batch = None
        self._homogeneous_edge_index = None

        self.dim = ponita_dim
        self.output_dim = output_dim
        self.output_dim_vec = output_dim_vec

        self.ponita = Ponita(
            input_dim=input_dim_node,  # 4 + 3,  # 4 scalars and 3 vectors
            dim=ponita_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            output_dim_vec=output_dim_vec,
            num_ori=num_ori,
            basis_dim=None,
            degree=degree,
            widening_factor=widening_factor,
            layer_scale=None,
            multiple_readouts=False,
            last_feature_conditioning=False,
            task_level="node",
            attention=attention,
            only_upper_hemisphere=only_upper_hemisphere,
        )

        self.linear = Linear(hidden_dim, output_dim + output_dim_vec)

    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    def homogeneous_batch(self, graph: HeteroData):
        if self._homogeneous_batch is None or self._homogeneous_batch.size(0) != graph.num_nodes:
            batch_list = []
            for i in range(len(graph)):
                batch_list.append(torch.full((graph[i].num_nodes,), i, dtype=torch.long))
            self._homogeneous_batch = torch.cat(batch_list, dim=0).to(self.device)
        return self._homogeneous_batch

    def homogeneous_edge_index(self, graph: HeteroData):
        if self._homogeneous_edge_index is None or self._homogeneous_batch.size(0) != graph.num_nodes:
            edge_index_list = []
            accum_idx = 0
            for i in range(len(graph)):
                homo_graph = graph[i].to_homogeneous()
                edge_index = homo_graph.edge_index + accum_idx
                edge_index_list.append(edge_index)
                accum_idx += graph[i].num_nodes
            self._homogeneous_edge_index = torch.cat(edge_index_list, dim=1).to(self.device)
        return self._homogeneous_edge_index

    def forward(self, data, input_vector, **kwargs):
        return self.one_step(data, input_vector, **kwargs)

    def one_step(
        self,
        graph: HeteroData,
        u_dict: Dict[str, torch.Tensor],
        **ignored,
    ):
        with torch.no_grad():
            scalar_dict, vector_dict = u_dict
            x_scalar_list = []
            x_vec_list = []
            x_full_list = []
            pos_list = []
            edge_attr_list = []

            batch_size = len(graph)
            for node_type in graph.node_types:
                scalar = scalar_to_sphere(scalar_dict[node_type], self.ponita.ori_grid)

                vector = vector_dict[node_type].view(scalar.shape[0], -1, 3)
                vector = vector[..., :2] if self.dim == 2 else vector
                vector = vec_to_sphere(vector, self.ponita.ori_grid)

                x_scalar_list.append(scalar.reshape(batch_size, -1, *scalar.shape[1:]))
                x_vec_list.append(vector.reshape(batch_size, -1, *vector.shape[1:]))
                x_full_list.append(scalar_dict[node_type].reshape(batch_size, -1, *scalar_dict[node_type].shape[1:]))

                pos = graph[node_type].pos.reshape(batch_size, -1, 3)
                pos = pos[..., :2] if self.dim == 2 else pos
                pos_list.append(pos)

            scalar = torch.cat(x_scalar_list, dim=1).to(self.device)
            vector = torch.cat(x_vec_list, dim=1).to(self.device)
            pos = torch.cat(pos_list, dim=1).to(self.device)

            # for edge_type in graph.edge_types:
            #     edge_attr = graph[edge_type].edge_attr
            #     edge_attr_list.append(edge_attr.reshape(batch_size, -1, edge_attr.shape[-1]))

            # edge_attr = torch.cat(edge_attr_list, dim=1).to(self.device)  # (batch_size, num_edges, input_dim_edge)
            # edge_attr = edge_attr.reshape(-1, edge_attr.shape[-1])  # (batch_size * num_edges, input_dim_edge)

            edge_index = self.homogeneous_edge_index(graph)
            batch = self.homogeneous_batch(graph)

        # Forward pass
        x = torch.cat([scalar, vector], dim=-1)
        x = x.reshape(-1, *x.shape[2:])
        pos = pos.reshape(-1, pos.shape[2])
        hidden = self.ponita(x, pos, edge_index, batch=batch)
        hidden = hidden.reshape(batch_size, -1, *hidden.shape[1:])

        output = self.linear(hidden)
        out_scalar, out_vec = output.split([self.output_dim, self.output_dim_vec], dim=-1)

        # Average over the orientations
        hidden = hidden.mean(dim=-2)
        out_scalar = out_scalar.mean(dim=-2)
        out_vec = torch.einsum("b n o c, o d -> b n c d", out_vec, self.ponita.ori_grid) / self.ponita.num_ori

        # Mask out the output nodes
        hidden = hidden[:, graph.output_mask]
        out_scalar = out_scalar[:, graph.output_mask]  # (batch_size, num_points, dim_feedforward)
        out_vec = out_vec[:, graph.output_mask]
        out = out_vec * out_scalar.unsqueeze(-1)
        if self.dim == 2:
            out = torch.cat([out, torch.zeros_like(out[..., :1])], dim=-1)
        return out.reshape(-1, out.shape[-1]), hidden.reshape(-1, hidden.shape[-1])
