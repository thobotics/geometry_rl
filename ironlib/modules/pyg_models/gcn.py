from typing import List, Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU


import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import MLP, fps, global_max_pool, radius, MessagePassing, GCNConv, GATv2Conv
from torch_geometric.utils import add_self_loops

from con_mgn.models.networks.mpnn import ProcessorLayer


class GCN(torch.nn.Module):
    def __init__(
        self,
        input_dim_node,
        output_dim,
        concat_global=False,
        **ignored,
    ):
        super().__init__()
        self.output_dim = output_dim
        self._device = None
        self._homogeneous_batch = None
        self._homogeneous_edge_index = None
        self.concat_global = concat_global
        input_dim_edge = 7
        hidden_dim = 64

        # Linear transformation to match dimensions if they are different
        self.node_lin = torch.nn.Linear(input_dim_node, hidden_dim)
        # self.edge_lin = torch.nn.Linear(input_dim_edge, hidden_dim)
        self.edge_lin = torch.nn.Sequential(
            Linear(input_dim_edge, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
        )

        self.conv1 = ProcessorLayer(64, 64, update_edge=False)
        self.conv2 = ProcessorLayer(64, 64, update_edge=False)

        input_dim = 64 * 2 if self.concat_global else 64
        self.mlp = torch.nn.Linear(input_dim, self.output_dim)

    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    def homogeneous_batch(self, graph: HeteroData):
        if self._homogeneous_batch is None or self._homogeneous_batch.size(0) != graph.num_nodes:
            batch_list = []
            for i in range(len(graph)):
                batch_list.append(torch.full((graph[i].num_nodes,), i, dtype=torch.long))
            self._homogeneous_batch = torch.cat(batch_list, dim=0).to(self.device())
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
            self._homogeneous_edge_index = torch.cat(edge_index_list, dim=1).to(self.device())
        return self._homogeneous_edge_index

    def output_mask(self, key, graph: HeteroData):
        start_idx = graph.node_offsets[key]
        end_idx = start_idx + graph[key].num_nodes
        return slice(start_idx, end_idx)

    def forward(self, data, input_vector, **kwargs):
        return self.one_step(data, input_vector, **kwargs)

    def one_step(
        self,
        graph: HeteroData,
        u_dict: Dict[str, torch.Tensor],
        **ignored,
    ):
        with torch.no_grad():
            pos_list = []
            x_list = []
            edge_attr_list = []

            # sub_graph = graph.node_type_subgraph(["actuator", "hole", "target"])
            sub_graph = graph

            batch_size = len(sub_graph)
            for node_type in sub_graph.node_types:
                pos = sub_graph[node_type].pos
                u = u_dict[node_type]

                pos_list.append(pos.reshape(batch_size, -1, 3))
                x_list.append(u.reshape(batch_size, -1, u.shape[-1]))

            pos = torch.cat(pos_list, dim=1).to(self.device())
            x = torch.cat(x_list, dim=1).to(self.device())  # (batch_size, num_points, input_dim_node)
            x = x.reshape(-1, x.shape[-1])  # (batch_size * num_points, input_dim_node)

            for edge_type in sub_graph.edge_types:
                edge_attr = sub_graph[edge_type].edge_attr
                edge_attr_list.append(edge_attr)

            edge_attr = torch.cat(edge_attr_list, dim=0).to(self.device())
            edge_index = self.homogeneous_edge_index(sub_graph)
            batch = self.homogeneous_batch(sub_graph)

        x = self.node_lin(x)
        edge_attr = self.edge_lin(edge_attr)

        h, _ = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        h, _ = self.conv2(x=h, edge_index=edge_index, edge_attr=edge_attr)

        if self.concat_global:
            g = global_max_pool(h, batch)

            h = h.reshape(batch_size, -1, h.shape[-1])  # (batch_size, num_points, hidden_dim)
            h = h[:, graph.output_mask]

            g = g.repeat(h.shape[1], 1, 1).transpose(0, 1)  # (batch_size, num_points, hidden_dim)
            h = torch.cat([g, h], dim=2)

            h = h.reshape(-1, h.shape[-1])  # (batch_size * num_points, hidden_dim)
            return self.mlp(h)
        else:
            h = h.reshape(batch_size, -1, h.shape[-1])  # (batch_size, num_points, hidden_dim)
            h = h[:, graph.output_mask].reshape(-1, h.shape[-1])
            return self.mlp(h)
