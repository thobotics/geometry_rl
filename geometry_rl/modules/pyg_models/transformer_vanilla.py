from typing import List, Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import MLP


class TransformerVanilla(nn.Module):
    def __init__(
        self,
        input_dim_node,
        output_dim,
        num_layers=2,
        num_heads=2,
        hidden_dim=64,
        dropout=0.1,
        concat_global=False,
        **ignored,
    ):
        super(TransformerVanilla, self).__init__()
        self.input_dim = input_dim_node
        self._device = None
        self.concat_global = concat_global

        self.cls_token = nn.Parameter(torch.randn(1, 1, output_dim), requires_grad=True)

        self.embedding = nn.Linear(input_dim_node, hidden_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        input_final_dim = hidden_dim * 2 if self.concat_global else hidden_dim

        self.fc_out = MLP([input_final_dim, output_dim], norm=None)

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

        x = self.embedding(x)

        if self.concat_global:
            # Prepare cls_token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, dim_feedforward)
            x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_points + 1, dim_feedforward)

            x = x.permute(1, 0, 2)
            h = self.transformer_encoder(x)

            h = h.permute(1, 0, 2)  # (batch_size, num_points + 1, dim_feedforward)
            cls_output = h[:, 0]

            output_mask = slice(graph.output_mask.start + 1, graph.output_mask.stop + 1)
            h = h[:, output_mask]  # (batch_size, num_points, dim_feedforward)

            g = cls_output.repeat(h.shape[1], 1, 1).transpose(0, 1)  # (batch_size, num_points, dim_feedforward)
            h = torch.cat([g, h], dim=-1)
            h = h.reshape(-1, h.shape[-1])  # (batch_size * num_points, dim_feedforward)
            return self.fc_out(h)
        else:
            x = x.permute(1, 0, 2)
            h = self.transformer_encoder(x)
            h = h.permute(1, 0, 2)  # (batch_size, num_points + 1, dim_feedforward)
            h = h[:, graph.output_mask]  # (batch_size, num_points, dim_feedforward)
            h = h.reshape(-1, h.shape[-1])
            return self.fc_out(h)
