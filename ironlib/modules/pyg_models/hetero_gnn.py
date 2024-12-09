from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from con_mgn.models.mgn import MeshGraphNet
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HeteroConv

from con_mgn.models.networks.hetero_mpnn import HeteroProcessorLayer


class HeteroGNN(MeshGraphNet):
    def __init__(
        self,
        input_dim_node,
        input_dim_edge,
        hidden_dim,
        latent_dim,
        output_dim,
        node_encoder_layers,
        edge_encoder_layers,
        node_decoder_layers,
        node_type_mapping,
        edge_type_mapping,
        edge_level_mapping,
        message_passing,
        num_messages,
        shared_processor=False,
        shared_node_encoder=True,
        shared_edge_encoder=True,
        device="cuda",
        **ignored,
    ):
        super(MeshGraphNet, self).__init__()

        self.input_dim_node = input_dim_node
        self.input_dim_edge = input_dim_edge
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_messages = num_messages
        self.shared_processor = shared_processor
        self.node_type_mapping = node_type_mapping
        self.edge_type_mapping = edge_type_mapping
        self.device = device

        # self.node_encoder = self._create_node_encoder(input_dim_node, hidden_dim, latent_dim, node_encoder_layers).to(
        #     device
        # )
        self.edge_encoder = self._create_edge_encoder(input_dim_edge, hidden_dim, latent_dim, edge_encoder_layers).to(
            device
        )

        self.node_encoder = nn.Linear(self.input_dim_node, latent_dim, False)
        # self.edge_encoder = nn.Linear(self.input_dim_edge, latent_dim, False)

        self.processor = nn.ModuleList()
        for k in range(num_messages):
            level_processor = {}
            for l, edge_level in enumerate(edge_level_mapping):
                pl = message_passing[l][k]

                if pl is not None:
                    for src, level, dest in edge_type_mapping:
                        if level == edge_level:
                            level_processor[src, level, dest] = pl.to(device)

            self.processor.append(HeteroProcessorLayer(level_processor))

        # self.decoder = self._create_decoder(latent_dim, hidden_dim, output_dim, node_decoder_layers).to(device)
        self.decoder = nn.Linear(latent_dim, output_dim, False)

    def one_step(
        self,
        graph: HeteroData,
        u_dict: Dict[str, torch.Tensor],
        u: torch.Tensor = None,
        u_properties: torch.Tensor = None,
    ):
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """
        latent_dict = {node_type: self.node_encoder(u_dict[node_type]) for node_type in graph.node_types}

        edge_attr_dict = {edge_type: self.edge_encoder(graph[edge_type].edge_attr) for edge_type in graph.edge_types}

        for i in range(self.num_messages):
            latent_dict, edge_attr_dict = self.processor[i](
                latent_dict=latent_dict,
                edge_index_dict=graph.edge_index_dict,
                edge_attr_dict=edge_attr_dict,
            )

        latent = latent_dict[graph.output_mask_key]
        return self.decoder(latent)
