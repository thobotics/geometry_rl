import torch
import torch.nn as nn
from torch.nn import LayerNorm, Linear, ReLU
from torch_geometric.data import Data
from .mpnn import ProcessorLayer


class BaseGNN(torch.nn.Module):
    def __init__(
        self,
        input_dim_node,
        input_dim_edge,
        hidden_dim,
        latent_dim,
        output_dim,
        num_messages,
        node_encoder_layers,
        edge_encoder_layers,
        node_decoder_layers,
        shared_processor=False,
        device="cuda",
        **ignored,
    ):
        super(BaseGNN, self).__init__()
        """
        Graph Networks model. This model is built upon Deepmind's 2021 paper.
        This model consists of three parts: (1) Preprocessing: encoder (2) Processor
        (3) postproccessing: decoder. Encoder has an edge and node decoders respectively.
        Processor has two processors for edge and node respectively. Note that edge attributes have to be
        updated first. Decoder is only for nodes.

        Input_dim: dynamic variables + node_type + node_position
        Hidden_dim: 128 in deepmind's paper
        Output_dim: dynamic variables: velocity changes (1)

        """
        self.input_dim_node = input_dim_node
        self.input_dim_edge = input_dim_edge
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_messages = num_messages
        self.shared_processor = shared_processor

        self.node_encoder = self._create_node_encoder(
            input_dim_node, hidden_dim, latent_dim, node_encoder_layers
        ).to(device)
        self.edge_encoder = self._create_edge_encoder(
            input_dim_edge, hidden_dim, latent_dim, edge_encoder_layers
        ).to(device)
        self.processor = self._create_processor(latent_dim, num_messages)
        self.decoder = self._create_decoder(
            latent_dim, hidden_dim, output_dim, node_decoder_layers
        ).to(device)

    def reset(self, batch_size: int = 1, device: torch.device = "cuda"):
        pass

    def _create_node_encoder(
        self, input_dim_node, hidden_dim, latent_dim, num_layers=1
    ):
        node_encoder = nn.ModuleList()
        for _ in range(num_layers):
            node_encoder.append(Linear(input_dim_node, hidden_dim))
            node_encoder.append(ReLU())
            input_dim_node = hidden_dim
        node_encoder.append(Linear(hidden_dim, latent_dim))
        node_encoder.append(LayerNorm(latent_dim))
        node_encoder = nn.Sequential(*node_encoder)

        # node_encoder = nn.Linear(input_dim_node, latent_dim)

        return node_encoder

    def _create_edge_encoder(
        self, input_dim_edge, hidden_dim, latent_dim, num_layers=1
    ):
        edge_encoder = nn.ModuleList()
        for _ in range(num_layers):
            edge_encoder.append(Linear(input_dim_edge, hidden_dim))
            edge_encoder.append(ReLU())
            input_dim_edge = hidden_dim
        edge_encoder.append(Linear(hidden_dim, latent_dim))
        edge_encoder.append(LayerNorm(latent_dim))
        edge_encoder = nn.Sequential(*edge_encoder)
        
        # edge_encoder = nn.Linear(input_dim_edge, latent_dim)

        return edge_encoder

    def _create_processor(self, latent_dim, num_messages=1):

        assert num_messages >= 1, "Number of message passing layers is not >=1"

        if self.shared_processor:
            processor = ProcessorLayer(latent_dim, latent_dim)
        else:
            processor = nn.ModuleList()
            for _ in range(num_messages):
                processor.append(ProcessorLayer(latent_dim, latent_dim))
        return processor

    def _create_decoder(self, latent_dim, hidden_dim, output_dim, num_layers=1):
        decoder = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                decoder.append(Linear(latent_dim, hidden_dim))
            else:
                decoder.append(Linear(hidden_dim, hidden_dim))
            decoder.append(ReLU())
        decoder.append(Linear(hidden_dim, output_dim))
        decoder = nn.Sequential(*decoder)

        return decoder

    def forward(
        self,
        **kwargs,
    ) -> torch.Tensor:
        return self.one_step(**kwargs)

    def one_step(
        self,
        graph: Data,
        u_vector: torch.Tensor,
        u: torch.Tensor = None,
        u_properties: torch.Tensor = None,
        num_messages: int = 0,
        decoding: bool = True,
    ):
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """

        if num_messages == 0:
            num_messages = self.num_messages

        node_attr = u_vector
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr

        latent = self.node_encoder(node_attr)
        edge_attr = self.edge_encoder(edge_attr)

        if self.shared_processor:
            for _ in range(num_messages):
                latent, edge_attr = self.processor(latent, edge_index, edge_attr)
        else:
            for i in range(num_messages):
                latent, edge_attr = self.processor[i](latent, edge_index, edge_attr)

        if decoding:
            output = self.decoder(latent)
        else:
            output = latent

        return output
