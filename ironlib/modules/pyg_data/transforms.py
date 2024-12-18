from typing import List, Optional, Tuple

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform, Cartesian, Distance


@functional_transform("edge_categorical")
class EdgeCategorical(BaseTransform):
    r"""Converts edge types into categorical edge attributes.
    (functional name: :obj:`edge_categorical`)

    Args:
        size (int): The number of edge types.
        cat (bool, optional): If set to :obj:`False`, edge types will be
            overwritten instead of concatenated. (default: :obj:`True`)
    """

    def __init__(self, size: int, cat: bool = True):
        self.size = size
        self.cat = cat

    def __call__(self, data: Data) -> Data:
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        edge_type = data.edge_type
        edge_type = torch.eye(self.size, device=edge_type.device)[edge_type]

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, edge_type.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = edge_type

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@functional_transform("hetero_node_categorical")
class HeteroNodeCategorical(BaseTransform):
    def __init__(self, attribute_name: str, cat: bool = True):
        super().__init__()
        self.attribute_name = attribute_name
        self.cat = cat

    def __call__(self, data: HeteroData) -> HeteroData:
        for node_type in data.node_types:
            # Extract the node attributes for the current edge type
            if hasattr(data[node_type], self.attribute_name):
                node_attr = getattr(data[node_type], self.attribute_name)
            else:
                node_attr = None

            #  Determine the size based on the number of unique node types
            size = len(data.node_types)
            # Create a one-hot encoding for the current edge type
            index = data.node_types.index(node_type)
            node_type_one_hot = torch.zeros(data[node_type].num_nodes, size)
            node_type_one_hot[:, index] = 1

            # Concatenate with existing edge attributes if necessary
            if node_attr is not None and self.cat:
                node_attr = node_attr.view(-1, 1) if node_attr.dim() == 1 else node_attr
                setattr(
                    data[node_type],
                    self.attribute_name,
                    torch.cat([node_attr, node_type_one_hot], dim=-1),
                )
            else:
                setattr(data[node_type], self.attribute_name, node_type_one_hot)

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(attribute_name='{self.attribute_name}')"


@functional_transform("hetero_edge_categorical")
class HeteroEdgeCategorical(BaseTransform):
    def __init__(
        self, cat: bool = True, edge_types: Optional[List[Tuple[str, str, str]]] = None
    ):
        super().__init__()
        self.cat = cat
        self.edge_types = edge_types

    def __call__(self, data: HeteroData) -> HeteroData:
        edge_types = self.edge_types if self.edge_types is not None else data.edge_types

        # Loop over all edge types in the HeteroData object
        for edge_type in edge_types:
            # Extract the edge attributes for the current edge type
            if hasattr(data[edge_type], "edge_attr"):
                edge_attr = data[edge_type].edge_attr
            else:
                edge_attr = None

            # Determine the size based on the number of unique edge types
            size = len(data.edge_types)
            # Create a one-hot encoding for the current edge type
            index = data.edge_types.index(edge_type)
            edge_type_one_hot = torch.zeros(
                (data[edge_type].edge_index.size(1), size),
                device=data[edge_type].edge_index.device,
            )
            edge_type_one_hot[:, index] = 1

            # Concatenate with existing edge attributes if necessary
            if edge_attr is not None and self.cat:
                edge_attr = edge_attr.view(-1, 1) if edge_attr.dim() == 1 else edge_attr
                data[edge_type].edge_attr = torch.cat(
                    [edge_attr, edge_type_one_hot], dim=-1
                )
            else:
                data[edge_type].edge_attr = edge_type_one_hot

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(cat={self.cat})"


class HeteroCartesian(Cartesian):
    def forward(self, data: HeteroData) -> HeteroData:
        for edge_type in data.edge_types:
            src, _, dst = edge_type
            assert data[src].pos is not None and data[dst].pos is not None

            # Create a temporary Data object for each edge type
            temp_data = Data(
                pos=torch.cat([data[src].pos, data[dst].pos], dim=0),
                edge_index=data[edge_type].edge_index,
                edge_attr=(
                    data[edge_type].edge_attr
                    if "edge_attr" in data[edge_type].keys()
                    else None
                ),
            )

            # Apply the Cartesian transform to the temporary Data object
            temp_data = super().forward(temp_data)

            # Assign the transformed edge attributes back to the original HeteroData object
            data[edge_type].edge_attr = temp_data.edge_attr

        return data


class HeteroDistance(Distance):
    def forward(self, data: HeteroData) -> HeteroData:
        for edge_type in data.edge_types:
            src, _, dst = edge_type
            assert data[src].pos is not None and data[dst].pos is not None

            # Create a temporary Data object for each edge type
            temp_data = Data(
                pos=torch.cat([data[src].pos, data[dst].pos], dim=0),
                edge_index=data[edge_type].edge_index,
                edge_attr=(
                    data[edge_type].edge_attr
                    if "edge_attr" in data[edge_type].keys()
                    else None
                ),
            )

            # Apply the Distance transform to the temporary Data object
            temp_data = super().forward(temp_data)

            # Assign the transformed edge attributes back to the original HeteroData object
            data[edge_type].edge_attr = temp_data.edge_attr

        return data
