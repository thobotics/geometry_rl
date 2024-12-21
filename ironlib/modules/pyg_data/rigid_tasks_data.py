from typing import List, Dict, Tuple

import enum
import torch
import torch_geometric
from torch_geometric.data import Data, HeteroData

from .transforms import (
    HeteroNodeCategorical,
    HeteroEdgeCategorical,
    HeteroCartesian,
    HeteroDistance,
)
from torch_geometric.transforms import Compose
from dataclasses import dataclass
from .base_data import BaseData
from .utils import noise_like


class NodeType(str, enum.Enum):
    PARTICLES = "object_geometry"
    ACTUATOR = "grippers"
    TARGET = "target_geometry"


class EdgeLevel(str, enum.Enum):
    INTERNAL = "internal"
    TASK = "task"
    AGENT = "agent"


class EdgeType(Tuple[str, str, str], enum.Enum):
    PARTICLES_INTERNAL_PARTICLES = (
        NodeType.PARTICLES,
        EdgeLevel.INTERNAL,
        NodeType.PARTICLES,
    )
    ACTUATOR_AGENT_ACTUATOR = (
        NodeType.ACTUATOR,
        EdgeLevel.AGENT,
        NodeType.ACTUATOR,
    )
    PARTICLES_TASK_ACTUATOR = (
        NodeType.PARTICLES,
        EdgeLevel.TASK,
        NodeType.ACTUATOR,
    )


@dataclass
class RigidTasksData(BaseData):

    def __init__(
        self,
        observation_dim: Dict,
        observation_names: Dict,
        full_graph_obs: bool = False,
        dist_as_pos: bool = False,
        output_mask_key: str = None,
        training_noise: bool = False,
        training_noise_std: float = 1e-2,
        concat_input_vector: bool = True,
        angular_velocity: bool = True,
        knn_k: int = 3,
        knn_to_actuators_k: int = -1,
        **kwargs,
    ):
        super().__init__(
            output_mask_key=output_mask_key,
            training_noise=training_noise,
            training_noise_std=training_noise_std,
            concat_input_vector=concat_input_vector,
        )

        self.transform = Compose(
            [
                HeteroCartesian(norm=False),
                HeteroDistance(norm=False),
            ]
        )

        self.observation_dim = {k: list(map(lambda x: x[0], vs)) for k, vs in observation_dim.items()}
        self.observation_names = observation_names
        self.full_graph_obs = full_graph_obs

        self.dist_as_pos = dist_as_pos
        self.angular_velocity = angular_velocity
        self.knn_k = knn_k
        self.knn_to_actuators_k = knn_to_actuators_k

        self.node_type_list = [name for name in NodeType if name != NodeType.TARGET]

    def _preprocess_input(
        self,
        scalars: torch.Tensor,
        position_vectors: torch.Tensor,
        velocity_vectors: torch.Tensor,
        norm_position_vectors: torch.Tensor,
        norm_velocity_vectors: torch.Tensor,
        infos: torch.Tensor,
        train: bool = True,
        **ignored,
    ) -> Dict:

        batch_size = scalars.shape[0]

        scalars_list = torch.split(scalars, self.observation_dim["scalars"], dim=1)
        position_vectors_list = torch.split(position_vectors, self.observation_dim["position_vectors"], dim=1)
        velocity_vectors_list = torch.split(velocity_vectors, self.observation_dim["velocity_vectors"], dim=1)
        norm_position_vectors_list = torch.split(norm_position_vectors, self.observation_dim["position_vectors"], dim=1)
        norm_velocity_vectors_list = torch.split(norm_velocity_vectors, self.observation_dim["velocity_vectors"], dim=1)
        infos_list = torch.split(infos, self.observation_dim["infos"], dim=1)

        inputs_dict = {
            "scalars": {},
            "position_vectors": {},
            "velocity_vectors": {},
            "norm_position_vectors": {},
            "norm_velocity_vectors": {},
            "infos": {},
            "batch_size": batch_size,
            "device": scalars.device,
            "train": train,
        }

        for i, scalar in enumerate(scalars_list):
            key_name = self.observation_names["scalars"][i]
            inputs_dict["scalars"][key_name] = scalar

        for i, position_vector in enumerate(position_vectors_list):
            key_name = self.observation_names["position_vectors"][i]
            inputs_dict["position_vectors"][key_name] = position_vector.reshape(batch_size, -1, 3)

        for i, velocity_vector in enumerate(velocity_vectors_list):
            key_name = self.observation_names["velocity_vectors"][i]
            inputs_dict["velocity_vectors"][key_name] = velocity_vector.reshape(batch_size, -1, 3)

        for i, norm_position_vector in enumerate(norm_position_vectors_list):
            key_name = self.observation_names["position_vectors"][i]
            inputs_dict["norm_position_vectors"][key_name] = norm_position_vector.reshape(batch_size, -1, 3)

        for i, norm_velocity_vector in enumerate(norm_velocity_vectors_list):
            key_name = self.observation_names["velocity_vectors"][i]
            inputs_dict["norm_velocity_vectors"][key_name] = norm_velocity_vector.reshape(batch_size, -1, 3)

        for i, info in enumerate(infos_list):
            key_name = self.observation_names["infos"][i]
            inputs_dict["infos"][key_name] = info

        return inputs_dict

    def construct_input_vector(
        self,
        data: Data,
        scalars: Dict[str, torch.Tensor],
        position_vectors: Dict[str, torch.Tensor],
        velocity_vectors: Dict[str, torch.Tensor],
        norm_position_vectors: Dict[str, torch.Tensor],
        norm_velocity_vectors: Dict[str, torch.Tensor],
        infos: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
        train: bool = True,
        **ignored,
    ) -> torch.Tensor:
        input_vector_dict = {}
        scalar_dict = {}
        vector_dict = {}

        for node_type in self.node_type_list:

            # Properties
            one_hot_type = data[node_type].properties

            # Position
            pos_vec = data[node_type].norm_pos

            if train and self.training_noise:
                pos_vec += noise_like(pos_vec, self.training_noise_std)

            # Corresponding position
            if node_type == NodeType.PARTICLES:
                target = norm_position_vectors[NodeType.TARGET].reshape(-1, 3)
                corresponding_pos = data[node_type].norm_pos - target if self.dist_as_pos else target
                corresponding_pos = corresponding_pos

                if train and self.training_noise:
                    corresponding_pos += noise_like(corresponding_pos, self.training_noise_std)
            else:
                corresponding_pos = torch.zeros_like(data[node_type].norm_pos)

            # Velocity
            if node_type in norm_velocity_vectors:
                if node_type == NodeType.PARTICLES:
                    velocity = norm_velocity_vectors[node_type].repeat_interleave(
                        norm_position_vectors[node_type].shape[1], dim=1
                    )
                    if self.angular_velocity:
                        angular_velocity = norm_velocity_vectors[f"{node_type}_angular"].repeat_interleave(
                            norm_position_vectors[node_type].shape[1], dim=1
                        )
                    else:
                        angular_velocity = torch.zeros_like(velocity)
                    velocity = velocity.reshape(-1, 3)
                    angular_velocity = angular_velocity.reshape(-1, 3)
                else:
                    velocity = norm_velocity_vectors[node_type].reshape(-1, 3)
                    if self.angular_velocity:
                        angular_velocity = norm_velocity_vectors[f"{node_type}_angular"].reshape(-1, 3)
                    else:
                        angular_velocity = torch.zeros_like(velocity)
                if train and self.training_noise:
                    velocity += noise_like(velocity, self.training_noise_std)
                    angular_velocity += noise_like(angular_velocity, self.training_noise_std)
            else:
                velocity = torch.zeros_like(data[node_type].norm_pos)
                angular_velocity = torch.zeros_like(data[node_type].norm_pos)

            scalars = one_hot_type
            vectors = torch.cat([pos_vec, corresponding_pos, velocity, angular_velocity], dim=1)
            input_vector = torch.cat([scalars, vectors], dim=1)

            scalar_dict[node_type] = scalars
            vector_dict[node_type] = vectors
            input_vector_dict[node_type] = input_vector

        if self.concat_input_vector:
            return input_vector_dict
        else:
            return scalar_dict, vector_dict

    def _update_placeholders(
        self,
        scalars: Dict[str, torch.Tensor],
        position_vectors: Dict[str, torch.Tensor],
        velocity_vectors: Dict[str, torch.Tensor],
        norm_position_vectors: Dict[str, torch.Tensor],
        norm_velocity_vectors: Dict[str, torch.Tensor],
        infos: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
        train: bool = True,
        **ignored,
    ) -> Data:
        data = self.example_data.clone()

        for node_type in self.node_type_list:
            data[node_type].pos = position_vectors[node_type].reshape(-1, 3)
            data[node_type].norm_pos = norm_position_vectors[node_type].reshape(-1, 3)

        data = self.transform(data)
        return data.to(device)

    def _should_reconstruct_placeholders(self, batch_size, **ignored) -> bool:
        return self._example_data is None or len(self._example_data) != batch_size

    def _construct_placeholders(
        self,
        scalars: Dict[str, torch.Tensor],
        position_vectors: Dict[str, torch.Tensor],
        velocity_vectors: Dict[str, torch.Tensor],
        norm_position_vectors: Dict[str, torch.Tensor],
        norm_velocity_vectors: Dict[str, torch.Tensor],
        infos: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
        **ignored,
    ):

        hetero_data_list = []

        num_points = infos["object_num_points"].long()
        num_edges = infos["object_num_edges"].long()

        for i in range(batch_size):
            hetero_data = HeteroData()

            for node_type in NodeType:
                hetero_data[node_type].pos = position_vectors[node_type][i]
                hetero_data[node_type].norm_pos = norm_position_vectors[node_type][i]

            actuators = position_vectors[NodeType.ACTUATOR][i]
            points = position_vectors[NodeType.PARTICLES][i]

            hetero_data[EdgeType.PARTICLES_INTERNAL_PARTICLES].edge_index = torch_geometric.nn.knn_graph(
                points[: num_points[i]], k=self.knn_k
            )

            if actuators.shape[0] > 1:
                actuator_actuator_edges = []
                for j in range(actuators.shape[0]):
                    for k in range(actuators.shape[0]):
                        if j != k:
                            actuator_actuator_edges.append([j, k])
                actuator_actuator_edges = torch.tensor(actuator_actuator_edges, device=device, dtype=torch.long).T
                hetero_data[EdgeType.ACTUATOR_AGENT_ACTUATOR].edge_index = actuator_actuator_edges
            else:
                hetero_data[EdgeType.ACTUATOR_AGENT_ACTUATOR].edge_index = torch.tensor(
                    [], device=device, dtype=torch.long
                ).reshape(2, 0)

            if self.knn_to_actuators_k > 0:
                points_actuator_edges = []
                for k in range(actuators.shape[0]):
                    knn_edges = torch_geometric.nn.knn(
                        points[: num_points[i]], actuators[k][None], self.knn_to_actuators_k
                    ).flip(0)
                    knn_edges[1] = k
                    points_actuator_edges.append(knn_edges)
                points_actuator_edges = torch.cat(points_actuator_edges, dim=1)
            else:
                points_actuator_edges = []
                for j in range(num_points[i].item()):
                    for k in range(actuators.shape[0]):
                        points_actuator_edges.append([j, k])
                points_actuator_edges = torch.tensor(points_actuator_edges, device=device, dtype=torch.long).T
                hetero_data[EdgeType.PARTICLES_TASK_ACTUATOR].edge_index = points_actuator_edges

            hetero_data = hetero_data.coalesce().to(device)
            hetero_data_list.append(hetero_data)

        hetero_batch = torch_geometric.data.Batch.from_data_list(hetero_data_list)
        transform = Compose(
            [
                HeteroNodeCategorical("properties"),
                HeteroEdgeCategorical(),
            ]
        )
        hetero_batch = transform(hetero_batch)
        hetero_batch = hetero_batch.to(device)

        # Select the node types to keep
        hetero_batch = hetero_batch.node_type_subgraph(self.node_type_list)

        # Extra information
        hetero_batch.output_mask_key = self._output_mask_key
        hetero_batch.output_mask = self.output_mask(hetero_batch[0], self._output_mask_key)
        hetero_batch.homo_batch = None
        hetero_batch.device = device

        self._example_data = hetero_batch
