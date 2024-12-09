from typing import Dict, Tuple
from abc import abstractmethod

import torch
from torch_geometric.data import Data
from dataclasses import dataclass


@dataclass
class BaseData:
    _example_data: Data = None
    _data_dict: dict = None

    def __init__(
        self,
        *,
        output_mask_key: str = None,
        training_noise: bool = False,
        training_noise_std: float = 1e-2,
        concat_input_vector: bool = True,
    ):
        self._output_mask_key = output_mask_key
        self.training_noise = training_noise
        self.training_noise_std = training_noise_std
        self.concat_input_vector = concat_input_vector

    @property
    def data_dict(self):
        if self._data_dict is not None:
            return self._data_dict

    @property
    def example_data(self):
        if self._example_data is not None:
            return self._example_data

    def output_mask(self, data: Data, key: str) -> slice:
        if key is not None:
            start_idx = data.node_offsets[key]
            end_idx = start_idx + data[key].num_nodes
            return slice(start_idx, end_idx)
        else:
            return slice(None)

    def build_data(self, *args, **kwargs) -> Tuple[Data, torch.Tensor]:
        input_dict = self._preprocess_input(*args, **kwargs)

        if self._should_reconstruct_placeholders(**input_dict):
            self._construct_placeholders(**input_dict)

        with torch.no_grad():
            data = self._update_placeholders(**input_dict)
            input_vector = self.construct_input_vector(data, **input_dict)

        return data, input_vector

    @abstractmethod
    def _update_placeholders(self, **kwargs) -> Data:
        raise NotImplementedError

    @abstractmethod
    def _construct_placeholders(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _should_reconstruct_placeholders(self, **kwargs) -> bool:
        return self._example_data is None

    @abstractmethod
    def _preprocess_input(self, *args, **kwargs) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def construct_input_vector(self, data: Data, **kwargs) -> torch.Tensor:
        raise NotImplementedError
