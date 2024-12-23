import torch

from collections import OrderedDict
from tensordict import TensorDict


def remap_dict_key(dict, from_key, to_key):
    """Remaps the policy to the observation space."""

    dict[to_key] = dict.pop(from_key)


def extract_tensors_from_a_dict(input_dict):
    tensors = type(input_dict)()
    for key, value in input_dict.items():
        if isinstance(value, (torch.Tensor, TensorDict)):
            # Directly store tensors or TensorDicts
            tensors[key] = value
        elif isinstance(value, (dict, OrderedDict)):
            # Recursively process nested dictionaries
            nested_tensors = extract_tensors_from_a_dict(value)
            if nested_tensors:
                # Only add the key if there are nested tensors
                tensors[key] = nested_tensors
    return tensors


def recursively_merge_dict(loaded_dict, current_dict, share_memory_if_possible=True):
    for key, value in current_dict.items():
        if key not in loaded_dict:
            loaded_dict[key] = value
        if share_memory_if_possible and isinstance(value, TensorDict):
            loaded_dict[key].share_memory_()
        elif isinstance(value, dict):
            recursively_merge_dict(loaded_dict[key], value)


def recursively_share_memory(loaded_dict, share_memory_if_possible=True):
    for key, value in loaded_dict.items():
        if share_memory_if_possible and isinstance(value, TensorDict):
            loaded_dict[key].share_memory_()
        elif isinstance(value, dict):
            recursively_merge_dict(loaded_dict[key], value)
