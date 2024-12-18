import torch

from .critic import BaseCritic
from .gnn_vf_net import GNNVFNet


def get_critic(
    critic_type: str,
    dim: int = 0,
    device: torch.device = "cpu",
    dtype=torch.float32,
    **kwargs,
):
    """
    Critic network factory
    Args:
        critic_type: what type of critic, one of 'base' (standard Q-learning), 'double' (double Q-learning),
                     and 'duelling' (Duelling Double Q-learning).
        dim: input dimensionality.
        device: torch device
        dtype: torch dtype
        **kwargs: critic arguments

    Returns:
        Critic instance
    """
    # Value-networks
    if critic_type == "gnn":
        vf = GNNVFNet(**kwargs)
        critic = BaseCritic(vf)
        return critic.to(device, dtype)
    else:
        raise ValueError(
            f"Invalid value_loss type {critic_type}. Select one of 'base', 'double', 'duelling'."
        )
