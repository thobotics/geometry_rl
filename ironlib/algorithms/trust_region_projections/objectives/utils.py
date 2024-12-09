import torch
from torchrl.objectives.utils import distance_loss


def _clip_value_loss(
    old_state_value: torch.Tensor,
    state_value: torch.Tensor,
    clip_value: torch.Tensor,
    target_return: torch.Tensor,
    loss_value: torch.Tensor,
    loss_critic_type: str,
):
    """Value clipping method for loss computation.
    This method computes a clipped state value from the old state value and the state value,
    and returns the most pessimistic value prediction between clipped and non-clipped options.
    It also computes the clip fraction.
    """
    state_value_clipped = old_state_value + (state_value - old_state_value).clamp(
        -clip_value, clip_value
    )
    loss_value_clipped = distance_loss(
        target_return,
        state_value_clipped,
        loss_function=loss_critic_type,
    )
    # Chose the most pessimistic value prediction between clipped and non-clipped
    loss_value = torch.max(loss_value, loss_value_clipped)
    with torch.no_grad():
        clip_fraction = (
            (state_value / old_state_value).clamp(1 - clip_value, 1 + clip_value).abs()
        )
    return loss_value, clip_fraction
