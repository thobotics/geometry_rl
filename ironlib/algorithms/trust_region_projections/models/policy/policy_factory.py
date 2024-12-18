import torch as ch

from .gnn_gaussian_policy_diag import GNNGaussianPolicyDiag


def get_policy_network(
    policy_type,
    proj_type,
    squash=False,
    device: ch.device = "cpu",
    dtype=ch.float32,
    **kwargs,
):
    """
    Policy network factory
    Args:
        policy_type: 'full' or 'diag' covariance
        proj_type: Which projection is used.
        squash: Gaussian policy with tanh transformation
        device: torch device
        dtype: torch dtype
        **kwargs: policy arguments

    Returns:
        Gaussian Policy instance
    """

    if policy_type == "gnn_diag":
        policy = GNNGaussianPolicyDiag(**kwargs)
    else:
        raise ValueError(
            f"Invalid policy type {policy_type}. Select one of 'full', 'diag'."
        )

    return policy.to(device, dtype)
