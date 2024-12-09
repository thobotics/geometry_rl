import torch as ch

from .gaussian_policy_diag import GaussianPolicyDiag
from .gaussian_policy_diag_squashed import GaussianPolicyDiagSquashed
from .gaussian_policy_full import GaussianPolicyFull
from .gaussian_policy_full_squashed import GaussianPolicyFullSquashed
from .gaussian_policy_sqrt import GaussianPolicySqrt
from .gaussian_policy_sqrt_squashed import GaussianPolicySqrtSquashed

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

    if squash:
        if policy_type == "full":
            policy = (
                GaussianPolicySqrtSquashed(**kwargs)
                if "w2" in proj_type
                else GaussianPolicyFullSquashed(**kwargs)
            )
        elif policy_type == "diag":
            policy = GaussianPolicyDiagSquashed(**kwargs)
        else:
            raise ValueError(
                f"Invalid policy type {policy_type}. Select one of 'full', 'diag'."
            )
    else:
        if policy_type == "full":
            policy = (
                GaussianPolicySqrt(**kwargs)
                if "w2" in proj_type
                else GaussianPolicyFull(**kwargs)
            )
        elif policy_type == "diag":
            policy = GaussianPolicyDiag(**kwargs)
        elif policy_type == "gnn_diag":
            policy = GNNGaussianPolicyDiag(**kwargs)
        else:
            raise ValueError(
                f"Invalid policy type {policy_type}. Select one of 'full', 'diag'."
            )

    return policy.to(device, dtype)
