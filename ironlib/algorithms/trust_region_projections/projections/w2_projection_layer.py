import torch as ch
from typing import Tuple

from ..models.policy.abstract_gaussian_policy import AbstractGaussianPolicy
from .base_projection_layer import (
    BaseProjectionLayer,
    mean_projection,
)
from ..utils.projection_utils import (
    gaussian_wasserstein_commutative,
)


class WassersteinProjectionLayer(BaseProjectionLayer):
    def _trust_region_projection(
        self,
        policy: AbstractGaussianPolicy,
        p: Tuple[ch.Tensor, ch.Tensor],
        q: Tuple[ch.Tensor, ch.Tensor],
        eps: ch.Tensor,
        eps_cov: ch.Tensor,
        **kwargs
    ):
        """
        runs wasserstein projection layer and constructs sqrt of covariance
        Args:
            **kwargs:
            policy: policy instance
            p: current distribution
            q: old distribution
            eps: (modified) kl bound/ kl bound for mean part
            eps_cov: (modified) kl bound for cov part

        Returns:
            mean, cov sqrt
        """
        mean, sqrt = p
        old_mean, old_sqrt = q
        batch_shape = mean.shape[:-1]

        ####################################################################################################################
        # precompute mean and cov part of W2, which are used for the projection.
        # Both parts differ based on precision scaling.
        # If activated, the mean part is the maha distance and the cov has a more complex term in the inner parenthesis.
        mean_part, cov_part = gaussian_wasserstein_commutative(
            policy, p, q, self.scale_prec
        )

        ####################################################################################################################
        # project mean (w/ or w/o precision scaling)

        proj_mean = mean_projection(mean, old_mean, mean_part, eps)

        ####################################################################################################################
        # project covariance (w/ or w/o precision scaling)

        cov_mask = cov_part > eps_cov

        if cov_mask.any():
            # gradient issue with ch.where, it executes both paths and gives NaN gradient.
            eta = ch.ones(batch_shape, dtype=sqrt.dtype, device=sqrt.device)
            eta[cov_mask] = ch.sqrt(cov_part[cov_mask] / eps_cov) - 1.0
            eta = ch.max(-eta, eta)

            new_sqrt = (sqrt + ch.einsum("i,ijk->ijk", eta, old_sqrt)) / (
                1.0 + eta + 1e-16
            )[..., None, None]
            proj_sqrt = ch.where(cov_mask[..., None, None], new_sqrt, sqrt)
        else:
            proj_sqrt = sqrt

        return proj_mean, proj_sqrt

    def trust_region_value(self, policy, p, q):
        """
        Computes the Wasserstein distance between two Gaussian distributions p and q_values.
        Returns:
            mean and covariance part
        """
        return gaussian_wasserstein_commutative(
            policy, p, q, scale_prec=self.scale_prec
        )
