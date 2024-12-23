import torch as ch
from typing import Tuple

from ..models.policy.abstract_gaussian_policy import AbstractGaussianPolicy
from .base_projection_layer import (
    BaseProjectionLayer,
)
from ..utils.projection_utils import (
    gaussian_wasserstein_non_commutative,
)
from ..utils.torch_utils import sqrtm_newton


class WassersteinProjectionLayerNonCommuting(BaseProjectionLayer):
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
        runs Wasserstein projection layer for non commuting case and constructs sqrt of covariance
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
        dim = mean.shape[-1]

        ####################################################################################################################
        # precompute mean and cov part of W2, which are used for the projection.
        # Both parts differ based on precision scaling.
        # If activated, the mean part is the maha distance and the cov has a more complex term in the inner parenthesis.
        mean_part, cov_part, eigvals, eigvecs = gaussian_wasserstein_non_commutative(
            policy, p, q, self.scale_prec, return_eig=True
        )
        mask = mean_part + cov_part > eps + eps_cov

        # if nothing has to be projected skip computation
        if (~mask).all():
            return mean, sqrt

        t = ch.ones(batch_shape, dtype=mean.dtype, device=mean.device)
        t[mask] = ch.sqrt((eps + eps_cov) / (mean_part[mask] + cov_part[mask] + 1e-16))

        ################################################################################################################
        # mean projection
        proj_mean = ch.where(
            mask[..., None], (1 - t)[..., None] * old_mean + t[..., None] * mean, mean
        )

        ################################################################################################################
        # covariance projection
        old_cov = policy.covariance(old_sqrt)
        I = ch.eye(dim, dtype=mean.dtype, device=mean.device)

        # Compute sqrt(c)^-1 = V @ sqrt(1/D) @ V^T
        prod_inv = ch.zeros_like(sqrt, dtype=mean.dtype, device=mean.device) + I
        prod_inv[mask] = (
            eigvecs[mask]
            @ (1 / eigvals[mask].sqrt()).diag_embed(0, -2, -1)
            @ eigvecs[mask].permute(0, 2, 1)
        )
        W = sqrt @ prod_inv @ sqrt
        d = (1 - t)[..., None, None] * I + t[..., None, None] * W
        proj_sqrt = ch.where(mask[..., None, None], sqrtm_newton(d @ old_cov @ d), sqrt)

        return proj_mean, proj_sqrt

    def trust_region_value(self, policy, p, q):
        """
        Computes the non commuting Wasserstein distance between two Gaussian distributions p and q_values.
        Returns:
            mean and covariance part
        """
        return gaussian_wasserstein_non_commutative(
            policy, p, q, scale_prec=self.scale_prec
        )
