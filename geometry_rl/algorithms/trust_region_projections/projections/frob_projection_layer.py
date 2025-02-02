import torch as ch
from typing import Tuple

from ..models.policy.abstract_gaussian_policy import AbstractGaussianPolicy
from .base_projection_layer import BaseProjectionLayer, mean_projection
from ..utils.projection_utils import gaussian_frobenius


class FrobeniusProjectionLayer(BaseProjectionLayer):
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
        runs frobenius projection layer and constructs cholesky of covariance

        Args:
            policy: policy instance
            p: current distribution
            q: old distribution
            eps: (modified) kl bound/ kl bound for mean part
            eps_cov: (modified) kl bound for cov part
            beta: (modified) entropy bound
            **kwargs:
        Returns: mean, cov cholesky
        """

        mean, chol = p
        old_mean, old_chol = q
        batch_shape = mean.shape[:-1]

        ####################################################################################################################
        # precompute mean and cov part of frob projection, which are used for the projection.
        mean_part, cov_part, cov, cov_old = gaussian_frobenius(policy, p, q, self.scale_prec, True)

        ################################################################################################################
        # mean projection maha/euclidean

        proj_mean = mean_projection(mean, old_mean, mean_part, eps)

        ################################################################################################################
        # cov projection frobenius

        cov_mask = cov_part > eps_cov

        if cov_mask.any():
            # alpha = ch.where(fro_norm_sq > eps_cov, ch.sqrt(fro_norm_sq / eps_cov) - 1., ch.tensor(1.))
            eta = ch.ones(batch_shape, dtype=chol.dtype, device=chol.device)
            eta[cov_mask] = ch.sqrt(cov_part[cov_mask] / eps_cov) - 1.0
            eta = ch.max(-eta, eta)

            new_cov = (cov + ch.einsum("i,ijk->ijk", eta, cov_old)) / (1.0 + eta + 1e-16)[..., None, None]
            proj_chol = ch.where(cov_mask[..., None, None], ch.cholesky(new_cov), chol)
        else:
            proj_chol = chol

        return proj_mean, proj_chol

    def trust_region_value(self, policy, p, q):
        """
        Computes the Frobenius metric between two Gaussian distributions p and q_values.
        Returns:
            mean and covariance part of
        """
        return gaussian_frobenius(policy, p, q, self.scale_prec)

    def get_trust_region_loss(
        self,
        policy: AbstractGaussianPolicy,
        p: Tuple[ch.Tensor, ch.Tensor],
        proj_p: Tuple[ch.Tensor, ch.Tensor],
    ):
        # mean_diff = (p[0] - proj_p[0]).pow(2).sum(-1)
        # TODO: mean_diff = policy.maha(p[0], proj_p[0], b_old_std)
        mean_diff = policy.maha(p[0], proj_p[0], p[1])
        if policy.contextual_std:
            # Compute MSE because Frob tends to make values explode
            cov_diff = (p[1] - proj_p[1]).pow(2).sum((-1, -2))
            delta_loss = (mean_diff + cov_diff).mean()
        else:
            delta_loss = mean_diff.mean()

        return delta_loss * self.trust_region_coeff
