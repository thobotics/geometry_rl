import logging

import copy
import numpy as np
import torch as ch
from typing import Tuple, Union

from ..utils.projection_utils import gaussian_kl
from ..models.policy.abstract_gaussian_policy import (
    AbstractGaussianPolicy,
)
from .base_projection_layer import (
    BaseProjectionLayer,
)
from ..utils.torch_utils import torch_batched_trace

logger = logging.getLogger("papi_projection")


class PAPIProjection(BaseProjectionLayer):
    def __init__(
        self,
        proj_type: str = "",
        mean_bound: float = 0.015,
        cov_bound: float = 0.0,
        trust_region_coeff: float = 0.0,
        scale_prec: bool = False,
        entropy_schedule: Union[None, str] = None,
        action_dim: Union[None, int] = None,
        total_train_steps: Union[None, int] = None,
        target_entropy: float = 0.0,
        temperature: float = 0.5,
        entropy_eq: bool = False,
        entropy_first: bool = True,
        cpu: bool = True,
        dtype: ch.dtype = ch.float32,
    ):
        assert entropy_first
        super().__init__(
            proj_type,
            mean_bound,
            cov_bound,
            trust_region_coeff,
            scale_prec,
            entropy_schedule,
            action_dim,
            total_train_steps,
            target_entropy,
            temperature,
            entropy_eq,
            entropy_first,
            cpu,
            dtype,
        )

        self.last_policies = []

    def __call__(self, policy, p, q, step=0, *args, **kwargs):
        if kwargs.get("obs"):
            self._papi_steps(policy, q, **kwargs)
        else:
            return p

    def _trust_region_projection(
        self,
        policy: AbstractGaussianPolicy,
        p: Tuple[ch.Tensor, ch.Tensor],
        q: Tuple[ch.Tensor, ch.Tensor],
        eps: Union[ch.Tensor, float],
        eps_cov: Union[ch.Tensor, float],
        **kwargs,
    ):
        """
        runs papi projection layer and constructs sqrt of covariance
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

        mean, chol = p
        old_mean, old_chol = q
        intermed_mean = kwargs.get("intermed_mean")

        dtype = mean.dtype
        device = mean.device

        dim = mean.shape[-1]

        ################################################################################################################
        # Precompute basic matrices

        # Joint bound
        eps += eps_cov

        I = ch.eye(dim, dtype=dtype, device=device)
        old_precision = ch.cholesky_solve(I, old_chol)[0]
        logdet_old = policy.log_determinant(old_chol)
        cov = policy.covariance(chol)

        ################################################################################################################
        # compute expected KL
        maha_part, cov_part = gaussian_kl(policy, p, q)
        maha_part = maha_part.mean()
        cov_part = cov_part.mean()

        if intermed_mean is not None:
            maha_intermediate = (
                0.5 * policy.maha(intermed_mean, old_mean, old_chol).mean()
            )
            mm = ch.min(maha_part, maha_intermediate)

        ################################################################################################################
        # matrix rotation/rescaling projection
        if maha_part + cov_part > eps + 1e-6:
            old_cov = policy.covariance(old_chol)

            maha_delta = eps if intermed_mean is None else (eps - mm)
            eta_rot = maha_delta / ch.max(
                maha_part + cov_part, ch.tensor(1e-16, dtype=dtype, device=device)
            )
            new_cov = (1 - eta_rot) * old_cov + eta_rot * cov
            proj_chol = ch.cholesky(new_cov)

            # recompute covariance part of KL for new chol
            trace_term = (
                0.5 * (torch_batched_trace(old_precision @ new_cov) - dim).mean()
            )  # rotation difference
            entropy_diff = 0.5 * (logdet_old - policy.log_determinant(proj_chol)).mean()

            cov_part = trace_term + entropy_diff

        else:
            proj_chol = chol

        ################################################################################################################
        # mean interpolation projection
        if maha_part + cov_part > eps + 1e-6:
            if intermed_mean is not None:
                a = 0.5 * policy.maha(mean, intermed_mean, old_chol).mean()
                b = (
                    0.5
                    * (
                        (mean - intermed_mean)
                        @ old_precision
                        @ (intermed_mean - old_mean).T
                    ).mean()
                )
                c = maha_intermediate - ch.max(
                    eps - cov_part, ch.tensor(0.0, dtype=dtype, device=device)
                )
                eta_mean = (
                    -b
                    + ch.sqrt(
                        ch.max(
                            b * b - a * c, ch.tensor(1e-16, dtype=dtype, device=device)
                        )
                    )
                ) / ch.max(a, ch.tensor(1e-16, dtype=dtype, device=device))
            else:
                eta_mean = ch.sqrt(
                    ch.max(eps - cov_part, ch.tensor(1e-16, dtype=dtype, device=device))
                    / ch.max(maha_part, ch.tensor(1e-16, dtype=dtype, device=device))
                )
        else:
            eta_mean = ch.tensor(1.0, dtype=dtype, device=device)

        return eta_mean, proj_chol

    def _papi_steps(
        self,
        policy: AbstractGaussianPolicy,
        q: Tuple[ch.Tensor, ch.Tensor],
        obs: ch.Tensor,
        lr_schedule,
        lr_schedule_vf=None,
    ):
        """
        Take PAPI steps after PPO finished its steps. Policy parameters are updated in-place.
        Args:
            policy: policy instance
            q: old distribution
            obs: collected observations from sampling
            lr_schedule: lr schedule for policy
            lr_schedule_vf: lr schedule for vf

        Returns:

        """
        assert not policy.contextual_std

        # save latest policy in history
        self.last_policies.append(copy.deepcopy(policy))

        ################################################################################################################
        # policy backtracking: out of last n policies and current one find one that satisfies the kl constraint

        intermed_policy = None
        n_backtracks = 0

        for i, pi in enumerate(reversed(self.last_policies)):
            p_prime = pi(obs)
            mean_part, cov_part = pi.kl_divergence(p_prime, q)
            if (mean_part + cov_part).mean() <= self.mean_bound + self.cov_bound:
                intermed_policy = pi
                n_backtracks = i
                break

        ################################################################################################################
        # LR update

        # reduce learning rate when appropriate policy not within the last 4 epochs
        if n_backtracks >= 4 or intermed_policy is None:
            # Linear learning rate annealing
            lr_schedule.step()
            if lr_schedule_vf:
                lr_schedule_vf.step()

        if intermed_policy is None:
            # pop last policy and make it current one, as the updated one was poor
            # do not keep last policy in history, otherwise we could stack the same policy multiple times.
            if len(self.last_policies) >= 1:
                policy.load_state_dict(self.last_policies.pop().state_dict())
                logger.warning(
                    f"No suitable policy found in backtracking of {len(self.last_policies)} policies."
                )
            return

        ################################################################################################################
        # PAPI iterations

        # We assume only non contextual covariances here, therefore we only need to project for one
        q = (q[0], q[1][:1])  # (means, covs[:1])

        # This is A from Alg. 2 [Akrour et al., 2019]
        intermed_weight = intermed_policy.get_last_layer().detach().clone()
        # This is A @ phi(s)
        intermed_mean = p_prime[0].detach().clone()

        entropy = policy.entropy(q)
        entropy_bound = (
            obs.new_tensor([-np.inf])
            if entropy / self.initial_entropy > 0.5
            else entropy - (self.mean_bound + self.cov_bound)
        )

        for _ in range(20):
            eta, proj_chol = self._projection(
                intermed_policy,
                (p_prime[0], p_prime[1][:1]),
                q,
                self.mean_bound,
                self.cov_bound,
                entropy_bound,
                intermed_mean=intermed_mean,
            )
            intermed_policy.papi_weight_update(eta, intermed_weight)
            intermed_policy.set_std(proj_chol[0])
            p_prime = intermed_policy(obs)

        policy.load_state_dict(intermed_policy.state_dict())
