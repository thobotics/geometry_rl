import contextlib

import math
import warnings
from dataclasses import dataclass
from typing import Tuple

import torch
from tensordict.nn import dispatch, ProbabilisticTensorDictSequential, TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.utils import NestedKey
from torch import distributions as d
from torchrl.objectives.ppo import PPOLoss
from torchrl.objectives.utils import distance_loss

from .utils import _clip_value_loss


class TRPLLoss(PPOLoss):
    """TRPL loss.

    The clipped importance weighted loss is computed as follows:
        loss = -min( weight * advantage, min(max(weight, 1-eps), 1+eps) * advantage)

    Args:
        actor (ProbabilisticTensorDictSequential): policy operator.
        critic (ValueOperator): value operator.

    Keyword Args:
        clip_epsilon (scalar, optional): weight clipping threshold in the TRPL loss equation.
            default: 0.2
        entropy_bonus (bool, optional): if ``True``, an entropy bonus will be added to the
            loss to favour exploratory policies.
        samples_mc_entropy (int, optional): if the distribution retrieved from the policy
            operator does not have a closed form
            formula for the entropy, a Monte-Carlo estimate will be used.
            ``samples_mc_entropy`` will control how many
            samples will be used to compute this estimate.
            Defaults to ``1``.
        entropy_coef (scalar, optional): entropy multiplier when computing the total loss.
            Defaults to ``0.01``.
        critic_coef (scalar, optional): critic loss multiplier when computing the total
            loss. Defaults to ``1.0``.
        loss_critic_type (str, optional): loss function for the value discrepancy.
            Can be one of "l1", "l2" or "smooth_l1". Defaults to ``"smooth_l1"``.
        normalize_advantage (bool, optional): if ``True``, the advantage will be normalized
            before being used. Defaults to ``False``.
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``, ie. gradients are propagated to shared
            parameters for both policy and critic losses.
        advantage_key (str, optional): [Deprecated, use set_keys(advantage_key=advantage_key) instead]
            The input tensordict key where the advantage is
            expected to be written. Defaults to ``"advantage"``.
        value_target_key (str, optional): [Deprecated, use set_keys(value_target_key=value_target_key) instead]
            The input tensordict key where the target state
            value is expected to be written. Defaults to ``"value_target"``.
        value_key (str, optional): [Deprecated, use set_keys(value_key) instead]
            The input tensordict key where the state
            value is expected to be written. Defaults to ``"state_value"``.

    .. note:
      The advantage (typically GAE) can be computed by the loss function or
      in the training loop. The latter option is usually preferred, but this is
      up to the user to choose which option is to be preferred.
      If the advantage key (``"advantage`` by default) is not present in the
      input tensordict, the advantage will be computed by the :meth:`~.forward`
      method.

        >>> ppo_loss = ClipPPOLoss(actor, critic)
        >>> advantage = GAE(critic)
        >>> data = next(datacollector)
        >>> losses = ppo_loss(data)
        >>> # equivalent
        >>> advantage(data)
        >>> losses = ppo_loss(data)

      A custom advantage module can be built using :meth:`~.make_value_estimator`.
      The default is :class:`~torchrl.objectives.value.GAE` with hyperparameters
      dictated by :func:`~torchrl.objectives.utils.default_value_kwargs`.

        >>> ppo_loss = ClipPPOLoss(actor, critic)
        >>> ppo_loss.make_value_estimator(ValueEstimators.TDLambda)
        >>> data = next(datacollector)
        >>> losses = ppo_loss(data)

    .. note::
      If the actor and the value function share parameters, one can avoid
      calling the common module multiple times by passing only the head of the
      value network to the PPO loss module:

        >>> common = SomeModule(in_keys=["observation"], out_keys=["hidden"])
        >>> actor_head = SomeActor(in_keys=["hidden"])
        >>> value_head = SomeValue(in_keys=["hidden"])
        >>> # first option, with 2 calls on the common module
        >>> model = ActorCriticOperator(common, actor_head, value_head)
        >>> loss_module = PPOLoss(model.get_policy_operator(), model.get_value_operator())
        >>> # second option, with a single call to the common module
        >>> loss_module = PPOLoss(ProbabilisticTensorDictSequential(model, actor_head), value_head)

      This will work regardless of whether separate_losses is activated or not.

    """

    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential,
        critic_network: TensorDictModule,
        *,
        projection: TensorDictModule = None,
        clip_epsilon: float = 0.2,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        trust_region_coef: float = 1.0,
        loss_critic_type: str = "smooth_l1",
        normalize_advantage: bool = True,
        gamma: float = None,
        separate_losses: bool = False,
        clip_value: float = None,
        **kwargs,
    ):
        super(TRPLLoss, self).__init__(
            actor_network,
            critic_network,
            entropy_bonus=entropy_bonus,
            samples_mc_entropy=samples_mc_entropy,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            loss_critic_type=loss_critic_type,
            normalize_advantage=normalize_advantage,
            gamma=gamma,
            separate_losses=separate_losses,
            **kwargs,
        )

        if clip_value is not None:
            if isinstance(clip_value, float):
                clip_value = torch.tensor(clip_value)
            elif isinstance(clip_value, torch.Tensor):
                if clip_value.numel() != 1:
                    raise ValueError(f"clip_value must be a float or a scalar tensor, got {clip_value}.")
            else:
                raise ValueError(f"clip_value must be a float or a scalar tensor, got {clip_value}.")
        self.register_buffer("clip_value", clip_value)

        self.trust_region_coef = trust_region_coef
        self.register_buffer("clip_epsilon", torch.tensor(clip_epsilon))
        self.projection = projection
        self._global_steps = 0

    @property
    def _clip_bounds(self):
        return (
            math.log1p(-self.clip_epsilon),
            math.log1p(self.clip_epsilon),
        )

    @property
    def out_keys(self):
        if self._out_keys is None:
            keys = ["loss_objective"]
            if self.entropy_bonus:
                keys.extend(["entropy", "loss_entropy"])
            if self.loss_critic:
                keys.append("loss_critic")
            keys.append("ESS")
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values

    def loss_critic(self, tensordict: TensorDictBase) -> torch.Tensor:
        # TODO: if the advantage is gathered by forward, this introduces an
        # overhead that we could easily reduce.
        if self.separate_losses:
            tensordict = tensordict.detach()
        try:
            target_return = tensordict.get(self.tensor_keys.value_target)
        except KeyError:
            raise KeyError(
                f"the key {self.tensor_keys.value_target} was not found in the input tensordict. "
                f"Make sure you provided the right key and the value_target (i.e. the target "
                f"return) has been retrieved accordingly. Advantage classes such as GAE, "
                f"TDLambdaEstimate and TDEstimate all return a 'value_target' entry that "
                f"can be used for the value loss."
            )

        if self.clip_value:
            try:
                old_state_value = tensordict.get(self.tensor_keys.value)
            except KeyError:
                raise KeyError(
                    f"clip_value is set to {self.clip_value}, but "
                    f"the key {self.tensor_keys.value} was not found in the input tensordict. "
                    f"Make sure that the value_key passed to PPO exists in the input tensordict."
                )

        with self.critic_network_params.to_module(self.critic_network) if self.functional else contextlib.nullcontext():
            state_value_td = self.critic_network(tensordict)

        try:
            state_value = state_value_td.get(self.tensor_keys.value)
        except KeyError:
            raise KeyError(
                f"the key {self.tensor_keys.value} was not found in the input tensordict. "
                f"Make sure that the value_key passed to PPO is accurate."
            )

        loss_value = distance_loss(
            target_return,
            state_value,
            loss_function=self.loss_critic_type,
        )

        clip_fraction = None
        if self.clip_value:
            loss_value, clip_fraction = _clip_value_loss(
                old_state_value,
                state_value,
                self.clip_value.to(state_value.device),
                target_return,
                loss_value,
                self.loss_critic_type,
            )
        return self.critic_coef * loss_value

    def _log_weight_and_projection(self, tensordict: TensorDictBase) -> Tuple[torch.Tensor, d.Distribution]:
        # current log_prob of actions
        action = tensordict.get(self.tensor_keys.action)
        if action.requires_grad:
            raise RuntimeError(f"tensordict stored {self.tensor_keys.action} requires grad.")

        previous_dist = self.actor_network.build_dist_from_params(tensordict)
        with self.actor_network_params.to_module(self.actor_network) if self.functional else contextlib.nullcontext():
            current_dist = self.actor_network.get_dist(tensordict)

        p = (current_dist.mean.cpu(), current_dist.covariance_matrix.cpu())
        q = (previous_dist.mean.cpu(), previous_dist.covariance_matrix.cpu())
        policy = self.actor_network.get_submodule("0").module
        proj_p = self.projection(policy, p, q, self._global_steps)
        dist = current_dist.__class__(proj_p[0].to(action.device), proj_p[1].to(action.device))
        log_prob = dist.log_prob(action)

        prev_log_prob = tensordict.get(self.tensor_keys.sample_log_prob)
        if prev_log_prob.requires_grad:
            raise RuntimeError("tensordict prev_log_prob requires grad.")

        log_weight = (log_prob - prev_log_prob).unsqueeze(-1)
        return log_weight, dist, policy, p, proj_p

    def log_tr_metrics(self, projection, c_policy, p, q):
        """
        Execute additional regression steps to match policy output and projection.
        The policy parameters are updated in-place.
        """
        metrics_dict = {}
        # get prediction before the regression to compare to regressed policy
        with torch.no_grad():
            constraints_initial_dict = projection.compute_metrics(c_policy, p, q, step=self._global_steps)
        metrics_dict["kl"] = constraints_initial_dict["kl"].item()
        metrics_dict["constraint"] = constraints_initial_dict["constraint"].item()
        metrics_dict["mean_constraint"] = constraints_initial_dict["mean_constraint"].item()
        metrics_dict["mean_constraint_max"] = constraints_initial_dict["mean_constraint_max"].item()
        metrics_dict["cov_constraint"] = constraints_initial_dict["cov_constraint"].item()
        metrics_dict["cov_constraint_max"] = constraints_initial_dict["cov_constraint_max"].item()
        metrics_dict["entropy"] = constraints_initial_dict["entropy"].item()
        metrics_dict["entropy_diff"] = constraints_initial_dict["entropy_diff"].item()

        return metrics_dict

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self._cached_critic_params_detached,
                target_params=self.target_critic_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        if self.normalize_advantage and advantage.numel() > 1:
            loc = advantage.mean().item()
            scale = advantage.std().clamp_min(1e-6).item()
            advantage = (advantage - loc) / scale

        log_weight, dist, policy, p, proj_p = self._log_weight_and_projection(tensordict)

        # ESS for logging
        with torch.no_grad():
            # In theory, ESS should be computed on particles sampled from the same source. Here we sample according
            # to different, unrelated trajectories, which is not standard. Still it can give a idea of the dispersion
            # of the weights.
            lw = log_weight.squeeze()
            ess = (2 * lw.logsumexp(0) - (2 * lw).logsumexp(0)).exp()
            batch = log_weight.shape[0]

        gain1 = log_weight.exp() * advantage
        td_out = TensorDict({"loss_objective": -gain1.mean()}, [])

        # Note that trust_region_loss is already multiplied with the coefficient
        trust_region_loss = self.projection.get_trust_region_loss(policy, p, proj_p)
        td_out.set("loss_trust_region", trust_region_loss)

        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.mean().detach())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy.mean())
        if self.critic_coef:
            loss_critic = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic.mean())
        td_out.set("ESS", ess.mean() / batch)

        tr_metrics = self.log_tr_metrics(self.projection, policy, p, proj_p)
        for k, v in tr_metrics.items():
            td_out.set(k, v)
        return td_out
