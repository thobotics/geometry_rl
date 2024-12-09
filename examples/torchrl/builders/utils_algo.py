import warnings

import ast
import torch.nn
import torch.optim


from hydra.utils import get_class

from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torchrl.data import CompositeSpec
from torchrl.envs import ExplorationType
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator


def _parse_composite_keys(keys):
    composite_keys = []
    for key in keys:
        try:
            composite_keys.append(ast.literal_eval(key))
        except:
            composite_keys.append(key)

    return composite_keys


def _make_probabilistic_actor(
    input_shape,
    num_outputs,
    distribution_class,
    distribution_kwargs,
    proof_environment,
    default_obs_key,
    actor_hidden_dims,
    actor_activation,
):
    # Define policy architecture
    policy_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=actor_activation,
        out_features=num_outputs,  # predict only loc
        num_cells=actor_hidden_dims,
    )

    # Initialize policy weights
    for layer in policy_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    # Add state-independent normal scale
    policy_mlp = torch.nn.Sequential(
        policy_mlp,
        AddStateIndependentNormalScale(proof_environment.action_spec.shape[-1], scale_lb=1e-8),
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        TensorDictModule(
            module=policy_mlp,
            in_keys=[default_obs_key],
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        spec=CompositeSpec(action=proof_environment.action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    return policy_module


def _make_trpl_probabilistic_actor(
    input_shape,
    num_outputs,
    distribution_class,
    distribution_kwargs,
    proof_environment,
    default_obs_key,
    projection_type="ppo",
    **kwargs,
):
    from ironlib.algorithms.trust_region_projections.models.policy.policy_factory import (
        get_policy_network,
    )

    actor_in_features = kwargs.pop("in_features")
    actor_in_keys = _parse_composite_keys(actor_in_features)
    actor_in_features = sum([proof_environment.observation_spec[key].shape[-1] for key in actor_in_keys])

    policy = get_policy_network(
        proj_type=projection_type,
        squash=False,
        device=proof_environment.observation_spec[actor_in_keys[0]].device,
        dtype=proof_environment.observation_spec[actor_in_keys[0]].dtype,
        obs_dim=actor_in_features,
        action_dim=num_outputs,
        vf_model=None,
        **kwargs,
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        TensorDictModule(
            module=policy,
            in_keys=actor_in_keys,
            out_keys=["loc", "covariance_matrix"],
        ),
        in_keys=["loc", "covariance_matrix"],
        spec=CompositeSpec(action=proof_environment.action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    return policy_module


def make_ppo_models(proof_environment, config, total_network_updates):
    # Define input shape
    default_obs_key = list(proof_environment.observation_spec.keys())[0]
    input_shape = proof_environment.observation_spec[default_obs_key].shape

    # Define policy output distribution class
    num_outputs = proof_environment.action_spec.shape[-1]
    distribution_class = get_class(config["policy"].pop("distribution_class"))

    critic_activation = config["value"].pop("activation")
    critic_hidden_dims = config["value"].pop("hidden_sizes")

    if proof_environment.action_spec.space is not None:
        distribution_kwargs = {
            "min": proof_environment.action_spec.space.low,
            "max": proof_environment.action_spec.space.high,
        }
    else:
        # Unbounded action space
        distribution_kwargs = {}

    # Add probabilistic sampling of the actions
    if config["name"] == "trpl":
        from ironlib.algorithms.trust_region_projections.projections.projection_factory import (
            get_projection_layer,
        )

        policy_kwargs = config["policy"]
        policy_module = _make_trpl_probabilistic_actor(
            input_shape,
            num_outputs,
            distribution_class,
            distribution_kwargs,
            proof_environment,
            default_obs_key,
            config["projection"]["proj_type"],
            **policy_kwargs,
        )
        projection = get_projection_layer(
            action_dim=num_outputs,
            total_train_steps=total_network_updates,
            cpu=(False if proof_environment.observation_spec[default_obs_key].device == "cuda" else True),
            dtype=proof_environment.observation_spec[default_obs_key].dtype,
            **config["projection"],
        )
    else:
        policy_module = _make_probabilistic_actor(
            input_shape,
            num_outputs,
            distribution_class,
            distribution_kwargs,
            proof_environment,
            default_obs_key,
            config["policy"]["hidden_sizes"],
            get_class(config["policy"]["activation"]),
        )
        projection = None

    # Define value architecture
    critic_in_features = config["value"].pop("in_features")
    value_in_keys = _parse_composite_keys(critic_in_features)
    value_in_features = sum([proof_environment.observation_spec[key].shape[-1] for key in value_in_keys])
    value_mlp = MLP(
        in_features=value_in_features,
        activation_class=get_class(critic_activation),
        out_features=1,
        num_cells=critic_hidden_dims,
    )

    # Initialize value weights
    for layer in value_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    # Define value module
    value_module = ValueOperator(
        value_mlp,
        in_keys=value_in_keys,
    )

    return policy_module, value_module, projection
