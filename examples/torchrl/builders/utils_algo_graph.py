import warnings


import ast
import torch.nn
import torch.optim
import torch.nn.functional as F


from hydra.utils import get_class, instantiate

from tensordict.nn import TensorDictModule
from torchrl.data import CompositeSpec
from torchrl.envs import ExplorationType
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator


def _build_pyg_model(
    proof_environment,
    config,
    input_dim_node,
    input_dim_edge,
    node_type_class,
    edge_type_class,
    edge_level_class,
    device,
):
    if "message_passing" not in config:
        config["input_dim_node"] = input_dim_node
    else:
        message_passing_cfg = config.pop("message_passing")
        message_passing = []
        shared_message_passing = []
        for level, message_cfg in enumerate(message_passing_cfg):
            message_passing.append([])

            if config["shared_processor"]:
                shared_message_passing.append(instantiate(message_cfg["processor"]).to(device))

            for i in range(config["num_messages"]):
                if message_cfg["code"][i] == 1:
                    if config["shared_processor"]:
                        message_passing[level].append(shared_message_passing[level])
                    else:
                        message_passing[level].append(instantiate(message_cfg["processor"]).to(device))
                else:
                    message_passing[level].append(None)

        config["message_passing"] = message_passing
        config["input_dim_node"] = input_dim_node
        config["input_dim_edge"] = input_dim_edge
        config["node_type_mapping"] = node_type_class
        config["edge_type_mapping"] = edge_type_class
        config["edge_level_mapping"] = edge_level_class

    model = instantiate(config).to(device)

    return model


def _make_pyg_agent(
    proof_environment,
    config,
    device: torch.device = torch.device("cuda"),
):
    data_cfg = config["data"]

    observation_dim = proof_environment.env.observation_manager.group_obs_term_dim
    observation_names = proof_environment.env.observation_manager._group_obs_term_names
    data_cfg["base_data"]["observation_dim"] = observation_dim
    data_cfg["base_data"]["observation_names"] = observation_names

    base_data = instantiate(data_cfg["base_data"])

    node_type_class = get_class(data_cfg["node_type_class"])
    edge_type_class = get_class(data_cfg["edge_type_class"])
    edge_level_class = get_class(data_cfg["edge_level_class"])

    input_dim_node = len(node_type_class) + data_cfg["input_node_aux_dim"]
    input_dim_edge = len(edge_type_class) + data_cfg["input_edge_aux_dim"]

    model_cfg = config["model"]

    mgn = _build_pyg_model(
        proof_environment,
        model_cfg,
        input_dim_node,
        input_dim_edge,
        node_type_class,
        edge_type_class,
        edge_level_class,
        device,
    )

    return mgn, base_data


def _parse_composite_keys(keys):
    composite_keys = []
    for key in keys:
        try:
            composite_keys.append(ast.literal_eval(key))
        except:
            composite_keys.append(key)

    return composite_keys


def _make_probabilistic_actor(
    num_outputs,
    distribution_class,
    distribution_kwargs,
    mgn,
    hyper_data,
    proof_environment,
    projection_type="ppo",
    **kwargs,
):
    from ironlib.algorithms.trust_region_projections.models.policy.policy_factory import (
        get_policy_network,
    )

    actor_in_features = kwargs.pop("in_features")
    actor_in_keys = _parse_composite_keys(actor_in_features)
    out_dim = kwargs.pop("out_dim")
    if out_dim is not None:
        num_outputs = out_dim
        num_actuators = len(out_dim)
    else:
        num_actuators = proof_environment.action_space.shape[-1] // kwargs.pop("action_dim")

    policy = get_policy_network(
        proj_type=projection_type,
        squash=False,
        device=proof_environment.device,
        dtype=proof_environment.observation_spec.dtype,
        action_dim=num_outputs,
        num_actuators=num_actuators,  # 3 DoF per actuator
        vf_model=None,
        mgn=mgn,
        hyper_data=hyper_data,
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


def _make_value_module(
    critic_type,
    critic_in_features,
    mgn,
    hyper_data,
    proof_environment,
    **kwargs,
):
    from ironlib.algorithms.trust_region_projections.models.value.critic_factory import (
        get_critic,
    )

    value_in_keys = _parse_composite_keys(critic_in_features)
    value_in_features = sum([proof_environment.observation_spec[key].shape[-1] for key in value_in_keys])

    if critic_type == "gnn":
        value_net = get_critic(
            critic_type=critic_type,
            dim=value_in_features,
            mgn=mgn,
            hyper_data=hyper_data,
            **kwargs,
        )
    else:
        value_net = MLP(
            in_features=value_in_features,
            activation_class=get_class(kwargs["activation"]),
            out_features=1,
            num_cells=kwargs["hidden_dims"],
        )

    # Initialize value weights
    for layer in value_net.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    value_module = ValueOperator(
        value_net,
        in_keys=value_in_keys,
    )

    return value_module


def make_ppo_models(proof_environment, config, total_network_updates):
    # Define policy output distribution class
    num_outputs = proof_environment.action_spec.shape[-1]
    distribution_class = get_class(config["policy"].pop("distribution_class"))
    shared_critic = config["policy"].pop("shared_critic")

    critic_type = config["value"].pop("value_type")
    critic_in_features = config["value"].pop("in_features")

    if proof_environment.action_spec.space is not None:
        distribution_kwargs = {
            "min": proof_environment.action_spec.space.low,
            "max": proof_environment.action_spec.space.high,
        }
    else:
        # Unbounded action space
        distribution_kwargs = {}

    mgn, hyper_data = _make_pyg_agent(
        proof_environment=proof_environment,
        config=config["policy"].pop("pyg_agent"),
        device=proof_environment.device,
    )

    # Add probabilistic sampling of the actions
    policy_kwargs = config["policy"]
    policy_module = _make_probabilistic_actor(
        num_outputs,
        distribution_class,
        distribution_kwargs,
        mgn,
        hyper_data,
        proof_environment,
        config["projection"]["proj_type"],
        **policy_kwargs,
    )
    projection = None

    if config["name"] == "trpl":
        from ironlib.algorithms.trust_region_projections.projections.projection_factory import (
            get_projection_layer,
        )

        projection = get_projection_layer(
            action_dim=num_outputs,
            total_train_steps=total_network_updates,
            cpu=False if proof_environment.device == "cuda" else True,
            dtype=proof_environment.observation_spec.dtype,
            **config["projection"],
        )

    if critic_type == "gnn":
        mgn, hyper_data = _make_pyg_agent(
            proof_environment=proof_environment,
            config=config["value"].pop("pyg_agent"),
            device=proof_environment.device,
        )

    # Define value architecture
    value_module = _make_value_module(
        critic_type,
        critic_in_features,
        mgn,
        hyper_data,
        proof_environment,
        **config["value"],
    )

    return policy_module, value_module, projection
