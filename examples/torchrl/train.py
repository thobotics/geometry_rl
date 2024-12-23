from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig, OmegaConf

simulation_app = None


@hydra.main(
    version_base=None,
    config_name="cartpole_ppo_cfg",
    config_path=f"{os.getcwd()}/configs",
)
def main(cfg: "DictConfig"):  # noqa: F821
    """Start Isaac Sim Simulator first."""
    global simulation_app
    from geometry_rl.orbit.utils.omniverse_app import launch_app  # noqa

    simulation_app = launch_app(config=OmegaConf.to_container(cfg.simulator, resolve=True))

    """ Rest everything follows. """
    import time
    import torch.optim
    import tqdm
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    from collections import OrderedDict
    from datetime import datetime
    from tensordict import TensorDict
    from torchrl.collectors import SyncDataCollector
    from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
    from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
    from torchrl.objectives.value.advantages import GAE
    from torchrl.record.loggers import generate_exp_name, get_logger
    from torchrl.envs import ExplorationType, set_exploration_type
    from pprint import pformat

    from torchmetrics.regression import ExplainedVariance

    from geometry_rl.orbit.utils.tensordict import (
        extract_tensors_from_a_dict,
        recursively_merge_dict,
    )
    from builders.agent import AgentBuilder

    device = cfg.env.device
    num_mini_batches = cfg.collector.frames_per_batch // cfg.algorithm.objective.mini_batch_size
    total_network_updates = (
        (cfg.collector.total_frames // cfg.collector.frames_per_batch)
        * cfg.algorithm.objective.ppo_epochs
        * num_mini_batches
    )

    env_config = OmegaConf.to_container(cfg.env, resolve=True)
    algo_config = OmegaConf.to_container(cfg.algorithm, resolve=True)

    env, actor, critic, projection, loss_module = AgentBuilder.build(
        env_config=env_config,
        algo_config=algo_config,
        env_kwargs={},
        algo_kwargs={"total_network_updates": total_network_updates},
    )

    # Initialize the networks
    init_tensordict = env.reset()
    actor(init_tensordict)
    critic(init_tensordict)

    # Set the base directory and experiment name
    checkpoint_load_dir = cfg.logger.checkpoint.load_dir
    if checkpoint_load_dir is not None:
        abs_log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        logs_base_dir = (
            abs_log_dir[: abs_log_dir.find(cfg.experiment_name) - 1]
            if cfg.logger.log_dir is None
            else cfg.logger.log_dir
        )
        experiment_name = cfg.experiment_name
        log_dir = f"{logs_base_dir}/{experiment_name}/{checkpoint_load_dir}"
        model_checkpoint = (
            "model_checkpoint_best.pth"
            if cfg.logger.checkpoint.model_checkpoint is None
            else cfg.logger.checkpoint.model_checkpoint
        )

        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, model_checkpoint)

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)

            # For some reason, the env state dict is not loaded properly.
            # So, we need to manually merge the state dicts, and make it share the same memory.
            recursively_merge_dict(checkpoint["env"], env.state_dict(), share_memory_if_possible=True)
            env.load_state_dict(checkpoint["env"])

            actor.load_state_dict(checkpoint["actor"])
            critic.load_state_dict(checkpoint["critic"])

            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Create collector
    split_trajs = False

    collector = SyncDataCollector(
        create_env_fn=env,
        policy=actor,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        storing_device="cpu",
        max_frames_per_traj=-1,
        split_trajs=split_trajs,
    )

    # Create data buffer
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.collector.frames_per_batch),
        sampler=sampler,
        batch_size=cfg.algorithm.objective.mini_batch_size,
    )

    # Create loss and adv modules
    adv_module = GAE(
        gamma=cfg.algorithm.objective.gamma,
        lmbda=cfg.algorithm.objective.gae_lambda,
        value_network=critic,
        average_gae=False,
        shifted=True,  # This prevents the GAE using vmap, which is not supported by the GNN
    )

    explained_value_variance = ExplainedVariance()

    # Create optimizers
    actor_optim = torch.optim.Adam(actor.parameters(), lr=cfg.algorithm.optim.lr, eps=1e-5)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=cfg.algorithm.optim.lr, eps=1e-5)

    # Create logger
    logger = None
    if cfg.logger.backend:
        prefix_exp_name = {f"{k}={v}" for k, v in cfg.logger.experiment_name.items()}
        prefix_exp_name = "_".join(prefix_exp_name)

        exp_name = "_".join(
            (
                prefix_exp_name,
                datetime.now().strftime("%y_%m_%d-%H_%M_%S"),
            )
        )

        job_type = OrderedDict()
        for k, v in cfg.logger.job_keys.items():
            if v is not None:
                if k == "extra":
                    for extra in v[0].split(","):
                        k_extra, v_extra = extra.split("=")
                        job_type[k_extra] = OmegaConf.select(cfg, v_extra)
                else:
                    job_type[k] = OmegaConf.select(cfg, v)
        job_type = "_".join([f"{k}={v}" for k, v in job_type.items()])

        group = "_".join(OmegaConf.to_container(cfg.logger.group, resolve=True))

        logger = get_logger(
            cfg.logger.backend,
            logger_name=cfg.logger.log_dir,
            experiment_name=exp_name,
            wandb_kwargs={
                "project": f"geometry_rl_{cfg.logger.project}",
                "group": group,
                "job_type": job_type,
            },
        )

    # Main loop
    collected_frames = 0
    num_network_updates = 0
    start_time = time.time()
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    sampling_start = time.time()

    # extract cfg variables
    cfg_loss_ppo_epochs = cfg.algorithm.objective.ppo_epochs
    cfg_optim_anneal_lr = cfg.algorithm.optim.anneal_lr
    cfg_optim_lr = cfg.algorithm.optim.lr
    cfg_clip_grad_norm = cfg.algorithm.objective.clip_grad_norm
    cfg_max_grad_norm = cfg.algorithm.objective.max_grad_norm

    if cfg.algorithm.name == "ppo":
        cfg_loss_anneal_clip_eps = cfg.algorithm.objective.anneal_clip_epsilon
        cfg_loss_clip_epsilon = cfg.algorithm.objective.clip_epsilon
    else:
        cfg_loss_anneal_clip_eps = False
        cfg_loss_clip_epsilon = 0.2
    losses = TensorDict({}, batch_size=[cfg_loss_ppo_epochs, num_mini_batches])

    # Create checkpoint directory
    checkpoint_dir = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "checkpoints")

    # Check if the checkpoint directory exists, create it if not
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    highest_reward_saved = -float("inf")

    # Dump the config
    env.dump_env_cfg(
        os.path.join(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            "scene.yaml",
        ),
    )

    if logger:
        logger.log_hparams(cfg)
        logger.log_hparams({"scene": env.unwrapped.cfg.to_dict()})

    # Main training loop
    for i, data in enumerate(collector):
        log_info = {}
        sampling_time = time.time() - sampling_start
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(data.numel())

        # Get training rewards and episode lengths
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "done"]]
            log_info.update(
                {
                    "train/reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item() / len(episode_length),
                }
            )

        # Compute GAE
        with torch.no_grad():
            data = data.to(device)
            data = adv_module(data)
        data_reshape = data.reshape(-1)

        # Update the data buffer
        data_buffer.extend(data_reshape)

        training_start = time.time()
        for j in range(cfg_loss_ppo_epochs):
            for k, batch in enumerate(data_buffer):
                # Get a data batch
                batch = batch.to(device)

                # Linearly decrease the learning rate and clip epsilon
                alpha = 1.0
                alpha_eps = 1.0
                if cfg_optim_anneal_lr:
                    alpha = 1 - (num_network_updates / total_network_updates)
                    for group in actor_optim.param_groups:
                        group["lr"] = cfg_optim_lr * alpha
                    for group in critic_optim.param_groups:
                        group["lr"] = cfg_optim_lr * alpha
                if cfg.algorithm.name == "ppo" and cfg_loss_anneal_clip_eps:
                    alpha_eps = 1 - (num_network_updates / total_network_updates)
                    loss_module.clip_epsilon.copy_(cfg_loss_clip_epsilon * alpha_eps)
                loss_module._global_steps = num_network_updates
                num_network_updates += 1

                # Forward pass PPO loss
                loss = loss_module(batch)
                loss_types = ["loss_critic", "loss_objective"]
                if cfg.algorithm.objective.entropy_bonus:
                    loss_types.append("loss_entropy")
                if cfg.algorithm.name == "trpl":
                    loss_types.append("loss_trust_region")
                    loss_types.append("kl")
                    loss_types.append("constraint")
                    loss_types.append("mean_constraint")
                    loss_types.append("mean_constraint_max")
                    loss_types.append("cov_constraint")
                    loss_types.append("cov_constraint_max")
                    loss_types.append("entropy")
                    loss_types.append("entropy_diff")

                losses[j, k] = loss.select(*loss_types).detach()

                critic_loss = loss["loss_critic"]
                actor_loss = loss["loss_objective"]
                if cfg.algorithm.objective.entropy_bonus:
                    actor_loss += loss["loss_entropy"]
                if cfg.algorithm.name == "trpl":
                    actor_loss += loss["loss_trust_region"]

                # Backward pass
                actor_loss.backward()
                critic_loss.backward()

                # Clip gradients
                if cfg_clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg_max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg_max_grad_norm)

                # Update the networks
                actor_optim.step()
                critic_optim.step()
                actor_optim.zero_grad()
                critic_optim.zero_grad()

        # Get training losses and times
        training_time = time.time() - training_start
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            log_info.update({f"train/{key}": value.item()})
        log_info.update(
            {
                "train/explained_variance": explained_value_variance(data["state_value"], data["value_target"]).item(),
                "train/lr": alpha * cfg_optim_lr,
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
                "train/clip_epsilon": (
                    alpha_eps * cfg_loss_clip_epsilon if cfg_loss_anneal_clip_eps else cfg_loss_clip_epsilon
                ),
            }
        )

        # Save the model checkpoint
        if "train/reward" in log_info:
            highest_reward = log_info["train/reward"]
            if (
                cfg.logger.checkpoint.save_each_n_iter
                and cfg.logger.checkpoint.save_interval > 0
                and i % cfg.logger.checkpoint.save_interval == 0
            ):
                model_path = os.path.join(checkpoint_dir, f"model_checkpoint_{i}.pth")
                torch.save(
                    {
                        "env": extract_tensors_from_a_dict(env.state_dict()),
                        "actor": actor.state_dict(),
                        "critic": critic.state_dict(),
                        "reward": highest_reward,
                    },
                    model_path,
                )
                if logger:
                    logger.experiment.save(model_path)
            if cfg.logger.checkpoint.save_best and highest_reward > highest_reward_saved:
                model_path = os.path.join(checkpoint_dir, "model_checkpoint_best.pth")
                torch.save(
                    {
                        "env": extract_tensors_from_a_dict(env.state_dict()),
                        "actor": actor.state_dict(),
                        "critic": critic.state_dict(),
                        "reward": highest_reward,
                    },
                    model_path,
                )
                if logger:
                    logger.experiment.save(model_path)
                highest_reward_saved = highest_reward

        print(pformat(log_info))
        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)

        collector.update_policy_weights_()
        sampling_start = time.time()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
