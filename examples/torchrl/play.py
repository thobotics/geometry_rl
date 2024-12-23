from __future__ import annotations

import os

from omegaconf import DictConfig, OmegaConf
from hydra.experimental import compose
from hydra import initialize_config_dir
import json
import argparse

simulation_app = None


def find_latest_experiment_log_dir(base_dir, experiment_name):
    experiment_dir = os.path.join(base_dir, experiment_name)
    all_subdirs = [
        os.path.join(experiment_dir, d)
        for d in os.listdir(experiment_dir)
        if os.path.isdir(os.path.join(experiment_dir, d))
    ]
    latest_dir = sorted(all_subdirs, key=lambda x: os.path.basename(x), reverse=True)[0]
    return latest_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Hydra-based script with specific config.")
    parser.add_argument(
        "-cd",
        "--config_path",
        type=str,
        default="configs",
        help="Path to the Hydra configuration directory.",
    )
    parser.add_argument(
        "-cn",
        "--config_name",
        type=str,
        default="cartpole_ppo_cfg",
        help="Name of the Hydra configuration file.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/geometry_rl",
        help="Path to the Hydra configuration directory.",
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        help="Path to the experiment directory. Otherwise, the latest experiment is used.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="all",
    )
    parser.add_argument(
        "--eval_name",
        type=str,
        default="eval",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=16,
        help="Number of environments to run in parallel.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to run for evaluation.",
    )
    parser.add_argument(
        "--exploration_type",
        type=str,
        default="mode",
        help="Type of exploration to use during evaluation.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Whether to run the simulator in headless mode.",
    )
    parser.add_argument(
        "--save_data",
        action="store_true",
        default=False,
        help="Whether to run the simulator in headless mode.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="logs/data",
        help="Path to the directory to save the data.",
    )
    return parser.parse_args()


def main(
    cfg: "DictConfig",
    log_dir: str,
    checkpoint_name: str,
    eval_name: str,
    num_episodes: int = 5,
    save_data: bool = False,
    save_dir: str = "logs/data",
    exploration_type: str = "mode",
    **kwargs,
):  # noqa: F821
    """Start Isaac Sim Simulator first."""
    global simulation_app
    from geometry_rl.orbit.utils.omniverse_app import launch_app  # noqa

    simulation_app = launch_app(config=OmegaConf.to_container(cfg.simulator, resolve=True))

    """ Rest everything follows. """
    import time
    import torch.optim
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    from torchrl.envs import ExplorationType, set_exploration_type
    from geometry_rl.orbit.utils.tensordict import (
        recursively_merge_dict,
        recursively_share_memory,
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

    env_config["seed"] = 999

    env, actor, critic, projection, _ = AgentBuilder.build(
        env_config=env_config,
        algo_config=algo_config,
        env_kwargs={},
        algo_kwargs={"total_network_updates": total_network_updates},
    )

    env.reset()

    checkpoint_dir = os.path.join(log_dir, "checkpoints")

    if checkpoint_name != "all":
        checkpoint_names = [checkpoint_name]
    else:
        checkpoint_names = sorted(
            [
                name
                for name in os.listdir(checkpoint_dir)
                if name.startswith("model_checkpoint") and not name.endswith("best.pth")
            ],
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )

    test_rewards_dict = {}

    for i, checkpoint_name in enumerate(checkpoint_names):
        env.reset()
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        # Load the model checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            highest_reward = checkpoint["reward"]

            # For some reason, the env state dict is not loaded properly.
            # So, we need to manually merge the state dicts, and make it share the same memory.
            recursively_merge_dict(checkpoint["env"], env.state_dict(), share_memory_if_possible=True)
            recursively_share_memory(checkpoint["env"], share_memory_if_possible=True)
            env.load_state_dict(checkpoint["env"])

            actor.load_state_dict(checkpoint["actor"])

            print(f"Loaded checkpoint from {checkpoint_path} with highest reward: {highest_reward}")
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Get test rewards
        with torch.no_grad(), set_exploration_type(ExplorationType.from_str(exploration_type)):
            print(f"Evaluating the model with exploration_type={exploration_type}")

            env.eval()
            actor.eval()
            eval_start = time.time()
            if save_data:
                test_rewards = AgentBuilder.generate_data(actor, env, num_episodes=num_episodes, save_dir=save_dir)
            else:
                test_rewards = AgentBuilder.eval_model(actor, env, num_episodes=num_episodes)
            eval_time = time.time() - eval_start

            print(f"Test rewards: {test_rewards} took {eval_time:.2f} seconds to finish")
            test_rewards_dict[i] = test_rewards

    print(f"Final test rewards: {test_rewards_dict}")
    json_path = os.path.join(log_dir, f"{eval_name}.json")
    with open(json_path, "w") as f:
        json.dump(test_rewards_dict, f)


if __name__ == "__main__":
    args = parse_args()

    # Initialize Hydra and compose the config
    initialize_config_dir(f"{os.getcwd()}/{args.config_path}")
    cfg = compose(config_name=args.config_name)

    # Set the base directory and experiment name
    logs_base_dir = f"{os.getcwd()}/{args.log_dir}"
    experiment_name = cfg.experiment_name

    # Find the latest log directory for the experiment
    if args.experiment_dir is not None:
        latest_log_dir = f"{logs_base_dir}/{experiment_name}/{args.experiment_dir}"
    else:
        latest_log_dir = find_latest_experiment_log_dir(logs_base_dir, experiment_name)
    latest_cfg_path = os.path.join(latest_log_dir, ".hydra", "config.yaml")
    latest_cfg = OmegaConf.load(latest_cfg_path)
    merged_cfg = OmegaConf.merge(cfg, latest_cfg)

    # Override the number of environments and headless mode
    merged_cfg.env.num_envs = args.num_envs
    merged_cfg.simulator.headless = args.headless

    del args.log_dir
    del args.experiment_dir
    del args.config_path
    del args.config_name
    del args.num_envs
    del args.headless

    # Use merged_cfg as needed
    print(OmegaConf.to_yaml(merged_cfg))
    main(
        merged_cfg,
        latest_log_dir,
        **vars(args),
    )

    # close sim app
    if simulation_app is not None:
        simulation_app.close()
