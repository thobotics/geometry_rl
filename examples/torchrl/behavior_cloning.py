from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None,
    config_name="bc_cloth_insertion_graph_trpl_cfg",
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
    import wandb

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    from torch.utils.data import DataLoader
    from torchrl.envs import ExplorationType, set_exploration_type

    from geometry_rl.orbit.utils.tensordict import (
        extract_tensors_from_a_dict,
        recursively_merge_dict,
        recursively_share_memory,
    )
    from builders.agent import AgentBuilder

    # Initialize wandb
    wandb.init(project="cloth_hanging_bc")

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

    data_dict = torch.load("./logs/data/data.pt")

    recursively_merge_dict(data_dict["env"], env.state_dict(), share_memory_if_possible=True)
    recursively_share_memory(data_dict["env"], share_memory_if_possible=True)
    env.load_state_dict(data_dict["env"])

    td_data = data_dict["data"]
    td_data = td_data.reshape(
        -1,
    )

    split = int(len(td_data) * 0.8)

    train_dataloader = DataLoader(
        td_data[:split],
        batch_size=cfg.algorithm.objective.mini_batch_size,
        shuffle=True,
        collate_fn=lambda x: x,
    )

    test_dataloader = DataLoader(
        td_data[split:],
        batch_size=cfg.algorithm.objective.mini_batch_size,
        collate_fn=lambda x: x,
    )

    optimizer = torch.optim.Adam(
        list(actor.parameters()),
        lr=5e-4,
    )

    loss_fn = torch.nn.MSELoss()

    size = len(train_dataloader.dataset)
    actor.train()
    env.eval()

    device = actor.device

    for epoch in range(501):
        losses = []
        for batch, data in enumerate(train_dataloader):
            data = data.to(device)
            target = data["action"].clone().detach()

            td_out = actor(data)

            predict = td_out["action"]

            loss = loss_fn(predict, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        mean_loss = sum(losses) / len(losses)
        print(f"Epoch {epoch}: Loss: {mean_loss}")
        wandb.log({"loss": mean_loss})

        if epoch > 0 and epoch % 10 == 0:
            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                actor.eval()
                test_rewards = AgentBuilder.eval_model(actor, env, num_episodes=10)
                print(f"Epoch {epoch}: Test rewards: {test_rewards}")
                wandb.log({"test_rewards": test_rewards})
            actor.train()


if __name__ == "__main__":
    # run the main function
    main()
