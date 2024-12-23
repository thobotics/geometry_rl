from pathlib import Path
from .utils_env import *
from torchrl.objectives import ClipPPOLoss, KLPENPPOLoss
from geometry_rl.algorithms.trust_region_projections.objectives.trpl import TRPLLoss
from geometry_rl.algorithms.trust_region_projections.objectives.ppo import ClipPPOLoss2
from geometry_rl.orbit.utils.tensordict import extract_tensors_from_a_dict
from .utils_algo_graph import make_ppo_models


class AgentBuilder:
    """Builing agent given configs"""

    @classmethod
    def build(
        cls,
        env_config,
        algo_config,
        env_kwargs,
        algo_kwargs,
    ):

        device = env_config["device"]
        env = make_orbit_env(
            env_name=env_config["name"],
            device=device,
            batch_size=env_config["num_envs"],
            env_config=env_config,
            **env_kwargs,
        )

        actor, critic, projection = make_ppo_models(
            proof_environment=env,
            config=algo_config,
            **algo_kwargs,
        )

        actor, critic = actor.to(device), critic.to(device)

        if algo_config["name"] == "trpl":
            loss_module = TRPLLoss(
                actor_network=actor,
                critic_network=critic,
                projection=projection,
                clip_epsilon=algo_config["objective"]["clip_epsilon"],
                loss_critic_type=algo_config["objective"]["loss_critic_type"],
                trust_region_coef=algo_config["projection"]["trust_region_coeff"],
                entropy_coef=algo_config["objective"]["entropy_coef"],
                entropy_bonus=algo_config["objective"].get("entropy_bonus", False),
                critic_coef=algo_config["objective"]["critic_coef"],
                clip_value=algo_config["objective"]["clip_value"],
                normalize_advantage=True,
            )
        elif algo_config["name"] == "ppo":
            loss_module = ClipPPOLoss2(
                actor_network=actor,
                critic_network=critic,
                clip_epsilon=algo_config["objective"]["clip_epsilon"],
                loss_critic_type=algo_config["objective"]["loss_critic_type"],
                entropy_coef=algo_config["objective"]["entropy_coef"],
                entropy_bonus=algo_config["objective"].get("entropy_bonus", False),
                critic_coef=algo_config["objective"]["critic_coef"],
                clip_value=algo_config["objective"]["clip_value"],
                normalize_advantage=True,
            )
        elif algo_config["name"] == "kl_ppo":
            loss_module = KLPENPPOLoss(
                actor_network=actor,
                critic_network=critic,
                dtarg=algo_config["objective"]["dtarg"],
                beta=algo_config["objective"]["beta"],
                increment=algo_config["objective"]["increment"],
                decrement=algo_config["objective"]["decrement"],
                samples_mc_kl=algo_config["objective"]["samples_mc_kl"],
                loss_critic_type=algo_config["objective"]["loss_critic_type"],
                entropy_coef=algo_config["objective"]["entropy_coef"],
                critic_coef=algo_config["objective"]["critic_coef"],
                normalize_advantage=True,
            )

        return env, actor, critic, projection, loss_module

    @staticmethod
    def is_graph_env(env_name):
        return "graph" in env_name.lower().split("-")

    @staticmethod
    def eval_model(actor, test_env, num_episodes=3):
        test_rewards = []

        for _ in range(num_episodes):
            td_test = test_env.rollout(
                policy=actor,
                auto_reset=True,
                auto_cast_to_device=True,
                break_when_any_done=True,
                max_steps=1_000_000,
            )
            reward = td_test["next", "episode_reward"][td_test["next", "done"]]
            test_rewards.append(reward.cpu())
        del td_test
        return torch.cat(test_rewards, 0).mean().item()

    @staticmethod
    def generate_data(actor, test_env, num_episodes=3, save_dir=None):

        list_tensordict = []
        test_rewards = []
        for _ in range(num_episodes):
            td_test = test_env.rollout(
                policy=actor,
                auto_reset=True,
                auto_cast_to_device=True,
                break_when_any_done=True,
                max_steps=1_000_000,
            )
            reward = td_test["next", "episode_reward"][td_test["next", "done"]]
            test_rewards.append(reward.cpu())
            list_tensordict.append(td_test.copy().cpu())

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            td_data = torch.cat(list_tensordict)

            save_dict = {
                "data": td_data,
                "env": extract_tensors_from_a_dict(test_env.state_dict()),
            }
            torch.save(save_dict, save_dir / "data.pt")

        return torch.cat(test_rewards, 0).mean()
