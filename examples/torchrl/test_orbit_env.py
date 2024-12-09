from __future__ import annotations

import os
import hydra
from omegaconf import DictConfig, OmegaConf

simulation_app = None


@hydra.main(
    version_base=None,
    config_name="deformable_manipulation_ppo_cfg",
    config_path=f"{os.getcwd()}/configs",
)
def test_orbit_env(cfg: "DictConfig"):  # noqa: F821
    """Start Isaac Sim Simulator first."""
    global simulation_app
    from ironlib.orbit.utils.omniverse_app import launch_app  # noqa

    simulation_app = launch_app(config=OmegaConf.to_container(cfg.simulator, resolve=True))

    """ Rest everything follows. """
    from .builders.utils_env import make_orbit_env

    env = make_orbit_env(
        env_name=cfg.env.name,
        device=cfg.env.device,
        batch_size=cfg.env.num_envs,
        env_config=OmegaConf.to_container(cfg.env, resolve=True),
    )

    n_test_rollout = 5
    td_out = env.rollout(n_test_rollout)
    print(f"TensorDict output from {n_test_rollout} rollout \n{td_out}")
    env.close()


if __name__ == "__main__":
    test_orbit_env()

    # close sim app
    if simulation_app is not None:
        simulation_app.close()
