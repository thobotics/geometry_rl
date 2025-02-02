import os


def launch_app(config):
    """Launch an omniverse app.

    Args:
        config (Dict): Configuration for the app.

    Returns:
        SimulationApp: The omniverse app.
    """
    from omni.isaac.orbit.app import AppLauncher  # noqa: F401

    # load cheaper kit config in headless
    headless = config.get("headless", False)
    if headless:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
    else:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"

    # launch omniverse app
    app_launcher = AppLauncher(config, experience=app_experience)
    simulation_app = app_launcher.app

    return simulation_app
