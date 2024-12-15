from sample_environments.reach_ball_env import ReachBallEnv
from sample_environments.reach_center_env import ReachCenterEnv
from soccer_2d_env import Soccer2DEnv

class EnvironmentFactory:
    """
    Factory class for creating different types of soccer environments.
    """
    def __init__(self):
        """
        Initialize the EnvironmentFactory.
        """
        pass
    
    def create(self, env_name: str, render_mode: str, logger, log_dir: str) -> Soccer2DEnv:
        """
        Create an environment based on the given name.
        
        :param env_name: Name of the environment to create.
        :param render_mode: Mode for rendering the environment.
        :param logger: Logger instance for logging.
        :param log_dir: Directory for logs.
        :return: Instance of Soccer2DEnv.
        :raises ValueError: If the environment name is not found.
        """
        if env_name.lower() == "reachcenter":
            return ReachCenterEnv(render_mode=render_mode, logger=logger, log_dir=log_dir)
        elif env_name.lower() == "reachball":
            return ReachBallEnv(render_mode=render_mode, logger=logger, log_dir=log_dir)
        else:
            raise ValueError(f"Environment {env_name} not found.")