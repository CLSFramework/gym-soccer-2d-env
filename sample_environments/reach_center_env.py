import os
import random
import logging
import datetime

import numpy as np
from gym import spaces
from pyrusgeom.geom_2d import Vector2D, AngleDeg

import service_pb2 as pb2
from utils.logger_utils import setup_logger
from soccer_2d_env import Soccer2DEnv

log_dir = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

class ReachCenterEnv(Soccer2DEnv):
    """Environment where the agent tries to reach the center of the field."""

    def __init__(self, render_mode: str = None, logger: logging.Logger = None, log_dir: str = log_dir):
        """Initialize the ReachCenterEnv environment."""
        # Initialize logger
        self.logger = logger or setup_logger('ReachCenterEnv', log_dir, console_level=logging.DEBUG, file_level=logging.DEBUG)
        super(ReachCenterEnv, self).__init__(render_mode, logger=self.logger, log_dir=log_dir)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.previous_distance_to_center = 0.0
        self.step_number = 0
        
    def action_to_rpc_actions(self, action: int, player_state: pb2.State) -> list[pb2.PlayerAction]:
        """Convert an action into a list of RPC actions for the player."""
        self.logger.debug(f"action_to_rpc_actions: {action}")
        self.step_number += 1
        # Ensure action is an integer
        if isinstance(action, (list, tuple, np.ndarray)):
            if isinstance(action, np.ndarray):
                if action.size == 1:
                    action = action.item()
            else:
                action = action[0]
        # Map discrete action to relative direction
        relative_direction = (action * 360.0 / self.action_space.n) % 360.0 - 180.0
        return [pb2.PlayerAction(dash=pb2.Dash(power=100, relative_direction=relative_direction))]
    
    def state_to_observation(self, state: pb2.State) -> np.ndarray:
        """Convert the current state into an observation."""
        # Extract player's position and body direction
        player_pos = Vector2D(state.world_model.self.position.x, state.world_model.self.position.y)
        player_body = AngleDeg(state.world_model.self.body_direction)
        # Calculate angle to center and relative angle to player's body
        player_to_center = (Vector2D(0, 0) - player_pos).th()
        player_body_to_center = (player_to_center - player_body).degree()
        # Normalize observations
        obs = np.array([
            player_body_to_center / 180.0,            # Relative angle to center
            player_body.degree() / 180.0,     # Player's body direction
            player_pos.x() / 52.5,            # Normalized x position
            player_pos.y() / 34.0             # Normalized y position
        ])
        self.logger.debug(f"Observation: {obs}")
        return obs
    
    def check_trainer_observation(self, state: pb2.State) -> tuple[bool, float, dict]:
        """Check the trainer's observation and compute the reward and done flag."""
        # Get player's current position
        player_x = state.world_model.teammates[0].position.x
        player_y = state.world_model.teammates[0].position.y
        player_pos = Vector2D(player_x, player_y)
        # Calculate distance to center
        center_pos = Vector2D(0, 0)
        distance_to_center = center_pos.dist(player_pos)
        
        info = {'result': None}
        done, reward = False, self.previous_distance_to_center - distance_to_center
            
        # Check if player has reached the center
        if distance_to_center < 5.0:
            done = True
            reward += 10.0  # Reward for reaching the center
            info['result'] = 'Goal'
        # Check for out-of-bounds condition
        if player_pos.abs_x() > 52.5 or player_pos.abs_y() > 34.0:
            done = True
            reward -= -10.0  # Penalize for going out of bounds
            info['result'] = 'Out'
        # Check if maximum steps have been exceeded
        if self.step_number > 100:
            done = True
            reward -= 5.0  # Penalize for taking too long
            info['result'] = 'Timeout'
        # Log the status
        self.logger.debug(
            f"Center position: {center_pos}, Player position: {player_pos}, "
            f"Distance to center: {distance_to_center}, Previous distance: {self.previous_distance_to_center}, "
            f"Reward: {reward}, Done: {done}, Step number: {self.step_number}"
        )
        # Update previous distance
        self.previous_distance_to_center = distance_to_center
        
        return done, reward, info
    
    def abs_reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation."""
        player_observation, trainer_observation = self.env_reset()
        # Initialize distance tracker
        # self.previous_distance_to_center = Vector2D(0, 0).dist(
        #     Vector2D(trainer_observation.world_model.teammates[0].position.x, 
        #              trainer_observation.world_model.teammates[0].position.y)
        # )
        self.check_trainer_observation(trainer_observation)
        
        return player_observation
    
    def trainer_reset_actions(self) -> list[pb2.TrainerAction]:
        """Generate actions to reset the trainer's state."""
        self.step_number = 0
        # Randomize player's starting position and orientation
        random_x = random.randint(-50, 50)
        random_y = random.randint(-30, 30)
        random_body_direction = random.randint(0, 360)
        
        actions = [
            pb2.TrainerAction(do_move_player=pb2.DoMovePlayer(
                our_side=True,
                uniform_number=1,
                position=pb2.RpcVector2D(x=random_x, y=random_y),
                body_direction=random_body_direction
            )),
            pb2.TrainerAction(do_recover=pb2.DoRecover())
        ]
        return actions
