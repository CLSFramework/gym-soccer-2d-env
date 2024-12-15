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
    def __init__(self, render_mode=None, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = setup_logger('ReachCenterEnv', log_dir, console_level=logging.DEBUG, file_level=logging.DEBUG)
        super(ReachCenterEnv, self).__init__(render_mode, logger=self.logger)
        
        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.distance_to_ball = 0.0
        self.step_number = 0
        
    def action_to_rpc_actions(self, action: int, player_state: pb2.State) -> list[pb2.PlayerAction]:
        self.logger.debug(f"action_to_rpc_actions: {action}")
        self.step_number += 1
        if isinstance(action, (list, tuple, np.ndarray)):
            action = action[0]
        relative_direction = action * 22.5
        return [pb2.PlayerAction(dash=pb2.Dash(power=100, relative_direction=relative_direction))]
    
    def state_to_observation(self, state: pb2.State):
        player_pos = Vector2D(state.world_model.self.position.x, state.world_model.self.position.y)
        player_body = AngleDeg(state.world_model.self.body_direction)
        player_to_center = (Vector2D(0, 0) - player_pos).th()
        player_body_to_center = (player_to_center - player_body).degree() / 180.0
        obs = np.array([player_body_to_center,
                        player_body.degree() / 180.0, 
                        player_pos.x() / 52.5, 
                        player_pos.y() / 34.0
                        ])
        self.logger.debug(f"Observation: {obs}")
        return obs
    
    def check_trainer_observation(self, state: pb2.State):
        player_x = state.world_model.teammates[0].position.x
        player_y = state.world_model.teammates[0].position.y
        center_pos = Vector2D(0, 0)
        player_pos = Vector2D(player_x, player_y)
        distance_to_center = center_pos.dist(player_pos)
        
        info = {'result': None}
        done, reward = False, 0.0
        if distance_to_center > self.distance_to_ball:
            reward = -0.01
        else:
            reward = 0.001
            
        if distance_to_center < 5.0:
            done = True
            reward = 1.0
            info['result'] = 'Goal'
        if player_pos.abs_x() > 52.5 or player_pos.abs_y() > 34.0:
            done = True
            reward = -1.0
            info['result'] = 'Out'
        
        if self.step_number > 100:
            done = True
            info['result'] = 'Timeout'
        
        self.logger.debug(f"Center position: {center_pos}, Player position: {player_pos} Distance to ball: {distance_to_center} Previous distance: {self.distance_to_ball} Reward: {reward} Done: {done} Step number: {self.step_number} ")
        self.distance_to_ball = distance_to_center
        
        return done, reward, info
    
    def abs_reset(self):
        player_observation, trainer_observation = self.env_reset()
        self.check_trainer_observation(trainer_observation)
        
        return player_observation
    
    def trainer_reset_actions(self):
        self.step_number = 0
        random_x = random.randint(-50, 50)
        random_y = random.randint(-30, 30)
        random_body_direction = random.randint(0, 360)
        
        actions = [
            pb2.TrainerAction(do_move_player=pb2.DoMovePlayer(
                our_side=True, uniform_number=1, position=pb2.RpcVector2D(x=random_x, y=random_y), body_direction=random_body_direction)),
            pb2.TrainerAction(do_recover=pb2.DoRecover())
        ]
        return actions
