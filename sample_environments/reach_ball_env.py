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

class ReachBallEnv(Soccer2DEnv):
    def __init__(self, render_mode=None, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = setup_logger('ReachBallEnv', log_dir, console_level=logging.DEBUG, file_level=logging.DEBUG)
        super(ReachBallEnv, self).__init__(render_mode, logger=self.logger)
        
        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.distance_to_ball = 0.0
        self.step_number = 0
        
    def action_to_rpc_actions(self, action, player_state: pb2.State):
        self.logger.debug(f"action_to_rpc_actions: {action}")
        self.step_number += 1
        if isinstance(action, (list, tuple, np.ndarray)):
            action = action[0]
        relative_direction = action * 22.5
        return pb2.PlayerAction(dash=pb2.Dash(power=100, relative_direction=relative_direction))
    
    def state_to_observation(self, state: pb2.State):
        ball_pos = Vector2D(state.world_model.ball.position.x, state.world_model.ball.position.y)
        player_pos = Vector2D(state.world_model.self.position.x, state.world_model.self.position.y)
        player_body = AngleDeg(state.world_model.self.body_direction)
        player_to_ball = (ball_pos - player_pos).th()
        player_body_to_ball = (player_to_ball - player_body).degree() / 180.0
        obs = np.array([player_body_to_ball, 
                        player_body.degree() / 180.0, 
                        player_pos.x() / 52.5, 
                        player_pos.y() / 34.0, 
                        ball_pos.x() / 52.5, 
                        ball_pos.y() / 34.0])
        # obs = np.array([player_body_to_ball])
        self.logger.debug(f"Observation: {obs}")
        return obs
    
    def check_trainer_observation(self, state: pb2.State):
        ball_x = state.world_model.ball.position.x
        ball_y = state.world_model.ball.position.y
        player_x = state.world_model.teammates[0].position.x
        player_y = state.world_model.teammates[0].position.y
        ball_pos = Vector2D(ball_x, ball_y)
        player_pos = Vector2D(player_x, player_y)
        distance_to_ball = ball_pos.dist(player_pos)
        
        info = {'result': None}
        done, reward = False, 0.0
        if distance_to_ball > self.distance_to_ball:
            reward = -0.01
        else:
            reward = 0.001
            
        if distance_to_ball < 5.0:
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
        
        self.logger.debug(f"Ball position: {ball_pos}, Player position: {player_pos} Distance to ball: {distance_to_ball} Previous distance: {self.distance_to_ball} Reward: {reward} Done: {done} Step number: {self.step_number} ")
        self.distance_to_ball = distance_to_ball
        
        return done, reward, info
    
    def abs_reset(self):
        player_observation, trainer_observation = self.env_reset()
        self.check_trainer_observation(trainer_observation)
        
        return player_observation
    
    def trainer_reset_actions(self):
        self.step_number = 0
        random_x = random.randint(-50, 50)
        random_y = random.randint(-30, 30)
        ball_random_x = random.randint(-50, 50)
        ball_random_y = random.randint(-30, 30)
        random_body_direction = random.randint(0, 360)
        
        actions = []    
        action1 = pb2.TrainerAction(do_move_ball=pb2.DoMoveBall(position=pb2.RpcVector2D(x=ball_random_x, y=ball_random_y),
                                                                velocity=pb2.RpcVector2D(x=0, y=0)))
        actions.append(action1)
        action2 = pb2.TrainerAction(do_move_player=pb2.DoMovePlayer(
            our_side=True, uniform_number=1, position=pb2.RpcVector2D(x=random_x, y=random_y), body_direction=random_body_direction))
        actions.append(action2)
        action3 = pb2.TrainerAction(do_recover=pb2.DoRecover())
        actions.append(action3)
        return actions
