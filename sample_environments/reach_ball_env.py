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

# Set up logging directory
log_dir = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

class ReachBallEnv(Soccer2DEnv):
    def __init__(self, render_mode=None, logger=None, log_dir=log_dir, **kwargs):
        # Initialize logger
        self.logger = logger
        if self.logger is None:
            self.logger = setup_logger('ReachBallEnv', log_dir, console_level=logging.DEBUG, file_level=logging.DEBUG)
        super(ReachBallEnv, self).__init__(render_mode, logger=self.logger, log_dir=log_dir)
        
        # Read kwargs
        self.change_ball_position = kwargs.get('change_ball_position', True)
        self.change_ball_velocity = kwargs.get('change_ball_velocity', False)
        self.ball_position_x = kwargs.get('ball_position_x', 0)
        self.ball_position_y = kwargs.get('ball_position_y', 0)
        self.ball_speed = kwargs.get('ball_speed', 0)
        self.ball_direction = kwargs.get('ball_direction', 0)
        self.min_distance_to_ball = kwargs.get('min_distance_to_ball', 5.0)
        self.max_steps = kwargs.get('max_steps', 200)
        self.use_continuous_action = kwargs.get('use_continuous_action', True)
        self.action_space_size = kwargs.get('action_space_size', 16)
        self.use_turning = kwargs.get('use_turning', False)
        
        # Define action and observation spaces
        if self.use_continuous_action:
            if self.use_turning:
                self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1], dtype=np.float32),
                                               high=np.array([1, 1, 1, 1], dtype=np.float32),
                                               dtype=np.float32)
            else:
                self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(self.action_space_size)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.distance_to_ball = 0.0
        self.body_ball_angle_diff = 0.0
        self.step_number = 0
        
    def action_to_rpc_actions(self, action, player_state: pb2.State):
        self.logger.debug(f"action_to_rpc_actions: {action} {type(action)}")
        self.step_number += 1
        # # Convert action to a single integer if necessary
        if isinstance(action, np.ndarray):
            if action.size == 1:
                action = action.item()
        self.logger.debug(f"action_to_rpc_actions: {action}")
        # Calculate relative direction
        if self.use_continuous_action:
            if self.use_turning:
                action = np.clip(action, -1.0, 1.0)
                turn_prob = action[0]
                turn_angle = action[1]
                dash_prob = action[2]
                dash_angle = action[3]
                p = np.array([dash_prob, turn_prob])
                p = np.exp(p) / np.sum(np.exp(p))
                turn_selected = np.random.rand() < p[0]
                if turn_selected:
                    relative_direction = turn_angle * 180.0
                    self.logger.debug(f"Turn: {relative_direction}")
                    return pb2.PlayerAction(turn=pb2.Turn(relative_direction=relative_direction))
                else:
                    relative_direction = dash_angle * 180.0
                    self.logger.debug(f"Dash: {relative_direction}")
                    return pb2.PlayerAction(dash=pb2.Dash(power=100, relative_direction=relative_direction))
            else:
                relative_direction = action * 180.0
                return pb2.PlayerAction(dash=pb2.Dash(power=100, relative_direction=relative_direction))
        else:
            relative_direction = (action * 360.0 / self.action_space.n) % 360.0 - 180.0
            return pb2.PlayerAction(dash=pb2.Dash(power=100, relative_direction=relative_direction))
    
    def state_to_observation(self, state: pb2.State):
        # Extract positions and directions
        ball_pos = Vector2D(state.world_model.ball.position.x, state.world_model.ball.position.y)
        ball_velocity = Vector2D(state.world_model.ball.velocity.x, state.world_model.ball.velocity.y)
        ball_speed = ball_velocity.r()
        ball_direction = ball_velocity.th().degree()
        player_pos = Vector2D(state.world_model.self.position.x, state.world_model.self.position.y)
        player_body = AngleDeg(state.world_model.self.body_direction)
        player_to_ball = (ball_pos - player_pos).th()
        player_body_to_ball = (player_to_ball - player_body).degree()
        # Create observation array
        obs = np.array([player_body_to_ball / 180.0,
                        player_body.degree() / 180.0, 
                        player_pos.x() / 52.5, 
                        player_pos.y() / 34.0, 
                        ball_pos.x() / 52.5, 
                        ball_pos.y() / 34.0,
                        ball_speed / 3.0,
                        ball_direction / 360.0,
                        ball_velocity.x() / 3.0,
                        ball_velocity.y() / 3.0])
        self.logger.debug(f"State: {ball_pos=}, {ball_speed=}, {ball_direction=}, {player_pos=}, {player_body=}, "
                          f"{player_to_ball=}, {player_body_to_ball=}")
        self.logger.debug(f"To Observation: {obs}")
        return obs
    
    def check_trainer_observation(self, state: pb2.State):
        # Extract positions
        ball_x = state.world_model.ball.position.x
        ball_y = state.world_model.ball.position.y
        player_x = state.world_model.teammates[0].position.x
        player_y = state.world_model.teammates[0].position.y
        ball_pos = Vector2D(ball_x, ball_y)
        player_pos = Vector2D(player_x, player_y)
        distance_to_ball = ball_pos.dist(player_pos)
        player_body: AngleDeg = AngleDeg(state.world_model.teammates[0].body_direction)
        ball_direction: AngleDeg = (ball_pos - player_pos).th()
        body_ball_angle_diff: AngleDeg = ball_direction - player_body
        
        info = {'result': None}
        
        done, reward = False, 0.0
        
        distance_reward = self.distance_to_ball - distance_to_ball
        reward += distance_reward
        
        angle_reward = (AngleDeg(self.body_ball_angle_diff).abs() - body_ball_angle_diff.abs()) / 180.0
        reward += angle_reward
        
        # Check if the player reached the ball
        if distance_to_ball < self.min_distance_to_ball:
            done = True
            reward += 10.0
            info['result'] = 'Goal'
        # Check if the player is out of bounds
        if player_pos.abs_x() > 52.5 or player_pos.abs_y() > 34.0:
            done = True
            reward -= -10.0
            info['result'] = 'Out'
        # Check if the maximum number of steps is reached
        if self.step_number > self.max_steps:
            done = True
            reward -= 5.0
            info['result'] = 'Timeout'
        
        self.logger.debug(f"Ball position: {ball_pos}, Player position: {player_pos}, "
                          f"Distance to ball: {distance_to_ball} Previous distance: {self.distance_to_ball}, "
                          f"Body ball angle: {body_ball_angle_diff.degree()} Previous angle: {self.body_ball_angle_diff}, "
                          f"angle reward: {angle_reward}, distance reward: {distance_reward}, "
                          f"Reward: {reward} Done: {done} Step number: {self.step_number} ")
        
        self.distance_to_ball = distance_to_ball
        self.body_ball_angle_diff = body_ball_angle_diff.degree()
        
        return done, reward, info
    
    def abs_reset(self):
        # Reset environment and check initial observation
        player_observation, trainer_observation = self.env_reset()
        self.check_trainer_observation(trainer_observation)
        
        return player_observation
    
    def trainer_reset_actions(self):
        # Reset step number and generate random positions and directions
        self.step_number = 0
        random_x = random.randint(-50, 50)
        random_y = random.randint(-30, 30)
        random_body_direction = random.randint(0, 360)
        if self.change_ball_position:
            ball_random_x = random.randint(-50, 50)
            ball_random_y = random.randint(-30, 30)
        else:
            ball_random_x = self.ball_position_x
            ball_random_y = self.ball_position_y
        
        ball_velocity = self.get_ball_velocity(ball_random_x, ball_random_y)
        
        actions = []    
        # Move ball to random position
        action1 = pb2.TrainerAction(do_move_ball=pb2.DoMoveBall(position=pb2.RpcVector2D(x=ball_random_x, y=ball_random_y),
                                                                velocity=pb2.RpcVector2D(x=ball_velocity.x(), y=ball_velocity.y())))
        actions.append(action1)
        # Move player to random position
        action2 = pb2.TrainerAction(do_move_player=pb2.DoMovePlayer(
            our_side=True, uniform_number=1, position=pb2.RpcVector2D(x=random_x, y=random_y), body_direction=random_body_direction))
        actions.append(action2)
        # Recover player
        action3 = pb2.TrainerAction(do_recover=pb2.DoRecover())
        actions.append(action3)
        return actions

    def get_ball_velocity(self, ball_position_x, ball_position_y):
        # Ball should not go out of bounds after steps
        
        if self.change_ball_velocity:
            while True:
                ball_random_speed = random.random() * 3.0
                ball_random_direction = random.randint(0, 360)
                ball_velocity = Vector2D.from_polar(ball_random_speed, ball_random_direction)
                ball_travel = ball_random_speed * (1.0 - 0.96 ** self.max_steps) / (1.0 - 0.96)
                ball_position = Vector2D(ball_position_x, ball_position_y)
                ball_target = ball_position + Vector2D.from_polar(ball_travel, ball_random_direction)
                
                if abs(ball_target.x()) <= 52.5 and abs(ball_target.y()) <= 34.0:
                    break
        else:
            ball_random_speed = self.ball_speed
            ball_random_direction = self.ball_direction
            ball_velocity = Vector2D.from_polar(ball_random_speed, ball_random_direction)
        
        return ball_velocity