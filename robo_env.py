import gym
from gym import spaces
import numpy as np
from multiprocessing import Process, Queue, Lock, Manager
import subprocess
import service_pb2_grpc as pb2_grpc
import service_pb2 as pb2
import grpc
import os
from server import serve
import time
import threading
from utils.logger_utils import setup_logger
import logging
import datetime
import signal
from queue import Empty
from pyrusgeom.geom_2d import Vector2D, AngleDeg


log_dir = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
env_logger = setup_logger('RoboEnv', log_dir, console_level=logging.DEBUG, file_level=logging.DEBUG)

run_grpc_server = True
run_rcssserver = True
run_trainer_player = True

class RoboEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, render_mode=None):
        super(RoboEnv, self).__init__()
        env_logger.info("Initializing RoboEnv...")
        
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # create all queues
        self.player_state_queue = Queue()
        self.trainer_state_queue = Queue()
        self.trainer_action_queue = Queue()
        self.player_action_queue = Queue()
        
        self._latest_player_state = None
        self._latest_trainer_state = None
        
        # Run grpc server
        if run_grpc_server:
            env_logger.info("Starting gRPC server...")
            self.grpc_process = self._run_server()
            env_logger.info(f"gRPC server started in pid: {self.grpc_process.pid}")
        
        # Run rcssserver
        if run_rcssserver:
            env_logger.info("Starting RCSSServer...")
            self.rcssserver_process = self._run_rcssserver()
            env_logger.info(f"RCSSServer started in pid: {self.rcssserver_process.pid}")
            self.rcssserver_thread = threading.Thread(target=self.stream_output_to_logger, args=(self.rcssserver_process, 'server'))
            env_logger.info(f"RCSSServer thread started in pid: {self.rcssserver_thread.ident}")
            self.rcssserver_thread.start()
            env_logger.info(f"RCSSServer thread started in pid: {self.rcssserver_thread.ident}")
            time.sleep(10)
        
        # # Run player and trainer
        if run_trainer_player:
            env_logger.info("Starting Trainer and Player...")
            self.agents_process = self._run_trainer_player()
            env_logger.info(f"Trainer and Player started in pid: {self.agents_process.pid}")
            self.agents_thread = threading.Thread(target=self.stream_output_to_logger, args=(self.agents_process, 'Agents'))
            env_logger.info(f"Trainer and Player thread started in pid: {self.agents_thread.ident}")
            self.agents_thread.start()
            env_logger.info(f"Trainer and Player thread started in pid: {self.agents_thread.ident}")
        
        self._wait_for_agents()
        
    def _fake_player(self, action=pb2.PlayerAction(body_hold_ball=pb2.Body_HoldBall())):
        try:
            state: pb2.State = self.player_state_queue.get(block=False)
            if state is None:
                return (-1, -1)
            env_logger.debug(f"Player state cycle: {state.world_model.cycle}")
            self._latest_player_state = state
            self._send_action_to_player(action)
            return (state.world_model.cycle, state.world_model.stoped_cycle)
        except Empty as e:
            return (-1, -1)
        except Exception as e:
            env_logger.error(f"Error: {e}")
            return (-1, -1)
    
    def _fake_trainer(self, action=pb2.TrainerAction(do_change_mode=pb2.DoChangeMode(game_mode_type=pb2.GameModeType.PlayOn, side=pb2.Side.LEFT))):
        try:
            state = self.trainer_state_queue.get(block=False)
            if state is None:
                return (-1, -1)
            env_logger.debug(f"Trainer state cycle: {state.world_model.cycle}")
            self._latest_trainer_state = state
            self._send_action_to_trainer(action)
            return (state.world_model.cycle, state.world_model.stoped_cycle)
        except Empty as e:
            return (-1, -1)
        except Exception as e:
            env_logger.error(f"Error: {e}")
            return (-1, -1)
    
    latest_player_cycle = (-1, -1)
    latest_trainer_cycle = (-1, -1)
    def _wait_for_agents(self):
        RoboEnv.latest_player_cycle = (-1, -1)
        RoboEnv.latest_trainer_cycle = (-1, -1)
        while True:
            latest_player_cycle = self._fake_player()
            latest_trainer_cycle = self._fake_trainer()
            if latest_player_cycle is not None and latest_player_cycle[0] != -1:
                RoboEnv.latest_player_cycle = latest_player_cycle
            if latest_trainer_cycle is not None and latest_trainer_cycle[0] != -1:
                RoboEnv.latest_trainer_cycle = latest_trainer_cycle
            
            try:
                env_logger.debug(f"Player cycle: {RoboEnv.latest_player_cycle}, Trainer cycle: {RoboEnv.latest_trainer_cycle}")
                if RoboEnv.latest_player_cycle is None or RoboEnv.latest_trainer_cycle is None:
                    continue
                if RoboEnv.latest_player_cycle[0] == -1 or RoboEnv.latest_trainer_cycle[0] == -1:
                    continue
                if RoboEnv.latest_player_cycle[0] == RoboEnv.latest_trainer_cycle[0] and abs(RoboEnv.latest_player_cycle[1] - RoboEnv.latest_trainer_cycle[1]) < 2:
                    env_logger.info("Player and Trainer cycles are in sync.")
                    break
                time.sleep(0.01)
            except KeyboardInterrupt:
                break
        
    def env_reset(self):
        # Send reset action to trainer
        reset_actions = self.trainer_reset_actions()
        self._send_action_to_trainer(reset_actions)
        
        # Send fake action to player
        self._send_action_to_player(pb2.PlayerAction(body_hold_ball=pb2.Body_HoldBall()))
        
        # Wait for player observation
        player_state = self.player_state_queue.get()
        self._latest_player_state = player_state
        # Wait for trainer observation
        trainer_state = self.trainer_state_queue.get()
        self._latest_trainer_state = trainer_state
        
        if player_state.world_model.cycle != trainer_state.world_model.cycle:
            env_logger.error(f"SyncError: Player cycle: {player_state.world_model.cycle}, Trainer cycle: {trainer_state.world_model.cycle} are not in sync.")
            self._wait_for_agents()
            
        env_logger.debug(f"Reset Environment at cycle: ##{player_state.world_model.cycle}##")
        
        # Return player observation
        return self.state_to_observation(player_state), trainer_state
    
    def abs_reset(self):
        player_observation, trainer_observation = self.env_reset()
        return player_observation
        
    def reset(self):
        return self.abs_reset()
    
    def step(self, action):
        # Send action to player
        self._send_action_to_player(self.action_to_rpc_actions(action, self._latest_player_state))
        
        # Send fake action to trainer
        self._send_action_to_trainer(pb2.TrainerAction(do_change_mode=pb2.DoChangeMode(game_mode_type=pb2.GameModeType.PlayOn, side=pb2.Side.LEFT)))
        
        # Wait for player observation
        player_state:pb2.State = self.player_state_queue.get()
        self._latest_player_state = player_state
        player_observation = self.state_to_observation(player_state)
        
        # Wait for trainer observation
        trainer_state:pb2.State = self.trainer_state_queue.get()
        self._latest_trainer_state = trainer_state
        
        if player_state.world_model.cycle != trainer_state.world_model.cycle:
            env_logger.error(f"Player cycle: {player_state.world_model.cycle}, Trainer cycle: {trainer_state.world_model.cycle} are not in sync.")
            self._wait_for_agents()
        
        env_logger.debug(f"Step Environment at cycle: ##{player_state.world_model.cycle}##")
        
        # Check trainer observation
        # Calculate reward
        done, reward, info = self.check_trainer_observation(trainer_state)
        
        # Return player observation, reward, done, info
        return player_observation, reward, done, info
    
    def render(self, mode="human"):
        pass
    
    def close(self):
        env_logger.info("Closing RoboEnv...")
        # Kill all processes
        if run_grpc_server:
            env_logger.info("Terminating gRPC server...")
            self.grpc_process.terminate()
        if run_rcssserver:
            env_logger.info("Terminating RCSSServer...")
            self.kill_process_group(self.rcssserver_process)
            self.rcssserver_process.terminate()
            self.rcssserver_thread.join()
        if run_trainer_player:
            env_logger.info("Terminating Trainer and Player by killing process group...")
            self.kill_process_group(self.agents_process)
            env_logger.info("Terminating Trainer and Player by terminating process...")
            self.agents_process.terminate()
            env_logger.info("Terminating Trainer and Player by joining thread...")
            self.agents_thread.join()
    
    def _send_action_to_player(self, action):
        self.player_action_queue.put(action)
        
    def _send_action_to_trainer(self, action):
        self.trainer_action_queue.put(action)
        
    def action_to_rpc_actions(self, action, player_state: pb2.State):
        pass
    
    def state_to_observation(self, state: pb2.State):
        pass
    
    def check_trainer_observation(self, observation):
        pass
    
    def trainer_reset_actions(self):
        pass
    
    def _run_rcssserver(self):
        rcssserver_path = 'scripts/rcssserver/rcssserver'
        if not os.path.exists(rcssserver_path):
            raise FileNotFoundError(f"{rcssserver_path} does not exist.")
        if not os.access(rcssserver_path, os.X_OK):
            raise PermissionError(f"{rcssserver_path} is not executable. Check permissions.")

        process = subprocess.Popen(
            ['./rcssserver', '--server::synch_mode=true', '--server::auto_mode=true'],
            cwd='scripts/rcssserver',  # Corrected directory to where start.sh is located
            preexec_fn=os.setsid,  # Create a new session and set the process group ID
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT  # Capture stderr and redirect it to stdout
        )
        return process
    
    def _run_trainer_player(self):
        process = subprocess.Popen(
            ['bash', 'train.sh'],
            cwd='scripts/proxy',  # Corrected directory to where start.sh is located
            preexec_fn=os.setsid,  # Create a new session and set the process group ID
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT  # Capture stderr and redirect it to stdout
        )
        return process
    
    def stream_output_to_logger(self, process, prefix):
        # Stream output from the process and log it with a prefix
        logger = setup_logger(prefix, log_dir, console_level=None, file_level=logging.DEBUG)
        for line in iter(process.stdout.readline, b''):
            logger.info(line.decode().strip())
        process.stdout.close()
        
    def _run_grpc(self):
        manager = Manager()
        shared_lock = Lock()  # Create a Lock for synchronization
        shared_number_of_connections = manager.Value('i', 0)

        # Run the gRPC server in the current process
        serve(50051, shared_lock, shared_number_of_connections, 
                self.player_state_queue, self.trainer_state_queue, 
                self.trainer_action_queue, self.player_action_queue,
                log_dir)
        
    def _run_server(self):
        grpc_process = Process(target=self._run_grpc, args=())
        grpc_process.start()
        return grpc_process
        
    def kill_process_group(self, process):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # Send SIGTERM to the process group
        except ProcessLookupError:
            pass  # The process might have already exited
    
import random

class MyRoboEnv(RoboEnv):
    def __init__(self, render_mode=None):
        super(MyRoboEnv, self).__init__(render_mode)
        
        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.distance_to_ball = 0.0
        self.step_number = 0
        
    def action_to_rpc_actions(self, action, player_state: pb2.State):
        env_logger.debug(f"action_to_rpc_actions: {action}")
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
        env_logger.debug(f"Observation: {obs}")
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
        
        env_logger.debug(f"Ball position: {ball_pos}, Player position: {player_pos} Distance to ball: {distance_to_ball} Previous distance: {self.distance_to_ball} Reward: {reward} Done: {done} Step number: {self.step_number} ")
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

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

class InfoCollectorCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.infos = []  # To store the info dictionaries

    def _on_step(self) -> bool:
        # Collect the info dictionary from the environment
        if self.locals.get('infos') is not None:
            infos = self.locals.get('infos')
            for info in infos:
                if info['result'] and len(info['result']) > 0:
                    self.infos.append(info)
        return True  # Continue training
    
if __name__ == "__main__":
    print("Press Ctrl+C to exit...")
    try:
        env = MyRoboEnv(render_mode="human")
        model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
        info_collector = InfoCollectorCallback()
        model.learn(total_timesteps=100_000, callback=info_collector)
        model.ep_info_buffer
        env.close()
        
        import matplotlib.pyplot as plt

        results_types = ['Goal', 'Out', 'Timeout']
        results = [info['result'] for info in info_collector.infos]
        env_logger.debug(f"Results: {results}")
        # count results in each 100 episodes
        results_dict = {type: [] for type in results_types}
        for i in range(0, len(results), 100):
            for type in results_types:
                length = len(results[i:i+100])
                results_dict[type].append(results[i:i+100].count(type) / length)
                
        # plot results
        fig, ax = plt.subplots()
        ax.plot(results_dict['Goal'], label='Goal')
        ax.plot(results_dict['Out'], label='Out')
        ax.plot(results_dict['Timeout'], label='Timeout')
        ax.legend()
        plt.show()

        env = MyRoboEnv(render_mode="human")
        obs = env.reset()
        
        for _ in range(1000):
            action = model.predict(obs)
            obs, reward, done, info = env.step(action)
            env_logger.debug(f"Observation: {obs}, Reward: {reward}, Done: {done}, Info: {info}")
            if done:
                env.reset()
            
            # Keep the main process alive
            # env._fake_player(pb2.PlayerAction(dash=pb2.Dash(power=100, relative_direction=0)))
            # env._fake_trainer(action)
            time.sleep(0.0001)  # Adjust sleep time as needed
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down...")
    finally:
        env.close()
        print("Environment closed successfully.")

# def _wait_for_agents(self):
    # Wait for player observation O-1
    # Wait for trainer observation O-1
    
# def reset(self):
    # Send reset action to trainer A-1
    # Send fake action to player A-1
    
    # Wait for player observation O-2
    # Wait for trainer observation O-2
    # Send fake action to trainer A-2

# def step(self, action):
    # Send action to player A-2
    
    # Wait for player observation O-3
    # Wait for trainer observation O-3
    
    # Check trainer observation
    # if not done:
        # Send fake action to trainer A-3

# def step(self, action):
    # Send action to player A-3

    # Wait for player observation O-4
    # Wait for trainer observation O-4
    
    # Check trainer observation
    # is done
    
# def reset(self):
    # Send reset action to trainer A-4
    # Send fake action to player A-4
    
    # Wait for player observation O-5
    # Wait for trainer observation O-5
    # Send fake action to trainer A-5








# def wait:
    # Wait for player observation O-1
    # Wait for trainer observation O-1
    
# def reset(self):
    # Send reset action to trainer A-1
    # Send fake action to player A-1
    
    # Wait for player observation O-2
    # Wait for trainer observation O-2

# def step(self, action):
    # Send action to player A-2
    # Send fake action to trainer A-2
    
    # Wait for player observation O-3
    # Wait for trainer observation O-3
    
    # Check trainer observation

# def step(self, action):
    # Send action to player A-3
    # Send fake action to trainer A-3
    
    # Wait for player observation O-4
    # Wait for trainer observation O-4
    
    # Check trainer observation
    # is done
    
# def reset(self):
    # Send reset action to trainer A-4
    # Send fake action to player A-4
    
    # Wait for player observation O-5
    # Wait for trainer observation O-5

