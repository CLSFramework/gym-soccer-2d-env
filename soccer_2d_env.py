import datetime
import logging
import os
import signal
import subprocess
import threading
import time
from multiprocessing import Lock, Manager, Process, Queue
from queue import Empty

import gym
import numpy as np
from gym import spaces

import service_pb2 as pb2
from server import serve
from utils.logger_utils import setup_logger


log_dir = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

class Soccer2DEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, render_mode=None, 
                 run_grpc_server=True, 
                 run_rcssserver=True, 
                 run_trainer_player=True,
                 logger=None):
        super(Soccer2DEnv, self).__init__()
        self.logger = logger
        if self.logger is None:
            self.logger = setup_logger('Soccer2DEnv', log_dir, console_level=logging.DEBUG, file_level=logging.DEBUG)
        self.logger.info("Initializing Soccer2DEnv...")
        
        self.render_mode = render_mode
        self.run_grpc_server = run_grpc_server
        self.run_rcssserver = run_rcssserver
        self.run_trainer_player = run_trainer_player
        
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
        if self.run_grpc_server:
            self.logger.info("Starting gRPC server...")
            self.grpc_process = self._run_server()
            self.logger.info(f"gRPC server started in pid: {self.grpc_process.pid}")
        
        # Run rcssserver
        if self.run_rcssserver:
            self.logger.info("Starting RCSSServer...")
            self.rcssserver_process = self._run_rcssserver()
            self.logger.info(f"RCSSServer started in pid: {self.rcssserver_process.pid}")
            self.rcssserver_thread = threading.Thread(target=self.stream_output_to_logger, args=(self.rcssserver_process, 'server'))
            self.logger.info(f"RCSSServer thread started in pid: {self.rcssserver_thread.ident}")
            self.rcssserver_thread.start()
            self.logger.info(f"RCSSServer thread started in pid: {self.rcssserver_thread.ident}")
            time.sleep(10)
        
        # # Run player and trainer
        if self.run_trainer_player:
            self.logger.info("Starting Trainer and Player...")
            self.agents_process = self._run_trainer_player()
            self.logger.info(f"Trainer and Player started in pid: {self.agents_process.pid}")
            self.agents_thread = threading.Thread(target=self.stream_output_to_logger, args=(self.agents_process, 'Agents'))
            self.logger.info(f"Trainer and Player thread started in pid: {self.agents_thread.ident}")
            self.agents_thread.start()
            self.logger.info(f"Trainer and Player thread started in pid: {self.agents_thread.ident}")
        
        self._wait_for_agents()
        
    def _fake_player(self, action=pb2.PlayerAction(body_hold_ball=pb2.Body_HoldBall())):
        try:
            state: pb2.State = self.player_state_queue.get(block=False)
            if state is None:
                return (-1, -1)
            self.logger.debug(f"Player state cycle: {state.world_model.cycle}")
            self._latest_player_state = state
            self._send_action_to_player(action)
            return (state.world_model.cycle, state.world_model.stoped_cycle)
        except Empty as e:
            return (-1, -1)
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return (-1, -1)
    
    def _fake_trainer(self, action=pb2.TrainerAction(do_change_mode=pb2.DoChangeMode(game_mode_type=pb2.GameModeType.PlayOn, side=pb2.Side.LEFT))):
        try:
            state = self.trainer_state_queue.get(block=False)
            if state is None:
                return (-1, -1)
            self.logger.debug(f"Trainer state cycle: {state.world_model.cycle}")
            self._latest_trainer_state = state
            self._send_action_to_trainer(action)
            return (state.world_model.cycle, state.world_model.stoped_cycle)
        except Empty as e:
            return (-1, -1)
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return (-1, -1)
    
    latest_player_cycle = (-1, -1)
    latest_trainer_cycle = (-1, -1)
    def _wait_for_agents(self):
        Soccer2DEnv.latest_player_cycle = (-1, -1)
        Soccer2DEnv.latest_trainer_cycle = (-1, -1)
        while True:
            latest_player_cycle = self._fake_player()
            latest_trainer_cycle = self._fake_trainer()
            if latest_player_cycle is not None and latest_player_cycle[0] != -1:
                Soccer2DEnv.latest_player_cycle = latest_player_cycle
            if latest_trainer_cycle is not None and latest_trainer_cycle[0] != -1:
                Soccer2DEnv.latest_trainer_cycle = latest_trainer_cycle
            
            try:
                self.logger.debug(f"Player cycle: {Soccer2DEnv.latest_player_cycle}, Trainer cycle: {Soccer2DEnv.latest_trainer_cycle}")
                if Soccer2DEnv.latest_player_cycle is None or Soccer2DEnv.latest_trainer_cycle is None:
                    continue
                if Soccer2DEnv.latest_player_cycle[0] == -1 or Soccer2DEnv.latest_trainer_cycle[0] == -1:
                    continue
                if Soccer2DEnv.latest_player_cycle[0] == Soccer2DEnv.latest_trainer_cycle[0] and abs(Soccer2DEnv.latest_player_cycle[1] - Soccer2DEnv.latest_trainer_cycle[1]) < 2:
                    self.logger.info("Player and Trainer cycles are in sync.")
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
            self.logger.error(f"SyncError: Player cycle: {player_state.world_model.cycle}, Trainer cycle: {trainer_state.world_model.cycle} are not in sync.")
            self._wait_for_agents()
            
        self.logger.debug(f"Reset Environment at cycle: ##{player_state.world_model.cycle}##")
        
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
            self.logger.error(f"Player cycle: {player_state.world_model.cycle}, Trainer cycle: {trainer_state.world_model.cycle} are not in sync.")
            self._wait_for_agents()
        
        self.logger.debug(f"Step Environment at cycle: ##{player_state.world_model.cycle}##")
        
        # Check trainer observation
        # Calculate reward
        done, reward, info = self.check_trainer_observation(trainer_state)
        
        # Return player observation, reward, done, info
        return player_observation, reward, done, info
    
    def render(self, mode="human"):
        pass
    
    def close(self):
        self.logger.info("Closing RoboEnv...")
        # Kill all processes
        if self.run_grpc_server:
            self.logger.info("Terminating gRPC server...")
            self.grpc_process.terminate()
        if self.run_rcssserver:
            self.logger.info("Terminating RCSSServer...")
            self.kill_process_group(self.rcssserver_process)
            self.rcssserver_process.terminate()
            self.rcssserver_thread.join()
        if self.run_trainer_player:
            self.logger.info("Terminating Trainer and Player by killing process group...")
            self.kill_process_group(self.agents_process)
            self.logger.info("Terminating Trainer and Player by terminating process...")
            self.agents_process.terminate()
            self.logger.info("Terminating Trainer and Player by joining thread...")
    
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
    