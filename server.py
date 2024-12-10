from concurrent import futures
from time import sleep
import service_pb2_grpc as pb2_grpc
import service_pb2 as pb2
from typing import Union
from multiprocessing import Manager, Lock
from utils.logger_utils import setup_logger
import logging
import grpc
import argparse
import datetime
from pyrusgeom.geom_2d import Vector2D
from multiprocessing import Queue


console_logging_level = logging.INFO
file_logging_level = logging.DEBUG

main_logger = None
log_dir = None


class GrpcAgent:
    def __init__(self, agent_type, uniform_number, logger,
                 player_state_queue, trainer_state_queue, 
                 trainer_action_queue, player_action_queue) -> None:
        self.agent_type: pb2.AgentType = agent_type
        self.uniform_number: int = uniform_number
        self.server_params: Union[pb2.ServerParam, None] = None
        self.player_params: Union[pb2.PlayerParam, None] = None
        self.player_types: dict[int, pb2.PlayerType] = {}
        self.debug_mode: bool = False
        self.logger: logging.Logger = logger
        self.player_state_queue: Queue = player_state_queue
        self.trainer_state_queue: Queue = trainer_state_queue
        self.trainer_action_queue: Queue = trainer_action_queue
        self.player_action_queue: Queue = player_action_queue
    
    def GetAction(self, state: pb2.State):
        self.logger.debug(f"================================= cycle={state.world_model.cycle}.{state.world_model.stoped_cycle} =================================")
        # self.logger.debug(f"State: {state}")
        if self.agent_type == pb2.AgentType.PlayerT:
            return self.GetPlayerActions(state)
        elif self.agent_type == pb2.AgentType.CoachT:
            return self.GetCoachActions(state)
        elif self.agent_type == pb2.AgentType.TrainerT:
            return self.GetTrainerActions(state)
        
    def GetPlayerActions(self, state: pb2.State):
        try:
            actions = []
            self.player_state_queue.put(state)
            action = self.player_action_queue.get()
            
            if action is not None:
                if isinstance(action, list):
                    for a in action:
                        actions.append(a)
                else:
                    actions.append(action)
            
            self.logger.debug(f"Actions: {actions}")
            
            return pb2.PlayerActions(actions=actions, ignore_preprocess=True)
        except Exception as e:
            self.logger.error(f"Error in GetPlayerActions: {e}")
            return pb2.PlayerActions(actions=[])
    
    def GetBestPlannerAction(self, request: pb2.BestPlannerActionRequest) -> int:
        self.logger.debug(f"GetBestPlannerAction cycle:{request.state.world_model.cycle} pairs:{len(request.pairs)} unum:{request.state.register_response.uniform_number}")
        best_index = max(request.pairs.items(), key=lambda x: x[1].evaluation)[0]
        best_action = request.pairs[best_index].action
        
        while best_action.parent_index and best_action.parent_index > 0:
            best_action = request.pairs[best_action.parent_index].action
        
        res = pb2.BestPlannerActionResponse(index=best_action.index)
        return res
    
    def GetCoachActions(self, state: pb2.State):
        actions = []
        actions.append(pb2.CoachAction(do_helios_substitute=pb2.DoHeliosSubstitute()))
        return pb2.CoachActions(actions=actions)
    
    def GetTrainerActions(self, state: pb2.State):
        try:
            actions = []
            self.trainer_state_queue.put(state)
            action = self.trainer_action_queue.get()
            
            if action is not None:
                if isinstance(action, list):
                    for a in action:
                        actions.append(a)
                else:
                    actions.append(action)
            
            self.logger.debug(f"Actions: {actions}")
            
            return pb2.TrainerActions(actions=actions)
        except Exception as e:
            self.logger.error(f"Error in GetTrainerActions: {e}")
            return pb2.TrainerActions(actions=[])
    
    def SetServerParams(self, server_params: pb2.ServerParam):
        self.logger.debug(f"Server params received unum {server_params.register_response.uniform_number}")
        # self.logger.debug(f"Server params: {server_params}")
        self.server_params = server_params
        
    def SetPlayerParams(self, player_params: pb2.PlayerParam):
        self.logger.debug(f"Player params received unum {player_params.register_response.uniform_number}")
        # self.logger.debug(f"Player params: {player_params}")
        self.player_params = player_params
        
    def SetPlayerType(self, player_type: pb2.PlayerType):
        self.logger.debug(f"Player type received unum {player_type.register_response.uniform_number}")
        # self.logger.debug(f"Player type: {player_type}")
        self.player_types[player_type.id] = player_type
        
class GameHandler(pb2_grpc.GameServicer):
    def __init__(self, shared_lock, shared_number_of_connections,
                 player_state_queue, trainer_state_queue, 
                 trainer_action_queue, player_action_queue) -> None:
        self.agents: dict[int, GrpcAgent] = {}
        self.shared_lock = shared_lock
        self.shared_number_of_connections = shared_number_of_connections
        self.player_state_queue: Queue = player_state_queue
        self.trainer_state_queue: Queue = trainer_state_queue
        self.trainer_action_queue: Queue = trainer_action_queue
        self.player_action_queue: Queue = player_action_queue

    def GetPlayerActions(self, state: pb2.State, context):
        main_logger.debug(f"GetPlayerActions unum {state.register_response.uniform_number} at {state.world_model.cycle}")
        res = self.agents[state.register_response.client_id].GetAction(state)
        return res

    def GetCoachActions(self, state: pb2.State, context):
        main_logger.debug(f"GetCoachActions coach at {state.world_model.cycle}")
        res = self.agents[state.register_response.client_id].GetAction(state)
        return res

    def GetTrainerActions(self, state: pb2.State, context):
        main_logger.debug(f"GetTrainerActions trainer at {state.world_model.cycle}")
        res = self.agents[state.register_response.client_id].GetAction(state)
        return res

    def SendServerParams(self, serverParams: pb2.ServerParam, context):
        main_logger.debug(f"Server params received unum {serverParams.register_response.uniform_number}")
        self.agents[serverParams.register_response.client_id].SetServerParams(serverParams)
        res = pb2.Empty()
        return res

    def SendPlayerParams(self, playerParams: pb2.PlayerParam, context):
        main_logger.debug(f"Player params received unum {playerParams.register_response.uniform_number}")
        self.agents[playerParams.register_response.client_id].SetPlayerParams(playerParams)
        res = pb2.Empty()
        return res

    def SendPlayerType(self, playerType: pb2.PlayerType, context):
        main_logger.debug(f"Player type received unum {playerType.register_response.uniform_number}")
        self.agents[playerType.register_response.client_id].SetPlayerType(playerType)
        res = pb2.Empty()
        return res

    def SendInitMessage(self, initMessage: pb2.InitMessage, context):
        main_logger.debug(f"Init message received unum {initMessage.register_response.uniform_number}")
        self.agents[initMessage.register_response.client_id].debug_mode = initMessage.debug_mode
        res = pb2.Empty()
        return res

    def Register(self, register_request: pb2.RegisterRequest, context):
        try:
            with self.shared_lock:
                main_logger.info(f"received register request from team_name: {register_request.team_name} "
                    f"unum: {register_request.uniform_number} "
                    f"agent_type: {register_request.agent_type}")
                self.shared_number_of_connections.value += 1
                main_logger.info(f"Number of connections {self.shared_number_of_connections.value}")
                team_name = register_request.team_name
                uniform_number = register_request.uniform_number
                agent_type = register_request.agent_type
                register_response = pb2.RegisterResponse(client_id=self.shared_number_of_connections.value,
                                        team_name=team_name,
                                        uniform_number=uniform_number,
                                        agent_type=agent_type)
                logger = setup_logger(f"agent{register_response.uniform_number}_{register_response.client_id}", log_dir)
                self.agents[self.shared_number_of_connections.value] = GrpcAgent(agent_type, uniform_number, logger,
                                                                                self.player_state_queue, self.trainer_state_queue, 
                                                                                self.trainer_action_queue, self.player_action_queue)
        except Exception as e:
            import traceback
            main_logger.error(f"Error in Register: {e}")
            main_logger.error(traceback.format_exc())
        return register_response

    def SendByeCommand(self, register_response: pb2.RegisterResponse, context):
        main_logger.debug(f"Bye command received unum {register_response.uniform_number}")
        # with shared_lock:
        self.agents.pop(register_response.client_id)
            
        res = pb2.Empty()
        return res
    
    def GetBestPlannerAction(self, pairs: pb2.BestPlannerActionRequest, context):
        main_logger.debug(f"GetBestPlannerAction cycle:{pairs.state.world_model.cycle} pairs:{len(pairs.pairs)} unum:{pairs.register_response.uniform_number}")
        res = self.agents[pairs.register_response.client_id].GetBestPlannerAction(pairs)
        return res
    

def serve(port, shared_lock, shared_number_of_connections, 
          player_state_queue, trainer_state_queue, trainer_action_queue, player_action_queue,
          log_dir2):
    global main_logger, log_dir
    log_dir = log_dir2
    main_logger = setup_logger("pmservice", log_dir, console_level=console_logging_level, file_level=file_logging_level)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=22))
    game_service = GameHandler(shared_lock, shared_number_of_connections,
                               player_state_queue, trainer_state_queue, trainer_action_queue, player_action_queue)
    pb2_grpc.add_GameServicer_to_server(game_service, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    main_logger.info(f"Starting server on port {port}")
    
    server.wait_for_termination()

    
