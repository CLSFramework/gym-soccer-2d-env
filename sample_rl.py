import os
import time
import logging
import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from utils.logger_utils import setup_logger
from sample_env import SampleEnv
import matplotlib.pyplot as plt


log_dir = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
logger = setup_logger('SampleRL', log_dir, console_level=logging.DEBUG, file_level=logging.DEBUG)

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
        env = SampleEnv(render_mode="human", logger=logger)
        model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
        info_collector = InfoCollectorCallback()
        model.learn(total_timesteps=1_000_000, callback=info_collector)
        model.ep_info_buffer
        env.close()
        
        results_types = ['Goal', 'Out', 'Timeout']
        results = [info['result'] for info in info_collector.infos]
        logger.debug(f"Results: {results}")
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

        env = SampleEnv(render_mode="human")
        obs = env.reset()
        
        for _ in range(1000):
            action = model.predict(obs)
            obs, reward, done, info = env.step(action)
            logger.debug(f"Observation: {obs}, Reward: {reward}, Done: {done}, Info: {info}")
            if done:
                env.reset()
            
            time.sleep(0.0001)  # Adjust sleep time as needed
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down...")
    finally:
        env.close()
        print("Environment closed successfully.")
