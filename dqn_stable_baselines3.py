import os
import time
import logging
import datetime
from stable_baselines3 import DQN
from utils.logger_utils import setup_logger
import matplotlib.pyplot as plt
from utils.info_collector_callback import InfoCollectorCallback
from sample_environments.environment_factory import EnvironmentFactory


log_dir = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
logger = setup_logger('SampleRL', log_dir, console_level=logging.DEBUG, file_level=logging.DEBUG)

env_name = "ReachCenter"

if __name__ == "__main__":
    print("Press Ctrl+C to exit...")
    try:
        env = EnvironmentFactory().create(env_name, render_mode=False, logger=logger, log_dir=log_dir)
        model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
        info_collector = InfoCollectorCallback()
        
        def train(total_timesteps):
            model.learn(total_timesteps=total_timesteps, callback=info_collector)
            model.ep_info_buffer
        
        def test(total_timesteps):
            obs = env.reset()
            results = {'Goal': 0, 'Out': 0, 'Timeout': 0}
            for _ in range(total_timesteps):
                action = model.predict(obs)
                obs, reward, done, info = env.step(action)
                logger.debug(f"Observation: {obs}, Reward: {reward}, Done: {done}, Info: {info}")
                if done:
                    logger.info(f"Episode done. Info: {info}")
                    if info['result']:
                        results[info['result']] += 1
                    env.reset()
                
                time.sleep(0.0001)  # Adjust sleep time as needed
            
            logger.info(f"#Test results: {results}")
        
        train(10000)
        
        info_collector.plot_print_results(info_collector, logger, file_name=os.path.join(log_dir, 'results'))
        
        test(1000)
        
        env.close()
        
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down...")
    finally:
        env.close()
        print("Environment closed successfully.")
