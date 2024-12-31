import os
import time
import logging
import datetime
from stable_baselines3 import DQN, DDPG
from utils.logger_utils import setup_logger
import matplotlib.pyplot as plt
from utils.info_collector_callback import InfoCollectorCallback
from sample_environments.environment_factory import EnvironmentFactory


log_dir = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
logger = setup_logger('SampleRL', log_dir, console_level=logging.DEBUG, file_level=logging.DEBUG)
train_logger = setup_logger('Train', log_dir, console_level=logging.DEBUG, file_level=logging.DEBUG)
test_logger = setup_logger('Test', log_dir, console_level=logging.DEBUG, file_level=logging.DEBUG)

env_name = "reachball"
kewargs = {
    'change_ball_position': True,
    'change_ball_velocity': True,
    'ball_position_x': 0,
    'ball_position_y': 0,
    'ball_speed': 0,
    'ball_direction': 0,
    'min_distance_to_ball': 5.0,
    'max_steps': 200,
    'action_space_size': 16,
    'use_continuous_action': False,
    'action_space_size': 16,
    'use_turning': False
}

if __name__ == "__main__":
    print("Press Ctrl+C to exit...")
    try:
        env = EnvironmentFactory().create(env_name, render_mode=False, logger=logger, log_dir=log_dir, **kewargs)
        if kewargs['use_continuous_action']:
            model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, use_sde=True)
        else:
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
            
            test_logger.info(f"#Test results: {results}")
            
            episode_count = results['Goal'] + results['Out'] + results['Timeout']
            return results['Goal'] / episode_count, results['Out'] / episode_count, results['Timeout'] / episode_count
        
        test_results = []
        test_results.append(test(2000))
        for i in range(10):
            train(5000)
            
            info_collector.plot_print_results(train_logger, file_name=os.path.join(log_dir, 'results'))
            
            test_results.append(test(2000))
        
        env.close()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot([r[0] for r in test_results], label='Goal')
        ax.plot([r[1] for r in test_results], label='Out')
        ax.plot([r[2] for r in test_results], label='Timeout')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Percentage")
        ax.set_title("Test Results")
        ax.legend()
        plt.show()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down...")
    finally:
        env.close()
        print("Environment closed successfully.")