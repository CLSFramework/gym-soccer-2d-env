import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt


def wrap_angle_deg(angle):
    """
    Wrap an angle in degrees to the range [-180, 180].
    """
    # Equivalent to ((angle + 180) % 360) - 180, 
    # but using numpy's mod might be safer for negative angles
    wrapped = (angle + 180) % 360 - 180
    return wrapped

def angle_to_point_deg(x, y, target_x=0.0, target_y=0.0):
    """
    Return the angle in degrees from (x, y) to (target_x, target_y),
    measured counterclockwise from the positive X-axis, in [-180, 180].
    """
    dx = target_x - x
    dy = target_y - y
    angle_rad = np.arctan2(dy, dx)  # range [-pi, pi]
    angle_deg = np.degrees(angle_rad)  # convert to degrees
    angle_deg = wrap_angle_deg(angle_deg)
    return angle_deg

class GoToCenterEnv(gym.Env):
    """
    A custom Gym environment where an agent starts at a random position (x,y)
    in the field [-5.5, 52.5] x [-34, 34], facing a random angle in [-180, 180].
    The agent has a discrete action space of size 16, where taking action n
    means:
        - new_body_angle = old_body_angle + n*(360/16)
        - move forward 1 meter in that new direction
    The episode ends if:
        1) The agent goes out of bounds,
        2) The agent is within distance 5 of the center,
        3) 100 steps have passed.
    The state (observation) is a 4D vector:
        [ angle_diff_to_center/180, body_angle/180, x/52.5, y/34 ]
    We use a shaping reward that encourages moving closer to the center each step
    and provides a bonus/penalty on termination.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, continuous=False):
        super(GoToCenterEnv, self).__init__()
        self.continuous = continuous
        # --- Action Space ---
        if self.continuous:
            # Continuous action space: [dash_angle]
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            # 16 discrete actions: 0..15
            self.action_space = spaces.Discrete(16)
        
        # --- Observation Space ---
        # 4 features, each in range [-1, 1]
        low_obs = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        high_obs = np.array([1.0,  1.0,  1.0,  1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        
        # Field boundaries
        self.x_min, self.x_max = -52.5, 52.5
        self.y_min, self.y_max = -34.0, 34.0
        
        # Episode constraints
        self.max_steps = 200
        self.min_distance_to_center = 5.0
        
        # Internal state
        self.x = 0.0
        self.y = 0.0
        self.body_angle_deg = 0.0
        self.step_count = 0
        
        self.episode_history = []
        
        # Keep track of distance to center from previous step for shaping
        self.prev_distance = None
        
        # For rendering (optional)
        self.viewer = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_history = []
        # Randomize agent position in the field
        self.x = np.random.uniform(self.x_min, self.x_max)
        self.y = np.random.uniform(self.y_min, self.y_max)
        # Randomize body angle in [-180, 180]
        self.body_angle_deg = np.random.uniform(-180, 180)
        # Reset step counter
        self.step_count = 0
        
        # Compute initial distance to center
        self.prev_distance = np.sqrt(self.x**2 + self.y**2)
        
        self.episode_history.append((self.x, self.y, self.body_angle_deg, 0.0))
        
        # Return both observation and info dict
        return self._get_obs(), {}
    
    def step(self, action):
        """
        Execute one step in the environment.
        """
        if self.continuous:
            action = np.clip(action, -1.0, 1.0)
            action = action.item()
            action *= 180.0
        else:
            action = int(action)
            action = action * (360.0 / 16.0) - 180.0
        
        logger.debug(f"Step {self.step_count} | Action: {action}")
        
        direction_angle = wrap_angle_deg(self.body_angle_deg + action)

        # Convert movement direction to Cartesian deltas
        rad = np.radians(direction_angle)
        dx = np.cos(rad)
        dy = np.sin(rad)

        # Move forward by 1 meter in the current direction
        self.x += dx
        self.y += dy

        # Compute new distance to center
        distance_to_center = np.sqrt(self.x**2 + self.y**2)

        # Reward: Difference in distance to center
        reward = self.prev_distance - distance_to_center

        self.step_count += 1

        status = ''
        # Termination conditions
        terminated = False
        truncated = False

        if (self.x < self.x_min or self.x > self.x_max or
            self.y < self.y_min or self.y > self.y_max):
            terminated = True
            reward -= 10.0  # Penalty for leaving the field
            status = 'Out'

        elif distance_to_center < self.min_distance_to_center:
            terminated = True
            reward += 10.0  # Bonus for reaching the center
            status = 'Goal'

        elif self.step_count >= self.max_steps:
            truncated = True
            reward -= 5.0  # Penalty for timeout
            status = 'Timeout'

        # Update previous distance for next step
        self.prev_distance = distance_to_center
        
        # Store step in episode history
        self.episode_history.append((self.x, self.y, self.body_angle_deg, reward))
        
        info = {'result': status}
        obs = self._get_obs()
        logger.debug(f"obs = {obs}, reward = {reward}, done = {terminated}, truncated = {truncated}, info = {info}")
        return obs, reward, terminated or truncated, terminated or truncated, info

    def _get_obs(self):
        """
        Observation:
          0) angle_diff_to_center / 180
          1) body_angle_deg / 180
          2) x / 52.5
          3) y / 34
        """
        # Compute angle from agent to center
        angle_to_center_deg = angle_to_point_deg(self.x, self.y, 0.0, 0.0)
        
        # Difference in angles (wrapped to [-180, 180])
        angle_diff_deg = wrap_angle_deg(angle_to_center_deg - self.body_angle_deg)
        
        # Normalize each component
        angle_diff_norm = angle_diff_deg / 180.0
        body_angle_norm = self.body_angle_deg / 180.0
        x_norm = self.x / 52.5
        y_norm = self.y / 34.0
        
        return np.array([angle_diff_norm, body_angle_norm, x_norm, y_norm], dtype=np.float32)    
    
    def render(self, mode='human'):
        """
        Render the environment with a simple visualization.
        """
        if not hasattr(self, 'fig'):
            # Initialize the plot only once
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            self.ax.set_xlim(self.x_min, self.x_max)
            self.ax.set_ylim(self.y_min, self.y_max)

            # Draw field
            self.ax.plot([-5.5, -5.5, 52.5, 52.5, -5.5], [-34, 34, 34, -34, -34], color='black')
            self.ax.axhline(0, color='red', linestyle='--')
            self.ax.axvline(0, color='red', linestyle='--')

            self.agent_point, = self.ax.plot([], [], 'bo', label='Agent')
            self.agent_arrow = self.ax.arrow(0, 0, 0, 0, head_width=1, head_length=1.5, fc='blue', ec='blue')
            self.ax.legend()
            self.ax.set_title("Agent's Position and Field")
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")

        # Update agent position and orientation
        self.agent_point.set_data([self.x], [self.y])
        self.agent_arrow.remove()  # Remove the old arrow
        rad = np.radians(self.body_angle_deg)
        self.agent_arrow = self.ax.arrow(self.x, self.y, np.cos(rad), np.sin(rad), head_width=1, head_length=1.5, fc='blue', ec='blue')

        # Refresh the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        
    def render_all(self):
        """
        Render the entire episode history.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)

        # Draw field
        ax.plot([-5.5, -5.5, 52.5, 52.5, -5.5], [-34, 34, 34, -34, -34], color='black')
        ax.axhline(0, color='red', linestyle='--')
        ax.axvline(0, color='red', linestyle='--')

        # Draw agent trajectory
        x_vals, y_vals = zip(*[(x, y) for x, y, _, _ in self.episode_history])
        ax.plot(x_vals, y_vals, color='blue', label='Agent Trajectory')
        ax.scatter(x_vals[0], y_vals[0], c='green', label='Start')
        ax.scatter(x_vals[-1], y_vals[-1], c='red', label='End')

        # Draw agent orientation
        for x, y, body_angle_deg, _ in self.episode_history:
            rad = np.radians(body_angle_deg)
            ax.arrow(x, y, np.cos(rad), np.sin(rad), head_width=1, head_length=1.5, fc='blue', ec='blue')

        ax.legend()
        ax.set_title("Agent's Trajectory and Orientation")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.show()
        
    def close(self):
        if self.viewer is not None:
            plt.close(self.fig)
            self.viewer.close()
            self.viewer = None
            

# env = GoToCenterEnv()

# while True:
#     obs = env.reset()
#     done = False
#     total_reward = 0.0
#     while not done:
#         print("#" * 40)
#         print(f"Step {env.step_count}")
#         print(f"Position: ({env.x:.2f}, {env.y:.2f}) | Body angle: {env.body_angle_deg:.2f}")
#         print(f"Observation: {obs}")
#         # env.render()
#         # action = int(input("Enter action (0..15): "))
#         action = env.action_space.sample()
#         obs, reward, done, _ = env.step(action)
#         total_reward += reward
#         print(f"Reward: {reward:.2f} | Total reward: {total_reward}")
#     env.render_all()
        
        
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

# env = GoToCenterEnv(continuous=True)
# obs = env.reset()
# while True:
#     # action = env.action_space.sample()
#     logger.debug(f"x: {env.x}, y: {env.y}, body_angle_deg: {env.body_angle_deg}")
#     logger.debug(f"Observation: {obs}")
#     action = float(input("Enter action [-1 to 1]: "))
#     obs, reward, done, _, info = env.step(action)
#     logger.debug(f"Next x: {env.x}, y: {env.y}, body_angle_deg: {env.body_angle_deg}")
#     logger.debug(f"Next Observation: {obs}, Action: {action}, Reward: {reward}, Done: {done}, Info: {info}")
#     env.render()
    
#     if done:
#         logger.info(f"Episode done. Info: {info}")
#         env.reset()
#     time.sleep(0.0001)  # Adjust sleep time as needed

if __name__ == "__main__":
    print("Press Ctrl+C to exit...")
    try:
        env = GoToCenterEnv(continuous=True)
        # model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
        model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
        info_collector = InfoCollectorCallback()
        
        def train(total_timesteps):
            model.learn(total_timesteps=total_timesteps, callback=info_collector)
            model.ep_info_buffer
        
        def test(total_timesteps):
            obs, _ = env.reset()
            results = {'Goal': 0, 'Out': 0, 'Timeout': 0}
            for _ in range(total_timesteps):
                action, _ = model.predict(obs)
                obs, reward, done, _, info = env.step(action)
                logger.debug(f"Observation: {obs}, Action: {action}, Reward: {reward}, Done: {done}, Info: {info}")
                if done:
                    logger.info(f"Episode done. Info: {info}")
                    if info['result']:
                        results[info['result']] += 1
                    env.reset()
                # env.render()
                
                time.sleep(0.0001)  # Adjust sleep time as needed
            
            test_logger.info(f"#Test results: {results}")
            
            # return percentage of goals, outs, and timeouts
            episode_count = results['Goal'] + results['Out'] + results['Timeout']
            return results['Goal'] / episode_count, results['Out'] / episode_count, results['Timeout'] / episode_count
        
        test_results = []
        test_results.append(test(2000))
        for i in range(50):
            train(10000)
            
            info_collector.plot_print_results(train_logger, file_name=os.path.join(log_dir, 'results'))
            
            test_results.append(test(2000))
        
        env.close()
        
        # Plot test results
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
