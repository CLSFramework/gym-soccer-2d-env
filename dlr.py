import gymnasium as gym
from simple_env import CustomFieldEnv
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
                if len(info['result']) > 0:
                    self.infos.append(info)
        return True  # Continue training
    
# env = gym.make("CartPole-v1", render_mode="human")
env = CustomFieldEnv(n=5, render_mode="human")
# Initialize the callback
info_collector = InfoCollectorCallback()

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000, callback=info_collector)
model.ep_info_buffer

# vec_env = model.get_env()
obs = env.reset()
env.render()
for i in range(100):
    print(f"Step {i}")
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    # VecEnv resets automatically
    if done:
        break
      

env.close()

import matplotlib.pyplot as plt

results_types = ['Goal', 'Outside', 'MaxSteps']
results = [info['result'] for info in info_collector.infos]
# count results in each 100 episodes
results_dict = {type: [] for type in results_types}
for i in range(0, len(results), 100):
    for type in results_types:
        length = len(results[i:i+100])
        results_dict[type].append(results[i:i+100].count(type) / length)
        
# plot results
fig, ax = plt.subplots()
ax.plot(results_dict['Goal'], label='Goal')
ax.plot(results_dict['Outside'], label='Outside')
ax.plot(results_dict['MaxSteps'], label='MaxSteps')
ax.legend()
plt.show()



