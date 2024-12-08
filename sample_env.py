import gym
from gym import spaces
import numpy as np

class CustomFieldEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, n=5, render_mode=None):
        super(CustomFieldEnv, self).__init__()
        self.n = n  # Size of the field (n x n)
        self.render_mode = render_mode
        # Action space: [x, y] where x and y can be -1, 0, or 1
        # self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.int8)
        # Descrete action space
        self.action_space = spaces.Discrete(4)
        
        # Observation space: agent position (x, y) and goal position (x, y)
        self.observation_space = spaces.Box(low=0, high=n-1, shape=(4,), dtype=np.int32)
        
        self.reset()
        
    def reset(self):
        # Initialize agent and goal positions
        self.agent_pos = np.random.randint(0, self.n, size=2)
        self.goal_pos = np.random.randint(0, self.n, size=2)
        while np.array_equal(self.agent_pos, self.goal_pos):
            self.goal_pos = np.random.randint(0, self.n, size=2)
        
        self.steps = 0
        self.previous_distance = self._distance_to_goal()
        return self._get_obs()
    
    def _get_obs(self):
        # Combine agent position and goal position as the observation
        return np.concatenate([self.agent_pos, self.goal_pos])
    
    def _distance_to_goal(self):
        # Calculate Manhattan distance to goal
        return np.sum(np.abs(self.agent_pos - self.goal_pos))
    
    def step(self, action):
        self.steps += 1
        
        # Update agent position based on action
        # action = np.clip(action, -1, 1)  # Ensure actions are within valid range
        # self.agent_pos += action
        
        # Discrete action space
        if action == 0:
            self.agent_pos[0] -= 1
        elif action == 1:
            self.agent_pos[0] += 1
        elif action == 2:
            self.agent_pos[1] -= 1
        elif action == 3:
            self.agent_pos[1] += 1
        
        # Calculate reward
        distance_to_goal = self._distance_to_goal()
        result = ''
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 1  # Goal reached
            done = True
            print(f"Goal reached in {self.steps} steps")
            result = 'Goal'
        elif np.any(self.agent_pos < 0) or np.any(self.agent_pos >= self.n):
            reward = -1  # Agent went outside the field
            done = True
            print(f"Agent went outside the field in {self.steps} steps")
            result = 'Outside'
        else:
            # Reward based on closeness to goal
            if distance_to_goal < self.previous_distance:
                reward = 0.1  # Positive for getting closer
            else:
                reward = -0.1  # Negative for moving away
            done = False
        
        self.previous_distance = distance_to_goal
        
        # End episode if max steps reached
        if self.steps >= 100:
            done = True
            print("Max steps reached")
            result = 'MaxSteps'
        
        return self._get_obs(), reward, done, {'result': result}
    
    def render(self, mode='human'):
        if self.render_mode != "human":
            raise NotImplementedError("Only `human` render mode is supported.")
        field = np.full((self.n, self.n), '.', dtype=str)
        field[self.goal_pos[0], self.goal_pos[1]] = 'G'
        if np.all(self.agent_pos >= 0) and np.all(self.agent_pos < self.n):
            field[self.agent_pos[0], self.agent_pos[1]] = 'A'
        print("\n".join(" ".join(row) for row in field))
        print()

# Example usage
# if __name__ == "__main__":
#     env = CustomFieldEnv(n=5)
#     obs = env.reset()
#     env.render()
#     done = False
#     while not done:
#         action = env.action_space.sample()  # Random action
#         obs, reward, done, _ = env.step(action)
#         print(f"Action: {action}, Reward: {reward}, Done: {done}")
#         env.render()
