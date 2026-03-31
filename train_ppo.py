import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

class TradingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        self.state = np.random.rand(3)
        return self.state, {}

    def step(self, action):
        sentiment, price, signal = self.state

        if action == 2 and sentiment > 0.6:
            reward = 1
        elif action == 0 and sentiment < 0.4:
            reward = 1
        else:
            reward = -1

        self.state = np.random.rand(3)
        return self.state, reward, False, False, {}

env = TradingEnv()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)

model.save("models/ppo_model")

print("PPO model saved successfully!")