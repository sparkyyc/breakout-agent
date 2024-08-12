import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

# Load the trained PPO model
model = DQN.load("dqn_breakout_extended")

# Create the environment
env_id = 'ALE/Breakout-v5'
    
# Create the environment
env = gym.make(env_id, render_mode='human')
env = Monitor(env)
env = DummyVecEnv([lambda: env])  # Use DummyVecEnv for single environment
env = VecTransposeImage(env)  # Handle image-based observations


# Reset the environment
obs = env.reset()

done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
