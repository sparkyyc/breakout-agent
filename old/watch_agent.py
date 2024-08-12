import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo

# Load the trained DQN model
model = DQN.load("dqn_breakout_extended")

# Create the environment
env_id = 'ALE/Breakout-v5'
env = gym.make(env_id, render_mode='rgb_array')
env = Monitor(env)

# Wrap the environment to record video
video_folder = './final-video/'
env = RecordVideo(env, video_folder, episode_trigger=lambda episode_id: True)
obs = env.reset()

done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

# Close the environment and video recorder
env.close()

print(f"Video saved at {video_folder}")
