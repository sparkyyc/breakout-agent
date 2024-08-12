import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import matplotlib.pyplot as plt
import imageio

# Main script using PPO with extended training and adjusted parameters
if __name__ == '__main__':
    env_id = 'ALE/Breakout-v5'
    
    # Create the environment
    env = gym.make(env_id)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])  # Use DummyVecEnv for single environment
    env = VecTransposeImage(env)  # Handle image-based observations

    model = PPO('CnnPolicy', env, verbose=1, learning_rate=1e-4, n_steps=256, batch_size=64, n_epochs=10)

    total_timesteps = 400000  # Increase total timesteps for extended training
    model.learn(total_timesteps=total_timesteps)

    # Save the updated model
    model.save("ppo_breakout_extended")

    # Plot rewards over time
    rewards = model.ep_info_buffer if hasattr(model, 'ep_info_buffer') else []
    if rewards:
        mean_rewards = [info['r'] for info in rewards]
        plt.plot(mean_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('PPO Training Rewards Over Time (Extended)')
        plt.savefig('ppo_breakout_training_rewards_extended.png')

    env = gym.make(env_id, render_mode='rgb_array')
    env = Monitor(env)
    frames = []
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        total_reward += reward
        done = terminated or truncated

    env.close()
