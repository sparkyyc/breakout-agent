import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import matplotlib.pyplot as plt
import imageio

# Main script using PPO
if __name__ == '__main__':
    env_id = 'ALE/Breakout-v5'
    
    # Create the environment
    env = gym.make(env_id)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])  # Use DummyVecEnv for single environment
    env = VecTransposeImage(env)  # Handle image-based observations

    # Create the PPO model
    model = PPO('CnnPolicy', env, verbose=1, learning_rate=2.5e-4, n_steps=128, batch_size=64, n_epochs=4)

    # Train the model
    total_timesteps = 200000
    model.learn(total_timesteps=total_timesteps)

    # Save the updated model
    model.save("ppo_breakout")

    # Plot rewards over time
    rewards = model.ep_info_buffer if hasattr(model, 'ep_info_buffer') else []
    if rewards:
        mean_rewards = [info['r'] for info in rewards]
        plt.plot(mean_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('PPO Training Rewards Over Time')
        plt.savefig('ppo_breakout_training_rewards.png')
        plt.show()

    # Record a video of the agent playing the game
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

    # Save video as gif
    imageio.mimsave('ppo_breakout_agent.gif', frames, fps=30)
    
    # Save video as mp4
    video_path = "ppo_breakout_agent.mp4"
    imageio.mimsave(video_path, frames, fps=30, format="mp4")
    
    print(f"Total reward for this episode: {total_reward}")
    print(f"Video saved at {video_path}")
