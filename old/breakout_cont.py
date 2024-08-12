import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import matplotlib.pyplot as plt
import imageio

# Main script to continue training
if __name__ == '__main__':
    env_id = 'ALE/Breakout-v5'
    
    # Create the environment
    env = gym.make(env_id)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])  # Use DummyVecEnv for single environment
    env = VecTransposeImage(env)  # Handle image-based observations

    # Load the existing model
    model = DQN.load("dqn_breakout", env=env)

    # Continue training
    additional_timesteps = 100000  # Number of additional timesteps to train
    model.learn(total_timesteps=additional_timesteps)

    # Save the updated model
    model.save("dqn_breakout_updated")

    # Plot rewards over time 
    rewards = model.ep_info_buffer if hasattr(model, 'ep_info_buffer') else []
    if rewards:
        mean_rewards = [info['r'] for info in rewards]
        plt.plot(mean_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards Over Time (Continued)')
        plt.savefig('breakout_training_rewards_continued.png')
        # plt.show()

    # Record a video of the agent playing the game
    env = gym.make(env_id, render_mode='rgb_array')
    env = Monitor(env)
    frames = []
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = model.predict(obs, deterministic=True)[0]
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        total_reward += reward
        done = terminated or truncated

    env.close()

    # Save video as gif
    imageio.mimsave('breakout_agent_continued.gif', frames, fps=30)
    print(f"Total reward for this episode: {total_reward}")
