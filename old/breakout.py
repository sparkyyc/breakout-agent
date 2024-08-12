import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import matplotlib.pyplot as plt
import imageio

# Learning rate scheduler function
def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

# Main script
if __name__ == '__main__':
    env_id = 'ALE/Breakout-v5'
    
    # Create the environment
    env = gym.make(env_id)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])  # Use DummyVecEnv for single environment
    env = VecTransposeImage(env)  # Handle image-based observations

    # Setup the learning rate schedule
    learning_rate = linear_schedule(2.5e-4)  # Start with a higher learning rate

    # Initialize the DQN model
    model = DQN(
        'CnnPolicy',
        env,
        verbose=1,
        buffer_size=10000,
        learning_starts=1000,
        target_update_interval=1000,
        learning_rate=learning_rate,
        exploration_fraction=0.6,  # Start with more exploration
        exploration_final_eps=0.05,  # Decay exploration to 0.05
        batch_size=64,
        train_freq=(4, "step")
    )

    # Train the model without early stopping
    total_timesteps = 100000
    model.learn(total_timesteps=total_timesteps)

    # Save the final model
    model.save("dqn_breakout")

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
    imageio.mimsave('breakout_agent.gif', frames, fps=30)
    print(f"Total reward for this episode: {total_reward}")

    # Plot rewards over time
    rewards = model.ep_info_buffer if hasattr(model, 'ep_info_buffer') else []
    if rewards:
        mean_rewards = [info['r'] for info in rewards]
        plt.plot(mean_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards Over Time')
        plt.savefig('breakout_training_rewards.png')
        # plt.show()
