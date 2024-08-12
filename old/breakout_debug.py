import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

if __name__ == '__main__':
    env_id = 'ALE/Breakout-v5'
    
    # Create the environment
    env = gym.make(env_id)
    env = Monitor(env)

    # Simplified model with basic setup
    model = DQN(
        'CnnPolicy',
        env,
        verbose=1,
        buffer_size=5000,
        learning_starts=1000,
        target_update_interval=500,
        learning_rate=2.5e-4,
        exploration_fraction=0.6,
        exploration_final_eps=0.05,
        batch_size=32,
        train_freq=(4, "step")
    )

    # Train the model
    total_timesteps = 10000  # Reduced for quick testing
    model.learn(total_timesteps=total_timesteps)

    # Evaluate the agent
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = model.predict(obs, deterministic=True)[0]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Total reward for this episode: {total_reward}")

    env.close()
