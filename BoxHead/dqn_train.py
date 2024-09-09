import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import boxhead_env

import os
from stable_baselines3 import DQN

# train dqn algorithm on the game
def train_dqn(lr=0.001, buffer_size=1000, batch_size=64, target_upd_interval=1000):
    # store trained models and logs
    train_model_dir = "train/models"
    train_log_dir = "train/log"

    os.makedirs(train_model_dir, exist_ok=True)
    os.makedirs(train_log_dir, exist_ok=True)
    reward_log = []

    env = gym.make('boxhead', render_mode='human')

    model = DQN('MlpPolicy', env, tensorboard_log=train_log_dir, 
                learning_rate=lr, buffer_size=buffer_size, batch_size=batch_size, target_update_interval=target_upd_interval)

    # test per number of timesteps
    TIMESTEPS = 10000
    iters = 0
    while True:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False) # train
        model.save(f"{train_model_dir}/dqn_{TIMESTEPS*iters}") # Save a trained model every TIMESTEPS

        # Save the reward log periodically
        with open(f"{train_log_dir}/reward_log.txt", "w") as f:
            for item in reward_log:
                f.write("%s\n" % item)


# test the trained DQN algorithm
def test_dqn(render=True):
    env = gym.make('boxhead', render_mode='human' if render else None)

    model = DQN.load('models/dqn_10000.zip', env=env)

    obs = env.reset()[0]
    terminated = False
    total_reward = 0

    while True:
        action, _ = model.predict(observation=obs, deterministic=True)
        obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward

        if terminated:
            break

    print(f"Total Reward: {total_reward}")
    return total_reward

# evaluate the DQN algorithm
def evaluate_dqn(hyperparams, num_episodes=10, total_timesteps=1000):
    total_rewards = []
    for _ in range(num_episodes):
        env = gym.make('boxhead', render_mode='human')
        model = DQN('MlpPolicy', env, **hyperparams)
        model.learn(total_timesteps=total_timesteps)
        obs = env.reset()[0]
        done = False
        total_reward = 0

        while True:
            action, _ = model.predict(observation=obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

            if done:
                break
        total_rewards.append(total_reward)

    avg_reward = sum(total_rewards) / num_episodes
    return avg_reward

def grid_search():
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'buffer_size': [1000, 5000, 10000],
        'batch_size': [32, 64, 128],
        'target_update_interval': [100, 500, 1000]
    }

    best_params = None
    best_avg_reward = -np.inf

    for learning_rate in param_grid['learning_rate']:
        for buffer_size in param_grid['buffer_size']:
            for batch_size in param_grid['batch_size']:
                for target_update_interval in param_grid['target_update_interval']:
                    params = {
                        'learning_rate': learning_rate,
                        'buffer_size': buffer_size,
                        'batch_size': batch_size,
                        'target_update_interval': target_update_interval
                    }

                    avg_reward = evaluate_dqn(params)
                    print(f"Hyperparams: {params}, Avg Reward: {avg_reward}")

                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        best_params = params

    print(f"Best Parameters: {best_params}, Best Avg Reward: {best_avg_reward}")

 
def plot_rewards(reward_log):
    plt.plot(reward_log)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Over Time')
    plt.show()


if __name__ == '__main__':
    train_dqn()
    # test_dqn()
    # evaluate_dqn()
    # grid_search()
