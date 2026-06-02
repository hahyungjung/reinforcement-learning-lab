import os
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def create_environment():
    return gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=False,
    )


def train_q_learning(
    episodes=3000,
    learning_rate=0.8,
    discount_factor=0.95,
    epsilon=1.0,
    epsilon_decay=0.999,
    min_epsilon=0.01,
):
    env = create_environment()

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    q_table = np.zeros((num_states, num_actions))
    rewards_per_episode = []
    epsilon_history = []

    for episode in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            old_value = q_table[state, action]
            future_best_value = np.max(q_table[next_state])

            new_value = old_value + learning_rate * (
                reward + discount_factor * future_best_value - old_value
            )

            q_table[state, action] = new_value

            state = next_state
            total_reward += reward

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        rewards_per_episode.append(total_reward)
        epsilon_history.append(epsilon)

    env.close()

    return q_table, rewards_per_episode, epsilon_history


def calculate_moving_average(values, window_size=100):
    if len(values) < window_size:
        return values

    return np.convolve(
        values,
        np.ones(window_size) / window_size,
        mode="valid",
    )


def evaluate_agent(q_table, episodes=100):
    env = create_environment()
    success_count = 0

    for _ in range(episodes):
        state, info = env.reset()
        done = False

        while not done:
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done and reward == 1:
                success_count += 1

    env.close()

    return success_count / episodes


def run_experiments():
    experiments = {
        "Fast Decay (0.95)": 0.95,
        "Medium Decay (0.995)": 0.995,
        "Slow Decay (0.999)": 0.999,
    }

    results = {}

    for experiment_name, epsilon_decay in experiments.items():
        print(f"Running experiment: {experiment_name}")

        q_table, rewards, epsilon_history = train_q_learning(
            episodes=3000,
            epsilon_decay=epsilon_decay,
        )

        success_rate = evaluate_agent(q_table, episodes=100)

        results[experiment_name] = {
            "q_table": q_table,
            "rewards": rewards,
            "epsilon_history": epsilon_history,
            "success_rate": success_rate,
        }

        print(f"{experiment_name} Success Rate: {success_rate:.2%}")
        print()

    return results


def plot_reward_comparison(results):
    os.makedirs("outputs/figures", exist_ok=True)

    plt.figure(figsize=(10, 6))

    for experiment_name, result in results.items():
        moving_average = calculate_moving_average(result["rewards"])
        plt.plot(moving_average, label=experiment_name)

    plt.title("Day 4 — Epsilon Decay Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward over 100 Episodes")
    plt.legend()
    plt.tight_layout()

    save_path = "outputs/figures/day04_epsilon_decay_comparison.png"
    plt.savefig(save_path)
    plt.show()

    print(f"Saved reward comparison chart to: {save_path}")


def plot_epsilon_history(results):
    os.makedirs("outputs/figures", exist_ok=True)

    plt.figure(figsize=(10, 6))

    for experiment_name, result in results.items():
        plt.plot(result["epsilon_history"], label=experiment_name)

    plt.title("Day 4 — Epsilon Decay History")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.legend()
    plt.tight_layout()

    save_path = "outputs/figures/day04_epsilon_history.png"
    plt.savefig(save_path)
    plt.show()

    print(f"Saved epsilon history chart to: {save_path}")


def main():
    results = run_experiments()

    print("=== Final Success Rates ===")
    for experiment_name, result in results.items():
        print(f"{experiment_name}: {result['success_rate']:.2%}")

    plot_reward_comparison(results)
    plot_epsilon_history(results)


if __name__ == "__main__":
    main()