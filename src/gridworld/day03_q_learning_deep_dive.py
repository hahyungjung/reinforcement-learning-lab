import os
import random
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


ACTION_LABELS = {
    0: "Left",
    1: "Down",
    2: "Right",
    3: "Up",
}

ACTION_ARROWS = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑",
}


def create_environment():
    """
    FrozenLake-v1 is a small Grid World environment.

    S: Start
    F: Frozen safe tile
    H: Hole
    G: Goal

    is_slippery=False makes the environment deterministic.
    This is easier for Day 2 because the result of each action is predictable.
    """
    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=False,
    )
    return env


def get_grid_map(env):
    """
    Convert Gymnasium FrozenLake map into a readable 2D list.
    """
    raw_map = env.unwrapped.desc
    grid_map = []

    for row in raw_map:
        decoded_row = [cell.decode("utf-8") for cell in row]
        grid_map.append(decoded_row)

    return grid_map


def state_to_position(state, grid_size=4):
    """
    Convert state number into row and column position.

    In FrozenLake 4x4:
    state 0  -> row 0, col 0
    state 1  -> row 0, col 1
    state 4  -> row 1, col 0
    state 15 -> row 3, col 3
    """
    row = state // grid_size
    col = state % grid_size
    return row, col


def draw_grid_world(
    env,
    agent_state,
    title="Grid World",
    q_table=None,
    pause_time=0.5,
    save_path=None,
):
    """
    Draw Grid World with Matplotlib.

    If q_table is provided, the best action for each state is displayed as an arrow.
    """
    grid_map = get_grid_map(env)
    grid_size = len(grid_map)

    fig, ax = plt.subplots(figsize=(6, 6))

    for row in range(grid_size):
        for col in range(grid_size):
            tile = grid_map[row][col]

            if tile == "S":
                face_color = "#DDEEFF"
                label = "S"
            elif tile == "F":
                face_color = "#EEEEEE"
                label = ""
            elif tile == "H":
                face_color = "#FFCCCC"
                label = "H"
            elif tile == "G":
                face_color = "#CCFFCC"
                label = "G"
            else:
                face_color = "white"
                label = tile

            rect = patches.Rectangle(
                (col, grid_size - 1 - row),
                1,
                1,
                linewidth=1.5,
                edgecolor="black",
                facecolor=face_color,
            )
            ax.add_patch(rect)

            ax.text(
                col + 0.5,
                grid_size - 1 - row + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
            )

            if q_table is not None and tile not in ["H", "G"]:
                state = row * grid_size + col
                best_action = int(np.argmax(q_table[state]))
                arrow = ACTION_ARROWS[best_action]

                ax.text(
                    col + 0.5,
                    grid_size - 1 - row + 0.2,
                    arrow,
                    ha="center",
                    va="center",
                    fontsize=20,
                )

    agent_row, agent_col = state_to_position(agent_state, grid_size)

    agent_circle = patches.Circle(
        (
            agent_col + 0.5,
            grid_size - 1 - agent_row + 0.5,
        ),
        0.25,
        facecolor="#4477AA",
        edgecolor="black",
        linewidth=1.5,
    )
    ax.add_patch(agent_circle)

    ax.text(
        agent_col + 0.5,
        grid_size - 1 - agent_row + 0.5,
        "A",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="white",
    )

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show(block=False)
    plt.pause(pause_time)
    plt.close(fig)


def print_basic_environment_info(env):
    print("=== Environment Info ===")
    print("Environment: FrozenLake-v1")
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)
    print()
    print("Action Mapping:")
    for action, label in ACTION_LABELS.items():
        print(f"{action} = {label}")
    print()


def watch_random_agent_visual(max_steps=15):
    """
    Show a random agent moving in Grid World using Matplotlib.
    This agent does not learn yet.
    """
    env = create_environment()
    state, info = env.reset(seed=42)

    print("=== Random Agent Visualization ===")

    draw_grid_world(
        env,
        agent_state=state,
        title="Random Agent — Start",
        pause_time=1.0,
    )

    for step in range(max_steps):
        action = env.action_space.sample()

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(
            f"Step {step + 1}: "
            f"State={state}, "
            f"Action={action}({ACTION_LABELS[action]}), "
            f"Next State={next_state}, "
            f"Reward={reward}"
        )

        draw_grid_world(
            env,
            agent_state=next_state,
            title=(
                f"Random Agent — Step {step + 1} | "
                f"Action: {ACTION_LABELS[action]} | Reward: {reward}"
            ),
            pause_time=0.7,
        )

        state = next_state

        if done:
            print("Random agent episode ended.")
            break

    env.close()


def train_q_learning(
    episodes=5000,
    learning_rate=0.8,
    discount_factor=0.95,
    epsilon=1.0,
    epsilon_decay=0.999,
    min_epsilon=0.01,
):
    """
    Train a Q-learning agent.

    Q-table shape:
    rows    = states
    columns = actions

    q_table[state, action] means:
    how valuable it is to take this action in this state.
    """
    env = create_environment()

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    q_table = np.zeros((num_states, num_actions))
    rewards_per_episode = []

    for episode in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy strategy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            old_value = q_table[state, action]
            future_best_value = np.max(q_table[next_state])

            # Q-learning update formula
            new_value = old_value + learning_rate * (
                reward + discount_factor * future_best_value - old_value
            )

            q_table[state, action] = new_value

            state = next_state
            total_reward += reward

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

    env.close()

    return q_table, rewards_per_episode


def evaluate_agent(q_table, episodes=100):
    """
    Evaluate the trained agent.
    The agent always chooses the action with the highest Q-value.
    """
    env = create_environment()
    success_count = 0

    for episode in range(episodes):
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


def watch_trained_agent_visual(q_table, max_steps=20):
    """
    Show the trained agent moving in Grid World using Matplotlib.
    """
    env = create_environment()
    state, info = env.reset(seed=42)
    done = False

    print("=== Trained Agent Visualization ===")

    draw_grid_world(
        env,
        agent_state=state,
        title="Trained Agent — Start",
        q_table=q_table,
        pause_time=1.0,
    )

    for step in range(max_steps):
        action = int(np.argmax(q_table[state]))

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(
            f"Step {step + 1}: "
            f"State={state}, "
            f"Action={action}({ACTION_LABELS[action]}), "
            f"Next State={next_state}, "
            f"Reward={reward}"
        )

        draw_grid_world(
            env,
            agent_state=next_state,
            title=(
                f"Trained Agent — Step {step + 1} | "
                f"Action: {ACTION_LABELS[action]} | Reward: {reward}"
            ),
            q_table=q_table,
            pause_time=0.8,
        )

        state = next_state

        if done:
            print("Trained agent episode ended.")
            break

    env.close()


def plot_training_progress(rewards_per_episode):
    """
    Plot training progress using moving average reward.
    """
    os.makedirs("outputs/figures", exist_ok=True)

    window_size = 100

    if len(rewards_per_episode) >= window_size:
        moving_average = np.convolve(
            rewards_per_episode,
            np.ones(window_size) / window_size,
            mode="valid",
        )
    else:
        moving_average = rewards_per_episode

    plt.figure(figsize=(10, 5))
    plt.plot(moving_average)
    plt.title("Q-Learning Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.tight_layout()

    save_path = "outputs/figures/day02_training_progress.png"
    plt.savefig(save_path)
    plt.show()

    print(f"Saved training progress chart to: {save_path}")


def save_q_table(q_table):
    """
    Save trained Q-table as a NumPy file.
    """
    os.makedirs("outputs/q_tables", exist_ok=True)

    save_path = "outputs/q_tables/day02_frozenlake_q_table.npy"
    np.save(save_path, q_table)

    print(f"Saved Q-table to: {save_path}")


def save_policy_image(q_table):
    """
    Save final policy visualization as an image.
    """
    env = create_environment()
    state, info = env.reset(seed=42)

    save_path = "outputs/figures/day02_final_policy.png"

    draw_grid_world(
        env,
        agent_state=state,
        title="Final Learned Policy",
        q_table=q_table,
        pause_time=1.0,
        save_path=save_path,
    )

    env.close()

    print(f"Saved final policy image to: {save_path}")


def main():
    env = create_environment()
    print_basic_environment_info(env)
    env.close()

    # 1. Watch random movement before training
    watch_random_agent_visual(max_steps=15)

    # 2. Train Q-learning agent
    print("=== Training Q-learning Agent ===")
    q_table, rewards_per_episode = train_q_learning(episodes=5000)

    print()
    print("=== Trained Q-table ===")
    print(q_table)
    print()

    # 3. Evaluate trained agent
    success_rate = evaluate_agent(q_table, episodes=100)
    print(f"Success Rate: {success_rate:.2%}")
    print()

    # 4. Watch trained movement after training
    watch_trained_agent_visual(q_table)

    # 5. Save outputs
    plot_training_progress(rewards_per_episode)
    save_policy_image(q_table)
    save_q_table(q_table)


if __name__ == "__main__":
    main()