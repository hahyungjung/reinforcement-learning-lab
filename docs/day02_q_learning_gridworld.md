# Day 2 — Grid World with Q-Learning

## Date

2026-05-25

## Topic

Q-learning practice using Gymnasium FrozenLake-v1

## What I Built

I implemented a simple Grid World reinforcement learning example using Gymnasium's FrozenLake-v1 environment. The project shows how a random agent behaves before learning and how a Q-learning agent improves its behavior through repeated training.

## Main Components

| Component | Purpose |
|---|---|
| Environment | Provides the Grid World, state, reward, and episode rules |
| Random Agent | Shows the behavior of an agent before learning |
| Training Q-learning Agent | Learns better actions by updating the Q-table |
| Trained Q-table | Stores learned values for each state-action pair |
| Success Rate | Evaluates how often the trained agent reaches the goal |

## Q-learning Formula

Q(s, a) ← Q(s, a) + α [r + γ max Q(s', a') - Q(s, a)]

## Key Concepts

- Grid World
- FrozenLake-v1
- Q-table
- Q-value
- Learning rate
- Discount factor
- Epsilon-greedy strategy
- Exploration vs Exploitation
- Success rate

## What I Observed

Before training, the random agent moved without strategy and could easily fall into a hole. After training, the Q-learning agent learned a safe path to the goal.

Example trained path:

0 → 4 → 8 → 9 → 13 → 14 → 15

## Output Files

- `outputs/figures/day02_training_progress.png`
- `outputs/figures/day02_final_policy.png`
- `outputs/q_tables/day02_frozenlake_q_table.npy`

## Key Takeaway

Q-learning allows an agent to learn which action is better in each state by repeatedly interacting with the environment and updating the Q-table.