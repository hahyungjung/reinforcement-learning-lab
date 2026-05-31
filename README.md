# Reinforcement Learning Lab

This repository documents my hands-on study of Reinforcement Learning, starting from basic concepts and gradually moving toward game AI.

## Goal

The goal of this project is to understand Reinforcement Learning by building small environments, implementing core algorithms, and documenting the learning process step by step.

This repository is designed as both a study record and a technical portfolio.

---

## Study Roadmap

1. Reinforcement Learning Basics
2. Grid World Environment
3. Q-Learning
4. Epsilon-Greedy Strategy
5. Visualization of Agent Behavior
6. Custom Grid World Environment
7. Tic-Tac-Toe AI
8. Simple Chess AI Concepts
9. Deep Q-Networks
10. AlphaZero-style Learning Concepts

---

## Current Progress

| Day   | Topic                         | Status    |
| ----- | ----------------------------- | --------- |
| Day 1 | Reinforcement Learning Basics | Completed |
| Day 2 | Grid World with Q-Learning    | Completed |
| Day 3 | Q-Learning Deep Dive          | Completed |
| Day 4 | Exploration vs Exploitation   | Planned   |

---

## Completed Work

### Day 1 — Reinforcement Learning Basics

Studied the basic structure of Reinforcement Learning.

Main concepts:

* Agent
* Environment
* State
* Action
* Reward
* Episode
* Policy
* Value
* Q-value
* Exploration
* Exploitation

### Day 2 — Grid World with Q-Learning

Implemented a simple Grid World example using Gymnasium's `FrozenLake-v1` environment.

Main features:

* Random Agent demo
* Q-learning training loop
* Q-table update
* Success rate evaluation
* Matplotlib Grid World visualization
* Final policy image export

### Day 3 — Q-Learning Deep Dive

Reviewed the Q-learning formula and studied how Q-values are updated during training.

Main concepts:

* Q-learning update formula
* Q-table update logic
* `max Q(s', a')`
* Learning rate
* Discount factor
* Epsilon-greedy strategy
* Exploration vs Exploitation

---

## Project Structure

```text
docs/       Study notes and curriculum
src/        Main Python source code
outputs/    Generated figures and results
tests/      Test files
```

---

## Tech Stack

* Python
* Gymnasium
* NumPy
* Matplotlib
* GitHub
* Markdown

---

## Main Output Examples

The project currently generates visual outputs such as:

```text
outputs/figures/day02_training_progress.png
outputs/figures/day02_final_policy.png
```

These outputs show the Q-learning training progress and the final learned policy in the Grid World environment.

---

## Next Step

Day 4 will focus on Exploration vs Exploitation by experimenting with epsilon and epsilon decay.

The next goal is to understand how the agent balances trying new actions and using the best-known actions during training.
