# Day 5 — Custom Grid World Environment

## Date

2026-05-28

## Topic

Building a custom Grid World environment from scratch

---

## Goal

The goal of Day 5 was to understand how a reinforcement learning environment works internally by building a simple Grid World environment without using Gymnasium.

In the previous days, Gymnasium provided the environment. In Day 5, I implemented the core environment logic directly.

---

## Why Build a Custom Environment?

In Gymnasium, the environment handles many things automatically:

* Current state management
* Action processing
* Next state calculation
* Reward calculation
* Episode termination
* Environment reset

By building a custom environment, I can understand how reinforcement learning environments are structured internally.

---

## Custom Grid World Design

The custom environment uses a 4x4 grid.

```text
S . . .
. W . .
. . W .
. . . G
```

| Symbol | Meaning        |
| ------ | -------------- |
| S      | Start position |
| W      | Wall           |
| G      | Goal           |
| .      | Empty space    |
| A      | Agent position |

---

## State

In this custom environment, the state is represented as the agent's position.

Example:

```text
state = (row, col)
```

The agent starts at:

```text
(0, 0)
```

The goal is located at:

```text
(3, 3)
```

---

## Action

The agent can choose one of four actions.

| Action Number | Action |
| ------------- | ------ |
| 0             | Up     |
| 1             | Down   |
| 2             | Left   |
| 3             | Right  |

---

## Reward Design

The reward system is defined as follows:

| Situation               | Reward |
| ----------------------- | -----: |
| Reaching the goal       |    +10 |
| Hitting a wall          |     -1 |
| Moving outside the grid |     -1 |
| Normal movement         |   -0.1 |

The purpose of this reward design is to encourage the agent to reach the goal while avoiding invalid moves and walls.

---

## Core Environment Functions

### `reset()`

The `reset()` function starts a new episode.

It moves the agent back to the start position.

```python
state = env.reset()
```

---

### `step(action)`

The `step(action)` function applies an action to the environment.

It returns:

```python
next_state, reward, done
```

| Return Value | Meaning                         |
| ------------ | ------------------------------- |
| `next_state` | Agent's new position            |
| `reward`     | Reward from the action          |
| `done`       | Whether the episode is finished |

---

### `render()`

The `render()` function displays the current Grid World state.

It helps visualize where the agent is located.

---

## Difference from Gymnasium

| Area                | Gymnasium Environment    | Custom Environment               |
| ------------------- | ------------------------ | -------------------------------- |
| State management    | Handled automatically    | Implemented manually             |
| Action logic        | Built in                 | Implemented manually             |
| Reward logic        | Built in                 | Designed manually                |
| Episode termination | Built in                 | Implemented manually             |
| Learning purpose    | Use existing environment | Understand environment structure |

---

## What I Built

I built a simple custom Grid World environment with:

* A start position
* A goal position
* Wall positions
* Four possible actions
* Reward logic
* Episode termination logic
* Text-based rendering

---

## Key Takeaway

A reinforcement learning environment is mainly built around two important functions:

```text
reset()
step(action)
```

The `reset()` function starts a new episode, and the `step(action)` function processes the agent's action and returns the result.

The core environment flow is:

```text
Current State → Action → Next State → Reward → Done
```

---

## Next Step

Day 6 will focus on connecting this custom Grid World environment with a Q-learning agent.
