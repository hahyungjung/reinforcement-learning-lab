# Day 3 — Q-Learning Deep Dive

## Date

2026-05-26

## Topic

Understanding the Q-learning formula, Q-table updates, and epsilon-greedy strategy

---

## Goal

The goal of Day 3 was to understand how Q-learning updates the Q-table and how each parameter affects the learning process.

---

## Key Questions

1. What does the Q-learning formula mean?
2. How is the Q-table updated?
3. What does `max Q(s', a')` mean?
4. Why is epsilon-greedy strategy needed?
5. How do learning rate, discount factor, and epsilon affect training?

---

## Q-learning Formula

```text
Q(s, a) ← Q(s, a) + α [r + γ max Q(s', a') - Q(s, a)]