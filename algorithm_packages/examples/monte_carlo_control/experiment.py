from __future__ import annotations

import gymnasium as gym
import numpy as np


def get_spec():
    return {
        "parameters": {
            "episodes": {
                "type": "int",
                "default": 2000,
                "min": 100,
                "max": 10000,
                "step": 100,
                "label": "Episodes",
                "help": "Number of complete FrozenLake episodes."
            },
            "gamma": {
                "type": "float",
                "default": 0.99,
                "min": 0.1,
                "max": 1.0,
                "step": 0.01,
                "label": "Discount Factor (γ)"
            },
            "epsilon": {
                "type": "float",
                "default": 0.2,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "label": "Exploration (ε)"
            },
            "slippery": {
                "type": "bool",
                "default": False,
                "label": "Enable Slippery Ice"
            },
            "seed": {
                "type": "int",
                "default": 7,
                "min": 0,
                "max": 100000,
                "step": 1,
                "label": "Random Seed"
            }
        }
    }


def _choose_action(q_values, epsilon, action_count, rng):
    if rng.random() < epsilon:
        return int(rng.integers(action_count))
    best = np.flatnonzero(q_values == np.max(q_values))
    return int(rng.choice(best))


def run(parameters, reporter):
    episodes = parameters["episodes"]
    gamma = parameters["gamma"]
    epsilon = parameters["epsilon"]
    slippery = parameters["slippery"]
    seed = parameters["seed"]

    env = gym.make("FrozenLake-v1", is_slippery=slippery)
    env.action_space.seed(seed)
    rng = np.random.default_rng(seed)

    state_count = env.observation_space.n
    action_count = env.action_space.n
    q_table = np.zeros((state_count, action_count), dtype=float)
    returns_sum = np.zeros_like(q_table)
    returns_count = np.zeros_like(q_table)

    rewards = []
    success_rates = []
    episode_lengths = []
    progress_interval = max(1, episodes // 100)

    try:
        for episode_index in range(episodes):
            state, _ = env.reset(seed=seed + episode_index)
            trajectory = []

            for _ in range(100):
                action = _choose_action(
                    q_table[state], epsilon, action_count, rng
                )
                next_state, reward, terminated, truncated, _ = env.step(action)
                trajectory.append((state, action, float(reward)))
                state = next_state
                if terminated or truncated:
                    break

            returns = [0.0] * len(trajectory)
            discounted_return = 0.0
            for index in range(len(trajectory) - 1, -1, -1):
                discounted_return = trajectory[index][2] + gamma * discounted_return
                returns[index] = discounted_return

            visited = set()
            for index, (visited_state, visited_action, _) in enumerate(trajectory):
                state_action = (visited_state, visited_action)
                if state_action in visited:
                    continue
                visited.add(state_action)
                returns_sum[state_action] += returns[index]
                returns_count[state_action] += 1
                q_table[state_action] = (
                    returns_sum[state_action] / returns_count[state_action]
                )

            episode_reward = sum(step[2] for step in trajectory)
            rewards.append(episode_reward)
            episode_lengths.append(len(trajectory))
            window = rewards[-100:]
            success_rates.append(float(np.mean(window)))

            if (
                (episode_index + 1) % progress_interval == 0
                or episode_index + 1 == episodes
            ):
                reporter.progress(
                    episode_index + 1,
                    episodes,
                    f"Episode {episode_index + 1}/{episodes}"
                )
                reporter.metric(
                    "success_rate",
                    success_rates[-1],
                    step=episode_index + 1
                )
    finally:
        env.close()

    return {
        "metrics": {
            "episode_reward": rewards,
            "success_rate_100": success_rates,
            "episode_length": episode_lengths
        },
        "summary": {
            "episodes": episodes,
            "final_success_rate": round(success_rates[-1], 3),
            "learned_state_actions": int(np.count_nonzero(returns_count))
        },
        "artifacts": []
    }
