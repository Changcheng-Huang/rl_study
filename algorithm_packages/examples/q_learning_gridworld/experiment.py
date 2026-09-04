from __future__ import annotations

import random


LAYOUT = ("SFFF", "FHFH", "FFFH", "HFFG")
ACTIONS = {
    0: (0, -1),
    1: (1, 0),
    2: (0, 1),
    3: (-1, 0),
}


def get_spec():
    return {
        "parameters": {
            "episodes": {
                "type": "int",
                "default": 2000,
                "min": 100,
                "max": 20000,
                "step": 100,
                "label": "Episodes",
            },
            "alpha": {
                "type": "float",
                "default": 0.1,
                "min": 0.01,
                "max": 1.0,
                "step": 0.01,
                "label": "Learning Rate (α)",
            },
            "gamma": {
                "type": "float",
                "default": 0.99,
                "min": 0.1,
                "max": 0.999,
                "step": 0.01,
                "label": "Discount Factor (γ)",
            },
            "epsilon": {
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "label": "Initial Exploration (ε)",
            },
            "epsilon_decay": {
                "type": "float",
                "default": 0.995,
                "min": 0.9,
                "max": 0.9999,
                "step": 0.0001,
                "label": "Exploration Decay",
            },
            "seed": {
                "type": "int",
                "default": 7,
                "min": 0,
                "max": 100000,
                "step": 1,
                "label": "Random Seed",
            },
        },
        "presentation": {
            "task": {
                "mission": (
                    "Cross the frozen lake from Start (S) to Goal (G) without "
                    "falling into a Hole (H)."
                ),
                "dynamics": [
                    "Actions are deterministic: left, down, right, and up.",
                    "Moving beyond the grid keeps the agent in the current cell.",
                    "Goal and Hole cells end the episode.",
                ],
                "rewards": [
                    "Goal: +1 and the episode ends.",
                    "Safe cell or boundary: 0.",
                    "Hole: 0 and the episode ends.",
                ],
            },
            "environment_map": {
                "kind": "grid",
                "layout": list(LAYOUT),
                "legend": {
                    "S": {
                        "label": "START",
                        "role": "start",
                        "terminal": False,
                        "color": "#8ecae6",
                    },
                    "F": {
                        "label": "ICE",
                        "role": "normal",
                        "terminal": False,
                        "icon": "❄️",
                    },
                    "H": {
                        "label": "HOLE",
                        "role": "hazard",
                        "terminal": True,
                        "color": "#444444",
                        "text_color": "#ffffff",
                    },
                    "G": {
                        "label": "GOAL",
                        "role": "goal",
                        "terminal": True,
                        "color": "#66cc66",
                    },
                },
                "actions": {
                    "0": {"label": "Left", "arrow": "←"},
                    "1": {"label": "Down", "arrow": "↓"},
                    "2": {"label": "Right", "arrow": "→"},
                    "3": {"label": "Up", "arrow": "↑"},
                },
            },
        },
    }


def _step(state, action):
    rows = len(LAYOUT)
    columns = len(LAYOUT[0])
    row, column = divmod(state, columns)
    delta_row, delta_column = ACTIONS[action]
    next_row = min(max(row + delta_row, 0), rows - 1)
    next_column = min(max(column + delta_column, 0), columns - 1)
    next_state = next_row * columns + next_column
    symbol = LAYOUT[next_row][next_column]
    return next_state, 1.0 if symbol == "G" else 0.0, symbol in {"G", "H"}


def _best_action(values, random_source):
    maximum = max(values)
    choices = [index for index, value in enumerate(values) if value == maximum]
    return random_source.choice(choices)


def run(parameters, reporter):
    episodes = int(parameters["episodes"])
    alpha = float(parameters["alpha"])
    gamma = float(parameters["gamma"])
    epsilon_start = float(parameters["epsilon"])
    epsilon_decay = float(parameters["epsilon_decay"])
    random_source = random.Random(int(parameters["seed"]))
    q_values = [[0.0 for _ in ACTIONS] for _ in range(16)]
    episode_rewards = []
    smoothed_success = []
    report_interval = max(1, episodes // 100)

    for episode in range(episodes):
        state = 0
        total_reward = 0.0
        epsilon = max(0.02, epsilon_start * (epsilon_decay**episode))
        for _ in range(100):
            if random_source.random() < epsilon:
                action = random_source.randrange(len(ACTIONS))
            else:
                action = _best_action(q_values[state], random_source)
            next_state, reward, done = _step(state, action)
            target = reward
            if not done:
                target += gamma * max(q_values[next_state])
            q_values[state][action] += alpha * (
                target - q_values[state][action]
            )
            state = next_state
            total_reward += reward
            if done:
                break

        episode_rewards.append(total_reward)
        window = episode_rewards[-100:]
        success_rate = sum(window) / len(window)
        smoothed_success.append(success_rate)
        reporter.metric("success_rate", success_rate, step=episode + 1)
        if (episode + 1) % report_interval == 0 or episode + 1 == episodes:
            reporter.progress(
                episode + 1,
                episodes,
                f"Training episode {episode + 1}/{episodes}",
            )

    terminal_states = {
        index
        for index, symbol in enumerate("".join(LAYOUT))
        if symbol in {"G", "H"}
    }
    state_values = []
    best_actions = []
    for state, values in enumerate(q_values):
        if state in terminal_states:
            state_values.append(None)
            best_actions.append(None)
        else:
            state_values.append(max(values))
            best_actions.append(_best_action(values, random_source))

    return {
        "metrics": {"success_rate": smoothed_success},
        "summary": {
            "episodes": episodes,
            "final_success_rate": smoothed_success[-1],
            "successful_episodes": int(sum(episode_rewards)),
        },
        "artifacts": [],
        "views": {
            "policy_grid": {
                "state_values": state_values,
                "best_actions": best_actions,
            }
        },
    }
