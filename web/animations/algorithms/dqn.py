from typing import Dict, Any


def get_entry() -> Dict[str, Dict[str, Any]]:
    """Return a single-entry dict used by the animation library registry."""
    return {'Deep Q-Network (DQN)': {'derivation_steps': [{'latex': "y = r + \\gamma\\max_{a'}Q(s',a';\\theta^-)",
                                                'text': 'DQN uses a target value y computed from the reward and the '
                                                        'next-state greedy action.',
                                                'title': 'Start from TD target y'},
                                               {'latex': 'L(\\theta) = \\mathbb{E}\\big[(y - Q(s,a;\\theta))^2\\big]',
                                                'text': 'Train the network to make Q(s,a;θ) match the target y (MSE '
                                                        'over samples).',
                                                'title': 'Define squared error loss'},
                                               {'latex': '\\theta^-\\ \\text{is updated periodically from}\\ \\theta',
                                                'text': 'Keep the target relatively fixed for a while to reduce the '
                                                        'moving-target problem during training.',
                                                'title': 'Why use a target network θ⁻'}],
                          'description': '\n'
                                         '**DQN** approximates Q(s,a) with a neural network when tables are '
                                         'infeasible.\n'
                                         'It stabilizes learning via **experience replay** and a **target network**.\n',
                          'file': 'DQNDemo',
                          'folder': 'DQN',
                          'highlights': ['Neural net replaces the Q-table (function approximation).',
                                         'Replay buffer helps stabilize learning.',
                                         'Target network parameters θ⁻ lag behind θ.'],
                          'latex': "L(\\theta)=\\mathbb{E}\\Big[(r+\\gamma\\max_{a'}Q(s',a';\\theta^-)-Q(s,a;\\theta))^2\\Big]",
                          'title': 'Deep Q-Network (DQN)'}}
