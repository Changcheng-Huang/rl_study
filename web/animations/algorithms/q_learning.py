from typing import Dict, Any


def get_entry() -> Dict[str, Dict[str, Any]]:
    """Return a single-entry dict used by the animation library registry."""
    return {'Q-Learning': {'derivation_steps': [{'latex': ["\\delta = r + \\gamma\\max_{a'}Q(s',a') - Q(s,a)",
                                                'Q(s,a)\\leftarrow Q(s,a)+\\alpha\\,\\delta'],
                                      'text': 'Q-learning update is a gradient-like step driven by a TD error δ.',
                                      'title': 'Define TD error (off-policy target)'},
                                     {'latex': "\\text{target uses }\\max_{a'}Q(s',a')\\ \\text{even if "
                                               "}a'\\sim\\varepsilon\\text{-greedy}",
                                      'text': 'Even if the behavior policy explores, the target assumes the greedy '
                                              'next action (the max).',
                                      'title': 'Why it is off-policy'}],
                'description': '\n'
                               '**Q-Learning** learns Q(s,a) from experience and is **off-policy** because its target '
                               'uses a greedy max over next actions.\n',
                'file': 'QLearningDemo',
                'folder': 'QLearning',
                'highlights': ['Update target uses the best future action (the max).',
                               'Even with exploration, learning pushes toward greedy behavior.',
                               'In risky tasks, Q-learning may learn aggressive shortest paths.'],
                'latex': "Q(s,a)\\leftarrow Q(s,a)+\\alpha\\,[r+\\gamma\\max_{a'}Q(s',a')-Q(s,a)]",
                'title': 'Q-Learning (Off-Policy)'}}
