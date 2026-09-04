from typing import Dict, Any


def get_entry() -> Dict[str, Dict[str, Any]]:
    """Return a single-entry dict used by the animation library registry."""
    return {'SARSA': {'derivation_steps': [{'latex': ["\\delta = r + \\gamma Q(s',a') - Q(s,a)",
                                           'Q(s,a)\\leftarrow Q(s,a)+\\alpha\\,\\delta'],
                                 'text': 'SARSA uses the next action actually taken a′ to form the TD target.',
                                 'title': 'Define TD error (on-policy target)'},
                                {'latex': "\\text{Q-learning uses }\\max_{a'}Q(s',a')\\quad\\text{SARSA uses }Q(s',a')",
                                 'text': 'Replace the max over actions with the sampled next action; exploration risk '
                                         'is included in the target.',
                                 'title': 'Contrast with Q-learning'}],
           'description': '\n'
                          '**SARSA** is on-policy TD control: it updates using the next action actually taken under '
                          'the current behavior policy.\n',
           'file': 'SARSADemo',
           'folder': 'SARSA',
           'highlights': ['Targets the next sampled action a′ (no max operator).',
                          'Under ε-greedy exploration, SARSA tends to learn safer policies.',
                          'Great to compare with Q-learning in risky environments.'],
           'latex': "Q(s,a)\\leftarrow Q(s,a)+\\alpha\\,[r+\\gamma Q(s',a')-Q(s,a)]",
           'title': 'SARSA (On-Policy)'}}
