from typing import Dict, Any


def get_entry() -> Dict[str, Dict[str, Any]]:
    """Return a single-entry dict used by the animation library registry."""
    return {'Policy Iteration': {'derivation_steps': [{'latex': 'V^\\pi(s)=\\mathbb{E}\\big[R_{t+1}+\\gamma V^\\pi(S_{t+1})\\mid '
                                                     'S_t=s, A_t=\\pi(s)\\big]',
                                            'text': 'For a fixed policy π, the value equals expected return following '
                                                    'π.',
                                            'title': 'Start from Bellman expectation equation (fixed policy)'},
                                           {'latex': "V^\\pi(s)=\\sum_{s',r} p(s',r\\mid s,\\pi(s))\\,[r + \\gamma "
                                                     "V^\\pi(s')]",
                                            'text': 'Convert the expectation into a sum over next state and reward '
                                                    "using the known model p(s',r|s,a).",
                                            'title': 'Expand the expectation using transition dynamics'},
                                           {'latex': "\\pi_{new}(s)\\in\\arg\\max_a\\sum_{s',r} p(s',r\\mid "
                                                     "s,a)\\,[r+\\gamma V^\\pi(s')]",
                                            'text': 'Given V^π, choose actions that maximize one-step lookahead '
                                                    'return; this defines an improved policy.',
                                            'title': 'Policy improvement step'}],
                      'description': '\n'
                                     '**Policy Iteration** is a model-based dynamic programming method that '
                                     'alternates:\n'
                                     '\n'
                                     '1) **Policy Evaluation**: compute the value function under the current policy\n'
                                     '\n'
                                     '2) **Policy Improvement**: update the policy greedily with respect to that value '
                                     'function\n'
                                     '\n'
                                     'Repeat until the policy stops changing.\n',
                      'file': 'PolicyIteration',
                      'folder': 'Policy_iteration',
                      'highlights': ['During Evaluation: arrows (policy) stay fixed, values update.',
                                     'During Improvement: arrows change to become greedier.',
                                     'Repeat until arrows stop changing (convergence).'],
                      'latex': "V^\\pi(s)=\\sum_{s',r} p(s',r\\mid s,\\pi(s))\\,[r + \\gamma V^\\pi(s')]",
                      'title': 'Policy Iteration'}}
