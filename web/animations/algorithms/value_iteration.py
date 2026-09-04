from typing import Dict, Any


def get_entry() -> Dict[str, Dict[str, Any]]:
    """Return a single-entry dict used by the animation library registry."""
    return {'Value Iteration': {'derivation_steps': [{'latex': 'V^*(s)=\\max_a Q^*(s,a)',
                                           'text': 'For the optimal policy, the best action at s achieves V*(s).',
                                           'title': 'Optimal value relates to optimal action-value'},
                                          {'latex': "Q^*(s,a)=\\sum_{s',r} p(s',r\\mid s,a)\\,[r+\\gamma V^*(s')]",
                                           'text': 'Action-value equals expected immediate reward plus discounted '
                                                   'optimal value of the next state.',
                                           'title': 'One-step lookahead definition of Q*'},
                                          {'latex': "V^*(s)=\\max_a\\sum_{s',r} p(s',r\\mid s,a)\\,[r+\\gamma V^*(s')]",
                                           'text': 'Combine the two equations to obtain the optimality backup used by '
                                                   'value iteration.',
                                           'title': 'Substitute Q* into V* (Bellman optimality equation)'},
                                          {'latex': 'V_{k+1}=\\mathcal{T}^*V_k',
                                           'text': 'Value iteration repeatedly applies the backup operator until '
                                                   'changes are small.',
                                           'title': 'Iterative application'}],
                     'description': '\n'
                                    '**Value Iteration** applies the Bellman optimality backup directly to the value '
                                    'function.\n'
                                    '\n'
                                    'When V converges, the greedy policy extracted from V is optimal.\n',
                     'file': 'ValueIterationGeneral',
                     'folder': 'Value_iteration',
                     'highlights': ['Values propagate outward from goal/terminal states (ripple effect).',
                                    'The greedy action becomes clearer as V stabilizes.',
                                    'Policy is derived after (or during late) iterations.'],
                     'latex': "V_{k+1}(s)\\leftarrow\\max_a\\sum_{s',r} p(s',r\\mid s,a)\\,[r+\\gamma V_k(s')]",
                     'title': 'Value Iteration'}}
