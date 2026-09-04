"""Algorithm registry for the Animation Library.

Each algorithm lives in its own module under `algorithms/`.

Public API:
- get_animation_data(): returns the merged ANIMATION_DATA dict used by the UI
"""

from typing import Dict, Any

from .policy_iteration import get_entry as _policy_iteration
from .value_iteration import get_entry as _value_iteration
from .q_learning import get_entry as _q_learning
from .sarsa import get_entry as _sarsa
from .dqn import get_entry as _dqn


def get_animation_data() -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    for fn in (_policy_iteration, _value_iteration, _q_learning, _sarsa, _dqn):
        data.update(fn())
    return data


__all__ = ["get_animation_data"]
