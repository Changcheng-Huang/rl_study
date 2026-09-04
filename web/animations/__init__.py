"""Animations module (package).

Keeps backward compatibility with:
    from animations import show_animation_library

Also exposes get_animation_data() for programmatic access.
"""

from .library import show_animation_library
from .algorithms import get_animation_data

__all__ = ["show_animation_library", "get_animation_data"]
