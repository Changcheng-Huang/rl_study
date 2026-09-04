import os
import re
from typing import Optional, List, Dict, Any

import streamlit as st

# ==========================================
# 1. Utility Functions
# ==========================================
def _score_quality(q: str):
    """Prefer higher resolution/fps labels first (e.g., 1080p60)."""
    m = re.search(r"(\d+)p(\d+)", q)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(\d+)p", q)
    if m:
        return (int(m.group(1)), 0)
    return (0, 0)

def pick_best_quality(scene_folder: str, video_name: str) -> Optional[str]:
    """Auto-pick the best available quality folder for a given scene/video."""
    base_dir = os.path.join("media", "videos", scene_folder)
    if not os.path.isdir(base_dir):
        return None

    qualities: List[str] = []
    for q in os.listdir(base_dir):
        mp4 = os.path.join(base_dir, q, f"{video_name}.mp4")
        if os.path.exists(mp4):
            qualities.append(q)

    if not qualities:
        return None

    qualities.sort(key=_score_quality, reverse=True)
    return qualities[0]

def get_manim_video(scene_folder: str, video_name: str, quality: Optional[str] = None) -> Optional[str]:
    if quality is None:
        quality = pick_best_quality(scene_folder, video_name)
    if quality is None:
        return None
    path = os.path.join("media", "videos", scene_folder, quality, f"{video_name}.mp4")
    return path if os.path.exists(path) else None

def render_derivation_steps(steps: List[Dict[str, Any]]):
    """Render derivation as 'real derivation': text paragraphs + separate LaTeX blocks."""
    for i, step in enumerate(steps, start=1):
        title = step.get("title")
        text = step.get("text")
        latex = step.get("latex")
        # number each step for clarity
        if title:
            st.markdown(f"**Step {i}: {title}**")
        else:
            st.markdown(f"**Step {i}**")
        if text:
            st.markdown(text)
        if latex:
            # allow single string or list of latex blocks
            if isinstance(latex, list):
                for block in latex:
                    st.latex(block)
            else:
                st.latex(latex)
        if i != len(steps):
            st.markdown("---")
