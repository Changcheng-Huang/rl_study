import os

import streamlit as st

from .common import pick_best_quality, get_manim_video, render_derivation_steps
from .algorithms import get_animation_data
from algorithm_registry.integration import imported_animation_data


def show_animation_library():
    """Render the Animation Library page."""

    st.header("Algorithm Animations")
    st.caption("Manim-based videos that visualize how values and policies evolve during learning.")

    animation_data = get_animation_data()
    for label, imported in imported_animation_data().items():
        unique_label = label
        if unique_label in animation_data:
            unique_label = f"{label} (Imported)"
        animation_data[unique_label] = imported

    st.sidebar.markdown("## Select a Video")
    keys = list(animation_data.keys())
    selected_key = st.sidebar.radio("Algorithm", keys)
    data = animation_data[selected_key]

    st.subheader(data["title"])

    left, right = st.columns([1.45, 1.0], gap="large")

    with left:
        if data.get("video_path"):
            video_path = data["video_path"]
        else:
            quality = pick_best_quality(data["folder"], data["file"])
            video_path = get_manim_video(data["folder"], data["file"], quality=quality)

        if video_path:
            st.video(video_path)
        else:
            st.error("Video file not found.")
            st.info("Tip: check your Manim output folder under 'media/videos/'.")
            if data.get("folder"):
                st.code(os.path.join("media", "videos", data["folder"]), language="text")

    with right:
        with st.container(border=True):
            st.markdown("#### What to look for")
            if data.get("highlights"):
                st.markdown("\n".join([f"- {item}" for item in data["highlights"]]))
            else:
                st.caption("No highlight notes provided yet.")

        with st.container(border=True):
            st.markdown("#### Mathematical core")
            if data.get("latex"):
                st.latex(data["latex"])
            else:
                st.caption("No formula was supplied for this imported animation.")
            symbols = data.get("symbols", [])
            if symbols:
                st.markdown(
                    "**Symbols:** "
                    + ", ".join(
                        f"{item['symbol']}={item['meaning']}" for item in symbols
                    )
                )
            else:
                st.write("Symbols: s=state, a=action, r=reward, γ=discount, α=learning rate, ε=exploration")

    st.divider()
    tab_intro, tab_derivation = st.tabs(["Concept", "Derivation Notes"])

    with tab_intro:
        st.markdown(data["description"])
        st.markdown("---")
        st.markdown("**Suggested viewing flow**")
        viewing_flow = data.get("viewing_flow", [])
        if viewing_flow:
            st.markdown("\n".join(f"- {item}" for item in viewing_flow))
        else:
            st.markdown(
                "- Watch once without pausing (get the intuition).\n"
                "- Watch again while tracking one state's value / Q-value change.\n"
                "- Finally, map the visual changes back to the equations."
            )

    with tab_derivation:
        steps = data.get("derivation_steps")
        if steps:
            render_derivation_steps(steps)
        else:
            st.info("Derivation notes are not available for this video yet.")
