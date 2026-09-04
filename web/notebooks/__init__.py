import streamlit as st

from .common import page_header
from . import dp, td, dqn
from algorithm_registry.integration import imported_algorithms


CHAPTERS = {
    "1. Dynamic Programming (DP)": dp,
    "2. Temporal Difference (TD)": td,
    "3. Deep Q-Networks (DQN)": dqn,
}


def show_notebook_module():
    page_header(
        "Theory",
        "Bridge **math intuition** ↔ **implementable updates**. Choose a chapter and read it like a mini-lecture.",
    )

    imported = {
        f"Imported · {algorithm.manifest.name}": algorithm
        for algorithm in imported_algorithms()
    }

    # ---- Sidebar (simple + consistent) ----
    st.sidebar.markdown("## Chapters")
    topic = st.sidebar.radio("Select:", list(CHAPTERS.keys()) + list(imported.keys()))
    st.sidebar.divider()

    # ---- Routing ----
    if topic in CHAPTERS:
        CHAPTERS[topic].render()
    else:
        algorithm = imported[topic]
        st.subheader(algorithm.manifest.name)
        st.caption(algorithm.manifest.summary)
        theory_path = algorithm.path / algorithm.manifest.theory_file
        st.markdown(theory_path.read_text(encoding="utf-8"))
