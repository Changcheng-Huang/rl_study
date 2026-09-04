import streamlit as st

from .common import right_card, render_quiz


DP_QUIZ = [
    {
        "q": "Which assumption is required for Dynamic Programming (DP) methods?",
        "options": [
            "A. The agent must explore to learn the model.",
            "B. The environment dynamics p(s', r | s, a) are known.",
            "C. Rewards must be dense (non-sparse).",
            "D. The state space must be continuous.",
        ],
        "answer": 1,
        "explain": "DP is model-based planning: it assumes the transition/reward model is known so Bellman backups can be computed.",
    },
    {
        "q": "In the Bellman optimality backup for V*(s), what does the max over actions represent?",
        "options": [
            "A. The action actually sampled under ε-greedy.",
            "B. Averaging over all actions equally.",
            "C. Choosing the best action to maximize expected return.",
            "D. Choosing the action that minimizes immediate reward.",
        ],
        "answer": 2,
        "explain": "The optimality equation defines V*(s) as the best achievable return from s, so it takes the maximum over actions.",
    },
    {
        "q": "What is the procedural difference between Policy Iteration and Value Iteration?",
        "options": [
            "A. Policy Iteration alternates evaluation and improvement; Value Iteration applies optimal backups directly to V.",
            "B. Policy Iteration is model-free; Value Iteration is model-based.",
            "C. Value Iteration requires exploration; Policy Iteration does not.",
            "D. They are identical algorithms with different names.",
        ],
        "answer": 0,
        "explain": "Policy Iteration has two phases (evaluate π then improve π). Value Iteration repeatedly applies the optimal backup to V and derives a greedy policy.",
    },
    {
        "q": "Why can DP methods plan without exploration?",
        "options": [
            "A. Because they approximate Q with a neural network.",
            "B. Because they reuse a replay buffer.",
            "C. Because they can compute expectations using the known model.",
            "D. Because they assume γ = 0.",
        ],
        "answer": 2,
        "explain": "With p(s',r|s,a) available, DP can compute expected returns via Bellman backups rather than sampling transitions through exploration.",
    },
]


def render():
    st.subheader("1. Dynamic Programming (DP)")
    st.caption("Model-based planning with known dynamics. Keywords: Bellman equation, backup, bootstrapping.")

    left, right = st.columns([1.6, 1.0], gap="large")

    with right:
        right_card(
            "Key ideas",
            bullets=[
                "DP assumes you know the environment model: p(s', r | s, a).",
                "You can plan without exploration by repeatedly applying Bellman backups.",
                "Policy Iteration: evaluate then improve. Value Iteration: combine them.",
            ],
        )
        right_card(
            "When to use",
            bullets=[
                "Small/medium discrete MDPs with known dynamics.",
                "As a gold-standard baseline for RL environments (ground-truth optimal).",
            ],
        )

    with left:
        tab_concept, tab_math, tab_code, tab_check = st.tabs(
            ["Concept", "Math", "Pseudocode", "Checkpoint"]
        )

        with tab_concept:
            st.markdown(
                """
### 1.1 Premise
Dynamic Programming methods assume the agent has a *perfect map* of the world:
- Transition model: **p(s′, r | s, a)**
- Reward function is included in that model (or available separately)

Because the rules are known, we don’t need to explore blindly—we can **compute** expected returns.
                """
            )
            st.info("DP is the planning baseline: it solves the MDP directly via repeated backups.")

        with tab_math:
            st.markdown("### 1.2 Bellman Optimality Backup (state value)")
            st.latex(r"V^*(s)=\max_a\sum_{s',r}p(s',r\mid s,a)\,[r+\gamma V^*(s')]")

            st.markdown("### 1.3 Policy Iteration vs. Value Iteration (high level)")
            st.markdown(
                "- **Policy Iteration**: alternate *Policy Evaluation* and *Policy Improvement*.\n"
                "- **Value Iteration**: repeatedly apply the optimality backup; derive policy from V."
            )

        with tab_code:
            st.markdown("### 1.4 Value Iteration (pseudocode)")
            st.code(
                """# Initialize V(s) arbitrarily
while delta > theta:
    delta = 0
    for s in States:
        v = V[s]
        V[s] = max_a sum_{s',r} p(s',r|s,a) * (r + gamma * V[s'])
        delta = max(delta, abs(v - V[s]))
""",
                language="python",
            )

        with tab_check:
            render_quiz("dp", DP_QUIZ)
