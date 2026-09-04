import streamlit as st

from .common import right_card, render_quiz


TD_QUIZ = [
    {
        "q": "What is the intuition behind the TD error?",
        "options": [
            "A. It is the difference between two sampled actions.",
            "B. It is a prediction error: target minus current estimate.",
            "C. It is the difference between two policies.",
            "D. It is the gradient of the reward function.",
        ],
        "answer": 1,
        "explain": "TD methods update from a prediction error (target - estimate). This error tells you how surprised the agent is.",
    },
    {
        "q": "Why is SARSA considered on-policy?",
        "options": [
            "A. It never explores.",
            "B. Its target uses the next action actually taken under the behavior policy (a').",
            "C. It requires a known transition model.",
            "D. It always uses max over next actions.",
        ],
        "answer": 1,
        "explain": "SARSA targets r + gamma * Q(s', a') where a' comes from the current behavior policy (e.g., epsilon-greedy).",
    },
    {
        "q": "What makes Q-learning off-policy?",
        "options": [
            "A. It uses r + gamma * max_a' Q(s', a') as the target regardless of which action was taken next.",
            "B. It uses Monte Carlo returns only.",
            "C. It can only learn in deterministic environments.",
            "D. It requires a replay buffer.",
        ],
        "answer": 0,
        "explain": "Even while the behavior policy explores, the Q-learning target assumes the greedy next action via max, so it learns the greedy policy off-policy.",
    },
    {
        "q": "In CliffWalking with epsilon-greedy exploration, which behavior is more typical?",
        "options": [
            "A. Q-learning often learns safer routes than SARSA.",
            "B. SARSA often learns safer routes than Q-learning.",
            "C. Both always learn identical routes.",
            "D. Neither can learn because rewards are negative.",
        ],
        "answer": 1,
        "explain": "SARSA's update includes the risk of exploration (on-policy), so it tends to prefer safer paths under epsilon-greedy behavior.",
    },
]


def render():
    st.subheader("2. Temporal Difference (TD) Learning")
    st.caption("Model-free learning from experience. Keywords: TD error, on-policy, off-policy.")

    left, right = st.columns([1.6, 1.0], gap="large")

    with right:
        right_card(
            "Key ideas",
            bullets=[
                "TD learns from samples (no model needed).",
                "Bootstrapping: update uses current estimates as targets.",
                "SARSA is on-policy (uses actual next action). Q-learning is off-policy (uses max).",
            ],
        )
        right_card(
            "Mental model",
            bullets=[
                "SARSA learns a policy that matches its exploration behavior (often safer).",
                "Q-learning learns the greedy optimal policy, even while exploring (often riskier).",
            ],
        )

    with left:
        tab_concept, tab_math, tab_code, tab_check = st.tabs(
            ["Concept", "Math", "Pseudocode", "Checkpoint"]
        )

        with tab_concept:
            st.markdown(
                """
### 2.1 Why TD?
In most real-world problems we **don't** know the transition model. TD methods learn from interaction.

TD combines:
1) **Monte Carlo**: learn from experience samples  
2) **DP**: bootstrap from existing value estimates
                """
            )

        with tab_math:
            st.markdown("### 2.2 SARSA (On-policy)")
            st.latex(r"Q(s,a)\leftarrow Q(s,a)+\alpha\,[r+\gamma Q(s',a')-Q(s,a)]")

            st.markdown("### 2.3 Q-learning (Off-policy)")
            st.latex(r"Q(s,a)\leftarrow Q(s,a)+\alpha\,[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]")

            st.info("Key difference: SARSA targets Q(s′,a′) from the behavior policy; Q-learning targets max over actions.")

        with tab_code:
            st.markdown("**SARSA update**")
            st.code(
                """next_action = choose_action(next_state)  # ε-greedy
target = reward + gamma * Q[next_state][next_action]
Q[state][action] += alpha * (target - Q[state][action])
""",
                language="python",
            )

            st.markdown("**Q-learning update**")
            st.code(
                """target = reward + gamma * max(Q[next_state])
Q[state][action] += alpha * (target - Q[state][action])
""",
                language="python",
            )

            st.info("Difference: SARSA uses Q(s′,a′) (on-policy), while Q-learning uses maxₐ′Q(s′,a′) (off-policy).")

        with tab_check:
            render_quiz("td", TD_QUIZ)
