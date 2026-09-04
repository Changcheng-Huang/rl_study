import streamlit as st

from .common import right_card, render_quiz


DQN_QUIZ = [
    {
        "q": "What is the key motivation for using a neural network in DQN?",
        "options": [
            "A. To make the environment deterministic.",
            "B. To approximate Q(s,a) when a table is infeasible in large/high-dimensional state spaces.",
            "C. To eliminate the need for discounting.",
            "D. To remove exploration entirely.",
        ],
        "answer": 1,
        "explain": "Tabular Q-learning does not scale to large or continuous state spaces; DQN uses function approximation Q(s,a;theta).",
    },
    {
        "q": "What problem does experience replay mainly address?",
        "options": [
            "A. It removes the need for a target network.",
            "B. It makes training data less correlated and closer to i.i.d. by sampling random past transitions.",
            "C. It guarantees optimal exploration.",
            "D. It computes exact Bellman backups using a model.",
        ],
        "answer": 1,
        "explain": "Replay breaks correlation between sequential samples and improves stability when training neural networks.",
    },
    {
        "q": "Why is a target network (theta-) used in DQN?",
        "options": [
            "A. To reduce moving-target instability by keeping the TD target relatively stable for some steps.",
            "B. To speed up rendering of animations.",
            "C. To make epsilon-greedy unnecessary.",
            "D. To make rewards non-sparse.",
        ],
        "answer": 0,
        "explain": "Bootstrapped targets change if the same network is used on both sides; a slowly updated target network stabilizes learning.",
    },
    {
        "q": "Which loss function is typically used to train the Q-network in basic DQN?",
        "options": [
            "A. Cross-entropy loss.",
            "B. Mean squared TD error between the target y and Q(s,a;theta).",
            "C. KL divergence between policies.",
            "D. Hinge loss.",
        ],
        "answer": 1,
        "explain": "DQN minimizes an MSE loss: (y - Q(s,a;theta))^2 where y = r + gamma * max_a' Q(s',a';theta-).",
    },
]


def render():
    st.subheader("3. Deep Q-Networks (DQN)")
    st.caption("Scaling Q-learning with function approximation. Keywords: replay buffer, target network.")

    left, right = st.columns([1.6, 1.0], gap="large")

    with right:
        right_card(
            "Key ideas",
            bullets=[
                "Replace Q-table with a neural network Q(s,a; θ).",
                "Replay buffer makes training data closer to i.i.d.",
                "Target network θ⁻ stabilizes the bootstrapped target.",
            ],
        )
        right_card(
            "Common failure modes",
            bullets=[
                "Divergence from moving targets (no target network).",
                "Correlated samples (no replay).",
                "Overestimation bias (often improved with Double DQN).",
            ],
        )

    with left:
        tab_concept, tab_math, tab_code, tab_check = st.tabs(
            ["Concept", "Math", "Pseudocode", "Checkpoint"]
        )

        with tab_concept:
            st.markdown(
                """
### 3.1 Curse of dimensionality
Tabular methods work for small discrete state spaces. With images or continuous states, tables are impossible.

DQN approximates the action-value function with a neural network:  
**Q(s,a; θ) ≈ Q*(s,a)**

### 3.2 Two stabilizers
- **Experience replay**: store transitions and sample random minibatches.
- **Target network**: compute targets using a slowly-updated copy of parameters.
                """
            )

        with tab_math:
            st.markdown("### 3.3 TD target and loss")
            st.latex(r"y = r + \gamma\max_{a'}Q(s',a';\theta^-)")
            st.latex(r"L(\theta)=\mathbb{E}\big[(y - Q(s,a;\theta))^2\big]")
            st.info("Intuition: fit Q(s,a;θ) to a relatively stable target y computed using θ⁻.")

        with tab_code:
            st.code(
                """# Pseudocode (high level)
replay = ReplayBuffer()
for step in range(num_steps):
    a = epsilon_greedy(Q, s)
    s2, r, done = env.step(a)
    replay.add(s, a, r, s2, done)

    batch = replay.sample(B)
    y = r + gamma * (1-done) * max_a' Q_target(s2, a')
    loss = mse(Q(s,a), y)
    optimizer.step(loss)

    if step % target_update == 0:
        Q_target.load_state_dict(Q.state_dict())
""",
                language="python",
            )

        with tab_check:
            render_quiz("dqn", DQN_QUIZ)
