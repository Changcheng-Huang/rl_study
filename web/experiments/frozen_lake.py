import streamlit as st
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ==========================================
# 1. Visualization & Training Utils (Local)
# ==========================================
def plot_heatmap_and_arrows(q_table, shape, custom_desc=None):
    """
    Frozen Lake 专用绘图函数
    [Color]: Inverted (Dark=Low Value, Light=High Value)
    """
    state_values = np.max(q_table, axis=1).reshape(shape)
    rows, cols = shape
    is_learned = not np.all(q_table == 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 修改1：cmap="YlGnBu_r" (反转颜色：深色代表低价值0，浅色代表高价值1)
    kwargs = {
        'vmin': 0, 'vmax': 1, 'cmap': "YlGnBu_r",
        'linewidths': 1, 'linecolor': '#eeeeee'
    }
    
    sns.heatmap(state_values, annot=False, cbar=True, ax=ax, alpha=0.9,
                cbar_kws={'label': 'Estimated State Value (V)'}, **kwargs)
    
    if custom_desc is None:
        custom_desc = ["SFFF", "FHFH", "FFFH", "HFFG"]

    for r in range(rows):
        for c in range(cols):
            s = r * cols + c
            best_action = np.argmax(q_table[s])
            value = state_values[r, c]
            
            # 修改2：文字颜色适配
            # 高价值(>0.5)是浅色背景，用黑字
            if is_learned:
                text_color = 'black' if value > 0.5 else 'white'
                ax.text(c + 0.05, r + 0.15, f"{value:.2f}", 
                        color=text_color, fontsize=7, ha='left', va='center')

            char = custom_desc[r][c]
            if char == 'H': 
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color='#444444', alpha=0.7))
                ax.text(c+0.5, r+0.5, "HOLE", color='white', ha='center', va='center', fontsize=8)
            elif char == 'G': 
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color='#66cc66', alpha=0.8))
                ax.text(c+0.5, r+0.5, "GOAL", color='white', weight='bold', ha='center', va='center', fontsize=12)
            elif char == 'S': 
                ax.text(c+0.5, r+0.85, "START", color='black', weight='bold', ha='center', va='center', fontsize=8)
            elif char == 'F': 
                ax.text(c+0.5, r+0.5, "❄️", color='#aaccff', ha='center', va='center', fontsize=14, alpha=0.6)

            if is_learned and char != 'H' and char != 'G' and value > 0.01:
                arrow_len = 0.25
                dx, dy = 0, 0
                if best_action == 0: dx = -arrow_len   # Left
                elif best_action == 1: dy = arrow_len    # Down
                elif best_action == 2: dx = arrow_len    # Right
                elif best_action == 3: dy = -arrow_len   # Up
                
                ax.arrow(c + 0.5, r + 0.5, dx, dy, 
                         head_width=0.08, head_length=0.08, fc='black', ec='black', alpha=0.6)

    title_prefix = "Learned Policy" if is_learned else "Initial Environment Map"
    ax.set_title(f"{title_prefix} (Frozen Lake)", fontsize=14, pad=10)
    ax.set_xlabel("Grid Column")
    ax.set_ylabel("Grid Row")
    return fig

def plot_learning_curve(data, title, optimal_line=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data, linewidth=2, color="#1f77b4", label="Agent Performance")
    if optimal_line is not None:
        ax.axhline(y=optimal_line, color='r', linestyle='--', alpha=0.6, label=f"Optimal ({optimal_line})")
        ax.legend()
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Success Rate", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    return fig

def get_action(env, q_table, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        values = q_table[state]
        return np.random.choice(np.flatnonzero(values == values.max()))

def train_agent(env, algorithm, episodes, alpha, gamma, epsilon, render_placeholder=None, eps_schedule=None):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards_history = []
    
    if render_placeholder:
        chart_container = render_placeholder.container()
        status_text = chart_container.empty()
        live_chart = chart_container.empty()
    
    progress_bar = st.progress(0)

    for i in range(episodes):
        # ε schedule (helps sparse-reward FrozenLake)
        if eps_schedule is None:
            epsilon_t = epsilon
        else:
            eps_start = float(eps_schedule.get('start', 1.0))
            eps_min = float(eps_schedule.get('min', 0.05))
            decay = float(eps_schedule.get('decay', 0.995))
            epsilon_t = max(eps_min, eps_start * (decay ** i))

        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = get_action(env, q_table, state, epsilon_t)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            target = reward + gamma * np.max(q_table[next_state])
            q_table[state, action] += alpha * (target - q_table[state, action])
            state = next_state
            total_reward += reward
        
        # Frozen Lake Reward is 0 or 1, so total_reward is 1 if success, 0 if fail
        rewards_history.append(total_reward)
        if (i+1) % (max(1, episodes // 20)) == 0:
            progress_bar.progress((i + 1) / episodes)
            if render_placeholder:
                status_text.write(f"⏳ Training... Episode {i+1}/{episodes}")
                win = 50 if len(rewards_history) > 50 else 1
                df = pd.DataFrame({"Reward (Mov Avg)": rewards_history})
                live_chart.line_chart(df.rolling(win, min_periods=1).mean())
                
    progress_bar.progress(1.0)
    if render_placeholder:
        status_text.success("Training Complete!")
    return q_table, rewards_history

# ==========================================
# 2. Main Experiment Logic
# ==========================================
def run():
    st.header("❄️ Frozen Lake")

    custom_map = ["SFFF", "FHFH", "FFFH", "HFFG"]

    # --- Intro ---
    col_desc, col_map = st.columns([1, 1.5], vertical_alignment="top")

    with col_desc:
        st.subheader("Task")
        st.markdown(
            """
            **Mission:** Cross the lake from **Start (S)** to **Goal (G)** without falling into **Holes (H)**.

            **Dynamics**
            - ❄️ If **Slippery** is enabled, actions may slip to a different direction.
            - 🏆 Reward is **+1 only at the goal** (sparse reward).
            """
        )

    with col_map:
        st.subheader("Environment Map")
        dummy_q = np.zeros((16, 4))
        st.pyplot(plot_heatmap_and_arrows(dummy_q, (4, 4), custom_desc=custom_map), clear_figure=True)

    st.divider()

    # --- Controls ---
    with st.expander("⚙️ Experiment Settings", expanded=True):
        left, right = st.columns([1, 2], vertical_alignment="top")

        with left:
            st.markdown("##### Environment")
            is_slippery = st.checkbox("Enable Slippery Ice", value=True, key="fl_slippery")
            if is_slippery:
                st.caption("Stochastic (Hard) — random transitions + sparse reward.")
            else:
                st.caption("Deterministic (Easy) — much easier to learn.")
            
            st.markdown("##### Exploration")
            use_eps_decay = st.checkbox(
                "Use ε Decay (Recommended for slippery)",
                value=is_slippery,
                key="fl_eps_decay"
            )

            st.markdown("##### Run")
            start_btn = st.button("Train Agent", type="primary", use_container_width=True, key="fl_btn")

        with right:
            st.markdown("##### Hyperparameters")
            c1, c2 = st.columns(2)

            with c1:
                gamma = st.slider("Discount Factor (γ)", 0.1, 0.99, 0.99, key="fl_gamma")
                alpha = st.slider("Learning Rate (α)", 0.01, 1.0, 0.10, key="fl_alpha")

            with c2:
                if use_eps_decay:
                    eps_start = st.slider("ε start", 0.1, 1.0, 1.0, key="fl_eps_start")
                    eps_min = st.slider("ε min", 0.0, 0.2, 0.05, key="fl_eps_min")
                    eps_decay = st.slider("ε decay per episode", 0.90, 0.999, 0.995, key="fl_eps_decay_rate")
                    epsilon = eps_start
                else:
                    epsilon = st.slider("Exploration (ε)", 0.0, 1.0, 0.30 if is_slippery else 0.10, key="fl_eps")
                    eps_start, eps_min, eps_decay = None, None, None

                episodes = st.number_input("Episodes", 100, 50000, 10000 if is_slippery else 2000, key="fl_ep")

            if is_slippery and not use_eps_decay:
                st.caption("Tip: slippery=True is hard with fixed ε. Consider ε decay or increase ε/episodes.")

    # --- Results ---
    vis_container = st.container()

    if start_btn:
        env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=is_slippery)

        eps_schedule = None
        if use_eps_decay:
            eps_schedule = {"start": eps_start, "min": eps_min, "decay": eps_decay}

        with vis_container:
            q_table, rewards = train_agent(
                env, "Q-Learning", episodes, alpha, gamma, epsilon,
                render_placeholder=vis_container,
                eps_schedule=eps_schedule
            )

            st.divider()
            tab1, tab2 = st.tabs(["🗺️ Policy Map", "📈 Training Curve"])

            with tab1:
                st.subheader("Learned Policy")
                st.pyplot(plot_heatmap_and_arrows(q_table, (4, 4), custom_desc=custom_map), clear_figure=True)

            with tab2:
                st.subheader("Success Rate (Smoothed)")
                success_rate = pd.Series(rewards).rolling(window=100, min_periods=1).mean()
                st.pyplot(
                    plot_learning_curve(success_rate, "Success Rate", optimal_line=0.7 if is_slippery else 1.0),
                    clear_figure=True
                )
