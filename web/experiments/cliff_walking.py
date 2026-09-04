import streamlit as st
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ==========================================
# 1. Visualization & Training Utils (Local)
# ==========================================
def plot_heatmap_and_arrows(q_table, shape):
    """
    Cliff Walking 专用绘图函数
    [Color]: Inverted (Dark=Low Value, Light=High Value)
    """
    state_values = np.max(q_table, axis=1).reshape(shape)
    rows, cols = shape
    is_learned = not np.all(q_table == 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 修改1：cmap="YlGnBu_r" (反转颜色：值越小越深蓝，值越大越浅黄)
    kwargs = {
        'vmin': -100, 'vmax': -10, 'cmap': "YlGnBu_r",
        'linewidths': 1, 'linecolor': '#eeeeee'
    }
    
    sns.heatmap(state_values, annot=False, cbar=True, ax=ax, alpha=0.9,
                cbar_kws={'label': 'Estimated State Value (V)'}, **kwargs)
    
    for r in range(rows):
        for c in range(cols):
            s = r * cols + c
            best_action = np.argmax(q_table[s])
            value = state_values[r, c]
            
            # 修改2：文字颜色适配
            # 因为高分(-13)现在是浅色背景，所以用黑字
            # 低分(-100)是深色背景，所以用白字
            if is_learned:
                text_color = 'black' if value > -20 else 'white'
                ax.text(c + 0.05, r + 0.15, f"{value:.0f}", 
                        color=text_color, fontsize=7, ha='left', va='center')

            # 地图元素绘制
            if r == 3 and c == 11: 
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color='#66cc66', alpha=0.8))
                ax.text(c+0.5, r+0.5, "GOAL", color='white', weight='bold', ha='center', va='center', fontsize=10)
            elif r == 3 and 0 < c < 11: 
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color='#ff4444', alpha=0.6))
                ax.text(c+0.5, r+0.5, "CLIFF", color='white', ha='center', va='center', fontsize=7)
            elif r == 3 and c == 0: 
                ax.text(c+0.5, r+0.8, "START", color='black', weight='bold', ha='center', va='center', fontsize=8)

            # 箭头绘制
            if is_learned and not (r == 3 and c == 11):
                arrow_len = 0.25
                dx, dy = 0, 0
                if best_action == 0: dy = -arrow_len    # Up
                elif best_action == 1: dx = arrow_len   # Right
                elif best_action == 2: dy = arrow_len   # Down
                elif best_action == 3: dx = -arrow_len  # Left
                
                ax.arrow(c + 0.5, r + 0.5, dx, dy, 
                         head_width=0.08, head_length=0.08, fc='black', ec='black', alpha=0.6)

    title_prefix = "Learned Policy" if is_learned else "Initial Environment Map"
    ax.set_title(f"{title_prefix} (Cliff Walking)", fontsize=14, pad=10)
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
    ax.set_ylabel("Total Reward", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    return fig

def get_action(env, q_table, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        values = q_table[state]
        return np.random.choice(np.flatnonzero(values == values.max()))

def train_agent(env, algorithm, episodes, alpha, gamma, epsilon, render_placeholder=None):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards_history = []
    
    if render_placeholder:
        chart_container = render_placeholder.container()
        status_text = chart_container.empty()
        live_chart = chart_container.empty()
    
    progress_bar = st.progress(0)

    for i in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        action = get_action(env, q_table, state, epsilon)
        
        while not done:
            if algorithm == "Q-Learning":
                action = get_action(env, q_table, state, epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                target = reward + gamma * np.max(q_table[next_state])
                q_table[state, action] += alpha * (target - q_table[state, action])
                state = next_state
            elif algorithm == "SARSA":
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_action = get_action(env, q_table, next_state, epsilon)
                target = reward + gamma * q_table[next_state, next_action]
                q_table[state, action] += alpha * (target - q_table[state, action])
                state = next_state
                action = next_action 
            total_reward += reward
        
        rewards_history.append(total_reward)
        if (i+1) % (max(1, episodes // 20)) == 0:
            progress_bar.progress((i + 1) / episodes)
            if render_placeholder:
                status_text.write(f"⏳ Training {algorithm}... Episode {i+1}/{episodes}")
                win = 50 if len(rewards_history) > 50 else 1
                df = pd.DataFrame({"Reward (Mov Avg)": rewards_history})
                live_chart.line_chart(df.rolling(win, min_periods=1).mean())
                
    progress_bar.progress(1.0)
    if render_placeholder:
        status_text.success(f"{algorithm} Training Complete!")
    return q_table, rewards_history

# ==========================================
# 2. Main Experiment Logic
# ==========================================
def run():
    st.header("🏔️Cliff Walking")

    # --- Intro ---
    col_desc, col_map = st.columns([1, 1.5], vertical_alignment="top")

    with col_desc:
        st.subheader("Task")
        st.markdown(
            """
            **Objective:** Go from **START** to **GOAL** as fast as possible.

            **Rewards**
            - 🦶 Step: **-1**
            - ☠️ Cliff: **-100** (falls reset you to START)

            **What to watch**
            - **Q-Learning:** often learns the shortest (risky) edge path.
            - **SARSA:** often learns a slightly longer but safer path.
            """
        )

    with col_map:
        st.subheader("Environment Map")
        dummy_q = np.zeros((48, 4))
        st.pyplot(plot_heatmap_and_arrows(dummy_q, (4, 12)), clear_figure=True)

    st.divider()

    # --- Controls ---
    with st.expander("⚙️ Experiment Settings", expanded=True):
        left, right = st.columns([1, 2], vertical_alignment="top")

        with left:
            st.markdown("##### Agent")
            algorithm = st.radio("Algorithm", ["Q-Learning", "SARSA"], key="cw_algo", horizontal=False)

            st.markdown("##### Run")
            start_btn = st.button("Train Agent", type="primary", use_container_width=True, key="cw_btn")

        with right:
            st.markdown("##### Hyperparameters")
            c1, c2 = st.columns(2)

            with c1:
                gamma = st.slider("Discount Factor (γ)", 0.1, 0.99, 0.90, key="cw_gamma")
                alpha = st.slider("Learning Rate (α)", 0.01, 1.0, 0.50, key="cw_alpha")

            with c2:
                epsilon = st.slider("Exploration (ε)", 0.0, 0.5, 0.10, key="cw_eps")
                episodes = st.number_input("Episodes", 10, 5000, 1000, key="cw_ep")

            st.caption("Tip: keep ε around 0.1 to highlight the risk difference between Q-Learning and SARSA.")

    # --- Results ---
    vis_container = st.container()

    if start_btn:
        env = gym.make("CliffWalking-v1")

        with vis_container:
            q_table, rewards = train_agent(
                env, algorithm, episodes, alpha, gamma, epsilon, render_placeholder=vis_container
            )

            st.divider()
            tab1, tab2 = st.tabs(["🗺️ Policy Map", "📈 Training Curve"])

            with tab1:
                st.subheader(f"Learned Policy ({algorithm})")
                if algorithm == "Q-Learning":
                    st.info("Typical behavior: shortest route near the cliff edge (high risk under exploration).")
                else:
                    st.info("Typical behavior: safer route a bit away from the cliff (more robust under exploration).")
                st.pyplot(plot_heatmap_and_arrows(q_table, (4, 12)), clear_figure=True)

            with tab2:
                st.subheader("Episode Return (Smoothed)")
                smooth_rewards = pd.Series(rewards).rolling(window=20, min_periods=1).mean()
                st.pyplot(
                    plot_learning_curve(smooth_rewards, f"{algorithm} Learning Curve", optimal_line=-13),
                    clear_figure=True
                )
