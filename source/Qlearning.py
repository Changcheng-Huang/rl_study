from manim import *
import numpy as np
import random


class QLearningDemo(Scene):
    def construct(self):
        config.max_files_cached = 5000

        # ✅ 4. 换种子：已修改为 42 (原为 7)
        self.seed = 42
        np.random.seed(self.seed)
        random.seed(self.seed)

        # 参数
        self.alpha = 0.5
        self.gamma = 0.9
        self.epsilon = 0.3
        self.max_epsilon = 0.3

        # Intro 页间切换 & 停留整体慢 3 倍
        self.intro_slow = 3.0

        self.grid_size = 5
        self.cell_size = 1.0
        self.grid_spacing = 1.1

        # 动作索引：0=UP,1=DOWN,2=LEFT,3=RIGHT
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_vecs = [UP, DOWN, LEFT, RIGHT]
        
        # 用于记录屏幕上显示的箭头方向
        self.arrow_directions = {}

        # 初始方向设为右
        self.initial_action_idx = 3

        self.visible_arrows = set()

        # 图表 state
        self.last_steps_point = None
        self.last_return_point = None
        self.steps_hist = []
        self.returns_hist = []

        # Intro
        self.play_intro()

        # Main
        self.play_q_learning()

    # ---------- helpers ----------
    def moving_avg(self, values, window=5):
        n = len(values)
        if n == 0:
            return 0.0
        k = min(window, n)
        return float(np.mean(values[-k:]))

    # 贪心动作：优先跟随屏幕上的箭头
    def best_action_det(self, r, c):
        q = self.q_table[r, c]
        max_val = np.max(q)
        best = np.flatnonzero(q == max_val)
        
        saved_act = self.arrow_directions.get((r, c), None)
        
        if saved_act is not None and saved_act in best:
            return int(saved_act)
            
        return int(np.random.choice(best))

    # 用于“显示策略箭头”的贪心选择
    def greedy_for_display(self, qvals, prefer_action_idx=None):
        max_val = float(np.max(qvals))
        best = np.flatnonzero(qvals == max_val)
        if len(best) == 1:
            return int(best[0])
        if prefer_action_idx is not None and prefer_action_idx in best:
            return int(prefer_action_idx)
        return int(np.random.choice(best))

    def get_timing(self, episode):
        if episode == 1:
            return dict(compass=0.14, move=0.34, q=0.22, hit=0.26, flash=0.22, reset=0.10)
        else:
            return dict(compass=0.07, move=0.16, q=0.11, hit=0.14, flash=0.12, reset=0.06)

    def get_step_result(self, curr_pos, action_idx, rewards):
        r, c = curr_pos
        di, dj = self.actions[action_idx]
        next_r, next_c = r + di, c + dj
        hit_wall = False

        if 0 <= next_r < self.grid_size and 0 <= next_c < self.grid_size:
            reward = rewards[next_r, next_c]
            next_pos = (next_r, next_c)
        else:
            reward = -0.1
            next_pos = curr_pos
            hit_wall = True

        return next_pos, reward, hit_wall

    # ---------- Intro (multi-page) ----------
    def play_intro(self):
        """Q-learning Intro"""
        self.title = Text("Q-Learning", font_size=48, color=BLUE).to_edge(UP)
        env_text = Text("Problem: 5x5 Grid Maze Navigation", font_size=32, color=TEAL).next_to(
            self.title, DOWN, buff=0.3
        )

        self.play(FadeIn(self.title), FadeIn(env_text))
        self.wait(0.3 * self.intro_slow)

        total_pages = 5

        def page_counter(n):
            return Text(f"{n}/{total_pages}", font_size=18, color=GREY_B).to_corner(DR).shift(
                UP * 0.35 + LEFT * 0.35
            )

        def fit_to_intro_area(mob, top_anchor, side_margin=0.7, bottom_margin=0.55, top_buff=0.55):
            max_w = config.frame_width - 2 * side_margin
            top_y = top_anchor.get_bottom()[1] - top_buff
            bottom_y = -config.frame_height / 2 + bottom_margin
            max_h = top_y - bottom_y

            if mob.width > max_w:
                mob.scale_to_fit_width(max_w)
            if mob.height > max_h:
                mob.scale_to_fit_height(max_h)

        def show_page(n, group, keep_env=True, hold=1.4):
            fit_to_intro_area(group, env_text)
            group.next_to(env_text, DOWN, buff=0.65).align_to(env_text, LEFT)

            counter = page_counter(n)
            fade_in_rt = 0.35 * self.intro_slow
            fade_out_rt = 0.30 * self.intro_slow

            self.play(FadeIn(group, shift=RIGHT), FadeIn(counter), run_time=fade_in_rt)
            self.wait(hold * self.intro_slow)
            if keep_env:
                self.play(FadeOut(group, shift=LEFT), FadeOut(counter), run_time=fade_out_rt)
            else:
                self.play(FadeOut(group, shift=LEFT), FadeOut(counter), FadeOut(env_text), run_time=fade_out_rt)

        # --- Page 1 ---
        p1_t = Text("1. What is Q-Learning?", font_size=30, color=YELLOW)
        p1_desc = Text(
            "Model-free TD Control: learn Q(s,a) from experience",
            font_size=22,
            color=GREY_B,
        )
        p1_header = VGroup(p1_t, p1_desc).arrange(DOWN, aligned_edge=LEFT, buff=0.12)

        # ✅ 1. 公式修复：单行显示，限制宽度
        p1_eq = MathTex(
            r"Q(s,a)\leftarrow Q(s,a)+\alpha\Big[r+\gamma\max_{a'}Q(s',a') - Q(s,a)\Big]",
            font_size=36, 
        )
        # 限制最大宽度为屏幕宽度减去一些边距，防止太宽
        if p1_eq.width > config.frame_width - 1.5:
             p1_eq.scale_to_fit_width(config.frame_width - 1.5)

        p1_bul = VGroup(
            Text("• No environment model needed (model-free)", font_size=22, color=WHITE),
            Text("• Update after every step: (s, a, r, s')", font_size=22, color=WHITE),
            Text("• Policy (greedy):  π(s) = argmax_a Q(s,a)", font_size=22, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)

        p1_group = VGroup(p1_header, p1_eq, p1_bul).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        p1_group.next_to(env_text, DOWN, buff=0.65).align_to(env_text, LEFT)

        show_page(1, p1_group, keep_env=True, hold=1.6)

        # --- Page 2 ---
        p2_t = Text("2. Environment & Rewards", font_size=30, color=YELLOW)
        def legend_item(color, label, value_text):
            box = Square(
                side_length=0.35, fill_color=color, fill_opacity=0.9, stroke_color=WHITE, stroke_width=1.5
            )
            t1 = Text(label, font_size=24, color=WHITE)
            t2 = Text(value_text, font_size=24, color=GREY_B)
            return VGroup(box, t1, t2).arrange(RIGHT, buff=0.25, aligned_edge=DOWN)

        L1 = legend_item(TEAL_E, "Goal", "+1.0 (terminal)")
        L2 = legend_item(MAROON_E, "Trap", "-1.0 (flash only)")
        L3 = legend_item(ORANGE, "Mud", "-0.5")
        L4 = legend_item(DARK_GRAY, "Step", "-0.04 per move")
        wall = Square(side_length=0.35, fill_color=BLACK, fill_opacity=0.0, stroke_color=RED, stroke_width=3)
        L5 = VGroup(
            wall,
            Text("Wall hit", font_size=24, color=WHITE),
            Text("-0.1 penalty (stay)", font_size=24, color=GREY_B),
        ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN)

        legend = VGroup(L1, L2, L3, L4, L5).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        start_dot = Dot(color=BLUE, radius=0.10)
        start_text = Text("Start at top-left (0,0)", font_size=24, color=WHITE)
        start_row = VGroup(start_dot, start_text).arrange(RIGHT, buff=0.25)
        p2_group = VGroup(p2_t, legend, start_row).arrange(DOWN, aligned_edge=LEFT, buff=0.38)
        show_page(2, p2_group, keep_env=True, hold=1.6)

        # --- Page 3 ---
        p3_t = Text("3. Action Selection (ε-greedy)", font_size=30, color=YELLOW)
        p3_eq = MathTex(
            r"a=\begin{cases}\text{random action} & \text{w.p. }\epsilon\\"
            r"\arg\max_a Q(s,a) & \text{w.p. }1-\epsilon\end{cases}",
            font_size=34,
        ).shift(RIGHT * 0.25)
        p3_note = VGroup(
            Text("• Q-table starts at 0 → all actions tie", font_size=24, color=WHITE),
            Text("• Tie-break is uniform random → initial movement is unbiased", font_size=24, color=WHITE),
            Text("• Purple agent = exploring (random action)", font_size=24, color=PURPLE),
            Text("• Blue agent = exploiting (greedy action)", font_size=24, color=BLUE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        demo_dots = VGroup(
            VGroup(Dot(color=PURPLE, radius=0.14), Text("Explore", font_size=22, color=PURPLE)).arrange(RIGHT, buff=0.2),
            VGroup(Dot(color=BLUE, radius=0.14), Text("Exploit", font_size=22, color=BLUE)).arrange(RIGHT, buff=0.2),
            VGroup(Dot(color=GREEN, radius=0.14), Text("Final demo", font_size=22, color=GREEN)).arrange(RIGHT, buff=0.2),
        ).arrange(RIGHT, buff=0.8)
        p3_group = VGroup(p3_t, p3_eq, p3_note, demo_dots).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        show_page(3, p3_group, keep_env=True, hold=1.7)

        # --- Page 4 ---
        p4_t = Text("4. What you will see on screen", font_size=30, color=YELLOW)
        p4_items = VGroup(
            Text("• Gold arrows in grid = greedy policy  argmax_a Q(s,a)", font_size=24, color=GOLD),
            Text("  (When exploring, the taken action may differ)", font_size=22, color=GREY_B),
            Text("• Number in each cell = max_a Q(s,a) (green/red = + / - )", font_size=24, color=WHITE),
            Text("• Right panel: Explore%, Episode, Steps, Action compass", font_size=24, color=WHITE),
            Text("• Charts: moving average of Steps & Return (first 24 episodes shown)", font_size=24, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        p4_group = VGroup(p4_t, p4_items).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        show_page(4, p4_group, keep_env=True, hold=1.6)

        # --- Page 5 ---
        p5_t = Text("5. Training Schedule", font_size=30, color=YELLOW)

        def stage_block(title, subtitle, color_fill):
            # ✅ 2. 文本框修复：宽度减小到 9.0 (原 11.5)，包裹更紧凑
            box_width = 9.0
            box = RoundedRectangle(
                corner_radius=0.15,
                height=1.15,
                width=box_width,
                fill_color=color_fill,
                fill_opacity=0.25,
                stroke_color=WHITE,
                stroke_width=2,
            )
            t = Text(title, font_size=24, color=WHITE)
            s = Text(subtitle, font_size=18, color=GREY_B)

            vg = VGroup(t, s).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
            
            # 适配宽度
            max_text_width = box_width - 0.6
            if vg.width > max_text_width:
                vg.scale_to_fit_width(max_text_width)
            
            vg.move_to(box.get_center()).align_to(box, LEFT).shift(RIGHT * 0.25)
            return VGroup(box, vg)

        st1 = stage_block(
            "Episodes 1–5  (visual, slow)",
            "up to 30 steps per episode • ε starts 0.30 and decays ×0.85",
            BLUE_E,
        )
        st2 = stage_block(
            "Episodes 6–24  (backend, fast)",
            "up to 50 steps per episode • ε fixed at 0.05 • charts update quickly",
            YELLOW_E,
        )
        st3 = stage_block(
            "Episode 25  (final demo)",
            "pure greedy run (green agent) • up to 12 steps",
            GREEN_E,
        )
        timeline = VGroup(st1, st2, st3).arrange(DOWN, aligned_edge=LEFT, buff=0.28)

        params_line = VGroup(
            MathTex(
                rf"\alpha={self.alpha:.1f}\quad \gamma={self.gamma:.1f}\quad \epsilon_0={self.max_epsilon:.2f}",
                font_size=30,
                color=YELLOW,
            ),
            Text(f"| Seeds = {self.seed}", font_size=22, color=GREY_B),
        ).arrange(RIGHT, buff=0.4)

        p5_group = VGroup(p5_t, timeline, params_line).arrange(DOWN, aligned_edge=LEFT, buff=0.38)
        
        fit_to_intro_area(p5_group, env_text)
        p5_group.next_to(env_text, DOWN, buff=0.65).align_to(env_text, LEFT)

        counter = page_counter(5)
        self.play(FadeIn(p5_group, shift=RIGHT), FadeIn(counter), run_time=0.35 * self.intro_slow)
        self.wait(1.9 * self.intro_slow)
        self.play(
            FadeOut(p5_group, shift=LEFT),
            FadeOut(counter),
            FadeOut(env_text),
            self.title.animate.scale(0.8),
            run_time=0.30 * self.intro_slow,
        )
        self.wait(0.15 * self.intro_slow)

    # ---------- dashboard ----------
    def create_dashboard(self):
        panel_bg = RoundedRectangle(
            corner_radius=0.2,
            height=6.4,
            width=6.0,
            fill_color=BLACK,
            fill_opacity=1.0,
            stroke_color=BLUE_E,
        )
        panel_bg.to_edge(RIGHT, buff=0.2).shift(DOWN * 0.4)

        bg_top = panel_bg.get_top()
        bg_center = panel_bg.get_center()
        bg_left = panel_bg.get_left()
        bg_right = panel_bg.get_right()
        bg_bottom = panel_bg.get_bottom()

        data_row_y = bg_top[1] - 0.6

        self.ep_tracker = ValueTracker(0)
        self.step_tracker = ValueTracker(0)
        self.eps_tracker = ValueTracker(self.epsilon)

        self.eps_label = Text("Explore:", font_size=14, color=PURPLE)
        self.eps_label.move_to(np.array([bg_left[0] + 1.02, data_row_y, 0]))
        self.eps_val = always_redraw(
            lambda: Text(f"{self.eps_tracker.get_value()*100:.0f}%", font_size=16, color=PURPLE).next_to(
                self.eps_label, RIGHT, buff=0.12
            )
        )

        self.ep_label = Text("Ep:", font_size=18, color=GREY_B)
        self.ep_label.move_to(np.array([bg_center[0] - 0.4, data_row_y, 0]))
        self.ep_val = always_redraw(
            lambda: Integer(int(self.ep_tracker.get_value()), font_size=24, color=WHITE).next_to(
                self.ep_label, RIGHT, buff=0.15
            )
        )

        self.step_label = Text("Steps:", font_size=18, color=GREY_B)
        self.step_label.move_to(np.array([bg_right[0] - 2.0, data_row_y, 0]))
        self.step_val = always_redraw(
            lambda: Integer(int(self.step_tracker.get_value()), font_size=24, color=WHITE).next_to(
                self.step_label, RIGHT, buff=0.15
            )
        )

        sep_line = Line(bg_left + RIGHT * 0.2, bg_right + LEFT * 0.2, color=GREY_E)
        sep_line.move_to(np.array([bg_center[0], data_row_y - 0.5, 0]))

        compass_label = Text("Action", font_size=16, color=YELLOW)
        compass_label.move_to(np.array([bg_center[0], data_row_y - 0.95, 0]))

        self.compass_arrows = VGroup()
        offsets = [UP, DOWN, LEFT, RIGHT]
        compass_center = np.array([bg_center[0], data_row_y - 1.85, 0])

        for i in range(4):
            arrow = Arrow(
                start=ORIGIN,
                end=offsets[i] * 0.6,
                color=DARK_GRAY,
                stroke_width=6,
                buff=0,
                max_tip_length_to_length_ratio=0.4,
            )
            if i == 0:
                arrow.move_to(compass_center + UP * 0.4)
            if i == 1:
                arrow.move_to(compass_center + DOWN * 0.4)
            if i == 2:
                arrow.move_to(compass_center + LEFT * 0.4)
            if i == 3:
                arrow.move_to(compass_center + RIGHT * 0.4)
            arrow.save_state()
            self.compass_arrows.add(arrow)

        chart_bottom_y = bg_bottom[1] + 0.70

        self.steps_ymin, self.steps_ymax = 0.0, 30.0
        self.steps_axes = Axes(
            x_range=[0, 24, 4],
            y_range=[self.steps_ymin, self.steps_ymax, 5],
            x_length=4.8,
            y_length=1.15,
            axis_config={"color": GREY, "stroke_width": 2, "include_tip": False},
            x_axis_config={"font_size": 11, "include_numbers": False},
            y_axis_config={"font_size": 11, "include_numbers": True},
        )

        self.ret_ymin, self.ret_ymax = -12.0, 2.0
        self.ret_axes = Axes(
            x_range=[0, 24, 4],
            y_range=[self.ret_ymin, self.ret_ymax, 2],
            x_length=4.8,
            y_length=1.05,
            axis_config={"color": GREY, "stroke_width": 2, "include_tip": False},
            x_axis_config={"font_size": 11, "include_numbers": True},
            y_axis_config={"font_size": 11, "include_numbers": True},
        )

        self.ret_axes.move_to(np.array([bg_center[0], chart_bottom_y + 0.45, 0]))
        self.steps_axes.next_to(self.ret_axes, UP, buff=0.35).align_to(self.ret_axes, LEFT)

        steps_title = Text("Steps", font_size=12, color=YELLOW).next_to(self.steps_axes, UP, buff=0.06).align_to(
            self.steps_axes, LEFT
        )
        ret_title = Text("Return", font_size=12, color=GREEN_B).next_to(self.ret_axes, UP, buff=0.06).align_to(
            self.ret_axes, LEFT
        )

        x_label = Text("Ep", font_size=12, color=WHITE).next_to(self.ret_axes.x_axis, UP, buff=0.05)
        y1_label = Text("S", font_size=12, color=YELLOW).next_to(self.steps_axes.y_axis, RIGHT, buff=0.05)
        y2_label = Text("R", font_size=12, color=GREEN_B).next_to(self.ret_axes.y_axis, RIGHT, buff=0.05)

        opt = self.optimal_steps
        y = float(opt)
        p0 = self.steps_axes.c2p(0, y)
        p1 = self.steps_axes.c2p(24, y)
        self.opt_line = DashedLine(p0, p1, dash_length=0.12, color=GREEN, stroke_width=2).set_z_index(1001)
        self.opt_label = Text(f"Optimal = {opt}", font_size=10, color=GREEN).next_to(
            self.opt_line, UP, buff=0.04
        ).align_to(self.opt_line, RIGHT)

        self.dashboard_group = VGroup(
            panel_bg,
            self.eps_label,
            self.eps_val,
            self.ep_label,
            self.ep_val,
            self.step_label,
            self.step_val,
            sep_line,
            compass_label,
            self.compass_arrows,
            self.steps_axes,
            steps_title,
            y1_label,
            self.ret_axes,
            ret_title,
            x_label,
            y2_label,
            self.opt_line,
            self.opt_label,
        )
        self.dashboard_group.set_z_index(1000)
        self.play(FadeIn(self.dashboard_group, shift=LEFT))

    def update_compass(self, action_idx, is_exploring):
        active_color = PURPLE if is_exploring else BLUE
        anims = []
        for i in range(4):
            if i == action_idx:
                anims.append(self.compass_arrows[i].animate.restore().set_color(active_color).scale(1.2))
            else:
                anims.append(self.compass_arrows[i].animate.restore())
        return anims

    def update_epsilon_percent(self, animate=True, run_time=0.25):
        if animate:
            self.play(self.eps_tracker.animate.set_value(self.epsilon), run_time=run_time)
        else:
            self.eps_tracker.set_value(self.epsilon)

    def update_charts(self, episode, steps, ep_return, window=5, animate=True):
        if episode > 24:
            return

        self.steps_hist.append(steps)
        self.returns_hist.append(ep_return)

        steps_ma = self.moving_avg(self.steps_hist, window=window)
        ret_ma = self.moving_avg(self.returns_hist, window=window)

        steps_plot = float(np.clip(steps_ma, self.steps_ymin, self.steps_ymax))
        ret_plot = float(np.clip(ret_ma, self.ret_ymin, self.ret_ymax))

        p_steps = self.steps_axes.c2p(episode, steps_plot)
        dot_steps = Dot(p_steps, radius=0.05, color=YELLOW).set_z_index(1002)
        line_steps = None
        if self.last_steps_point is not None:
            line_steps = Line(self.last_steps_point, p_steps, stroke_width=2, color=YELLOW).set_z_index(1002)
        self.last_steps_point = p_steps

        p_ret = self.ret_axes.c2p(episode, ret_plot)
        dot_ret = Dot(p_ret, radius=0.05, color=GREEN_B).set_z_index(1002)
        line_ret = None
        if self.last_return_point is not None:
            line_ret = Line(self.last_return_point, p_ret, stroke_width=2, color=GREEN_B).set_z_index(1002)
        self.last_return_point = p_ret

        if animate:
            anims = []
            if line_steps:
                anims.append(Create(line_steps))
            anims.append(Create(dot_steps))
            if line_ret:
                anims.append(Create(line_ret))
            anims.append(Create(dot_ret))
            self.play(*anims, run_time=0.28)
        else:
            if line_steps:
                self.add(line_steps)
            self.add(dot_steps)
            if line_ret:
                self.add(line_ret)
            self.add(dot_ret)

    # ---------- 全局刷新（策略=贪心；Goal 不画箭头） ----------
    def refresh_policy_arrows_and_values(self):
        anims = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                qvals = self.q_table[i, j]
                max_val = float(np.max(qvals))

                val_color = GREEN if max_val > 0.01 else (RED if max_val < -0.01 else WHITE)
                anims.append(self.q_val_displays[(i, j)].animate.set_value(max_val).set_color(val_color))

                arrow = self.arrows.get((i, j), None)
                if arrow is None:
                    continue

                best_act = self.greedy_for_display(qvals, prefer_action_idx=None)
                
                # 记录最后一次显示的箭头方向
                self.arrow_directions[(i, j)] = best_act
                
                new_vec = self.action_vecs[best_act]
                target = Arrow(
                    start=self.cells[(i, j)].get_center(),
                    end=self.cells[(i, j)].get_center() + new_vec * 0.4,
                    buff=0,
                    color=GOLD,
                    stroke_width=4,
                    max_tip_length_to_length_ratio=0.4,
                )

                if (i, j) not in self.visible_arrows:
                    arrow.become(target)
                    self.visible_arrows.add((i, j))
                    anims.append(FadeIn(arrow))
                else:
                    anims.append(Transform(arrow, target))

        if anims:
            self.play(LaggedStart(*anims, lag_ratio=0.003), run_time=1.2)

    # ---------- main ----------
    def play_q_learning(self):
        rewards = np.array(
            [
                [-0.04, -0.04, -0.04, -0.50, -1.00],
                [-0.04, -1.00, -0.04, -1.00, -1.00],
                [-0.04, -0.50, -0.04, -0.04, -0.04],
                [-1.00, -1.00, -1.00, -1.00, -0.04],
                [-1.00, -0.04, -0.04, -0.04, 1.00],
            ]
        )

        goal_idx = np.argwhere(np.isclose(rewards, 1.0))
        self.goal_pos = tuple(goal_idx[0]) if len(goal_idx) else (self.grid_size - 1, self.grid_size - 1)

        self.optimal_steps = 8
        self.q_table = np.zeros((self.grid_size, self.grid_size, 4))

        # --- 网格 ---
        grid_group = VGroup()
        self.cells = {}
        self.arrows = {}
        self.q_val_displays = {}
        center_offset = (self.grid_size - 1) * self.grid_spacing / 2

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pos = np.array(
                    [
                        j * self.grid_spacing - center_offset,
                        (self.grid_size - 1 - i) * self.grid_spacing - center_offset,
                        0,
                    ]
                )

                if np.isclose(rewards[i, j], 1.0):
                    fill_col = TEAL_E
                elif np.isclose(rewards[i, j], -1.0):
                    fill_col = MAROON_E
                elif np.isclose(rewards[i, j], -0.5):
                    fill_col = ORANGE
                else:
                    fill_col = DARK_GRAY

                cell = Square(
                    side_length=self.cell_size, color=WHITE, fill_color=fill_col, fill_opacity=0.8
                ).move_to(pos)
                self.cells[(i, j)] = cell

                txt = ""
                if np.isclose(rewards[i, j], 1.0):
                    txt = "+1"
                elif np.isclose(rewards[i, j], -1.0):
                    txt = "-1"
                elif np.isclose(rewards[i, j], -0.5):
                    txt = "-0.5"
                if txt:
                    grid_group.add(Text(txt, font_size=16, color=WHITE).move_to(pos + UR * 0.3))

                q_num = DecimalNumber(0.00, num_decimal_places=2, font_size=14, color=WHITE).move_to(
                    pos + DOWN * 0.25
                )
                self.q_val_displays[(i, j)] = q_num

                if (i, j) == self.goal_pos:
                    self.arrows[(i, j)] = None
                    grid_group.add(cell, q_num)
                else:
                    init_vec = self.action_vecs[self.initial_action_idx]
                    arrow = Arrow(
                        start=pos,
                        end=pos + init_vec * 0.4,
                        buff=0,
                        color=GOLD,
                        stroke_width=4,
                        max_tip_length_to_length_ratio=0.4,
                    ).set_opacity(0)
                    self.arrows[(i, j)] = arrow
                    grid_group.add(cell, q_num, arrow)

        grid_group.move_to(ORIGIN)
        self.play(Create(grid_group))
        self.play(grid_group.animate.scale(0.85).move_to(LEFT * 4.0))

        # --- 仪表盘 ---
        self.create_dashboard()
        self.update_epsilon_percent(animate=False)

        # --- Agent ---
        agent = Dot(color=BLUE, radius=0.2).move_to(self.cells[(0, 0)].get_center()).set_z_index(100)
        self.play(FadeIn(agent))

        path = TracedPath(agent.get_center, stroke_color=BLUE_A, stroke_width=2, dissipating_time=1.0)
        self.add(path)

        visual_episodes = 5
        total_episodes = 25
        max_steps_visual = 30
        max_steps_backend = 50
        max_steps_demo = 12

        # --- 可视化训练：ep 1-5 ---
        for episode in range(1, visual_episodes + 1):
            T = self.get_timing(episode)
            self.play(self.ep_tracker.animate.set_value(episode), run_time=0.25)

            curr_pos = (0, 0)
            agent.move_to(self.cells[curr_pos].get_center())

            path.clear_points()
            path.start_new_path(agent.get_center())

            steps = 0
            ep_return = 0.0
            self.step_tracker.set_value(0)

            done = False
            while not done and steps < max_steps_visual:
                r, c = curr_pos

                greedy_idx_before = self.best_action_det(r, c)

                # ε-greedy
                if np.random.rand() < self.epsilon:
                    action_idx = np.random.randint(4)
                    is_exploring = True
                    agent.set_color(PURPLE)
                else:
                    action_idx = greedy_idx_before
                    is_exploring = False
                    agent.set_color(BLUE)

                next_pos, reward, hit_wall = self.get_step_result(curr_pos, action_idx, rewards)

                self.play(*self.update_compass(action_idx, is_exploring), run_time=T["compass"])

                if hit_wall:
                    self.play(Indicate(agent, color=RED, scale_factor=1.2), run_time=T["hit"])
                else:
                    self.play(agent.animate.move_to(self.cells[next_pos].get_center()), run_time=T["move"])

                if np.isclose(reward, -1.0):
                    self.play(Flash(agent, color=RED, flash_radius=0.5, line_length=0.2), run_time=T["flash"])

                old_q = self.q_table[r, c, action_idx]
                max_next_q = np.max(self.q_table[next_pos[0], next_pos[1]])

                if np.isclose(rewards[next_pos[0], next_pos[1]], 1.0) and next_pos != curr_pos:
                    max_next_q = 0
                    done = True

                new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
                self.q_table[r, c, action_idx] = new_q

                ep_return += float(reward)
                steps += 1
                self.step_tracker.set_value(steps)

                # 更新该格的“策略箭头”
                qvals = self.q_table[r, c]
                max_q_val = float(np.max(qvals))
                best_act_idx = self.greedy_for_display(qvals, prefer_action_idx=action_idx)

                # 记录箭头方向
                self.arrow_directions[(r, c)] = best_act_idx

                val_color = GREEN if max_q_val > 0.01 else (RED if max_q_val < -0.01 else WHITE)

                arrow = self.arrows.get((r, c), None)

                if arrow is None:
                    self.play(
                        self.q_val_displays[(r, c)].animate.set_value(max_q_val).set_color(val_color),
                        run_time=T["q"],
                    )
                else:
                    new_vec = self.action_vecs[best_act_idx]
                    target_arrow = Arrow(
                        start=self.cells[(r, c)].get_center(),
                        end=self.cells[(r, c)].get_center() + new_vec * 0.4,
                        buff=0,
                        color=GOLD,
                        stroke_width=4,
                        max_tip_length_to_length_ratio=0.4,
                    )

                    if (r, c) not in self.visible_arrows:
                        arrow.become(target_arrow)
                        self.visible_arrows.add((r, c))
                        self.play(
                            self.q_val_displays[(r, c)].animate.set_value(max_q_val).set_color(val_color),
                            FadeIn(arrow),
                            run_time=T["q"],
                        )
                    else:
                        self.play(
                            self.q_val_displays[(r, c)].animate.set_value(max_q_val).set_color(val_color),
                            Transform(arrow, target_arrow),
                            run_time=T["q"],
                        )

                curr_pos = next_pos
                if done:
                    self.play(Flash(agent, color=TEAL, flash_radius=0.5, line_length=0.5), run_time=T["flash"])

                self.play(*[a.animate.restore() for a in self.compass_arrows], run_time=T["reset"])

            self.update_charts(episode, steps, ep_return, window=5, animate=True)

            self.epsilon = max(0.05, self.epsilon * 0.85)
            self.update_epsilon_percent(animate=True, run_time=0.18)

        # --- 后台训练：ep 6-24 ---
        # ✅ 3. 文字被挡修复：强制将 "Training rapidly..." 移到网格上方的左侧，避开右侧仪表盘
        fast_text = Text("Training rapidly...", font_size=24, color=YELLOW).move_to(LEFT * 3.0 + UP * 2.8)
        self.play(FadeIn(fast_text))

        self.epsilon = 0.05
        self.update_epsilon_percent(animate=True, run_time=0.2)

        for episode in range(visual_episodes + 1, total_episodes):  # 6..24
            self.ep_tracker.set_value(episode)

            curr_pos = (0, 0)
            steps = 0
            ep_return = 0.0
            done = False

            while not done and steps < max_steps_backend:
                r, c = curr_pos

                greedy_idx = self.best_action_det(r, c)
                if np.random.rand() < self.epsilon:
                    action_idx = np.random.randint(4)
                else:
                    action_idx = greedy_idx

                next_pos, reward, hit = self.get_step_result(curr_pos, action_idx, rewards)

                old_q = self.q_table[r, c, action_idx]
                max_next_q = np.max(self.q_table[next_pos[0], next_pos[1]])

                if np.isclose(rewards[next_pos[0], next_pos[1]], 1.0) and next_pos != curr_pos:
                    max_next_q = 0
                    done = True

                self.q_table[r, c, action_idx] = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)

                curr_pos = next_pos
                steps += 1
                ep_return += float(reward)

            self.step_tracker.set_value(steps)
            self.update_charts(episode, steps, ep_return, window=5, animate=False)
            self.wait(0.04)

        self.play(FadeOut(fast_text))

        # 全局刷新（策略箭头）
        self.refresh_policy_arrows_and_values()

        # --- 最终演示：Ep=25，最多 12 步 ---
        self.ep_tracker.set_value(total_episodes)
        self.step_tracker.set_value(0)

        self.remove(path)
        agent.move_to(self.cells[(0, 0)].get_center()).set_color(GREEN)

        new_path = TracedPath(agent.get_center, stroke_color=GREEN, stroke_width=4, dissipating_time=None).set_z_index(99)
        self.add(new_path)

        curr_pos = (0, 0)
        done = False
        steps = 0
        ep_return = 0.0

        while not done and steps < max_steps_demo:
            r, c = curr_pos
            action_idx = self.best_action_det(r, c)
            next_pos, reward, hit = self.get_step_result(curr_pos, action_idx, rewards)

            self.play(*self.update_compass(action_idx, False), run_time=0.10)

            if hit:
                self.play(Indicate(agent, color=RED), run_time=0.25)
            else:
                self.play(agent.animate.move_to(self.cells[next_pos].get_center()), run_time=0.35)
                curr_pos = next_pos

                if np.isclose(reward, 1.0):
                    self.play(Flash(agent, color=TEAL), run_time=0.45)
                    done = True
                elif np.isclose(reward, -1.0):
                    self.play(Flash(agent, color=RED, flash_radius=0.5, line_length=0.2), run_time=0.25)

            steps += 1
            ep_return += float(reward)
            self.step_tracker.set_value(steps)

            self.play(*[a.animate.restore() for a in self.compass_arrows], run_time=0.08)

        self.wait(2)