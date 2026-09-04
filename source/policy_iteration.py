from manim import *
import numpy as np

class PolicyIterationGeneral(Scene):
    def construct(self):
        # --- 0. 全局配置 (5x5) ---
        config.max_files_cached = 5000 
        self.gamma = 0.9        
        
        # 固定轮次配置: 第一轮 10 步，后续 5 步
        # 增加总轮次，确保策略能传导到起点
        self.pe_steps_schedule = [10, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5] 
        
        self.grid_size = 5      
        self.cell_size = 1.0    
        self.grid_spacing = 1.1 
        
        # 动作定义
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
        self.action_vecs = [UP, DOWN, LEFT, RIGHT]
        
        # --- 1. 播放理论介绍 ---
        self.play_intro()
        
        # --- 2. 播放核心网格演示 ---
        self.play_grid_world()

    # ---------- 新增/修改的 Intro 部分 ----------
    def play_intro(self):
        """
        Policy Iteration Intro (Re-designed to match QLearning style)
        Structure:
        1. Definition (Math)
        2. Environment (Legend)
        3. The Loop (Eval <-> Improve)
        4. Dashboard (Log Error chart)
        5. Schedule (k-steps)
        """
        # 设置节奏参数 (与 QLearning 保持一致)
        self.intro_slow = 3.0
        
        # 顶部标题
        self.title = Text("Policy Iteration", font_size=48, color=BLUE).to_edge(UP)
        env_text = Text("Problem: 5x5 Grid Maze Navigation", font_size=32, color=TEAL).next_to(
            self.title, DOWN, buff=0.3
        )

        self.play(FadeIn(self.title), FadeIn(env_text))
        self.wait(0.3 * self.intro_slow)

        total_pages = 5

        # --- Helpers (内部定义，保持命名空间整洁) ---
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

        # --- Page 1: What is Policy Iteration? ---
        p1_t = Text("1. What is Policy Iteration?", font_size=30, color=YELLOW)
        p1_desc = Text(
            "Model-based Dynamic Programming: Alternates two phases",
            font_size=22,
            color=GREY_B,
        )
        p1_header = VGroup(p1_t, p1_desc).arrange(DOWN, aligned_edge=LEFT, buff=0.12)

        # 公式
        p1_eq1 = MathTex(r"\textbf{1. Eval: } V(s) \leftarrow \sum P(s'|s,\pi)[R+\gamma V(s')]", font_size=32)
        p1_eq2 = MathTex(r"\textbf{2. Improve: } \pi(s) \leftarrow \arg\max_a \sum P(s'|s,a)[R+\gamma V(s')]", font_size=32)
        p1_eqs = VGroup(p1_eq1, p1_eq2).arrange(DOWN, aligned_edge=LEFT, buff=0.25).shift(RIGHT * 0.2)

        p1_bul = VGroup(
            Text("• Requires Environment Model (P, R)", font_size=22, color=WHITE),
            Text("• Policy Evaluation: calculates V for current π", font_size=22, color=WHITE),
            Text("• Policy Improvement: updates π to be greedy wrt V", font_size=22, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)

        p1_group = VGroup(p1_header, p1_eqs, p1_bul).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        show_page(1, p1_group, keep_env=True, hold=1.6)

        # --- Page 2: Environment & Rewards (Same as Q-Learning) ---
        p2_t = Text("2. Environment & Rewards", font_size=30, color=YELLOW)
        def legend_item(color, label, value_text):
            box = Square(
                side_length=0.35, fill_color=color, fill_opacity=0.9, stroke_color=WHITE, stroke_width=1.5
            )
            t1 = Text(label, font_size=24, color=WHITE)
            t2 = Text(value_text, font_size=24, color=GREY_B)
            return VGroup(box, t1, t2).arrange(RIGHT, buff=0.25, aligned_edge=DOWN)

        # 使用 policy_iteration.py 中定义的颜色 (TEAL_E, MAROON_E 等)
        L1 = legend_item(TEAL_E, "Goal", "+1.0 (terminal)")
        L2 = legend_item(MAROON_E, "Trap", "-1.0 (flash only)")
        L3 = legend_item(ORANGE, "Mud", "-0.5")
        L4 = legend_item(DARK_GRAY, "Step", "-0.04 per move")
        
        legend = VGroup(L1, L2, L3, L4).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        
        # Grid visual hint
        hint_text = Text("State Value V(s) shown in each cell", font_size=24, color=BLUE_B)
        
        p2_group = VGroup(p2_t, legend, hint_text).arrange(DOWN, aligned_edge=LEFT, buff=0.38)
        show_page(2, p2_group, keep_env=True, hold=1.6)

        # --- Page 3: The Algorithm Loop ---
        p3_t = Text("3. The Iteration Loop", font_size=30, color=YELLOW)
        
        # 创建循环图示
        eval_box = RoundedRectangle(height=1.2, width=3.0, corner_radius=0.2, color=BLUE)
        eval_txt = VGroup(Text("Evaluation", font_size=24, color=BLUE), Text("Update V(s)", font_size=20, color=GREY_B)).arrange(DOWN)
        eval_grp = VGroup(eval_box, eval_txt)
        
        imp_box = RoundedRectangle(height=1.2, width=3.0, corner_radius=0.2, color=GREEN)
        imp_txt = VGroup(Text("Improvement", font_size=24, color=GREEN), Text("Update π(s)", font_size=20, color=GREY_B)).arrange(DOWN)
        imp_grp = VGroup(imp_box, imp_txt)
        
        loop_grp = VGroup(eval_grp, imp_grp).arrange(RIGHT, buff=2.0)
        
        arrow_top = Arrow(eval_box.get_top(), imp_box.get_top(), path_arc=-1.0, color=WHITE)
        arrow_btm = Arrow(imp_box.get_bottom(), eval_box.get_bottom(), path_arc=-1.0, color=WHITE)
        
        lbl_top = Text("Converged?", font_size=16).next_to(arrow_top, UP)
        lbl_btm = Text("New Policy", font_size=16).next_to(arrow_btm, DOWN)
        
        diagram = VGroup(loop_grp, arrow_top, arrow_btm, lbl_top, lbl_btm)
        
        p3_note = Text("Iterates until policy stops changing (Stable)", font_size=22, color=WHITE)
        
        p3_group = VGroup(p3_t, diagram, p3_note).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        show_page(3, p3_group, keep_env=True, hold=1.6)

        # --- Page 4: Dashboard Logic ---
        p4_t = Text("4. Dashboard: Convergence Check", font_size=30, color=YELLOW)
        
        p4_items = VGroup(
            Text("• Left Grid: Displays V(s) and Policy Arrows (Gold/Green)", font_size=24, color=WHITE),
            Text("• Right Chart: Max Error (Log Scale)", font_size=24, color=TEAL),
            Text("    - Shows how quickly V(s) converges in Eval phase", font_size=20, color=GREY_B),
            Text("    - Y-axis: 10^0 down to 10^-4", font_size=20, color=GREY_B),
            Text("• Phase Indicator: Initialization -> Eval -> Improve", font_size=24, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        
        p4_group = VGroup(p4_t, p4_items).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        show_page(4, p4_group, keep_env=True, hold=1.6)

        # --- Page 5: Schedule & Parameters ---
        p5_t = Text("5. Demo Schedule", font_size=30, color=YELLOW)

        def stage_block(title, subtitle, color_fill):
            box_width = 9.0
            box = RoundedRectangle(
                corner_radius=0.15, height=1.15, width=box_width,
                fill_color=color_fill, fill_opacity=0.25,
                stroke_color=WHITE, stroke_width=2,
            )
            t = Text(title, font_size=24, color=WHITE)
            s = Text(subtitle, font_size=18, color=GREY_B)
            vg = VGroup(t, s).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
            
            # fit width
            if vg.width > box_width - 0.6: vg.scale_to_fit_width(box_width - 0.6)
            vg.move_to(box.get_center()).align_to(box, LEFT).shift(RIGHT * 0.25)
            return VGroup(box, vg)

        st1 = stage_block(
            "Iteration 1 (Slow)",
            "Evaluation runs for fixed k=10 steps • Visualizes error drop",
            BLUE_E,
        )
        st2 = stage_block(
            "Iteration 2+ (Faster)",
            "Evaluation runs for fixed k=5 steps • Policy updates quickly",
            GREEN_E,
        )
        st3 = stage_block(
            "Final Agent Run",
            "Agent follows the optimal policy (Green path) to Goal",
            GOLD_E,
        )
        timeline = VGroup(st1, st2, st3).arrange(DOWN, aligned_edge=LEFT, buff=0.28)

        params_line = VGroup(
            MathTex(rf"\gamma={self.gamma}\quad \text{{Grid}}=5\times5", font_size=30, color=YELLOW),
            Text(f"| Seeds = 42", font_size=22, color=GREY_B),
        ).arrange(RIGHT, buff=0.4)

        p5_group = VGroup(p5_t, timeline, params_line).arrange(DOWN, aligned_edge=LEFT, buff=0.38)
        
        fit_to_intro_area(p5_group, env_text)
        p5_group.next_to(env_text, DOWN, buff=0.65).align_to(env_text, LEFT)

        counter = page_counter(5)
        self.play(FadeIn(p5_group, shift=RIGHT), FadeIn(counter), run_time=0.35 * self.intro_slow)
        self.wait(1.9 * self.intro_slow)
        
        # End intro
        self.play(
            FadeOut(p5_group, shift=LEFT),
            FadeOut(counter),
            FadeOut(env_text),
            self.title.animate.scale(0.8),
            run_time=0.30 * self.intro_slow,
        )
        self.wait(0.15 * self.intro_slow)

    def get_static_color(self, reward):
        if reward == 1.0: return TEAL_E      
        if reward == -1.0: return MAROON_E   
        if reward == -0.5: return ORANGE     
        return DARK_GRAY                     

    def play_grid_world(self):
        # --- A. 环境初始化 ---
        rewards = np.array([
            [-0.04, -0.04, -0.04, -0.50, -1.00], 
            [-0.04, -1.00, -0.04, -1.00, -1.00], 
            [-0.04, -0.50, -0.04, -0.04, -0.04], 
            [-1.00, -1.00, -1.00, -1.00, -0.04], 
            [-1.00, -0.04, -0.04, -0.04,  1.00]  
        ])
        
        values = np.zeros((self.grid_size, self.grid_size))
        
        np.random.seed(42)
        policy = np.random.randint(0, 4, size=(self.grid_size, self.grid_size))
        
        # --- B. 构建网格 ---
        grid_group = VGroup()
        self.value_trackers = {} 
        self.cells = {} 
        self.arrows = {} 
        
        center_offset = (self.grid_size - 1) * self.grid_spacing / 2

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = j * self.grid_spacing - center_offset
                y = (self.grid_size - 1 - i) * self.grid_spacing - center_offset
                pos = np.array([x, y, 0])
                
                fill_color = self.get_static_color(rewards[i, j])
                cell = Square(side_length=self.cell_size, color=WHITE, fill_color=fill_color, fill_opacity=0.8)
                cell.move_to(pos)
                self.cells[(i, j)] = cell 
                
                if rewards[i, j] == 1.0: r_text, label_color = "Goal\n+1.0", YELLOW
                elif rewards[i, j] == -1.0: r_text, label_color = "Trap\n-1.0", WHITE
                elif rewards[i, j] == -0.5: r_text, label_color = "Mud\n-0.5", WHITE
                else: r_text, label_color = "-0.04", WHITE
                
                reward_label = Text(r_text, font_size=9, line_spacing=0.8, color=label_color)
                reward_label.move_to(pos + np.array([0, 0.32, 0]))
                reward_label.set_z_index(10)

                tracker = ValueTracker(values[i, j])
                self.value_trackers[(i, j)] = tracker
                
                # --- 修改处：保留 3 位小数，字号调小至 14 ---
                val_num = DecimalNumber(
                    values[i, j], num_decimal_places=3, show_ellipsis=False,
                    font_size=14, color=WHITE
                ).move_to(pos + np.array([0, -0.2, 0]))
                
                val_num.add_updater(lambda m, r=i, c=j: m.set_value(self.value_trackers[(r, c)].get_value()))
                val_num.set_z_index(10)
                
                grid_group.add(cell, reward_label, val_num)
                
                if rewards[i, j] != 1.0:
                    action_idx = policy[i, j]
                    vec = self.action_vecs[action_idx]
                    arrow = Arrow(
                        start=pos, end=pos + vec * 0.4, buff=0, 
                        color=GOLD, stroke_width=3, 
                        max_tip_length_to_length_ratio=0.4
                    )
                    arrow.set_opacity(0.6) 
                    self.arrows[(i, j)] = arrow
                    grid_group.add(arrow)

        grid_group.move_to(ORIGIN)
        self.play(Create(grid_group))
        self.play(FadeOut(self.title))
        self.play(grid_group.animate.scale(0.8).move_to(LEFT * 3.5))
        
        # --- C. 右侧信息面板 ---
        info_group = VGroup()
        phase_title = Text("Phase: Initialization", font_size=24, color=WHITE).to_edge(RIGHT, buff=1.0).shift(UP*2.5)
        iter_text = Text("Policy Iteration: 0", font_size=20).next_to(phase_title, DOWN, buff=0.3)
        status_text = Text("Ready", font_size=20, color=YELLOW).next_to(iter_text, DOWN, buff=0.3)
        
        # --- Log Scale Axis Construction ---
        y_height = 2.5
        x_width = 3.5
        
        # 轴线
        x_axis = Line(ORIGIN, RIGHT * x_width, color=BLUE)
        y_axis = Line(ORIGIN, UP * y_height, color=BLUE)
        axes_group = VGroup(x_axis, y_axis).next_to(status_text, DOWN, buff=0.8).shift(LEFT * 0.5)
        
        # X轴刻度 (0, 5, 10)
        x_labels = VGroup()
        for i in range(0, 11, 2):
            tick = Line(UP*0.1, DOWN*0.1, color=BLUE).move_to(axes_group[0].get_start() + RIGHT * (i/10) * x_width)
            label = Text(str(i), font_size=12).next_to(tick, DOWN, buff=0.1)
            x_labels.add(tick, label)
            
        # Y轴刻度 (Log Scale: 10^0, 10^-1, 10^-2, 10^-3, 10^-4)
        y_labels = VGroup()
        log_vals = [0, -1, -2, -3, -4]
        labels_tex = ["10^0", "10^{-1}", "10^{-2}", "10^{-3}", "10^{-4}"]
        
        for i, val in enumerate(log_vals):
            # Normalize -4~0 to 0~1
            norm_y = (val - (-4)) / 4
            pos = axes_group[1].get_start() + UP * norm_y * y_height
            tick = Line(LEFT*0.1, RIGHT*0.1, color=BLUE).move_to(pos)
            label = MathTex(labels_tex[i], font_size=16).next_to(tick, LEFT, buff=0.1)
            y_labels.add(tick, label)
            
        # 轴标签
        x_label_text = Text("Step", font_size=16).next_to(x_axis, DOWN, buff=0.4)
        y_label_text = Text("Max Error (Log)", font_size=16, color=WHITE).rotate(90*DEGREES).next_to(y_axis, LEFT, buff=0.6)
        
        # k 值文字
        k_text = Text("Fixed Steps: k=--", color=RED, font_size=20).next_to(x_axis, DOWN, buff=0.6)

        plot_group = VGroup(axes_group, x_labels, y_labels, x_label_text, y_label_text, k_text)
        info_group.add(phase_title, iter_text, status_text, plot_group)
        
        self.play(Write(info_group))
        
        # --- D. 主循环 ---
        policy_stable = False
        outer_iter = 0
        
        # 增加最大迭代次数，确保 Agent 能找到路
        while not policy_stable and outer_iter < 15:
            current_target_steps = self.pe_steps_schedule[min(outer_iter, len(self.pe_steps_schedule)-1)]
            outer_iter += 1
            
            # 更新 k 值文字
            new_k_text = Text(f"Fixed Steps: k={current_target_steps}", color=RED, font_size=20).move_to(k_text)
            
            self.play(
                iter_text.animate.become(Text(f"Policy Iteration: {outer_iter}", font_size=20).move_to(iter_text)),
                Transform(k_text, new_k_text),
                run_time=0.2
            )
            
            # --- Phase 1: Policy Evaluation ---
            self.play(
                phase_title.animate.become(Text("Phase: Policy Evaluation", font_size=24, color=YELLOW).move_to(phase_title)),
                status_text.animate.become(Text(f"Running for k={current_target_steps} steps", font_size=20, color=YELLOW).move_to(status_text)),
                *[a.animate.set_opacity(0.3) for a in self.arrows.values()], 
                run_time=0.5
            )
            
            plot_dots = VGroup()
            last_dot = None
            
            for pe_step in range(1, current_target_steps + 1):
                max_diff = 0
                new_values = values.copy()
                anims = []
                
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        if rewards[i, j] == 1.0: continue 

                        action_idx = policy[i, j]
                        di, dj = self.actions[action_idx]
                        ni, nj = i + di, j + dj
                        
                        if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                            r_next = rewards[ni, nj]
                            v_next = values[ni, nj]
                        else:
                            r_next = rewards[i, j] 
                            v_next = values[i, j]
                            
                        val = r_next + self.gamma * v_next
                        new_values[i, j] = val
                        
                        diff = abs(val - values[i, j])
                        if diff > max_diff:
                            max_diff = diff
                            
                        if diff > 0.0001:
                            anims.append(self.value_trackers[(i, j)].animate.set_value(val))
                
                values = new_values
                
                # --- 图表打点 (Log Scale) ---
                plot_err = max(max_diff, 1e-4)
                plot_err = min(plot_err, 1.0)
                
                log_err = np.log10(plot_err) # -4 to 0
                norm_y = (log_err - (-4)) / 4
                
                # X轴映射: step 1->0, step 10->width
                x_pos = (pe_step / 10) * x_width
                y_pos = norm_y * y_height
                
                dot_pos = axes_group[0].get_start() + RIGHT * x_pos + UP * y_pos
                new_dot = Dot(dot_pos, color=YELLOW, radius=0.06)
                plot_dots.add(new_dot)
                
                connect_line = None
                if last_dot:
                    connect_line = Line(last_dot.get_center(), new_dot.get_center(), color=YELLOW, stroke_width=2)
                    plot_dots.add(connect_line)
                last_dot = new_dot

                chart_anims = [Create(new_dot)]
                if connect_line: chart_anims.append(Create(connect_line))
                
                if anims:
                    self.play(
                        AnimationGroup(*chart_anims), 
                        LaggedStart(*anims, lag_ratio=0.01), 
                        run_time=0.4
                    )
                else:
                    self.play(AnimationGroup(*chart_anims), run_time=0.4)

            # 变绿
            if last_dot:
                self.play(last_dot.animate.set_color(GREEN).scale(1.5), run_time=0.3)
                self.play(last_dot.animate.scale(1/1.5), run_time=0.1)

            self.play(FadeOut(plot_dots))

            # --- Phase 2: Policy Improvement ---
            self.play(
                phase_title.animate.become(Text("Phase: Policy Improvement", font_size=24, color=GREEN).move_to(phase_title)),
                status_text.animate.become(Text("Greedy Update", font_size=20, color=GREEN).move_to(status_text)),
                *[a.animate.set_opacity(1.0) for a in self.arrows.values()],
                run_time=0.5
            )
            
            policy_stable = True
            change_anims = []
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if rewards[i, j] == 1.0: continue
                    
                    old_action = policy[i, j]
                    
                    q_values = []
                    for idx, (di, dj) in enumerate(self.actions):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                            r_next = rewards[ni, nj]
                            v_next = values[ni, nj]
                        else:
                            r_next = rewards[i, j]
                            v_next = values[i, j]
                        
                        q = r_next + self.gamma * v_next
                        q_values.append(q)
                    
                    best_action = np.argmax(q_values)
                    
                    if best_action != old_action:
                        policy_stable = False
                        policy[i, j] = best_action
                        
                        arrow = self.arrows[(i, j)]
                        new_vec = self.action_vecs[best_action]
                        center = self.cells[(i, j)].get_center()
                        
                        new_arrow = Arrow(
                            start=center, end=center + new_vec * 0.4, buff=0, 
                            color=GREEN, stroke_width=3, 
                            max_tip_length_to_length_ratio=0.4
                        )
                        change_anims.append(Transform(arrow, new_arrow))
            
            if change_anims:
                self.play(LaggedStart(*change_anims, lag_ratio=0.05), run_time=1.5)
            else:
                self.play(Flash(status_text, color=GREEN, run_time=0.8))
                
        # --- E. 结束与 Agent 演示 ---
        final_text = Text("Converged! Running Agent...", color=GREEN, font_size=24).next_to(status_text, DOWN)
        self.play(Write(final_text))
        
        self.simulate_agent(policy)

    def simulate_agent(self, policy):
        start_pos = (0, 0)
        curr_r, curr_c = start_pos
        
        start_center = self.cells[start_pos].get_center()
        
        # Z-index 300 确保在最顶层
        agent = Dot(color=BLUE, radius=0.25).move_to(start_center).set_z_index(300)
        path = TracedPath(agent.get_center, stroke_color=BLUE_A, stroke_width=4, stroke_opacity=0.8).set_z_index(299)
        
        self.add(path)
        self.play(FadeIn(agent))
        self.wait(0.5)
        
        steps = 0
        while steps < 20: # 增加步数上限以防万一
            action_idx = policy[curr_r, curr_c]
            di, dj = self.actions[action_idx]
            nr, nc = curr_r + di, curr_c + dj
            
            # 碰撞检测
            hit_wall = False
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                target_pos = self.cells[(nr, nc)].get_center()
                
                # 执行移动
                self.play(agent.animate.move_to(target_pos), run_time=0.6, rate_func=smooth)
                curr_r, curr_c = nr, nc
                
                if (curr_r, curr_c) == (4, 4):
                    self.play(Flash(agent, color=YELLOW, line_length=0.5))
                    break
            else:
                # 撞墙可视化：变红并轻微抖动
                hit_wall = True
                
            if hit_wall:
                self.play(Indicate(agent, color=RED, scale_factor=1.2), run_time=0.5)
            
            steps += 1
        
        self.wait(2)