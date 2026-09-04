from manim import *
import numpy as np


class ValueIterationGeneral(Scene):
    def construct(self):
        # --- 0. 全局配置 (5x5) ---
        config.max_files_cached = 5000 
        self.gamma = 0.9        # 折扣因子
        self.grid_size = 5      # 5x5 网格
        self.cell_size = 1.0    # 格子尺寸
        self.grid_spacing = 1.1 # 格子间距
        
        # --- 1. 播放理论介绍 ---
        self.play_intro()
        
        # --- 2. 播放核心网格演示 ---
        self.play_grid_world()

    # ---------- 新增/修改的 Intro 部分 (Value Iteration) ----------
    def play_intro(self):
        """
        Value Iteration Intro (Re-designed to match QLearning/PI style)
        Structure:
        1. Definition (Bellman Optimality)
        2. Environment (Legend)
        3. The Loop (Iterative Update)
        4. Dashboard (Max Error & Threshold)
        5. Schedule (Slow/Fast phases)
        """
        # 设置节奏参数 (与 QLearning 保持一致)
        self.intro_slow = 3.0
        
        # 顶部标题
        self.title = Text("Value Iteration", font_size=48, color=BLUE).to_edge(UP)
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

        # --- Page 1: What is Value Iteration? ---
        p1_t = Text("1. What is Value Iteration?", font_size=30, color=YELLOW)
        p1_desc = Text(
            "Model-based Dynamic Programming: Iteratively update V(s)",
            font_size=22,
            color=GREY_B,
        )
        p1_header = VGroup(p1_t, p1_desc).arrange(DOWN, aligned_edge=LEFT, buff=0.12)

        # 公式: Bellman Optimality Equation
        p1_eq = MathTex(
            r"V_{k+1}(s) \leftarrow \max_{a} \sum P(s'|s,a)[R + \gamma V_k(s')]", 
            font_size=34
        ).shift(RIGHT * 0.2)

        p1_bul = VGroup(
            Text("• Computes optimal Value V*(s) directly", font_size=22, color=WHITE),
            Text("• Combines Policy Eval & Improvement into one step", font_size=22, color=WHITE),
            Text("• Optimal Policy π* is derived greedily from V*", font_size=22, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)

        p1_group = VGroup(p1_header, p1_eq, p1_bul).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        show_page(1, p1_group, keep_env=True, hold=1.6)

        # --- Page 2: Environment & Rewards ---
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
        
        legend = VGroup(L1, L2, L3, L4).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        
        hint_text = Text("Numbers in grid = Estimate of V(s)", font_size=24, color=BLUE_B)
        
        p2_group = VGroup(p2_t, legend, hint_text).arrange(DOWN, aligned_edge=LEFT, buff=0.38)
        show_page(2, p2_group, keep_env=True, hold=1.6)

        # --- Page 3: The Algorithm Loop ---
        p3_t = Text("3. The Iteration Loop", font_size=30, color=YELLOW)
        
        # Diagram: V_k -> Bellman Update -> V_k+1
        box_k = RoundedRectangle(height=1.0, width=2.5, corner_radius=0.2, color=BLUE)
        txt_k = Text("Values V_k", font_size=20)
        grp_k = VGroup(box_k, txt_k)

        arrow = Arrow(LEFT, RIGHT, color=WHITE)
        label_update = Text("Bellman Update\n(Max over actions)", font_size=16, color=YELLOW).next_to(arrow, UP)

        box_k1 = RoundedRectangle(height=1.0, width=2.5, corner_radius=0.2, color=GREEN)
        txt_k1 = Text("Values V_k+1", font_size=20)
        grp_k1 = VGroup(box_k1, txt_k1)

        diagram = VGroup(grp_k, arrow, grp_k1).arrange(RIGHT, buff=0.5)
        # Re-position label after arrange
        label_update.next_to(arrow, UP)
        diagram.add(label_update)

        p3_bul = VGroup(
            Text("• Sweeps through all states every iteration", font_size=22, color=WHITE),
            Text("• Values propagate backwards from Goal", font_size=22, color=WHITE),
            Text("• Repeat until change < Threshold", font_size=22, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)

        p3_group = VGroup(p3_t, diagram, p3_bul).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        show_page(3, p3_group, keep_env=True, hold=1.6)

        # --- Page 4: Dashboard Logic ---
        p4_t = Text("4. Dashboard: Convergence Check", font_size=30, color=YELLOW)
        
        p4_items = VGroup(
            Text("• Right Chart: Max Error (V_new - V_old)", font_size=24, color=TEAL),
            Text("    - Measures stability of the value function", font_size=20, color=GREY_B),
            Text("    - Red Dashed Line: Threshold = 0.1", font_size=20, color=RED),
            Text("• Iteration Counter: Tracks number of sweeps", font_size=24, color=WHITE),
            Text("• Convergence: Stops when Error < Threshold", font_size=24, color=GREEN),
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
            "Iterations 1-10 (Slow)",
            "Visualizing the update spread • See values change",
            BLUE_E,
        )
        st2 = stage_block(
            "Iterations 11-40 (Fast)",
            "Rapid updates to reach convergence threshold",
            YELLOW_E,
        )
        st3 = stage_block(
            "Optimal Policy Extraction",
            "Derive policy from V* and run Agent to Goal",
            GREEN_E,
        )
        timeline = VGroup(st1, st2, st3).arrange(DOWN, aligned_edge=LEFT, buff=0.28)

        params_line = VGroup(
            MathTex(rf"\gamma={self.gamma}\quad \epsilon=0.1", font_size=30, color=YELLOW),
            Text(f"| Grid = 5x5", font_size=22, color=GREY_B),
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
        """根据地形类型返回固定颜色"""
        if reward == 1.0: return TEAL_E      # Goal
        if reward == -1.0: return MAROON_E   # Trap
        if reward == -0.5: return ORANGE     # Mud
        return DARK_GRAY                     # Normal path

    def play_grid_world(self):
        """主演示流程"""
        
        # --- A. 环境初始化 (不对称 & 唯一路径设计) ---
        # 视觉布局：Row 0 是最上方，Row 4 是最下方
        # Goal (4,4) 右下角, Start (0,0) 左上角
        
        # 只有一条蜿蜒的最优路径，其他路被 Trap 封死或被 Mud 减分
        rewards = np.array([
            # Col 0,   1,     2,     3,     4
            [-0.04, -0.04, -0.04, -0.50, -1.00], # Row 0 (Top) Start at (0,0)
            [-0.04, -1.00, -0.04, -1.00, -1.00], # Row 1 (Trap Wall)
            [-0.04, -0.50, -0.04, -0.04, -0.04], # Row 2 (Mud in middle)
            [-1.00, -1.00, -1.00, -1.00, -0.04], # Row 3 (Trap Wall)
            [-1.00, -0.04, -0.04, -0.04,  1.00]  # Row 4 (Bottom) Goal at (4,4)
        ])
        
        # 对应矩阵索引
        goal_pos_idx = (4, 4) 
        start_pos_idx = (0, 0)
        
        # 演示计算的目标格子：Goal 上方的格子 (3, 4)
        # Row 3, Col 4. 它是 Normal (-0.04)，下方是 Goal。
        demo_target_pos = (3, 4) 

        values = np.zeros((self.grid_size, self.grid_size))
        
        # --- B. 构建初始网格 ---
        grid_group = VGroup()
        self.value_trackers = {} 
        self.cells = {} 
        
        # 计算中心偏移量
        center_offset_x = (self.grid_size - 1) * self.grid_spacing / 2
        center_offset_y = (self.grid_size - 1) * self.grid_spacing / 2

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # 坐标映射修改：i=0(Top) -> Positive Y, i=4(Bottom) -> Negative Y
                x = j * self.grid_spacing - center_offset_x
                y = (self.grid_size - 1 - i) * self.grid_spacing - center_offset_y
                pos = np.array([x, y, 0])
                
                fill_color = self.get_static_color(rewards[i, j])
                
                cell = Square(side_length=self.cell_size, color=WHITE, fill_color=fill_color, fill_opacity=0.8)
                cell.move_to(pos)
                self.cells[(i, j)] = cell 
                
                # Label 优化：解决看不清和拥挤问题
                if rewards[i, j] == 1.0: 
                    r_text = "Goal\n+1.0"
                    label_color = YELLOW
                elif rewards[i, j] == -1.0: 
                    r_text = "Trap\n-1.0"
                    label_color = WHITE # 红色背景用白色字
                elif rewards[i, j] == -0.5: 
                    r_text = "Mud\n-0.5"
                    label_color = WHITE # 橙色背景用白色字
                else: 
                    # --- 2. 修改显示内容和颜色 ---
                    r_text = "-0.04"
                    label_color = WHITE 
                
                # 缩小字体 (9)，稍微上调位置 (0.32) 以防拥挤
                reward_label = Text(r_text, font_size=9, line_spacing=0.8, color=label_color)
                reward_label.move_to(pos + np.array([0, 0.32, 0]))

                # 价值数字
                tracker = ValueTracker(values[i, j])
                self.value_trackers[(i, j)] = tracker
                
                val_num = DecimalNumber(
                    values[i, j], 
                    num_decimal_places=2, 
                    show_ellipsis=False,
                    font_size=16, 
                    color=WHITE
                )
                val_num.move_to(pos + np.array([0, -0.2, 0]))
                
                val_num.add_updater(lambda m, r=i, c=j: m.set_value(self.value_trackers[(r, c)].get_value()))
                
                grid_group.add(cell, reward_label, val_num)

        grid_group.move_to(ORIGIN)
        self.play(Create(grid_group))
        
        # --- C. 微观演示 (One-Step Lookahead) ---
        self.play(FadeOut(self.title)) 
        
        # 演示 Goal 上方的格子
        self.visualize_one_step(grid_group, values, rewards, target_pos=demo_target_pos)
        
        # --- D. 布局转换 ---
        self.play(
            grid_group.animate.scale(0.8).move_to(LEFT * 3.5)
        )
        
        # 设置坐标轴
        ax = Axes(
            x_range=[0, 40, 10],       
            y_range=[0, 1.2, 0.2],    
            x_length=4.0,              
            y_length=3.0,
            axis_config={"color": BLUE, "include_numbers": True, "font_size": 18},
            tips=False
        ).move_to(RIGHT * 3.2)
        
        epsilon = 0.1
        threshold_line = DashedLine(
            start=ax.c2p(0, epsilon), 
            end=ax.c2p(40, epsilon), 
            color=RED
        )
        threshold_label = Text(f"Threshold: {epsilon}", font_size=16, color=RED).next_to(threshold_line, UP, buff=0.05).set_x(ax.c2p(20,0)[0])

        x_lbl = ax.get_x_axis_label("Iteration").scale(0.5)
        y_lbl = ax.get_y_axis_label("Max Error").scale(0.5).rotate(90*DEGREES).next_to(ax, LEFT, buff=0.1)

        plot_group = VGroup(ax, x_lbl, y_lbl, threshold_line, threshold_label)
        
        # --- 修改1: 右侧坐标系与文字全部一次性渲染 ---
        iter_label = Text("Iteration: 0", font_size=24).next_to(ax, UP, buff=0.5).shift(LEFT * 1.0)
        error_label = Text("Max Error: N/A", font_size=24, color=YELLOW).next_to(iter_label, RIGHT, buff=0.5)
        
        # 将所有UI元素打包，使用 FadeIn 一次性展示 (替代 Create/Write)
        all_plot_ui = VGroup(plot_group, iter_label, error_label)
        self.play(FadeIn(all_plot_ui))
        
        # --- E. 宏观演示: 价值迭代主循环 ---
        max_iterations = 40 
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
        last_dot = None 
        converged = False

        for k in range(max_iterations):
            new_values = values.copy()
            anim_entries = [] 
            
            if k < 10:                 
                current_run_time = 1.5 
                current_lag = 0.08      
            else:
                current_run_time = 0.2
                current_lag = 0.0
            
            self.play(Transform(iter_label, Text(f"Iteration: {k+1}", font_size=24).move_to(iter_label)), run_time=0.1)
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    
                    q_values = []
                    
                    for di, dj in actions:
                        ni, nj = i + di, j + dj
                        # Infinite Horizon 逻辑
                        if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                            r_next = rewards[ni, nj]
                            v_next = values[ni, nj] 
                        else:
                            r_next = rewards[i, j]
                            v_next = values[i, j]
                        
                        q = r_next + self.gamma * v_next
                        q_values.append(q)
                    
                    new_values[i, j] = max(q_values)
                    
                    change_amount = abs(new_values[i, j] - values[i, j])
                    
                    if change_amount > 0.001:
                        # 距离计算：现在目标在 (4,4)
                        dist_to_goal = abs(i - goal_pos_idx[0]) + abs(j - goal_pos_idx[1])
                        val_anim = self.value_trackers[(i, j)].animate.set_value(new_values[i, j])
                        anim_entries.append((dist_to_goal, val_anim))
            
            anim_entries.sort(key=lambda x: x[0])
            anim_data = [x[1] for x in anim_entries]

            diff_matrix = np.abs(new_values - values)
            max_delta = np.max(diff_matrix)
            
            new_error_label = Text(f"Max Error: {max_delta:.4f}", font_size=24, color=YELLOW).move_to(error_label)
            if max_delta < epsilon:
                new_error_label.set_color(GREEN)
            
            self.play(Transform(error_label, new_error_label), run_time=0.1)
            
            plot_y = max_delta if max_delta <= 1.2 else 1.2
            current_point = ax.c2p(k + 1, plot_y)
            new_dot = Dot(current_point, color=YELLOW, radius=0.06)
            
            if anim_data:
                self.play(
                    LaggedStart(*anim_data, lag_ratio=current_lag),
                    Create(new_dot),
                    run_time=current_run_time
                )
            else:
                self.play(Create(new_dot), run_time=current_run_time)
            
            if last_dot:
                self.add(Line(last_dot.get_center(), new_dot.get_center(), color=YELLOW, stroke_width=2))
            
            values = new_values
            last_dot = new_dot
            
            if max_delta < epsilon:
                converged = True
                break

        if converged:
            converged_label = Text(f"Converged!", color=GREEN, font_size=24) 
        else:
            converged_label = Text("Max Iter Reached", color=ORANGE, font_size=24)
        
        plot_ur = ax.c2p(40, 1.2)
        converged_label.move_to(plot_ur + np.array([-1.5, -0.5, 0]))

        self.play(Write(converged_label))
        self.wait(1)

        # --- G. 替换机器人寻路为：全局最优策略展示 (Arrows) ---
        for mobj in grid_group:
            if isinstance(mobj, DecimalNumber):
                mobj.clear_updaters()

        # 调用新方法展示每个格子的箭头
        self.show_optimal_policy(values, rewards)
        
        final_text = Text("Optimal Policy: Visualized by Arrows", color=GREEN, font_size=32).to_edge(DOWN)
        self.play(FadeIn(final_text))
        self.wait(1)

        # --- NEW H. 增加 Agent 寻路演示 ---
        self.play(FadeOut(final_text))
        self.simulate_agent(values, rewards)

    def show_optimal_policy(self, values, rewards):
        """在每个格子上绘制箭头，指示最优策略方向。若有多个最优方向，则绘制多个箭头。"""
        arrows_group = VGroup()
        
        # 动作对应向量: Up, Down, Left, Right
        # 这里的向量是 Manim 屏幕坐标系的向量
        directions = [UP, DOWN, LEFT, RIGHT]
        # 对应的矩阵索引变化 (di, dj)
        # i是行(Row), j是列(Col)。Row 0是Top, Row 4是Bottom。
        # 所以 UP 是 Row-1, DOWN 是 Row+1
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # 如果是 Goal，不画箭头，或者画一个圆点表示 Stay
                if rewards[i, j] == 1.0:
                    dot = Dot(color=YELLOW).move_to(self.cells[(i, j)].get_center())
                    arrows_group.add(dot)
                    continue
                
                q_values = []
                for di, dj in actions:
                    ni, nj = i + di, j + dj
                    
                    # 边界检查
                    if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                        r_next = rewards[ni, nj]
                        v_next = values[ni, nj]
                    else:
                        r_next = rewards[i, j]
                        v_next = values[i, j]
                        
                    q = r_next + self.gamma * v_next
                    q_values.append(q)
                
                max_q = max(q_values)
                
                # 找出所有等于 max_q 的方向（允许微小误差）
                best_indices = [idx for idx, q in enumerate(q_values) if abs(q - max_q) < 0.0001]
                
                # 在格子中心绘制箭头
                center = self.cells[(i, j)].get_center()
                
                for idx in best_indices:
                    vec = directions[idx]
                    # 箭头从中心出发，指向对应方向
                    # 使用较短的箭头以放在格子内
                    arrow = Arrow(
                        start=center, 
                        end=center + vec * 0.4, 
                        buff=0, 
                        color=YELLOW, 
                        max_tip_length_to_length_ratio=0.4,
                        stroke_width=3
                    )
                    arrows_group.add(arrow)
                    
        self.play(LaggedStart(Create(arrows_group), lag_ratio=0.02, run_time=3))

    def create_zoom_cell(self, reward, value, type_str, size):
        """创建一个放大的格子用于展示"""
        color = self.get_static_color(reward)
        cell = Square(side_length=size, color=WHITE, fill_color=color, fill_opacity=0.9)
        
        r_text = f"{reward}"
        if reward == 1.0: r_text = "+1.0"
        
        txt = Text(r_text, font_size=24, color=LIGHT_GREY).move_to(cell.get_center() + UP * 0.5)
        val = DecimalNumber(value, num_decimal_places=2, font_size=32, color=WHITE).move_to(cell.get_center() + DOWN * 0.2)
        
        return VGroup(cell, txt, val)

    def visualize_one_step(self, grid_group, values, rewards, target_pos):
        """可视化单步前瞻 (Updated for R(s') + gamma*V(s'))"""
        r, c = target_pos
        
        # --- 0. 高亮目标格子 ---
        target_cell_map = self.cells[(r, c)]
        
        selection_box = Square(side_length=self.cell_size, color=YELLOW, stroke_width=8, fill_opacity=0)
        selection_box.move_to(target_cell_map.get_center())
        
        selection_label = Text("Update Target", color=YELLOW, font_size=24).next_to(selection_box, UP, buff=0.1)
        
        self.play(
            Create(selection_box),
            Write(selection_label)
        )
        self.wait(0.5)
        self.play(FadeOut(selection_label))
        
        # 1. 移动主 Grid
        self.play(
            grid_group.animate.move_to(LEFT * 4.0).set_opacity(0.15),
            selection_box.animate.shift(LEFT * 4.0)
        )
        
        # 2. 构建右侧放大版视图
        zoom_group = VGroup()
        zoom_cell_size = 1.5 
        zoom_spacing_h = 3.0 
        zoom_spacing_v = 2.6 
        center_pos = RIGHT * 2.5 
        
        target_val = values[r,c]
        target_reward = rewards[r,c]
        center_cell_grp = self.create_zoom_cell(target_reward, target_val, "Target", zoom_cell_size)
        center_cell_grp.move_to(center_pos)
        
        center_val_mob = center_cell_grp[2]
        
        highlight = Square(side_length=zoom_cell_size, color=YELLOW, stroke_width=8, fill_opacity=0)
        highlight.move_to(center_pos)
        
        zoom_group.add(center_cell_grp, highlight)
        
        # 绘制邻居
        directions = [UP, DOWN, LEFT, RIGHT]
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
        
        for idx, (di, dj) in enumerate(actions):
            ni, nj = r + di, c + dj
            direction_vec = directions[idx]
            
            if idx < 2: pos = center_pos + direction_vec * zoom_spacing_v
            else: pos = center_pos + direction_vec * zoom_spacing_h
            
            if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                n_reward = rewards[ni,nj]
                n_val = values[ni,nj]
            else:
                n_reward = rewards[r,c]
                n_val = values[r,c]
            
            nb_cell = self.create_zoom_cell(n_reward, n_val, "Nb", zoom_cell_size)
            nb_cell.move_to(pos)
            zoom_group.add(nb_cell)

        # 标题更新
        title_text = Text("Calculating Q = R(next) + 0.9 * V(next)", color=YELLOW, font_size=32).next_to(zoom_group, UP, buff=0.2).shift(UP*0.5)
        
        self.play(FadeIn(zoom_group), Write(title_text))
        
        # 3. 演示计算过程
        arrows = VGroup()
        result_labels = VGroup() 
        q_vals = []
        
        for idx, (di, dj) in enumerate(actions):
            ni, nj = r + di, c + dj
            direction_vec = directions[idx]
            
            if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                r_immediate = rewards[ni, nj] # R of entering neighbor
                v_neighbor = values[ni, nj]
            else:
                r_immediate = rewards[r, c] # Wall -> R of entering self
                v_neighbor = values[r, c]
            
            q_val = r_immediate + self.gamma * v_neighbor
            q_vals.append(q_val)
            
            start = center_pos + direction_vec * (zoom_cell_size / 2)
            if idx < 2: spacing = zoom_spacing_v
            else: spacing = zoom_spacing_h
            end = center_pos + direction_vec * (spacing - zoom_cell_size / 2)
            
            arrow = Arrow(start, end, buff=0.1, color=BLUE_B, stroke_width=6, max_tip_length_to_length_ratio=0.2)
            arrows.add(arrow)
            
            formula_text = MathTex(
                f"{r_immediate} + 0.9({v_neighbor:.0f})", 
                font_size=24, 
                color=WHITE
            )
            if r_immediate > 0: formula_text.set_color(GREEN)
            elif r_immediate < 0: formula_text.set_color(RED)
            
            result_text = MathTex(f"= {q_val:.2f}", font_size=28, color=YELLOW)
            
            label_content = VGroup(formula_text, result_text).arrange(DOWN, buff=0.1)
            
            label_bg = BackgroundRectangle(label_content, color=BLACK, fill_opacity=0.9, buff=0.1)
            label_grp = VGroup(label_bg, label_content)
            
            midpoint = (start + end) / 2
            label_grp.move_to(midpoint)
            
            result_labels.add(label_grp)

        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.2))
        
        self.play(FadeIn(result_labels))
        self.wait(1)
        
        max_q = max(q_vals)
        max_idx = q_vals.index(max_q)
        
        target_grp = result_labels[max_idx]
        
        max_label = Text("max", color=RED, font_size=24).next_to(target_grp, UP if max_idx >= 2 else RIGHT, buff=0.1)
        
        self.play(
            target_grp.animate.scale(1.2).set_stroke(GREEN, 3), 
            Write(max_label),
            run_time=0.5
        )
        
        values[r, c] = max_q
        self.value_trackers[(r, c)].set_value(max_q)
        
        final_val = MathTex(f"V = {max_q:.2f}", color=GREEN, font_size=36).move_to(center_pos + DOWN * 0.3)
        final_val_bg = BackgroundRectangle(final_val, color=BLACK, fill_opacity=0.8, buff=0.1)
        final_group = VGroup(final_val_bg, final_val)
        
        self.play(
            FadeOut(center_val_mob),
            FadeIn(final_group)
        )
        self.wait(0.5)
        
        self.play(
            FadeOut(zoom_group), 
            FadeOut(arrows), 
            FadeOut(result_labels), 
            FadeOut(title_text),
            FadeOut(final_group), 
            FadeOut(selection_box),
            FadeOut(max_label),
            grid_group.animate.set_opacity(1)
        )

    def simulate_agent(self, values, rewards):
        """演示 Agent 从起点 (0,0) 利用计算出的价值函数走到终点"""
        # --- 1. 初始化 Agent ---
        start_pos = (0, 0)
        curr_r, curr_c = start_pos
        
        # 重新定位到 Grid 的当前位置 (它被移到了 LEFT * 3.5)
        # 注意：self.cells 里的对象已经被移走了，所以 get_center() 是最新的位置
        start_cell_center = self.cells[start_pos].get_center()
        
        agent = Dot(color=BLUE, radius=0.2).move_to(start_cell_center).set_z_index(100)
        path = TracedPath(agent.get_center, stroke_color=BLUE_A, stroke_width=4, stroke_opacity=0.8).set_z_index(99)
        
        agent_label = Text("Optimal Agent", font_size=20, color=BLUE).next_to(agent, UP, buff=0.2)
        
        self.play(FadeIn(agent), Write(agent_label))
        self.add(path)
        self.wait(0.5)
        self.play(FadeOut(agent_label))

        # --- 2. 寻路循环 ---
        # 动作: Up, Down, Left, Right
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
        steps = 0
        
        while steps < 20: # 安全上限
            # 贪婪选择: argmax [R + gamma * V_next]
            best_q = -np.inf
            best_move = None
            
            for di, dj in actions:
                ni, nj = curr_r + di, curr_c + dj
                
                # 越界检查
                if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                    r_next = rewards[ni, nj]
                    v_next = values[ni, nj]
                else:
                    # 撞墙 (假设原地不动)
                    r_next = rewards[curr_r, curr_c] 
                    v_next = values[curr_r, curr_c]
                
                q_val = r_next + self.gamma * v_next
                
                # 只有更好的才更新 (简单的 argmax)
                if q_val > best_q:
                    best_q = q_val
                    best_move = (ni, nj)
            
            # 执行移动
            if best_move:
                ni, nj = best_move
                # 如果最佳策略是撞墙（理论上不应该，除非全都是负反馈且无路可走），则不动
                if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                     target_pos = self.cells[(ni, nj)].get_center()
                     self.play(agent.animate.move_to(target_pos), run_time=0.5, rate_func=linear)
                     curr_r, curr_c = ni, nj
                
                # 到达终点 Goal (4, 4)
                if rewards[curr_r, curr_c] == 1.0:
                    self.play(Flash(agent, color=YELLOW, line_length=0.5))
                    
                    # --- 修改2: 文字移到底部边缘，防止遮挡 Grid ---
                    success_text = Text("Goal Reached!", color=YELLOW, font_size=32).to_edge(DOWN)
                    self.play(Write(success_text))
                    break
            
            steps += 1
            
        self.wait(2)