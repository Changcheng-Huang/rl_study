from manim import *
import numpy as np


class DQNDemo(Scene):
    """
    DQN 演示（GridWorld + Neural Net + Q 值柱状图）

    修复点（相对你原版 DQN.py）：
    - 不再用 cell.set_stroke(width=0) “取消高亮”，避免网格线逐步消失（原代码会把格子边框直接变为 0）；
    - 右侧柱状图不再依赖 bar_bg_group[b_idx*2] 这种脆弱索引，而是显式保存每个 bar 的背景框；
    - 目标 Q 的虚线位置按 target_q 值计算，不再固定 shift(UP*0.3)；
    - Intro 里修复了 Rectangle(grid_xstep=...) 这类不兼容参数，并避免 next_to(新建一个Rectangle/Circle) 导致的错位；
    - 自适应布局：左(网格) / 中(网络) / 右(柱状图) 互不遮挡。
    """

    def construct(self):
        # --- 0. 全局配置 ---
        self.grid_size = 3
        self.cell_size = 1.15
        self.grid_spacing = 1.25

        # --- 1. 理论介绍（参照 policy_iteration 的多页 Intro 风格） ---
        self.play_intro()

        # --- 2. 核心演示 ---
        self.play_dqn_process()

    # -------------------------------------------------------------------------
    # Intro
    # -------------------------------------------------------------------------
    def _mini_grid_in_square(self, side_len=2.0, step=0.5, stroke=GREY_B, stroke_width=1):
        """在一个正方形内画“表格线”，用于 Q-Table 的视觉示意。"""
        lines = VGroup()
        half = side_len / 2
        # 只画内部线（不含边框），避免太粗
        n = int(side_len / step)
        # 内部线的位置：-half + k*step, k=1..n-1
        for k in range(1, n):
            x = -half + k * step
            v = Line([x, -half, 0], [x, half, 0], color=stroke, stroke_width=stroke_width)
            h = Line([-half, x, 0], [half, x, 0], color=stroke, stroke_width=stroke_width)
            lines.add(v, h)
        return lines

    def play_intro(self):
        """参照 policy_iteration 的风格：多页介绍 + 页码 + 自适应排版。"""
        self.intro_slow = 2.4

        # 顶部标题
        title = Text("Deep Q-Network (DQN)", font_size=48, color=BLUE).to_edge(UP)
        env_text = Text(
            "Problem: 3x3 GridWorld Navigation", font_size=30, color=TEAL
        ).next_to(title, DOWN, buff=0.25)

        self.play(FadeIn(title), FadeIn(env_text))
        self.wait(0.25 * self.intro_slow)

        total_pages = 5

        # --- Helpers ---
        def page_counter(n: int):
            return Text(f"{n}/{total_pages}", font_size=18, color=GREY_B).to_corner(DR).shift(
                UP * 0.35 + LEFT * 0.35
            )

        def fit_to_intro_area(mob: Mobject, top_anchor: Mobject, side_margin=0.7, bottom_margin=0.55, top_buff=0.55):
            max_w = config.frame_width - 2 * side_margin
            top_y = top_anchor.get_bottom()[1] - top_buff
            bottom_y = -config.frame_height / 2 + bottom_margin
            max_h = top_y - bottom_y

            # 只缩小不放大
            if mob.width > max_w:
                mob.scale_to_fit_width(max_w)
            if mob.height > max_h:
                mob.scale_to_fit_height(max_h)

        def show_page(n: int, group: VGroup, keep_env=True, hold=1.5):
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

        # -----------------------------------------------------------------
        # Page 1: What is DQN?
        # -----------------------------------------------------------------
        p1_t = Text("1. What is DQN?", font_size=30, color=YELLOW)
        p1_desc = Text("Model-free Value-based RL (Q-learning + Neural Network)", font_size=22, color=GREY_B)
        p1_header = VGroup(p1_t, p1_desc).arrange(DOWN, aligned_edge=LEFT, buff=0.12)

        p1_eq1 = MathTex(r"Q_\theta(s,a)\approx Q^*(s,a)", font_size=36)
        p1_eq2 = MathTex(r"\pi(s)=\arg\max_a\,Q_\theta(s,a)", font_size=36)
        p1_eqs = VGroup(p1_eq1, p1_eq2).arrange(DOWN, aligned_edge=LEFT, buff=0.22)

        p1_bul = VGroup(
            Text("• Learns from interaction (no model P,R needed)", font_size=22, color=WHITE),
            Text("• Outputs Q-values for actions: U / D / L / R", font_size=22, color=WHITE),
            Text("• Uses TD target + loss to update network parameters", font_size=22, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)

        p1_group = VGroup(p1_header, p1_eqs, p1_bul).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        show_page(1, p1_group, keep_env=True, hold=1.55)

        # -----------------------------------------------------------------
        # Page 2: From Q-Table to Neural Net
        # -----------------------------------------------------------------
        p2_t = Text("2. From Q-Table to Neural Network", font_size=30, color=YELLOW)
        p2_desc = Text("Replace the table with a function approximator", font_size=22, color=GREY_B)
        p2_header = VGroup(p2_t, p2_desc).arrange(DOWN, aligned_edge=LEFT, buff=0.12)

        # Q-table -> NN diagram (静态展示，避免遮挡)
        qbox = Square(side_length=1.65, color=WHITE, stroke_width=2)
        qgrid = self._mini_grid_in_square(side_len=1.65, step=0.4125, stroke=GREY_C, stroke_width=1)
        cross = Cross(qbox, stroke_color=RED, stroke_width=7)
        qtable_label = Text("Q-Table", font_size=20).next_to(qbox, DOWN, buff=0.15)
        qtable_viz = VGroup(qbox, qgrid, cross, qtable_label)

        nn_circle = Circle(radius=0.45, color=BLUE, stroke_width=2).set_fill(BLUE_E, 0.35)
        nn_label = Text("Neural Net", font_size=20).next_to(nn_circle, DOWN, buff=0.15)
        nn_viz = VGroup(nn_circle, nn_label)

        ends = VGroup(qtable_viz, nn_viz).arrange(RIGHT, buff=2.2)
        arrow = Arrow(qtable_viz.get_right(), nn_viz.get_left(), buff=0.22, stroke_width=5)

        expl = VGroup(
            VGroup(Text("Approximate", font_size=22, color=YELLOW), MathTex("Q(s,a)", color=YELLOW).scale(0.9)).arrange(RIGHT, buff=0.20),
            Text("with a Neural Network", font_size=22, color=YELLOW),
        ).arrange(DOWN, buff=0.06)

        if expl.width > arrow.width * 0.98:
            expl.scale_to_fit_width(arrow.width * 0.98)
        expl.next_to(arrow, UP, buff=0.12)

        diagram = VGroup(ends, arrow, expl)
        diagram.shift(RIGHT * 0.55)  # 视觉居中

        p2_bul = VGroup(
            Text("• Q-table size grows with |S| × |A|", font_size=22, color=WHITE),
            Text("• Network generalizes across similar states", font_size=22, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)

        p2_group = VGroup(p2_header, diagram, p2_bul).arrange(DOWN, aligned_edge=LEFT, buff=0.42)
        show_page(2, p2_group, keep_env=True, hold=1.55)

        # -----------------------------------------------------------------
        # Page 3: State encoding & outputs
        # -----------------------------------------------------------------
        p3_t = Text("3. State Encoding & Outputs", font_size=30, color=YELLOW)
        p3_desc = Text("In this demo: state = (x, y) coordinates", font_size=22, color=GREY_B)
        p3_header = VGroup(p3_t, p3_desc).arrange(DOWN, aligned_edge=LEFT, buff=0.12)

        p3_eq1 = MathTex(r"s=(x,y)", font_size=40)
        p3_eq2 = MathTex(r"Q_\theta(s)=[Q_U,\,Q_D,\,Q_L,\,Q_R]", font_size=34)
        p3_eqs = VGroup(p3_eq1, p3_eq2).arrange(DOWN, aligned_edge=LEFT, buff=0.25)

        p3_bul = VGroup(
            Text("• 2 inputs → 2 state features (x and y)", font_size=22, color=WHITE),
            Text("• 4 outputs → Q-values for actions (U,D,L,R)", font_size=22, color=WHITE),
            Text("• Other option: one-hot state (3×3 = 9 dims)", font_size=22, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)

        p3_group = VGroup(p3_header, p3_eqs, p3_bul).arrange(DOWN, aligned_edge=LEFT, buff=0.42)
        show_page(3, p3_group, keep_env=True, hold=1.55)

        # -----------------------------------------------------------------
        # Page 4: TD Target & Loss
        # -----------------------------------------------------------------
        p4_t = Text("4. TD Target & Loss", font_size=30, color=YELLOW)
        p4_desc = Text("Train the network by minimizing the TD error", font_size=22, color=GREY_B)
        p4_header = VGroup(p4_t, p4_desc).arrange(DOWN, aligned_edge=LEFT, buff=0.12)

        p4_eq1 = MathTex(r"y=r+\gamma\max_{a'}Q_{\theta^-}(s',a')", font_size=34)
        p4_eq2 = MathTex(r"L=(y-Q_\theta(s,a))^2", font_size=36)
        p4_eqs = VGroup(p4_eq1, p4_eq2).arrange(DOWN, aligned_edge=LEFT, buff=0.22)

        # 小图例（对应你右侧柱状图的视觉符号）
        legend_items = VGroup(
            VGroup(Square(0.18, fill_opacity=1, fill_color=GREEN, stroke_width=0), Text("Q-value bars", font_size=22, color=WHITE)).arrange(RIGHT, buff=0.25),
            VGroup(Square(0.18, fill_opacity=1, fill_color=YELLOW, stroke_width=0), Text("Chosen action", font_size=22, color=WHITE)).arrange(RIGHT, buff=0.25),
            VGroup(DashedLine(LEFT * 0.2, RIGHT * 0.2, color=RED, dash_length=0.06, stroke_width=2), Text("TD Target", font_size=22, color=WHITE)).arrange(RIGHT, buff=0.25),
            VGroup(Line(LEFT * 0.2, RIGHT * 0.2, color=RED, stroke_width=4), Text("Loss (error)", font_size=22, color=WHITE)).arrange(RIGHT, buff=0.25),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)

        p4_group = VGroup(p4_header, p4_eqs, legend_items).arrange(DOWN, aligned_edge=LEFT, buff=0.42)
        show_page(4, p4_group, keep_env=True, hold=1.6)

        # -----------------------------------------------------------------
        # Page 5: Demo schedule
        # -----------------------------------------------------------------
        p5_t = Text("5. Demo Schedule", font_size=30, color=YELLOW)

        def stage_block(title_txt: str, subtitle_txt: str, color_fill):
            box_width = 9.0
            box = RoundedRectangle(
                corner_radius=0.15, height=1.10, width=box_width,
                fill_color=color_fill, fill_opacity=0.25,
                stroke_color=WHITE, stroke_width=2,
            )
            t = Text(title_txt, font_size=24, color=WHITE)
            s = Text(subtitle_txt, font_size=18, color=GREY_B)
            vg = VGroup(t, s).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
            if vg.width > box_width - 0.6:
                vg.scale_to_fit_width(box_width - 0.6)
            vg.move_to(box.get_center()).align_to(box, LEFT).shift(RIGHT * 0.25)
            return VGroup(box, vg)

        st1 = stage_block("Step A: Forward Pass", "State → Neural Net → 4 Q-values", BLUE_E)
        st2 = stage_block("Step B: Choose Action", "Pick argmax Q(s,a) (yellow bar)", TEAL_E)
        st3 = stage_block("Step C: TD Target & Loss", "Show target (red dashed) and loss", RED_E)
        st4 = stage_block("Step D: Agent Run", "After training: run an (approx.) optimal path", GREEN_E)
        timeline = VGroup(st1, st2, st3, st4).arrange(DOWN, aligned_edge=LEFT, buff=0.25)

        params_line = VGroup(
            MathTex(r"\gamma=0.9\quad \text{Grid}=3\times3", font_size=30, color=YELLOW),
            Text("| Actions = 4", font_size=22, color=GREY_B),
        ).arrange(RIGHT, buff=0.4)

        p5_group = VGroup(p5_t, timeline, params_line).arrange(DOWN, aligned_edge=LEFT, buff=0.38)
        show_page(5, p5_group, keep_env=False, hold=1.9)

        # End intro: 保留标题，缩小并固定到顶部
        self.play(title.animate.scale(0.85).to_edge(UP), run_time=0.30 * self.intro_slow)
        self.wait(0.15 * self.intro_slow)

    # -------------------------------------------------------------------------
    # Neural Net Viz
    # -------------------------------------------------------------------------
    def create_neural_net(self):
        """创建一个可视化的神经网络 VGroup"""
        layers = [2, 5, 4]  # 输入层(x,y), 隐藏层, 输出层(Q-values)

        neurons = VGroup()
        edges = VGroup()

        layer_x_spacing = 1.6
        neuron_y_spacing = 0.6

        layer_groups = []

        for i, layer_size in enumerate(layers):
            layer_group = VGroup()
            x = i * layer_x_spacing

            y_start = (layer_size - 1) * neuron_y_spacing / 2

            for j in range(layer_size):
                y = y_start - j * neuron_y_spacing
                circle = Circle(
                    radius=0.12,
                    color=WHITE,
                    stroke_width=2,
                    fill_color=BLACK,
                    fill_opacity=1,
                ).move_to([x, y, 0])
                layer_group.add(circle)

            layer_groups.append(layer_group)
            neurons.add(layer_group)

        # 连接线
        for i in range(len(layers) - 1):
            curr_layer = layer_groups[i]
            next_layer = layer_groups[i + 1]
            for n1 in curr_layer:
                for n2 in next_layer:
                    edge = Line(
                        n1.get_center(),
                        n2.get_center(),
                        stroke_width=1,
                        color=GREY_C,
                        stroke_opacity=0.55
                    )
                    edges.add(edge)

        net_group = VGroup(edges, neurons)

        input_label = Text("Input\n(State)", font_size=16, color=BLUE_B).next_to(layer_groups[0], UP, buff=0.15)
        output_label = Text("Output\n(Q-Values)", font_size=16, color=GREEN_B).next_to(layer_groups[-1], UP, buff=0.15)

        # 说明：输入层节点数量对应“状态的特征维度”。
        # 这里用 2 个节点表示 (x, y) 坐标编码（便于直观演示）。
        # 如果你想展示 one-hot（例如 3x3 网格=9 维），可以把 layers[0] 改成 9，
        # 并相应调整 neuron_y_spacing / layer_x_spacing 以免太拥挤。
        feature_labels = VGroup()
        xy_names = ["x", "y"]
        if len(layer_groups[0]) == 2:
            for node, name in zip(layer_groups[0], xy_names):
                feature_labels.add(
                    Text(name, font_size=14, color=GREY_B).next_to(node, LEFT, buff=0.12)
                )

        labels = VGroup(input_label, output_label, feature_labels)

        return net_group, layer_groups, edges, labels

    # -------------------------------------------------------------------------
    # GridWorld + Chart Builders
    # -------------------------------------------------------------------------
    def build_grid_world(self, rewards: np.ndarray):
        grid_group = VGroup()
        cells = {}
        label_group = VGroup()

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pos = np.array([(j - 1) * self.grid_spacing, (1 - i) * self.grid_spacing, 0])

                color = BLACK
                if rewards[i, j] == 1.0:
                    color = GREEN_E
                elif rewards[i, j] == -1.0:
                    color = RED_E
                elif rewards[i, j] == -0.5:
                    color = ORANGE

                cell = Square(
                    side_length=self.cell_size,
                    fill_color=color,
                    fill_opacity=0.45,
                    color=WHITE,
                    stroke_width=2,
                ).move_to(pos)

                cells[(i, j)] = cell
                grid_group.add(cell)

                # Label
                if rewards[i, j] == 1.0:
                    txt = "+1"
                elif rewards[i, j] == -1.0:
                    txt = "-1"
                elif rewards[i, j] == -0.5:
                    txt = "Mud"
                else:
                    txt = ""
                if txt:
                    t = Text(txt, font_size=18, color=WHITE).move_to(pos).set_z_index(5)
                    label_group.add(t)

        # 标题小一点，避免遮挡
        title = Text("GridWorld", font_size=24, color=WHITE).next_to(grid_group, UP, buff=0.25)
        full = VGroup(grid_group, label_group, title)
        return full, cells

    def build_q_chart(self, bar_names=("U", "D", "L", "R")):
        """右侧 Q 值柱状图（手工搭建，布局更稳）。"""
        BAR_BG_H = 3.0
        BAR_W = 0.42
        COL_BUFF = 0.35
        INNER_PAD = 0.06  # bar 离底部一点点距离
        VALUE_TO_HEIGHT = 2.55  # val * VALUE_TO_HEIGHT -> height，保持与你原版接近

        bar_columns = VGroup()
        bar_bgs = []
        bars = []
        action_labels = VGroup()

        for name in bar_names:
            bg = Rectangle(width=BAR_W, height=BAR_BG_H, stroke_color=GREY_B, stroke_width=1)
            bar = Rectangle(width=BAR_W, height=0.05, fill_color=GREEN, fill_opacity=0.85, stroke_width=0)
            # bar 初始放底部
            bar.move_to(bg.get_bottom() + UP * (INNER_PAD + bar.height / 2))

            lbl = Text(name, font_size=20, color=WHITE).next_to(bg, DOWN, buff=0.12)
            col = VGroup(bg, bar, lbl)

            bar_columns.add(col)
            bar_bgs.append(bg)
            bars.append(bar)
            action_labels.add(lbl)

        bar_columns.arrange(RIGHT, buff=COL_BUFF, aligned_edge=DOWN)

        chart_title = Text("Q-Values", font_size=24, color=YELLOW).next_to(bar_columns, UP, buff=0.25)
        chart = VGroup(chart_title, bar_columns)

        # 返回更新所需的引用
        meta = {
            "BAR_BG_H": BAR_BG_H,
            "BAR_W": BAR_W,
            "INNER_PAD": INNER_PAD,
            "VALUE_TO_HEIGHT": VALUE_TO_HEIGHT,
            "bar_bgs": bar_bgs,
            "bars": bars,
            "bar_columns": bar_columns,
        }
        return chart, meta

    def _place_three_panels(self, left_panel: Mobject, mid_panel: Mobject, right_panel: Mobject, buff=0.6):
        """把三块内容放进同一帧：左 / 中 / 右，尽量不互相遮挡。"""
        left_panel.to_edge(LEFT, buff=0.7)
        right_panel.to_edge(RIGHT, buff=0.7)

        # 安全边距：避免右侧/左侧的描边或注释被画面裁剪
        frame_right = self.camera.frame_width / 2 - 0.15
        if right_panel.get_right()[0] > frame_right:
            right_panel.shift(LEFT * (right_panel.get_right()[0] - frame_right))

        frame_left = -self.camera.frame_width / 2 + 0.15
        if left_panel.get_left()[0] < frame_left:
            left_panel.shift(RIGHT * (frame_left - left_panel.get_left()[0]))

        # 垂直对齐
        right_panel.move_to([right_panel.get_center()[0], left_panel.get_center()[1], 0])

        # 中间区域可用宽度
        left_x = left_panel.get_right()[0] + buff
        right_x = right_panel.get_left()[0] - buff
        mid_x = (left_x + right_x) / 2

        mid_panel.move_to([mid_x, left_panel.get_center()[1] - 0.25, 0])

        available_w = max(0.1, right_x - left_x)
        if mid_panel.width > available_w:
            mid_panel.scale(available_w / mid_panel.width)

        # 再次对齐（scale 会变）
        mid_panel.move_to([mid_x, left_panel.get_center()[1] - 0.25, 0])

    # -------------------------------------------------------------------------
    # Main DQN Process
    # -------------------------------------------------------------------------
    def play_dqn_process(self):
        # --- A. 布局初始化 ---
        rewards = np.array([
            [-0.04, -0.5, -0.04],
            [-0.04, -1.0,  -0.04],
            [-0.04, -0.04,  1.0]
        ])

        grid_panel, cells = self.build_grid_world(rewards)

        net_viz, layer_groups, edges, labels = self.create_neural_net()
        net_group_all = VGroup(net_viz, labels)

        chart_group, chart_meta = self.build_q_chart(bar_names=("U", "D", "L", "R"))

        # 在柱子上方显示 Q 数值（每一步动态更新）
        q_nums = []
        q_num_group = VGroup()
        for i, (bg, bar) in enumerate(zip(chart_meta["bar_bgs"], chart_meta["bars"])):
            dn = DecimalNumber(
                0.0,
                num_decimal_places=2,
                font_size=16,
                color=WHITE
            ).set_z_index(30)
            # 初始位置：靠近柱子底部上方一点，避免一开始高度太小导致重叠
            dn.move_to([bg.get_center()[0], bg.get_bottom()[1] + chart_meta["INNER_PAD"] + 0.25, 0])
            q_nums.append(dn)
            q_num_group.add(dn)
        chart_group.add(q_num_group)

        # 自适应放置三块面板
        self._place_three_panels(grid_panel, net_group_all, chart_group, buff=0.7)

        self.play(FadeIn(grid_panel), FadeIn(net_group_all), FadeIn(chart_group))

        # --- B. 演示过程 ---
        agent = Dot(color=BLUE_A, radius=0.15).set_z_index(20)
        agent.move_to(cells[(0, 0)].get_center())
        self.play(FadeIn(agent))

        steps = [
            # ((r,c), action_idx, reward, q_values_list)
            # Action: 0:U, 1:D, 2:L, 3:R
            ((0, 0), 1, -0.04, [0.10, 0.20, 0.10, 0.15]),
            ((1, 0), 1, -0.04, [0.20, 0.50, 0.10, 0.20]),
            ((2, 0), 3, -0.04, [0.10, 0.10, 0.20, 0.60]),
            ((2, 1), 3,  1.00, [0.10, 0.00, 0.30, 0.90]),
        ]

        bar_names = ["U", "D", "L", "R"]
        bars = chart_meta["bars"]
        bar_bgs = chart_meta["bar_bgs"]
        VALUE_TO_HEIGHT = chart_meta["VALUE_TO_HEIGHT"]
        INNER_PAD = chart_meta["INNER_PAD"]

        for idx, (pos, action_idx, reward, q_vals) in enumerate(steps):
            r, c = pos
            current_cell = cells[(r, c)]

            # 1) 当前 state 的高亮：用“覆盖框”，不改 cell stroke，避免网格消失
            hl = SurroundingRectangle(current_cell, buff=0, color=YELLOW, stroke_width=4).set_z_index(15)
            self.play(Create(hl), run_time=0.2)

            # 2) 前向传播动画 (Forward Pass)
            run_time = 0.9

            # 输入层闪烁
            self.play(*[n.animate.set_fill(YELLOW) for n in layer_groups[0]], run_time=0.18)

            # 信号通过连线
            edge_anims = [
                ShowPassingFlash(e.copy().set_color(YELLOW), time_width=0.25)
                for e in edges
            ]
            self.play(LaggedStart(*edge_anims, lag_ratio=0.08), run_time=run_time)

            # 输出层闪烁
            self.play(*[n.animate.set_fill(GREEN) for n in layer_groups[-1]], run_time=0.18)

            # 恢复神经元颜色（保持 fill_opacity 不变）
            restore = [n.animate.set_fill(BLACK) for n in layer_groups[0]] + \
                      [n.animate.set_fill(BLACK) for n in layer_groups[-1]]
            self.play(*restore, run_time=0.10)

            # 3) 更新右侧柱状图
            bar_anims = []
            for b_idx, val in enumerate(q_vals):
                target_h = max(0.02, float(val) * VALUE_TO_HEIGHT)
                bg = bar_bgs[b_idx]

                new_bar = Rectangle(
                    width=bg.width,
                    height=target_h,
                    fill_color=GREEN,
                    fill_opacity=0.85,
                    stroke_width=0
                )
                new_bar.move_to(bg.get_bottom() + UP * (INNER_PAD + target_h / 2))

                if b_idx == action_idx:
                    # 选中动作：改成黄色更直观
                    new_bar.set_fill(YELLOW, opacity=0.9)

                bar_anims.append(Transform(bars[b_idx], new_bar))

            # 同步更新每根柱子的数值标签
            num_anims = []
            for nb_idx, nb_val in enumerate(q_vals):
                nb_bg = bar_bgs[nb_idx]
                nb_h = max(0.02, float(nb_val) * VALUE_TO_HEIGHT)
                nb_y = nb_bg.get_bottom()[1] + INNER_PAD + nb_h + 0.18
                # 夹紧到柱状图背景范围内，避免越界
                nb_y = min(nb_bg.get_top()[1] - 0.12, nb_y)
                nb_y = max(nb_bg.get_bottom()[1] + 0.18, nb_y)

                nb_pos = [nb_bg.get_center()[0], nb_y, 0]
                nb_color = YELLOW if nb_idx == action_idx else WHITE

                num_anims.append(q_nums[nb_idx].animate.set_value(float(nb_val)).move_to(nb_pos).set_color(nb_color))

            self.play(*bar_anims, *num_anims, run_time=0.45)

            # 4) 执行动作：移动 agent 到下一格（按 steps 给定的下一个 state）
            if idx < len(steps) - 1:
                next_pos = steps[idx + 1][0]
            else:
                next_pos = (2, 2)

            self.play(agent.animate.move_to(cells[next_pos].get_center()), run_time=0.45)

            # 5) 反向传播 (Backprop) + Reward 信息
            info = Text(
                f"a={bar_names[action_idx]}   r={reward:+.2f}",
                font_size=22,
                color=GOLD
            ).to_corner(DR, buff=0.3).set_z_index(30)
            self.play(FadeIn(info, shift=DOWN * 0.15), run_time=0.2)

            # 目标 Q（演示用）
            target_q_val = float(q_vals[action_idx]) + 0.2
            bg = bar_bgs[action_idx]

            # 目标虚线位置按 target_q 算
            target_h = max(0.02, target_q_val * VALUE_TO_HEIGHT)
            y = bg.get_bottom()[1] + INNER_PAD + target_h
            target_line = DashedLine(
                start=[bg.get_left()[0], y, 0],
                end=[bg.get_right()[0], y, 0],
                dash_length=0.08,
                stroke_width=2,
                color=RED
            ).set_z_index(25)

            target_label = Text("TD Target", color=RED, font_size=15)
            target_val = DecimalNumber(target_q_val, num_decimal_places=2, font_size=15, color=RED)
            target_text = VGroup(target_label, target_val).arrange(RIGHT, buff=0.12)

            # 位置：TD Target 放在虚线上方、居中对齐当前柱子，避免与 Loss/柱子数值打架
            target_text.next_to(target_line, UP, buff=0.14)
            target_text.set_x(bg.get_center()[0])
            top_limit = bg.get_top()[1] - 0.08
            if target_text.get_top()[1] > top_limit:
                target_text.shift(DOWN * (target_text.get_top()[1] - top_limit))
            target_text.set_z_index(25)

            # Loss（误差）：预测 Q 与 TD Target 的差
            pred_h = max(0.02, float(q_vals[action_idx]) * VALUE_TO_HEIGHT)
            pred_y = bg.get_bottom()[1] + INNER_PAD + pred_h
            loss_seg = Line(
                start=[bg.get_right()[0], pred_y, 0],
                end=[bg.get_right()[0], y, 0],
                color=RED,
                stroke_width=4
            ).set_z_index(25)

            loss_val = abs(target_q_val - float(q_vals[action_idx]))
            loss_label = Text("Loss", color=RED, font_size=15)
            loss_num = DecimalNumber(loss_val, num_decimal_places=2, font_size=15, color=RED)
            loss_text = VGroup(loss_label, loss_num).arrange(RIGHT, buff=0.12)

            # 位置：Loss 放在误差线段侧边（默认左侧，贴边时自动改到右侧），并与 TD Target 错开
            loss_text.next_to(loss_seg, LEFT, buff=0.14)
            loss_text.set_y(loss_seg.get_center()[1])
            if loss_text.get_left()[0] < bg.get_left()[0] + 0.02:
                loss_text.next_to(loss_seg, RIGHT, buff=0.14)
                loss_text.set_y(loss_seg.get_center()[1])

            if abs(loss_text.get_center()[1] - target_text.get_center()[1]) < 0.22:
                loss_text.shift(DOWN * 0.22)

            # 防止出画面（右侧贴边时很常见）
            frame_r = self.camera.frame_width / 2 - 0.12
            frame_l = -self.camera.frame_width / 2 + 0.12
            if loss_text.get_right()[0] > frame_r:
                loss_text.shift(LEFT * (loss_text.get_right()[0] - frame_r))
            if loss_text.get_left()[0] < frame_l:
                loss_text.shift(RIGHT * (frame_l - loss_text.get_left()[0]))

            loss_text.set_z_index(25)


            self.play(
                Create(target_line),
                FadeIn(target_text),
                Create(loss_seg),
                FadeIn(loss_text),
                run_time=0.3
            )
            self.wait(0.2)

            # 回传闪烁（红色脉冲）
            self.play(*[e.animate.set_stroke(RED, opacity=0.85) for e in edges], run_time=0.18)
            self.play(*[e.animate.set_stroke(GREY_C, opacity=0.55) for e in edges], run_time=0.25)

            self.play(
                FadeOut(info),
                FadeOut(target_line),
                FadeOut(target_text),
                FadeOut(loss_seg),
                FadeOut(loss_text),
                FadeOut(hl),
                run_time=0.25
            )

        # --- C. 训练完成 ---
        self.play(
            FadeOut(agent),
            FadeOut(chart_group),
            FadeOut(net_group_all),
            grid_panel.animate.scale(1.12).move_to(ORIGIN).shift(DOWN * 0.15),
            run_time=0.6
        )

        final_text = Text("Training Complete: Optimal Policy", color=GREEN, font_size=32).to_edge(DOWN, buff=0.8)
        self.play(Write(final_text))
        self.wait(0.3)

        # 快速跑一遍最优路径
        agent.move_to(cells[(0, 0)].get_center())
        self.play(FadeIn(agent))

        path = TracedPath(agent.get_center, stroke_color=BLUE, stroke_width=4)
        self.add(path)

        for wp in [(1, 0), (2, 0), (2, 1), (2, 2)]:
            self.play(agent.animate.move_to(cells[wp].get_center()), run_time=0.35)

        self.play(Flash(agent, color=YELLOW))
        self.wait(1.5)
