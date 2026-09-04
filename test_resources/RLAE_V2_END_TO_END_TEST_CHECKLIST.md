# RLAE 原始资料到 RLEP 集成：端到端验收清单

测试输入：`rlae_double_q_learning_raw_material.md`

## A. 原始资料与 AlgorithmSpec

- [ ] 在 Manage Algorithms → Create v2 Draft 选择 **Generic scaffold**。
- [ ] 上传测试输入，页面能提取并预览文本。
- [ ] 点击 **Suggest AlgorithmSpec with Agent** 后，算法 ID 建议为
      `double-q-learning`，算法名为 `Double Q-Learning`。
- [ ] 建议包含两张 Q 表、交叉选择/评价、FrozenLake-v1 和核心更新公式。
- [ ] Verified source evidence 不为 0，且摘录可在原文件中逐字找到。
- [ ] 选择平台的 4×4 FrozenLake 教学场景，并确认其 provenance 为
      `platform_preset`。
- [ ] 由审核人修改并确认 AlgorithmSpec 后创建草稿。

## B. 四个学习模块

- [ ] 新建 Generic 草稿时 Theory、Notebook、Experiment 都是
      `not_generated`，不会把脚手架冒充已完成内容。
- [ ] 分别调用三个模块 Agent；每次只改变当前模块的状态。
- [ ] Theory 正确解释最大化偏差、交叉选择/评价和终止转移。
- [ ] Notebook 可下载、可解析，包含可运行代码、参数实验和自检问题。
- [ ] Experiment 通过接口校验，能报告进度、奖励与成功率，并返回策略网格。
- [ ] Animation 区域不会声称生成了视频，只提供规划与上传成品 MP4 的入口。
- [ ] 每个实际发布模块都能独立进入 awaiting_review、changes_requested、approved。

## C. 动画决策选项（关键需求）

- [ ] AI 能先展示至少三个可比较的候选方案，而不是直接覆盖成唯一详细分镜。
- [ ] 每个候选方案包含教学重点、时长、复杂度、成本和适用场景。
- [ ] RLAE 可以明确选中某一候选方案，并基于该选择生成或保存详细分镜。
- [ ] Creator Kit 可下载，包含任务书、分镜、旁白、公式核对和交付清单。
- [ ] 保存动画建议不会被当作已上传 MP4，也不会改变模块审核状态。

注意：以当前实现检查，系统会生成一份可编辑 Animation Creator Guidance，
但尚未看到“多候选方案 → RLAE 选择 → 生成详细分镜”的独立状态与交互。
因此本节前三项预期会暴露产品差距，不应误判为测试通过。

## D. 审核、安装与网站发现

- [ ] 未完成模块、未清除 `placeholder_content` 或未批准模块时，安装按钮禁用。
- [ ] 三个核心模块全部完成并通过人工审核后，发布锁可以按规则清除。
- [ ] 如果添加 Animation MP4，该模块未批准时不能安装；不添加 Animation 时
      不应阻塞三个核心模块的安装。
- [ ] 安装成功后，Theory 能发现新讲义。
- [ ] Jupyter Notebooks 能发现并预览/下载新 Notebook。
- [ ] RL Laboratory 能发现并运行 Double Q-Learning，最终进度到 100%。
- [ ] 若上传并批准 MP4，Animation 能发现并播放视频及显示相关元数据。

## E. 回归与失败路径

- [ ] 内置 Q-Learning、SARSA、DQN、Policy Iteration、Value Iteration 不受影响。
- [ ] Monte Carlo Control 示例仍可正常使用。
- [ ] 对 Theory 提出 Needs Changes 后，只解锁 Theory。
- [ ] 上传损坏 Notebook、语法错误 Experiment 或伪造 MP4 时，系统拒绝替换。
- [ ] 删除已安装 Double Q-Learning 时采用软删除，文件可从回收目录恢复。
- [ ] 运行项目完整自动化测试，记录通过数、失败数和失败原因。
