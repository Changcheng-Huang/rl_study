# RL Education Platform

这是一个使用 Streamlit、Gymnasium、Manim 和 Jupyter Notebook 构建的强化学习
教学平台。它包含预渲染动画、理论章节、Notebook 预览和可实时运行的实验。

## 本地运行

项目使用 Python 3.10 虚拟环境：

```bash
source .venv/bin/activate
streamlit run web/app.py
```

浏览器打开 Streamlit 给出的本地地址。侧边栏包含：

- **Animation**：观看预渲染算法动画。
- **Theory**：阅读理论内容和导入算法的 Markdown 讲义。
- **Jupyter Notebooks**：预览和下载 Notebook。
- **RL Laboratory**：运行内置和导入的算法实验。
- **Manage Algorithms**：校验、安装和删除算法 ZIP。

## 第二版草稿工作流

`Manage Algorithms` 现在提供 Schema v2 的完整审核闭环：

1. 在 **Create v2 Draft** 上传 Markdown、UTF-8 TXT 或可提取文本的 PDF；
2. 手工填写 AlgorithmSpec，或让 LangChain Agent 根据资料生成可编辑建议；
3. 专业人员确认 AlgorithmSpec，系统记录确认人、时间和内容哈希；
4. 创建审核草稿；Generic profile 此时只写入安全脚手架，不调用模块模型；
5. 可选生成 Animation Creator Guidance，下载分镜、元数据和外部制作清单；
6. 创作者在网站外制作 MP4，上传后填写或导入公式、重点和推导步骤；
7. 在 **Review Drafts** 分别生成、检查和批准 Theory、Notebook、Experiment，
   以及已声明的 Animation；
8. 填写侧栏 Reviewer name，批准全部已声明模块后执行安装；
9. 安装后的内容会自动出现在对应教学模块中。

通用模板生成的 Experiment 只是占位实现，带有
`placeholder_content` 发布阻塞标记。已经由 Agent 生成或人工替换的单个模块
可以立即审核批准；未完成的脚手架不能批准，整个草稿也不能安装。三个模块
都成功生成并通过文件校验后会自动清除全局发布锁；人工或混合完成时仍需
审核人确认。第二版 MVP 继续使用 Monte Carlo Control 作为内置验收样例。

所有 Agent 生成的 AlgorithmSpec、Theory、Notebook 教学文字、Experiment
用户可见文字、Animation Guidance 和 Creator Kit 均要求使用英文，不受上传
资料语言影响。用于溯源核验的 evidence excerpt 是唯一例外，必须保留上传资料
原文，不能翻译或改写。

上传资料本身不会触发网络调用。配置模型连接后，用户可以主动点击
**Suggest AlgorithmSpec with Agent**，由 LangChain `ChatOpenAI` 将提取出的
资料文本发送给兼容 OpenAI Chat Completions 格式的服务。Agent 返回结构化
表单建议和来源依据；用户仍需逐项检查并确认 AlgorithmSpec。Generic profile
创建时只保存 `not_generated` 脚手架，再到 Review Drafts 逐个调用模块 Agent；
Monte Carlo profile 继续复制仓库中的内置可信模块文件。

```bash
export ALGORITHM_AGENT_API_KEY="your_api_key_here"
export ALGORITHM_AGENT_BASE_URL="https://provider.example/v1"
export ALGORITHM_AGENT_MODEL="provider-model-name"
# function_calling 兼容范围较广；支持原生 JSON Schema 时可改为 json_schema
export ALGORITHM_AGENT_STRUCTURED_METHOD="function_calling"

# 可为三个模块统一指定模型，也可以使用 THEORY_AGENT_MODEL、
# NOTEBOOK_AGENT_MODEL、EXPERIMENT_AGENT_MODEL 分别覆盖。
export ALGORITHM_MODULE_AGENT_MODEL="provider-model-name"
# 可选；未设置时沿用提供商默认行为。
export ALGORITHM_MODULE_AGENT_ENABLE_THINKING="false"
uv run streamlit run web/app.py
```

直接使用 OpenAI 时可以省略 `ALGORITHM_AGENT_BASE_URL`，也兼容现有的
`OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `OPENAI_ALGORITHM_SPEC_MODEL`
变量。API key 只从环境变量读取，不写入草稿或仓库。超过 60,000 字符的资料
只发送开头和结尾；适配器、模型、结构化输出方式、响应 ID、证据、警告和
token 用量会记录在草稿 `generation.algorithm_spec_agent` 中。模块 Agent 的
模型、提示词版本、响应 ID、警告和 token 用量记录在
`generation.module_generations` 中。AlgorithmSpec Agent 只建议统一数据；
Theory、Notebook 和 Experiment 由各自模块 Agent 生成。Animation Planning
Agent 只生成制作建议和分镜，不生成视频或可执行代码。
Agent 请求不会自动重试或切换模型/结构化方式。失败时页面保留原文件，显示
request ID 和技术详情，由 Provider 明确点击 Retry generation 再次调用。

使用阿里云百炼 `qwen3.8-flash` 时，完整的环境变量、启动验证、当前实现边界和
后续计划见
[`docs/qwen3_8_flash_startup_and_roadmap.md`](docs/qwen3_8_flash_startup_and_roadmap.md)。

第二版当前也不会生成动画或执行 Manim 源码。Review Drafts 中的
**Animation** 工作区先从 AlgorithmSpec 生成三个概念方案，RLAE Provider 选择
一个方向后，再生成确定性制作清单或调用兼容 OpenAI 的 Planning Agent 细化
分镜。下载的 Creator Kit 只包含
面向制作者的英文文档：制作任务书、分镜表、旁白稿、公式核对、验收清单和
可选工具参考，不要求制作者阅读或填写 JSON。制作建议独立于 Animation 模块：
保存或重新生成建议不会创建、替换或解锁 MP4，也不影响安装门禁。用户用
Manim 或其他工具在网站外
制作好 MP4 后上传，平台负责格式校验、预览、完整展示元数据、审核状态和
安装。上传视频不会自动分析画面内容。

Planning Agent 默认继承 `ALGORITHM_AGENT_*` 连接，也可以用
`ANIMATION_PLANNING_AGENT_MODEL`、`ANIMATION_PLANNING_AGENT_API_KEY`、
`ANIMATION_PLANNING_AGENT_BASE_URL`、
`ANIMATION_PLANNING_AGENT_STRUCTURED_METHOD` 和
`ANIMATION_PLANNING_AGENT_ENABLE_THINKING` 单独覆盖。其输出和运行记录保存在
草稿内部的 `animation_guidance.json` 与
`generation.animation_guidance` 中，API key 不会写入文件。
内部 JSON 由表单自动映射，普通用户不直接编辑。

### GitHub / Colab Notebook 发布

内置与导入 Notebook 共用可配置的公开 GitHub 仓库。细粒度 Token 只需要
目标仓库的 Contents 写权限，并且只从环境变量读取：

```bash
export RLAE_NOTEBOOK_GITHUB_OWNER="your-github-owner"
export RLAE_NOTEBOOK_GITHUB_REPO="your-public-notebook-repository"
export RLAE_NOTEBOOK_GITHUB_BRANCH="main"
export RLAE_NOTEBOOK_GITHUB_ROOT="notebooks"
export RLAE_NOTEBOOK_GITHUB_TOKEN="fine-grained-token"
```

安装完成后，导入 Notebook 会发布到
`notebooks/{algorithm-id}/{version}/notebook.ipynb`。相同内容可安全重试；
同版本的不同内容不会被覆盖。GitHub 故障不会回滚本地安装，可在 Package
Manager 中重试。Token 不写入算法包或发布状态文件。

Agent 生成的 Notebook 会进行 nbformat、语法、依赖白名单和危险操作静态检查，
但平台不会执行单元格，因此界面会明确显示“static checks passed · not
executed”。

导入实验可通过声明式 `presentation` 和 `views.policy_grid` 展示 Task、初始
网格和训练后的 Learned Policy；数据在父进程重新校验后由平台统一绘制，
算法包仍不能执行自定义 Streamlit 页面。已安装的 Schema v2 算法可以创建
补丁版本修订草稿，完成审核并软删除旧 ID 后再安装新版。

草稿、拒绝记录和已安装包分别保存在：

```text
algorithm_packages/drafts/
algorithm_packages/rejected/
algorithm_packages/installed/
algorithm_packages/.trash/
```

被拒绝的草稿可在 **Rejected Drafts** 中查看和恢复；恢复后所有模块进入
`changes_requested`，需要重新审核。

模块进入 `changes_requested` 后，模块 Agent 会读取当前 AlgorithmSpec、原始
资料、当前文件和审核意见进行定向重新生成；也可以填写理由并执行
**Cancel Change Request**。取消不会覆盖模块文件，只会恢复待审核状态并记录
审核历史。Notebook 和 Experiment 的下载、修改、重新上传仍作为人工兜底。

导入实验的 `get_spec()` 和 `run()` 均在独立子进程执行，默认超时为
120 秒。该隔离只用于防止错误实验拖死 Streamlit 页面，不是面向不可信代码的
生产级沙箱。

## 导入示例算法

仓库提供 Monte Carlo Control 示例：

```bash
.venv/bin/python tools/build_algorithm_package.py \
  algorithm_packages/examples/monte_carlo_control
```

命令生成：

```text
dist/monte-carlo-control-1.0.0.zip
```

启动网站，进入 **Manage Algorithms**，选择这个 ZIP 并点击
**Install Package**。导入后：

- Theory 中出现 Monte Carlo Control 讲义。
- Jupyter Notebooks 中出现示例 Notebook。
- RL Laboratory 中出现可运行的 Monte Carlo Control 实验。
- 示例没有提供 MP4，因此不会出现在 Animation；可在 v2 草稿创建或审核页面
  上传制作完成的 MP4，审核通过并安装后自动加入。

如果上传错误，在 **Manage Algorithms** 的 Installed Algorithms 区域勾选
**Confirm removal**，再点击 **Remove**。删除采用软删除，文件会移动到
`algorithm_packages/.trash/`，不会立刻永久丢失。

完整 ZIP 格式和实验接口见
[`algorithm_packages/README.md`](algorithm_packages/README.md)。

Schema v2 的字段、审核状态和演示步骤见
[`docs/v2_mvp_guide.md`](docs/v2_mvp_guide.md)。

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```
