# Schema v2 MVP 操作与实现说明

## 工作流

第二版在 Schema v1 直接导入能力之外增加以下闭环：

```text
本地资料
→ AlgorithmSpec / manifest.json
→ Theory、Notebook、Experiment 草稿 + 可选 Animation MP4
→ 自动校验
→ 分模块审核
→ 标准 ZIP
→ 原子安装
→ 网站发现与隔离运行
```

启动平台：

```bash
source .venv/bin/activate
streamlit run web/app.py
```

进入 **Manage Algorithms** 后：

1. 在 **Create v2 Draft** 选择 Monte Carlo Control preset；
2. 上传资料，或使用内置 Monte Carlo Theory 作为演示来源；
3. 手工填写 AlgorithmSpec，或让 LangChain Agent 生成建议并逐项检查；
4. 系统根据支持环境推荐标准教学场景；用户可以接受平台模板、选择其他模板，
   或暂不生成地图；
5. 在侧栏填写 Reviewer name，确认 AlgorithmSpec 为模块唯一事实来源；
6. 创建审核草稿；Generic profile 此时只创建三个 `not_generated` 安全脚手架；
7. 可选上传已经制作完成的 Animation MP4，并填写公式、重点与推导步骤；
8. 在 **Review Drafts** 分别生成、预览和批准 Theory、Notebook、Experiment；
9. 点击 **Install Approved Draft**；
10. 到 Animation、Theory、Jupyter Notebooks 和 RL Laboratory 验证自动发现；
11. 在 Installed 标签页软删除算法。

同一算法 ID 已安装时，必须先软删除旧版本，系统不会覆盖或原地升级。

上传资料后默认仍只进行文本提取和预览，不会自动发起网络调用。配置环境变量
后，可以主动调用使用 OpenAI-compatible 接口的 LangChain Agent：

输出语言固定为英文。该规则覆盖 AlgorithmSpec、Theory、Notebook 教学文字、
Experiment 用户可见文字、Animation Guidance 与 Creator Kit。上传资料和可验证
来源摘录保留原始语言，以便审核人逐字核对证据。

```bash
export ALGORITHM_AGENT_API_KEY="your_api_key_here"
export ALGORITHM_AGENT_BASE_URL="https://provider.example/v1"
export ALGORITHM_AGENT_MODEL="provider-model-name"
export ALGORITHM_AGENT_STRUCTURED_METHOD="function_calling"
export ALGORITHM_MODULE_AGENT_MODEL="provider-model-name"
export ALGORITHM_MODULE_AGENT_ENABLE_THINKING="false"
uv run streamlit run web/app.py
```

- 点击 Suggest AlgorithmSpec with Agent 后，提取文本才会发送到所配置的服务；
- `ALGORITHM_AGENT_BASE_URL` 和模型名都可替换；直接使用 OpenAI 时 Base URL
  可以省略；
- 默认用兼容范围较广的 `function_calling`，服务支持原生 JSON Schema 时
  可以选择 `json_schema`；
- 模型请求不自动重试，也不自动切换结构化输出方式；失败后由用户手动重试；
- Agent 使用结构化输出建议表单字段，并返回可验证的来源摘录；
- 建议只预填可编辑表单，不会自动创建或批准草稿；
- 再次生成会产生新的模型调用，必须先确认覆盖；系统在当前浏览器会话中保留
  最近五次建议的模型、方法、证据数量、警告和 token 摘要；
- 来源摘录必须能在提取文本中匹配才计入 Verified source evidence。系统允许
  Markdown/PDF 换行空白和字母大小写差异，并保存资料中的实际原文；改写、
  翻译、公式替换仍会被拒绝。数量为 0 时界面显示高风险警告，但仍由专业
  人员决定是否继续；
- 超过 60,000 字符时只提交资料开头和结尾，并在界面显示警告；
- API key 只从环境变量读取，不保存到 manifest；
- LangChain 适配器、模型、结构化输出方式、响应 ID、证据、警告和 token
  用量保存在
  `generation.algorithm_spec_agent`，用于追溯；
- AlgorithmSpec 确认时记录确认人、UTC 时间和规范内容 SHA-256；
- Monte Carlo preset 仍复制内置模块文件；
- Generic profile 创建时不调用模块 Agent，只创建不可发布的确定性脚手架；
  模块 Agent 在 Review Drafts 中按模块单独调用。

三个模块 Agent 都使用 LangChain `ChatOpenAI` 和 OpenAI-compatible 接口。
默认继承 `ALGORITHM_AGENT_*` 连接，也支持以下逐层覆盖：

```text
THEORY_AGENT_MODEL / NOTEBOOK_AGENT_MODEL / EXPERIMENT_AGENT_MODEL
ALGORITHM_MODULE_AGENT_MODEL
ALGORITHM_AGENT_MODEL
```

API key、Base URL、结构化输出方法和 `ENABLE_THINKING` 也使用同样的角色级、
模块共享级、全局级优先顺序。模块生成记录保存在
`generation.module_generations.<module>[]`，包括模型、实际结构化输出方式、
提示词版本、响应 ID、时间、审核意见哈希、警告和 token 用量。
Qwen 模块模型在没有显式配置时默认关闭 thinking，以避免长文件生成耗尽
120 秒请求预算；可以通过角色级或共享级变量显式重新开启。Animation 视频
本阶段仍不由 Agent 生成。

### Animation Creator Guidance

Review Drafts 为每个草稿提供独立的动画制作规划区：

- Provider 先生成三个概念方案，每个方案展示教学重点、视觉方法、时长、复杂度、
  制作成本、适用场景和取舍；
- Provider 明确选择一个概念后，才能生成详细分镜；
- **Create AlgorithmSpec Starter** 不调用模型，按选中方案生成可编辑的分镜与
  制作检查单；
- **Generate with Planning Agent** 使用 LangChain `ChatOpenAI` 和兼容 OpenAI
  的结构化输出接口，按选中方案、AlgorithmSpec、原资料和制作备注细化分镜；
- **Download Creator Kit** 导出英文制作任务书、分镜表、旁白稿、公式核对、
  验收清单和可选工具参考；Creator Kit 不包含要求制作者操作的 JSON；
- 网站不会执行 Manim、FFmpeg、Python 或其他动画代码，也不会后台渲染视频；
- 保存或覆盖制作建议不会创建、修改或解锁 MP4，不改变任何模块审核状态，
  也不阻塞安装；只有上传成品 MP4 后才会声明 Animation 模块并进入审核。

Planning Agent 默认继承 `ALGORITHM_AGENT_*`，并按以下优先顺序覆盖：

```text
ANIMATION_PLANNING_AGENT_*
ANIMATION_AGENT_*
ALGORITHM_MODULE_AGENT_*
ALGORITHM_AGENT_*
OPENAI_API_KEY / OPENAI_BASE_URL
```

制作建议由网页表单编辑，普通用户不直接填写 JSON。系统内部仍将结构化结果
保存在草稿根目录 `animation_guidance.json`，哈希、生成来源、更新时间、
编辑者和 Agent 运行记录保存在 `generation.animation_guidance`。这些内容会写入
不可变审核历史，但不是需要 Approve 的发布模块。

## 草稿与审核规则

- 草稿目录使用 `<algorithm-id>-<semantic-version>` 命名。
- 上传的 MD、TXT 或 PDF 原件保存在草稿 `sources/` 中。
- PDF 仅提取文本，不执行 OCR；空文本、加密或无法解析的 PDF 会被拒绝。
- 本地来源记录 SHA-256，校验时会重新计算并比较。
- Reference URL 只保存，不发起网络请求。
- Approved 模块被锁定，必须先执行 Needs Changes 才能修改或重新生成。
- Generic 新草稿的三个核心模块初始状态为 `not_generated`。每次只生成一个
  模块；生成成功后该模块进入 `awaiting_review`，可以先审核批准，不必等待
  其他模块。
- 模块 Agent 重新生成时读取已确认 AlgorithmSpec、原资料、当前文件和最近的
  审核意见，只覆盖当前模块；其他模块状态不变。
- `algorithm.experiment_design` 保存教学场景及其来源。`source_derived` 表示资料
  明确给出，`platform_preset` 表示平台补充的标准场景，`agent_proposed` 表示
  等待专业复核的模型建议；平台模板不会伪装成上传资料的原文结论。
- 当前提供标准 4×4 FrozenLake 和 4×12 CliffWalking。创建页根据 Agent 提取的
  `supported_environments` 自动推荐，也允许选择“不生成地图，稍后由专家决定”。
- Review Drafts 可以单独更换 Experiment teaching scenario。这个操作只把
  Experiment 置为 `changes_requested`，不会撤销 Theory、Notebook 或 Animation
  的批准状态。
- 如果修改要求不再需要，可填写取消原因并执行 Cancel Change Request；
  当前文件不会被覆盖，模块恢复为 `awaiting_review`，动作会写入审核历史。
- Needs Changes 或 Cancel Change Request 完成后，当前模块保持展开。
- Theory 在线修改后，其状态会重置为 `awaiting_review`。
- Notebook 和 Experiment 可下载当前文件到本地修改，再上传替换文件；系统
  先在临时副本校验，成功后才覆盖当前模块。
- Needs Changes 是审核请求和解锁动作，不是在线代码编辑器。Notebook 和
  Experiment 的两条修订路径是按审核意见重新调用 Agent，或下载后本地修改并
  上传替换。
- Animation 视频不由系统生成。用户可以先生成并下载 Creator Guidance，在
  本机或专业制作环境中完成视频。用户上传完成的 MP4 后，系统校验扩展名、MP4
  文件头和 200 MiB 大小限制，并保存概念说明、公式、符号、重点、观看流程和
  推导步骤元数据。上传视频不会自动理解或改变这些文字。
- 导入 Experiment 可以在 `get_spec().presentation` 中声明 Task 和最多 20×20
  的字符网格，并在 `run().views.policy_grid` 返回状态价值和最优动作。平台在
  父进程校验后统一绘制初始地图和 Learned Policy，算法包不能执行自定义
  Streamlit 页面。
- Installed 中的 Schema v2 算法可以复制为默认补丁版本的修订草稿。原内容
  以 approved 状态带入；修改模块前执行 Needs Changes。修订审核完成后必须
  先软删除旧 ID，再安装新版。
- Animation 未声明时不阻塞安装；一旦添加，就会获得独立审核状态，必须
  Approve 才能安装。Approved Animation 也必须先 Needs Changes 才能替换、
  修改元数据或移除。
- 移除 Animation 会把 MP4 移入 `.trash/draft-assets/`，不是永久删除。
- 完整 AlgorithmSpec JSON 可以在审核页修改；保存后所有模块统一进入
  `changes_requested`。只修改实验教学场景时使用独立场景选择器，仅影响
  Experiment。
- 需要修改和拒绝草稿都必须填写理由。
- 所有审核动作都记录管理员名、前后状态、备注和 UTC 时间。
- Rejected Drafts 页面支持查看拒绝原因、恢复草稿或移入可恢复回收站。

通用模板会生成教学结构和 Experiment 接口脚手架，但不会伪造真实算法实现。
这类草稿初始带有 `placeholder_content` 全局安装阻塞标记。某个模块完成 Agent
生成、Theory 人工编辑或 Notebook/Experiment 上传替换后，即可独立审核批准；
仍是脚手架的模块不能批准。三个核心模块都由 Agent 成功生成并通过文件级
校验后，系统自动清除全局发布锁。人工编辑/上传与 Agent 混合完成时，由
审核人明确确认后清除。该机制验证结构和接口，不代表算法在专业意义上一定
正确。Monte Carlo Control 预置复用仓库中的可信实现。

## 安装与故障处理

三个核心模块以及任何已声明的 Animation 全部批准后，发布服务会：

1. 再次校验 AlgorithmSpec、来源哈希和模块文件；
2. 在临时副本中把模块状态更新为 `installed`；
3. 生成 `dist/<id>-<version>.zip`；
4. 使用 staging 目录安装并通过 `os.replace` 原子发布；
5. 成功后把原草稿移动到可恢复归档。

安装失败时不会删除草稿，也不会覆盖已有算法。已安装包的 Remove 操作仍然
移动到 `algorithm_packages/.trash/`。

## Experiment 运行边界

导入包的 `get_spec()` 和 `run(parameters, reporter)` 不在 Streamlit 主进程
执行。父进程启动独立 Python 子进程，通过进程间消息接收：

- progress；
- metric；
- 最终 result；
- 受控错误。

父进程重新校验参数、事件和结果。默认超时 120 秒，超时或子进程崩溃时终止
该实验并向页面返回错误。该机制用于故障隔离，仍只允许导入可信本地代码。

## 验证

```bash
.venv/bin/python -m unittest discover -s tests -v
```

测试覆盖 v1 回归、v2 Schema、资料提取、草稿状态、发布门禁、Monte Carlo
端到端安装运行、软删除、跨进程进度与指标、超时、崩溃以及 Streamlit 管理页
冒烟测试。
