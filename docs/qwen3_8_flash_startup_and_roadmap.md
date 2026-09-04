# Qwen3.8-Flash 启动、现状与后续实现计划

本文面向本项目当前采用阿里云百炼 `qwen3.8-flash` 的运行方式，说明如何配置、
启动和验证 Agent，同时记录当前实现边界及建议的后续开发顺序。

## 1. 推荐配置

当前建议统一使用以下组合：

```text
模型：qwen3.8-flash
接口：阿里云百炼 OpenAI-compatible Chat Completions
结构化输出：JSON Schema
思考模式：关闭
```

选择该组合的原因：

- 当前任务主要是资料提取、教学文本生成、Notebook 和 Experiment 草稿生成，
  `qwen3.8-flash` 的成本和延迟更适合作为默认模型；
- 项目依赖严格的 Pydantic 输出结构，`qwen3.8-flash` 支持 JSON Schema；
- 当前生成流程是单步结构化生成，并非需要长链推理的自主 Agent，关闭 thinking
  可以减少延迟和输出 Token；
- 当前模型输入全部是提取后的文本，多模态能力暂时不是运行前提。

## 2. 前置条件

1. 已开通阿里云百炼，并获得对应业务空间的 API Key；
2. 已取得业务空间 ID（WorkspaceId）；
3. 本机已安装 `uv`；
4. 当前目录为项目根目录。

首次运行或依赖发生变化时执行：

```bash
uv sync
```

## 3. 环境变量

### 3.1 华北 2（北京）推荐配置

将下面的 `<your-api-key>` 和 `<your-workspace-id>` 替换为实际值：

```bash
export DASHSCOPE_API_KEY="<your-api-key>"
export ALGORITHM_AGENT_API_KEY="$DASHSCOPE_API_KEY"
export ALGORITHM_AGENT_BASE_URL="https://<your-workspace-id>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

export ALGORITHM_AGENT_MODEL="qwen3.8-flash"
export ALGORITHM_AGENT_STRUCTURED_METHOD="json_schema"
export ALGORITHM_AGENT_ENABLE_THINKING="false"
```

项目本身读取 `ALGORITHM_AGENT_API_KEY` 或 `OPENAI_API_KEY`，不会直接读取
`DASHSCOPE_API_KEY`，所以上面的映射不能省略。

`ALGORITHM_AGENT_*` 是全局默认配置。Theory、Notebook、Experiment 和
Animation Planning Agent 会继承这些连接、模型、结构化输出和 thinking 配置，
因此当前不需要为每个 Agent 重复设置。AlgorithmSpec Agent 当前固定发送
`enable_thinking=false`，尚未读取同名 thinking 环境变量，但最终行为同样是关闭。

### 3.2 其他地域

API Key 与地域绑定，Base URL 必须与 API Key 所属地域一致。常用地址如下：

```text
北京：https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
新加坡：https://<WorkspaceId>.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
美国：https://<WorkspaceId>.us-east-1.maas.aliyuncs.com/compatible-mode/v1
德国：https://<WorkspaceId>.eu-central-1.maas.aliyuncs.com/compatible-mode/v1
日本：https://<WorkspaceId>.ap-northeast-1.maas.aliyuncs.com/compatible-mode/v1
```

旧的北京公共兼容地址仍可用于兼容场景：

```bash
export ALGORITHM_AGENT_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

生产环境优先使用业务空间专属域名。

### 3.3 按模块覆盖模型

只有在评测证明某个模块需要更强模型时才建议覆盖。例如保留 Flash 作为默认，
只把 Notebook 和 Experiment 提升到 Max：

```bash
export NOTEBOOK_AGENT_MODEL="qwen3.8-max"
export EXPERIMENT_AGENT_MODEL="qwen3.8-max"
```

角色级配置优先于模块共享配置，模块共享配置优先于全局配置：

```text
THEORY_AGENT_* / NOTEBOOK_AGENT_* / EXPERIMENT_AGENT_*
ALGORITHM_MODULE_AGENT_*
ALGORITHM_AGENT_*
```

Animation Planning 的覆盖顺序为：

```text
ANIMATION_PLANNING_AGENT_*
ANIMATION_AGENT_*
ALGORITHM_MODULE_AGENT_*
ALGORITHM_AGENT_*
```

## 4. 启动应用

环境变量必须和启动命令位于同一个 shell 会话。配置完成后执行：

```bash
uv run streamlit run web/app.py
```

也可以使用已经创建的虚拟环境：

```bash
source .venv/bin/activate
streamlit run web/app.py
```

仓库虽然忽略 `.env` 文件，但当前代码没有调用 `load_dotenv()`，因此仅创建
`.env` 不会自动生效。如果把变量保存在本地 `.env` 中，需要先通过可信方式将其
导入 shell；不要把 API Key 写入 README、源码、草稿或提交记录。

## 5. 首次运行验证

启动后进入 **Manage Algorithms**，按以下顺序做最小验证：

1. 在 **Create v2 Draft** 上传一个较短的 UTF-8 Markdown 或 TXT 文件；
2. 确认页面显示 Agent 已配置，模型为 `qwen3.8-flash`，Endpoint 为自定义地址；
3. 点击 **Suggest AlgorithmSpec with Agent**；
4. 检查返回内容是否成功填入表单、是否包含可匹配的来源证据；
5. 确认 AlgorithmSpec 并创建 Generic 草稿；
6. 在 **Review Drafts** 分别生成 Theory、Notebook 和 Experiment；
7. 检查每个模块的模型名、实际结构化方式、Token 用量和警告；
8. 完成审核后再安装草稿。

建议第一次只处理短文本，先排除 API Key、地域、WorkspaceId 和模型权限问题。

## 6. 常见问题

### Agent 显示未配置

确认当前 shell 中存在项目实际读取的变量：

```bash
test -n "$ALGORITHM_AGENT_API_KEY" && echo "API key is set"
test -n "$ALGORITHM_AGENT_BASE_URL" && echo "$ALGORITHM_AGENT_BASE_URL"
echo "$ALGORITHM_AGENT_MODEL"
```

不要输出 API Key 本身。

### 返回 401 或 403

- 检查 API Key 是否属于 Base URL 对应地域；
- 检查业务空间是否已经开通 `qwen3.8-flash`；
- 检查 WorkspaceId 是否填写正确。

### JSON Schema 请求失败

先确认模型名确实为 `qwen3.8-flash`。临时兼容可以改为：

```bash
export ALGORITHM_AGENT_STRUCTURED_METHOD="function_calling"
```

项目不会自动重试或自动切换结构化方式。失败时页面保留现有草稿文件并显示
request ID；如需改用 `json_mode`，请显式修改配置并重新启动后手动重试。

### 请求超时或生成耗时过长

确认已关闭 thinking：

```bash
export ALGORITHM_AGENT_ENABLE_THINKING="false"
```

Qwen 模块模型在没有显式配置时也会默认关闭 thinking，但生产启动脚本仍建议
明确设置，避免模型识别或配置继承变化造成行为漂移。

## 7. 当前实现情况

### 7.1 已实现能力

| 能力 | 当前状态 |
| --- | --- |
| OpenAI-compatible 接入 | 已实现，基于 LangChain `ChatOpenAI` |
| Qwen thinking 开关 | 已实现，通过 `extra_body.enable_thinking` 传递 |
| 全局、模块、角色级模型覆盖 | 已实现 |
| AlgorithmSpec 结构化建议 | 已实现，包含 Pydantic 校验和来源证据核验 |
| Theory、Notebook、Experiment 独立生成 | 已实现，可单独重生成和审核 |
| Animation Creator Guidance | 已实现确定性版本和 Planning Agent 版本 |
| 人工确认与分模块审核 | 已实现 |
| 生成追踪 | 已记录模型、方法、响应 ID、Token、警告和时间 |
| 安装前校验和软删除 | 已实现 |

整体已经形成可运行的 Human-in-the-loop MVP：模型生成的内容不会自动发布，
AlgorithmSpec 需要人工确认，各模块需要分别审核。

### 7.2 当前输入和上下文边界

- 来源文件只支持 Markdown、UTF-8 TXT 和可提取文本的 PDF；
- PDF 通过 `pypdf` 提取文本，不执行 OCR；
- AlgorithmSpec 最多向模型提交 60,000 字符，保留开头和结尾；
- 模块生成最多提交 40,000 字符来源和 24,000 字符当前模块；
- Animation Planning 最多提交来源前 30,000 字符；
- 没有文档切块、检索增强生成（RAG）或向量数据库。

因此，模型即使提供 1M 上下文，当前实现也不会把完整大型资料直接发送给它。
现阶段长上下文不是系统的主要瓶颈。

### 7.3 当前多模态边界

- 图片、扫描 PDF 页面不会发送给模型；
- PDF 中无法提取的公式、图表和示意图会丢失；
- MP4 只上传、校验、预览和安装，Agent 不分析视频内容；
- Animation Planning Agent 只读取文本、AlgorithmSpec 和人工备注。

所以当前系统实质上是纯文本 Agent 工作流。`qwen3.8-flash` 的多模态能力尚未被
使用，但为后续 OCR、图表理解和动画审核保留了模型能力空间。

### 7.4 工具调用现状

当前 `function_calling` 主要被 LangChain 用作结构化输出传输方式，而不是业务
工具系统。Agent 尚未获得联网搜索、文件检索、代码解释器或任意 shell 工具。

Experiment Agent 生成 Python 源码，但模型不会在生成过程中执行代码。生成后
由项目的校验和独立子进程运行机制负责验证，人工审核仍是发布前提。

### 7.5 已知技术债务

1. 三类 Agent 分别创建 `ChatOpenAI`，连接和模型参数存在重复实现；
2. Qwen 的识别依赖模型名以 `qwen` 开头，而不是显式 provider 配置；
3. AlgorithmSpec Agent 固定关闭 thinking，模块与 Planning Agent 使用环境变量，
   配置模型不完全统一；
4. 所有结构化方式失败后都需要人工重试或显式更改配置；
5. 没有启动前 API 连通性、模型权限和能力探测；
6. 没有统一记录请求延迟、失败类型和估算成本；
7. 长资料使用固定截断，可能遗漏正文中部的重要证据；
8. 目前没有基于真实教学样本的模型质量回归评测集。

## 8. 后续实现计划

### 阶段一：统一 Qwen 配置与启动诊断

目标：让错误在首次生成前被发现。

- 新增统一的模型工厂，三类 Agent 复用相同连接、超时、重试和 thinking 配置；
- 引入显式 `ALGORITHM_AGENT_PROVIDER=qwen`，不再只根据模型名前缀判断；
- 增加安全的配置诊断页或命令，只显示变量是否存在，不显示 API Key；
- 增加一次低成本连通性测试，检查 Endpoint、模型权限和结构化输出能力；
- 增加可提交的 `.env.example`，但继续禁止提交真实 `.env`；
- 为配置继承、thinking 开关和地域错误补充测试。

验收标准：新环境能够在一次诊断中明确区分缺少 Key、地域不匹配、模型未开通、
JSON Schema 不支持和网络超时。

### 阶段二：增强结构化输出可靠性

目标：减少生成失败和人工重试。

- 建立 `json_schema → function_calling → json_mode` 的受控降级链；
- 每次降级记录原始失败类别和最终实际方法；
- 对空响应、Schema 不匹配和截断响应采用有限次数重试；
- 为每类输出设置合理的最大输出 Token，避免 Notebook 或 Experiment 被截断；
- 对 Qwen3.8-Flash 建立固定版本的回归样例。

验收标准：代表性测试资料的结构化解析成功率可量化，并且所有自动降级都可追踪。

### 阶段三：质量、成本和模型路由评测

目标：用数据决定是否升级模型。

- 建立覆盖 AlgorithmSpec、Theory、Notebook、Experiment 的小型金标准数据集；
- 记录成功率、专家修改量、延迟、输入/输出 Token 和估算费用；
- 对比 `qwen3.8-flash` 与 `qwen3.8-max`；
- 默认继续使用 Flash，只在困难模块或首次失败后选择性升级 Max；
- 避免无条件多次生成，优先使用审核意见做定向修订。

验收标准：模型升级由质量提升或失败率下降触发，而不是依赖主观印象。

### 可选阶段四：仅在长资料确有遗漏时改进文本选取

这不是当前 MVP 的既定范围，也不要求引入向量数据库。只有真实用户上传的长
资料反复出现“正文中部关键信息被截断”的问题时，才启动这一阶段。

目标：用简单、可解释的方法改进固定的头尾截断。

- 对 Markdown、TXT 和 PDF 提取文本进行分块；
- 优先使用标题、页码、关键词和 BM25 等本地文本检索选择相关片段；
- 为 AlgorithmSpec 字段保留页码或段落定位；
- 模块生成只注入与当前任务有关的片段；
- 保持证据必须能够回查原文的约束；
- 不默认引入向量数据库。只有简单检索经过评测仍无法满足需求时，才单独论证
  是否需要 embedding 和向量存储。

验收标准：长文档正文中部的重要信息能够被检索并形成可验证证据，同时请求
Token 不随原文件长度线性增长。

### 阶段五：按需求引入多模态

目标：只为明确场景增加多模态成本和复杂度。

建议顺序：

1. 扫描 PDF OCR 和页面级引用；
2. 论文公式、表格和环境示意图理解；
3. 图片与提取文本的联合证据；
4. 最后再评估 MP4 抽帧和动画内容一致性审核。

验收标准：多模态结果必须能定位到具体页面、图片或视频时间点，且仍需人工审核。

## 9. 当前推荐决策

近期继续使用 `qwen3.8-flash + json_schema + enable_thinking=false`，不要因为模型
支持 1M 上下文就扩大单次输入。优先完成配置统一、结构化输出降级和质量评测。
当前不引入 RAG 或向量数据库；只有真实长资料测试证明固定截断影响质量后，才
先尝试分段和本地关键词/BM25 检索。只有需要扫描 PDF、图表或视频审核时，再
接入多模态输入。
