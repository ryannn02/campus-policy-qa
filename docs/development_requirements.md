# 校园制度问答助手开发需求

> 对应文档：`docs/requirements.md`（需求分析）  
> 本文目标：明确开发环境、技术栈、模块划分、里程碑与编码约束

## 一、开发环境与技术栈

- 编程语言与运行环境
  - Python：版本要求 `>=3.11`（见 `pyproject.toml`）。
  - 包管理：推荐使用 `uv` 或 `pip`，按 README 中的 `uv sync --extra dev` 完成依赖安装。

- 主要依赖库（核心运行时）
  - `langchain-core`：构建生成链、管理 Prompt 与输出解析。
  - `langchain-openai`：对接大模型 API（ChatCompletion/JSON 输出）。
  - `langchain-huggingface`：如需对接本地或 HuggingFace 模型。
  - `sentence-transformers`：中文向量化模型（例如 bge-small-zh-v1.5）。
  - `numpy`：向量运算与检索得分计算。
  - `pydantic`：数据模型与接口契约（如 `AnswerResult`、`Citation`）。
  - `python-dotenv`：加载 `.env` 中的配置与密钥。

- 可选依赖库（数据解析与开发）
  - `beautifulsoup4`、`pypdf`、`pymupdf`、`python-docx`：用于 PDF/Word/网页等原始制度文档的解析与文本抽取（在 `ingest` extra 中定义）。
  - `pytest`：单元测试与回归测试（在 `dev` extra 中定义）。

- 环境配置
  - `.env` 文件中至少包含：
    - `LLM_API_KEY`：大模型 API 密钥；
    - `CHAT_MODEL`：聊天模型名称（默认可使用 `gpt-4.1-mini`）；
    - `EMBEDDING_MODEL`：向量模型名称（默认 `bge-small-zh-v1.5`）。
  - 配置通过 `src/campus_qa/config.py` 中的 `Settings` 加载与集中管理。

## 二、模块划分与开发任务

### 2.1 核心包 `src/campus_qa`

- `config.py`
  - 职责：集中管理路径、模型名称、检索参数等基础配置。
  - 开发任务：
    - 已实现最小版 `Settings` 与 `settings` 实例，后续如有新参数应统一加在此处；
    - 保持配置简洁，不引入额外配置框架。

- `contracts.py`
  - 职责：定义 AnswerResult、Citation 等接口契约，以及拒答原因枚举 `RefuseReason`。
  - 开发任务：
    - 已实现基础模型与枚举；
    - 后续若扩展字段（如多路检索来源标签），需要保证向后兼容。

- `retriever.py`
  - 职责：完成制度条款的向量化与检索逻辑，返回标准 `Document` 对象与检索得分。
  - 开发任务：
    - 实现 bge 向量化和本地向量索引（暴力余弦即可满足当前数据量）；
    - 支持 Top-K 和 per_doc 限额；
    - 为拒答逻辑提供检索侧信号（如 `top_retrieval_score`）。

- `chain.py`
  - 职责：构建 LangChain 生成链：`ChatPromptTemplate → ChatOpenAI → StrOutputParser`。
  - 开发任务：
    - 设计和实现提示词模板（带版本号，存放于 `prompts/answer/`）；
    - 保证输出为合法 JSON 字符串，便于解析为 `AnswerResult`。

- `service.py`
  - 职责：编排整体问答流程：语言判断 → 检索 → 生成 → 引用回填 → 拒答判定。
  - 开发任务：
    - 提供主入口函数（例如 `answer(question: str) -> AnswerResult`）；
    - 集成 `config`、`contracts`、`retriever`、`chain` 四个模块；
    - 实现基础拒答策略，与 `RefuseReason` 枚举保持一致。

### 2.2 脚本与接口 `scripts/`

- `scripts/build_index`
  - 职责：从原始制度文档出发，生成 `data/chunks.jsonl` 与向量索引 `data/index/`。
  - 开发任务：
    - 调用解析脚本，将 PDF/Word/网页转为纯文本；
    - 按文件/章节/页码切分，再进行向量化与索引构建。

- `scripts.ask`
  - 职责：命令行问答入口，便于本地调试和回归测试。
  - 开发任务：
    - 从命令行读取问题，调用 `service.answer()`；
    - 将结果以 JSON 形式打印（包含 `answer`、`citations`、`refused` 等字段）。

- `scripts.serve`
  - 职责：提供 Web 界面与 JSON API。
  - 开发任务：
    - 选择轻量 Web 框架（如 FastAPI 或 Starlette）；
    - 实现问题输入端点和对应的 HTML 页面；
    - 确保返回结构与 `AnswerResult` 对齐。

- `scripts.eval_answer`
  - 职责：评测脚本入口，一键跑 Recall@5、有据可依率、拒答率、幻觉率等指标。
  - 开发任务：
    - 加载评测集 JSON；
    - 调用系统问答接口批量执行；
    - 汇总指标并输出 Markdown 报告与图表。

### 2.3 文档与测试

- 文档 `docs/`
  - 已有：`architecture.md`（架构说明）、`requirements.md`（需求分析）、`team_roles.md`（团队分工）。
  - 开发任务：
    - 后续补充部署说明、评测报告模板、badcase 归因表样例等。

- 测试 `tests/`
  - 现状：仅 `.gitkeep`，尚未创建具体测试文件。
  - 开发任务：
    - 首先为核心路径（检索、生成链、拒答逻辑）编写少量单元测试；
    - 确保关键逻辑在改动后可快速回归，而不追求全面覆盖。

## 三、开发里程碑与优先级

- 里程碑 M1：最小端到端原型
  - 完成 `config.py` 与 `contracts.py` 的基础实现（已完成初版）；
  - 在 `retriever.py`、`chain.py`、`service.py` 中实现一条最小可用的问答链；
  - 实现 `scripts.ask`，从命令行输入问题并获得带引用或拒答的 `AnswerResult`。

- 里程碑 M2：数据与索引流水线
  - 完成制度文档解析（至少覆盖部分代表性文档）；
  - 构建 `data/chunks.jsonl` 与 `data/index/`；
  - 对检索效果进行初步人工评估，修正切分策略与向量化参数。

- 里程碑 M3：Web/API 与评测体系
  - 完成基础 Web 界面和 JSON API；
  - 构建 ≥150 条评测集，并实现 `scripts.eval_answer`；
  - 生成首版评测报告，包含基础档指标（Recall@5、有据可依率、拒答率、幻觉率）。

- 里程碑 M4：优化与进阶功能
  - 根据评测结果调整检索与拒答阈值；
  - 有精力时引入 BM25 + 多路召回 + 重排；
  - 尝试查询改写与意图澄清，补充 badcase 归因与改进记录。

## 四、编码约束与开发原则

- 依赖与架构约束
  - 优先使用现有依赖和标准库实现需求，不随意增加新依赖；
  - LangChain 仅用于生成链与基础检索接口适配，不使用其文档加载器、切分器、向量库等重组件。

- 代码风格与可维护性
  - 使用类型注解（type hints）提高可读性与 IDE 友好度；
  - 核心数据结构通过 `pydantic` 或 `dataclass` 明确建模；
  - 避免过度抽象与提前设计复杂扩展点，先满足当前指标与场景。

- 测试与质量保障
  - 对非平凡逻辑（检索、拒答、解析）至少保留一个可运行的测试或自检函数；
  - 评测脚本视作「系统级回归测试」，在重要改动后运行一次并记录结果。

- 文档与协作
  - 与团队分工保持一致：各模块负责人在开发过程中同步更新 `docs/` 下相关说明；
  - 组长负责整合架构与需求，确保实现与 `requirements.md` 对齐。

