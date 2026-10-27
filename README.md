# 校园制度问答系统

上海第二工业大学校园制度问答助手：基于 LangChain + RAG，为师生提供有出处的制度条款回答。

**当前状态**：M1 里程碑已跑通。CLI / Web API / 评测脚本均可运行：bge 向量检索 → 阈值闸门 → LLM 生成 → 引用回填 → `AnswerResult`。拒答策略仍偏松（评测集 8 道应拒答题里答出 4 道、拒答率 0.50，见 `eval/report.md`），属 M2 待优化项。

首个里程碑：从 CLI 输入一个中文问题，到输出一条结构化的 `AnswerResult`，端到端跑通；首轮只追求"能跑 + 能演示"，检索/生成指标与进阶优化放到第二阶段。

## 目录

```
data/raw/        原始文件（pdf/docx/html/.doc），不入 Git
data/parsed/     解析后的纯文本，不入 Git
data/chunks.jsonl  切分后的条款（4190 块），入库
data/index/      向量索引，不入 Git，可重建
meta/            采集登记表 documents.csv（doc_id ↔ 来源 ↔ 状态），入库
eval/            测试问题集与回归用例
prompts/answer/  提示词，带版本号，v1 永不删
scripts/         命令行入口：ingest 解析、建索引、问答、评测、网页演示
src/campus_qa/   核心包
tests/           单元测试
docs/            架构与项目文档
```

## 分层

| 模块 | 职责 |
| --- | --- |
| `config.py` | 读 `.env`，集中路径与模型配置 |
| `contracts.py` | `AnswerResult` / `Citation` / 拒答 code，接口契约 |
| `retriever.py` | bge 向量化 + 本地索引召回，`per_doc` 限额 |
| `chain.py` | LangChain 生成链：`ChatPromptTemplate → ChatOpenAI → StrOutputParser` |
| `service.py` | 编排：语言/分数闸门 → 生成 → 引用回填 → 拒答 |
| `scripts/` | `ingest` / `build_index` / `ask` / `serve` / `eval_answer` |

## 快速开始

```bash
uv sync --extra dev                    # 安装依赖；或 .venv 已就绪则跳过
source .venv/bin/activate              # 激活虚拟环境（后续 python 命令在此环境执行；或用 uv run 前缀）
uv sync --extra ingest                 # 仅当要重新解析 data/raw 原始文件时安装解析依赖
cp .env.example .env                   # 若无 .env.example，按下方变量表自建 .env
python -m scripts.ingest               # 仅当 data/raw 有增删时：重建 data/parsed + chunks.jsonl
python -m scripts.build_index          # 仅当 data/index 缺失或语料变更时执行
python -m scripts.ask "图书馆能借几本书？"
python -m scripts.ask --json "图书馆能借几本书？"   # 输出 AnswerResult JSON
python -m scripts.serve                # http://127.0.0.1:8000（/health 存活、/api/stats 看 P95）
python -m scripts.eval_answer          # 跑 eval/questions.jsonl 并输出指标
python -m pytest -q                    # 单元测试
```

> 数据流水线（成员 A 负责）：`data/raw` 原始文件 → `python -m scripts.ingest` →
> `data/parsed/*.txt` + `data/chunks.jsonl`。`ingest` 只消费 `meta/documents.csv`
> 中 `status == "effective"` 的条目，不重新生成 `doc_id`，因此 `doc_id` 与向量
> 索引保持稳定对齐。语料增删请同步更新登记表与 `docs/data_manifest.md`。

配置读取顺序（`src/campus_qa/config.py`）：

| 项 | 优先级 |
| --- | --- |
| 密钥 | `LLM_API_KEY` → `DEEPSEEK_API_KEY` → `OPENAI_API_KEY` |
| 地址 | `LLM_BASE_URL` → `DEEPSEEK_BASE_URL` / `OPENAI_BASE_URL` |
| 模型 | `CHAT_MODEL` → `DEEPSEEK_MODEL` / `OPENAI_MODEL` |
| 向量模型 | `EMBEDDING_MODEL` → `EMBED_MODEL`，短名自动补 `BAAI/` |

当前 `.env` 实际走的是 DeepSeek（`deepseek-chat`）；`OPENAI_MODEL=gpt-4` 已被上游下线，切回 OpenAI 需改成在售模型名。

生成链不可用（无密钥 / 401 / 超时）时，`service` 退化为「原文摘录」回答，流程不中断；向量模型不可用时检索退化为内置样本的字符级匹配，并按低分拒答。

## Web / API

| 端点 | 说明 |
| --- | --- |
| `GET /` | 单页问答界面（输入框、答案、引用跳转原文、拒答提示） |
| `POST /api/answer` | 输入 `{"question": "..."}`，返回 `AnswerResult` JSON |
| `GET /health` | 存活探针 `{"status": "ok", ...}` |
| `GET /api/stats` | 问答时延统计（avg / P50 / P95 / P99，见 `docs/web_api_contract.md`） |

P95 响应时延目标：基础档 ≤8s、进阶档 ≤5s（见 `docs/requirements.md` 非功能需求）。

## 设计约束

- 引用元数据由代码回填，模型只能挑编号，编造引用在结构上不可能
- 没有依据就拒答，拒答理由是机读 code，不是自由文本
- 非中文提问暂时直接拒答
- 提示词进仓库、带版本号，改动新建 `vN+1.md`
