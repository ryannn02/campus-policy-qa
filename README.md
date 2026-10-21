# 校园制度问答系统

上海第二工业大学校园制度问答助手：基于 LangChain + RAG，为师生提供有出处的制度条款回答。

**当前状态**：项目骨架（目录、依赖、接口边界）。业务代码尚未开始。

## 目录

```
data/raw/        原始文件（pdf/docx/html/.doc），不入 Git
data/parsed/     解析后的纯文本，不入 Git
data/chunks/     chunks.jsonl，入库
data/index/      向量索引，不入 Git，可重建
eval/            测试问题集与回归用例
prompts/answer/  提示词，带版本号，v1 永不删
scripts/         命令行入口：建索引、问答、评测、网页演示
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
| `scripts/` | `build_index` / `ask` / `serve` / `eval_answer` |

## 快速开始（待实现）

```bash
uv sync --extra dev
cp .env.example .env          # 填 LLM_API_KEY
python -m scripts.build_index
python -m scripts.ask "图书馆能借几本书？"
python -m scripts.serve       # http://127.0.0.1:8000
```

## 设计约束

- 引用元数据由代码回填，模型只能挑编号，编造引用在结构上不可能
- 没有依据就拒答，拒答理由是机读 code，不是自由文本
- 非中文提问暂时直接拒答
- 提示词进仓库、带版本号，改动新建 `vN+1.md`
