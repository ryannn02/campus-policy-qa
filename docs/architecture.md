# 架构（现行实现）

> 本文描述**代码的实际状态**，不是设计意图。凡与代码不一致处，以代码为准并回来改本文。
> 最近核对：`data/chunks.jsonl` 3851 块 / 269 份文档，`eval/questions.jsonl` 162 题，
> `data/life/chunks.jsonl` 29 条生活语料（28 条已按 2026 指南核实），`pytest` 180 个用例通过（原 68 + M5 新增 105 + 行政制度路由 2 + 领域词 1 + 生活/寝室路由 1 + 金额方向/职责扩展 3）。

## 一、离线数据流

```
data/raw/*            原始文件（登记 316 条；其中 effective 269 份：pdf 37 / docx 25 / doc 11 / html 183 / txt 13，扫描件 1）
  │  meta/documents.csv  status=effective 的 269 条才进入解析
  │  scripts/ingest.py
  │    · 按扩展名分派解析器：PyMuPDF / python-docx / textutil|soffice / BeautifulSoup / 直读
  │    · PDF 文字层 <120 字判定为扫描件 → RapidOCR 逐页识别，页码按物理页保留
  │    · 清洗：噪声行（ICP 备案、浏览次数、页脚）、页码行、连续重复行、HTML 碎行合并
  ▼
data/parsed/{doc_id}.txt
  │  split_chunks：按「第X条/章/节」切，维护章/节标题栈，条款不跨块，回填 PDF 页码
  │  hard_split：单块上限 800 字，超长段落按句号拆开再贪心聚合
  │  merge_small：<50 字碎片并入上一块，<20 字丢弃
  ▼
data/chunks.jsonl     3851 块 / 269 文档，字段：
                      chunk_id, doc_id, dept, title, year, section, page, text, chars, source_url
  │  scripts/build_index.py：编码文本 = "title\ntext"，bge-small-zh-v1.5，归一化
  ▼
data/index/embeddings.npy（3851×512）+ manifest.json（model / chunks / dim / query_prefix）
```

解析失败明细写 `reports/parse_failures.csv`（当前 0 条）；正文为空的网页条目用
`python -m scripts.ingest --prune-empty` 标为 `excluded`。

## 二、在线问答链路

M5 之后有两条入口：**制度链路（原样保留）** 与 **统一入口（外挂新能力）**。

```
scripts/ask.py（默认 / --smart）        scripts/serve.py（兼容壳）
scripts/eval_answer.py（批量评测）      uvicorn → campus_qa.app.main
                    │                        │
                    │                        ▼
                    │            orchestrator.answer(question, session_id)
                    │                        │
                    │              router.classify（纯规则，兜底 policy）
                    │         ┌──────────┬─────┴─────┬───────────┐
                    │         ▼          ▼           ▼           ▼
                    │      tool       life       chitchat     policy
                    │   tools.run   life.store   service     service
                    │   （实时数据）（独立 BM25）（寒暄短路）   │
                    │                                        ▼
                    └──────────────► service.answer_question（未改一行）
                                              │
                                ┌─────────────┴─────────────┐
                                │ 闸门（顺序短路，命中即返回）│
                                │ empty → 寒暄 → 语言 → 任务型│
                                │ → 歧义 → 检索空 → 低分      │
                                └─────────────┬─────────────┘
                                              ▼
                        retriever.retrieve_candidates(question) -> List[Citation]
                           · 惰性加载 npy + manifest + chunks（长度取 min 对齐）
                           · 问题前置 manifest.query_prefix 后编码（512 维）
                           · 向量路：归一化 → 矩阵点积 → argsort
                           · BM25 路：字符级 2-gram 倒排打分（RETRIEVAL_ENABLE_BM25）
                           · RRF 融合两路 → top_k=5、per_doc=3
                                              ▼
                        chain.generate_answer_with_citations(question, candidates)
                           · 提示词：prompts/answer/v6.md（v6→v5→…→内置模板逐级回退）
                           · ChatPromptTemplate → ChatOpenAI → StrOutputParser
                           · 输出 JSON：{answer, citations:[编号], follow_up_questions:[]}
                                              ▼
                        service 回填：编号 [1..N] → 真实 Citation（片段/页码/URL）
                                              ▼
                                   AnswerResult（contracts.py）
                                   + intent / sources / tool_result（M5 可选字段）
```

**分层红线**：`policy` 分支只做一件事——把原问题原样交给 `service.answer_question`。
不改检索、不改提示词、不改拒答、不拼多轮上下文。生活库、工具、会话都在它旁边。

### M5 新增的四条支路

| 支路 | 实现 | 关键约束 |
| --- | --- | --- |
| 路由 | `router.classify` | 纯规则、可解释；识别不出兜底 `policy`，保证拒答行为与 M3 一致 |
| 生活库 | `life.LifeStore` | 独立语料 `data/life/chunks.jsonl` + 独立 BM25 索引；不参与制度召回 |
| 工具 | `tools/` | 天气（Open-Meteo）、时间、校历、公开电话；失败降级为可读提示，不编造 |
| 多轮 | `conversation.ConversationStore` | SQLite 只存问题/回答/来源；追问只在生活库拼上下文，制度链路不拼 |

生活库索引结构见 `src/campus_qa/life.py` 顶部注释：`doc_count / avg_length /
doc_lengths / postings`，语料变更后 `python -m scripts.build_life_index` 重建。
索引缺失时 `LifeStore` 在内存里就地重建，不阻塞启动。

## 三、LangChain 的边界（与代码一致）

实际只用了 3 个类，且只在 `chain.py`：

| 用到 | 位置 |
| --- | --- |
| `ChatPromptTemplate` | `chain._build_prompt` |
| `ChatOpenAI` | `chain.generate_answer_with_citations` |
| `StrOutputParser` | 同上 |

**没有用**：文档加载器、文本切分器、向量库、`Document`、`BaseRetriever`。
检索层不实现 LangChain 的 `BaseRetriever`，返回值也不是 `Document`，而是
`contracts.Citation`（pydantic）；内部用 `retriever._IndexedChunk` 承载 chunk 元数据。

不用的原因：

- 页码映射。语料需要「按条款切、再回算这一条落在哪一物理页」，通用切分器只能给到整份文件的页码粒度。
- 正文抽取。209 份 HTML 来自不同部门站群，必须按站点选择正文区（`HTML_SELECTORS`）再剔导航与噪声。
- 脏活边界。`.doc` 依赖外部 `textutil`/`soffice` 转换，扫描件依赖 OCR 并按 `(y, x)` 重排阅读顺序，加载器覆盖不到。
- 规模。3851×512 的暴力余弦是毫秒级，引入 FAISS/Chroma 只会多一份需要同步的状态。

> 早期文档曾写「检索结果仍然是标准 `Document`」，与实现不符，此处已更正。
> 若后续确需 LangChain 生态适配（例如接入 LangSmith 的检索追踪），最小改动是给
> `Citation` 加 `to_document()` 或包一层 `BaseRetriever` 适配器，对外契约不变。

## 四、模块职责

| 模块 | 职责 | 关键实现 |
| --- | --- | --- |
| `config.py` | 路径与模型配置 | 绝对路径锚定仓库根；密钥按 `LLM_*→DEEPSEEK_*→OPENAI_*` 回退；`bge-small-zh-v1.5` 自动补 `BAAI/`；`Settings` 为 frozen dataclass，默认 `top_k=5 / per_doc=3 / τ=0.35 / BM25 开、bm25_top_k=10` |
| `contracts.py` | 对外契约 | `RefuseReason` 8 个 code；`Citation`（含 `page/section/url/citation_index`）；`AnswerResult`（含 `follow_up_questions`） |
| `retriever.py` | 召回 | 惰性加载并 `min(chunks, embeddings)` 对齐；`_coerce_page` 清洗空字符串页码；模型加载失败只重试一次后停用；索引/模型不可用退化为 2 条内置样本的字符 Jaccard |
| `chain.py` | 生成 | 版本化提示词；容忍 ```json 围栏与前后废话的 JSON 截取；引用编号去重 + 越界丢弃；`confidence = top1 × (0.5 + 0.5 × 有效引用率)`；非法 JSON 抛 `InvalidModelOutputError` |
| `service.py` | 编排 | 9 条拒答路径（8 个 code，`no_evidence_in_corpus` 规则层与模型层各一）+ 寒暄短路；引用回填并写 `citation_index`；LLM 不可用时退化为原文摘录，流程不中断 |
| `router.py` | 意图路由 | 工具 > 制度 > 生活 > 闲聊 > 兜底制度；命中词随结果返回；追问可继承上一轮意图 |
| `life.py` | 生活库检索 | 字符 2-gram BM25；索引 `data/life/index.json`；索引缺失/损坏就地重建；加载失败不外溢 |
| `orchestrator.py` | 统一入口 | `answer(question, session_id)`；policy 原样调 service；life/tool/chitchat 各自成路；会话落库失败不影响回答 |
| `conversation.py` | 多轮存储 | SQLite `sessions` + `messages`；只存问题/回答/来源；删除会话即删消息；提供最近 N 轮上下文 |
| `safety.py` | 脱敏与隐私 | 手机号/学号/QQ/微信/身份证/姓名规则脱敏；`is_privacy_request` 拦截「查他人信息」；导入与渠道共用 |
| `tools/` | 实时工具 | 天气（Open-Meteo，免 key）、上海时区当前时间、按 `SEMESTER_START` 算教学周、公开电话表 |
| `app/main.py` | Web 服务 | FastAPI；`/api/answer` 兼容旧契约（非法 JSON 仍 400）；SSE 流式；会话 CRUD；静态页 |
| `channels/qq.py` | QQ 渠道 | OneBot 事件解析（text 段 / CQ 码清理）、message_id 去重、群聊按 group_id 复用匿名会话；不落用户身份 |
| `web/` | 前端 | 会话侧栏、SSE 渲染、来源卡片、深浅色、移动端折叠；全部 `textContent` 拼 DOM，无 innerHTML 注入 |

## 五、拒答与短路

`refuse_reason` 是机读 code，给用户看的话放 `answer` 或 `suggested_contact`：

| 顺序 | code | 触发方 | 含义 |
| --- | --- | --- | --- |
| 1 | `empty_question` | 代码 | 输入为空或纯空白 |
| — | *不拒答* | 代码 | 寒暄（你好 / 谢谢 / 你是谁 / 再见）直接礼貌回应 |
| 2 | `unsupported_language` | 代码 | 提问不含中文，或要求英文等非中文输出 |
| 3 | `no_evidence_in_corpus` | 代码 | 任务型请求（帮我写 / 代写 / 写一封） |
| 4 | `ambiguous_question` | 代码 | ≤10 字、含泛化词（奖学金/学分等）且无具体限定 |
| 5 | `retrieval_empty` | 代码 | 没召回任何条款 |
| 6 | `retrieval_low_score` | 代码 | top-1 低于 τ=0.35 |
| 7 | `invalid_model_output` | 代码 | 模型输出不是合法 JSON |
| 8 | `no_evidence_in_corpus` | 模型 | 模型判定条款支撑不住（返回空 answer） |
| 9 | `no_valid_citation` | 代码 | 有答案但没有任何有效引用编号 |

任务型请求与模型自判共用 `no_evidence_in_corpus`：前者是规则层拦截，后者是模型判定，语义上都是「库里没有依据」。

## 六、数据与版本治理

`meta/documents.csv` 是数据清单的权威来源，共 316 行：

| status | 数量 | 含义 | 是否进检索 |
| --- | --- | --- | --- |
| `effective` | 269 | 现行有效 | ✅（3851 块全部来自这批） |
| `archived` | 13 | 已被新版替代，留档追溯 | ❌ |
| `excluded` | 34 | 网页正文为空壳 | ❌ |
| 另：`data/removed/` | 21 份 | 招生/通知/重复/非制度文件，物理移出 | ❌ |

`doc_id` 由采集阶段生成，`ingest` 不重算，保证与已建索引稳定对齐；`archived` 文档
（如 2011《图书借阅规定》）因此天然不进索引，新旧交替靠登记表状态而非运行时过滤。

**已知缺口**：运行时不过滤年份或生效日期，`retriever`/`service` 均不读 `year`/`status`，
同主题多个 `effective` 版本可能同时进入候选（例如「学生手册（2023 版）」与
「图书外借管理办法（2026 修订）」可同现于一次回答）。

引用定位字段的结构性缺失（非解析错误）：

| 字段 | 缺失 | 原因 | 兜底 |
| --- | --- | --- | --- |
| `page` | 2269 / 3851 | 仅 PDF 有页码；HTML、Word、人工整理件无页码概念 | 前端展示 `section`（章 / 节 / 标题） |
| `source_url` | 3745 / 3851 | 仅「人工整理」归纳件无官网来源 | 仅展示文档标题 |

## 七、评测现状

`python -m scripts.eval_answer` 跑 `eval/questions.jsonl`（162 题，四类：answerable 101 /
unanswerable 36 / multi-hop 15 / ambiguous 10，gold 多版本标注覆盖 100%），指标写入 `eval/report.md`：

| 指标 | 当前 | 目标 | 状态 |
| --- | --- | --- | --- |
| `answer_rate` | 1.000 | — | — |
| `refusal_rate` | 0.913 | ≥0.50 | ✅ |
| `refuse_failure_rate` | 0.025 | ≤0.15 | ✅ |
| `grounded_rate` | 0.853 | ≥0.80 | ✅ |
| `recall_at_5` | 0.853 | ≥0.75 | ✅ |

口径说明：

- `recall_at_5` / `grounded_rate` 只在「可回答题且带 gold」的题上计算（分母 116 条）；`refuse_failure_rate` 以全量为分母。
- `grounded_rate` 采用 snippet 精确匹配（答案引用 chunk 全文须与 gold 标注 snippet 逐字一致）。为消除「同一制度条款在多份文件重复收录、条款号不同」导致的假阴性，gold 已做**多版本标注**（273 个元素 / 162 题），命中任一版本即算命中。
- 五项基础档指标全部达标（v5 提示词：恢复多引用 + 制度库外内容置空；检索跨文档同主题去重）。
- 误答用例归因见 `eval/badcase.md`。

M5 新增两套**不参与制度指标**的评测集，用于回归新链路：

| 文件 | 内容 | 口径 |
| --- | --- | --- |
| `eval/questions_daily.jsonl` | 37 条日常题（食堂/快递/校园卡/天气/电话/闲聊/制度） | 断言 `intent` 命中 + 关键信息出现在答案 |
| `eval/questions_multi_turn.jsonl` | 8 组多轮（追问继承、话题切换） | 断言每轮意图与上下文承接 |

制度指标只跑 `eval/questions.jsonl`，新增题集不会稀释 `refusal_rate` / `Recall@5`。

## 八、里程碑与后续

- **M1**（已验收）：从 CLI 输入中文问题，端到端输出结构化 `AnswerResult`。
- **M2**（已验收）：拒答收紧、评测集扩至 162 题、OCR 与遗留文件处置、可点击引用锚点、动态追问（见 `docs/m2_plan.md`）。
- **M3**（已验收，4/5 指标达标）：BM25 + 向量多路召回（RRF 融合）、gold 标注 100%、领域相关性拒答判定。`Recall@5` 0.802 ✅、`refuse_failure_rate` 0.019 ✅、`refusal_rate` 0.935 ✅；`grounded_rate` 0.776 距 0.80 差 0.024，如实记录、接受现状。
- **M4 剩余项**：
  1. `grounded_rate` 冲刺 0.80：提示词强化「优先引用与问题最直接相关的条款编号」（成员 B）；
  2. 运行时无版本时效过滤（见第六节缺口，成员 A/B）；
  3. 无 lint / 类型检查 / CI，测试靠本地 `python -m pytest -q`（成员 E）。
- **M5**（已落地）：意图路由（制度/生活/工具/闲聊）、独立生活库（BM25）、
  实时工具（天气/时间/校历/电话）、SQLite 多轮会话、FastAPI + SSE + 新前端、
  校园墙问题提炼管线、OneBot 渠道适配、脱敏与隐私边界、Docker 部署。
  **制度核心三件套（`service` / `retriever` / `chain`）未改动。**
- **M5 待接**：校园墙的具体入口（QQ 号 / 群号 / 小程序 / 接口）尚未确认，
  `scripts/crawl_wall.py` 已支持 HTTP 接口与离线 JSONL 两种来源，入口确认后
  只需补一个 fetcher；`channels/qq.py` 的 `run_forever` 同样等 OneBot 接入方式。

## 九、从旧项目继承

- `data/chunks.jsonl` 的初始版本（旧项目 `p1-rag` 产出，现已被 `scripts/ingest.py` 重跑覆盖为 3851 块 / 269 份）
- 拒答 code 与「引用编号由代码回填」的规则
- 提示词正文的初版思路（现行 `prompts/answer/v1~v3.md` 已在本仓库版本化管理）

旧项目 `p1-rag` 保留不动，作为对照与数据来源。
