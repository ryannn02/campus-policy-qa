# 校园制度问答系统

上海第二工业大学校园制度问答助手：基于 LangChain + RAG，为师生提供有出处的制度条款回答。

**当前状态**：M5 已落地。制度问答仍是原来的 RAG 链路（当前离线口径：`answer_rate` 1.000、`refusal_rate` 0.913、`refuse_failure_rate` 0.025、`Recall@5` 0.853、`grounded_rate` 0.853）；在此之上新增意图路由（制度/生活/工具/闲聊）、独立的校园生活库（BM25）、实时工具（天气/时间/校历/电话）、SQLite 多轮会话、FastAPI + 流式输出 + 新前端，以及校园墙「问题提炼」管线与 QQ 渠道适配。`pytest` 180 个用例通过。

首个里程碑：从 CLI 输入一个中文问题，到输出一条结构化的 `AnswerResult`，端到端跑通；首轮只追求"能跑 + 能演示"，检索/生成指标与进阶优化放到第二阶段。

**2026-11-24 数据更新**：制度库换版/增补 11 份文件——学生侧 5 份（学生转专业实施办法 2026 年修订、非学历继续教育项目管理实施细则 2026 年修订、研究生学业奖学金评定管理办法 2024 年修订、研究生学术成果奖学金实施细则、本专科生国家助学金实施办法 2024 年修订）；行政后勤侧 6 份（采购管理办法 2026 年修订、部门采购实施细则 2026 年修订、货物与服务类采购合同管理办法 2026 年修订、基本建设项目管理暂行办法修订、房屋设施维修项目管理暂行办法修订、房屋设施零星维修项目管理办法 2026 年修订）。同时将 2019/2021 三份基建、维修制度汇编（746 块）转为 `archived` 并从索引下线，避免旧版压过 2026 新档。当前语料 3851 块 / 269 份现行文档，`Recall@5` 0.853（99/116），行政新档首条引用 10/10。生活库 29 条按《2026 上工大校园生活服务指南》逐条校对：28 条已核实、1 条待现场确认，明细见 `data/life/verification_report.md`。

## 目录

```
data/raw/        原始文件（pdf/docx/html/.doc），不入 Git
data/parsed/     解析后的纯文本，不入 Git
data/chunks.jsonl  切分后的条款（3851 块 / 269 份文档），入库
data/index/      制度库向量索引，不入 Git，可重建
data/life/       校园生活库语料 chunks.jsonl + BM25 index.json（与制度库分离），校对报告见 verification_report.md
data/wall/       校园墙提炼后的问题集与抓取报告（不含发帖人信息）
meta/            采集登记表 documents.csv（doc_id ↔ 来源 ↔ 状态），入库
eval/            测试问题集与回归用例（含日常题 / 多轮题）
prompts/answer/  提示词，带版本号，v1 永不删
scripts/         命令行入口：ingest / build_index / build_life_index / crawl_wall / ask / serve / eval_answer
src/campus_qa/   核心包（含 app/ FastAPI 与 web/ 前端、channels/ 渠道）
tests/           单元测试
docs/            架构与项目文档
```

## 分层

| 模块 | 职责 |
| --- | --- |
| `config.py` | 读 `.env`，集中路径与模型配置（含生活库 / 会话库 / 天气 / 渠道） |
| `contracts.py` | `AnswerResult` / `Citation` / `RefuseReason`；M5 新增 `Intent` / `Source` / `ToolResult`（纯新增，旧字段不动） |
| `retriever.py` | 查询改写 + bge 向量 & BM25 多路召回（RRF 融合）+ 跨文档同主题去重 |
| `chain.py` | LangChain 生成链：`ChatPromptTemplate → ChatOpenAI → StrOutputParser` |
| `service.py` | 制度链路：语言/分数闸门 → 生成 → 引用回填 → 拒答 |
| `router.py` | 意图路由：制度 / 生活 / 工具 / 闲聊，兜底走制度，支持追问继承 |
| `life.py` | 生活库 BM25 检索（字符 2-gram），与制度库完全分离 |
| `orchestrator.py` | 统一入口 `answer()`：路由 + 制度 service + 生活库 + 工具 + 会话落库 |
| `conversation.py` | SQLite 会话与消息存储，提供最近 N 轮上下文 |
| `safety.py` | 手机号 / 学号 / QQ / 微信 / 身份证 / 姓名脱敏 + 查他人隐私拦截 |
| `tools/` | 天气（Open-Meteo）、当前时间、校历周次、公开常用电话 |
| `app/main.py` | FastAPI：`/api/answer`（契约不变）、`/api/answer/stream`、会话 CRUD、运维端点 |
| `channels/qq.py` | OneBot 事件 → 问答 → 回复；只处理问题与回答，不采集用户身份 |
| `web/` | 单页前端：单一会话（ID 存浏览器本地）、流式渲染、蓝白色来源标签、深浅色与响应式 |
| `scripts/` | `ingest` / `build_index` / `build_life_index` / `crawl_wall` / `ask` / `serve` / `eval_answer` |

## 快速开始

```bash
uv sync --extra dev                    # 安装运行 + 开发依赖（pytest、matplotlib 评测图表）
source .venv/bin/activate              # 激活虚拟环境（后续 python 命令在此环境执行；或用 uv run 前缀）
uv sync --extra ingest                 # 仅当要重新解析 data/raw 原始文件时安装解析依赖
cp .env.example .env                   # 若无 .env.example，按下方变量表自建 .env
python -m scripts.ingest               # 仅当 data/raw 有增删时：重建 data/parsed + chunks.jsonl
python -m scripts.build_index          # 仅当 data/index 缺失或语料变更时执行
python -m scripts.build_life_index     # 生活库语料变更后重建 BM25 索引
python -m scripts.ask "图书馆能借几本书？"
python -m scripts.ask --smart "食堂几点开门？"        # 走统一入口：路由 + 生活库 + 工具
python -m scripts.ask --json "图书馆能借几本书？"   # 输出 AnswerResult JSON
python -m scripts.serve                # 默认监听 0.0.0.0:8000：本机 http://127.0.0.1:8000，局域网 http://<本机IP>:8000
python -m scripts.eval_answer          # 跑 eval/questions.jsonl 并输出指标
python -m pytest -q                    # 单元测试（180 个）
```

### 校园墙 → 问题集（M5）

不导入现成 CSV，直接从校园墙抓原始内容，再提炼成问题：

```bash
# 校园墙有接口时（或先在 .env 配 WALL_SOURCE）
python -m scripts.crawl_wall --source "https://example.edu/wall/api" --limit 300

# 只有原始内容落盘时，离线验证提炼质量（JSONL，每行一个 {"text": "..."}）
python -m scripts.crawl_wall --raw-jsonl data/wall/raw_sample.jsonl --dry-run

# 产出
data/wall/questions.jsonl   # 已脱敏、已去重的问题集（带类目）
data/wall/report.json       # 抓取/提炼/去重统计
```

提炼管线：去 CQ 码与 @ → 切句 → 问句信号过滤（吗/呢/怎么/几点/多少…）→
脱敏（手机号/学号/QQ/微信/姓名）→ 统一问号 → 2-gram Jaccard 去重 → 类目归类。
**只留问题，不留发帖人**：任何身份字段在落盘前就被替换成占位符。

### Windows 快速开始

macOS / Linux 按上方步骤即可；Windows（PowerShell / CMD）注意以下差异：

```powershell
# 1. 装 Python 3.11 与 uv（或 pip），克隆仓库
git clone https://github.com/ryannn02/campus-policy-qa.git
cd campus-policy-qa

# 2. 安装依赖（uv 跨平台）
uv sync --extra dev

# 3. 配置密钥：Windows 无 cp，用 copy（CMD）或 Copy-Item（PowerShell）
copy .env.example .env            # CMD
Copy-Item .env.example .env       # PowerShell
# 编辑 .env 填入 DEEPSEEK_API_KEY

# 4. 重建向量索引（索引文件不入库；首次运行会下载 bge-small-zh-v1.5 模型，
#    国内网络可在 .env 里设 HF_ENDPOINT=https://hf-mirror.com）
python -m scripts.build_index

# 5. 使用
python -m scripts.ask "图书馆能借几本书？"
python -m scripts.serve            # 浏览器打开 http://127.0.0.1:8000；同一 Wi-Fi 下其他设备打开 http://<本机IP>:8000
                                  # 其他电脑打不开时：管理员运行 pwsh scripts/allow_lan_firewall.ps1 放行 8000 端口（仅本地子网）
python -m scripts.eval_answer
python -m pytest -q
```

Windows 差异说明：

- GPU：`build_index` 自动检测，Windows 无 MPS 会用 CPU（3851 条语料 CPU 编码约几分钟，可接受）；
- 重新解析原始文档（`python -m scripts.ingest`）需要 `data/raw` 原始文件（不入库，需自行同步）；
  其中 `.doc` 文件依赖 LibreOffice（装好后自动识别 `soffice`），PDF/DOCX/HTML 无需额外软件；
  只做问答 / 评测 / Web 演示不需要这一步；
- 索引目录若从 macOS 机器拷贝而来，`manifest.json` 里的绝对路径会自动回退到仓库内
  `data/chunks.jsonl`（`retriever.py` 已做跨机器兼容），但推荐直接重跑 `build_index`。

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

## 局域网访问（其他电脑 / 手机打开）

`scripts.serve` 默认监听 `0.0.0.0`，启动时会打印本机局域网地址（如 `http://192.168.x.x:8000`），
同一 Wi-Fi / 网段内的设备直接在浏览器输入该地址即可使用；只想本机用传 `--host 127.0.0.1`。

Windows 防火墙默认会拦截入站连接（尤其当前网络被标为「公用」时），首次使用需以管理员执行一次：

```powershell
pwsh scripts/allow_lan_firewall.ps1        # 放行本项目 .venv 的 python.exe、TCP 8000、仅本地子网
pwsh scripts/allow_lan_firewall.ps1 8011   # 换端口启动时同步放行对应端口
```

规则只对本地子网生效，不会把服务暴露到校园网之外；IP 变化无需重做规则。

### 回答精简（生活类）

生活库是 BM25 检索，命中段落往往是一整块服务说明。`life_summary.condense()` 会先用 LLM
把「问题 + 命中段落」压成只含所问要点的 1–3 句；无密钥 / 超时 / 输出为空时退回句级抽取
（按与问题的二元组重合度 + 时间/地点/流程类型加权，只保留得分不低于最高分 60% 的句子）。
`sources` 仍保留全部命中出处，前端只以蓝白标签显示名称。

> 注意：若只设置了 `DEEPSEEK_API_KEY` 而未设置 `DEEPSEEK_BASE_URL`，`config.py` 会默认使用
> `https://api.deepseek.com`；此前该地址为空会落到 OpenAI 官方地址并因网络不可达超时，
> 导致制度链路一直退化为「原文摘录」模式。

### 教学周次（按学期独立计数）

「现在第几周」读取 `data/academic_calendar.json`：春 / 夏 / 秋每个学期各自配置第 1 周周一（`start`）
与教学周数（`weeks`），今天落在哪个学期就按那个学期从第 1 周起算，学期之间视为假期；
校历文件缺失时才退回 `SEMESTER_START` 单一起点。新学期开始前往 `terms` 里追加一项即可。

```json
{ "name": "2026-2027 学年秋季学期", "start": "2026-09-21", "weeks": 20 }
```

## Web / API

| 端点 | 说明 |
| --- | --- |
| `GET /` | 单页问答界面（单一会话、流式输出、蓝白色来源标签、拒答提示、深浅色；顶栏「清空对话」重置） |
| `POST /api/answer` | 输入 `{"question": "...", "session_id": "可选"}`，返回 `AnswerResult` JSON（旧字段全保留，新增 `intent` / `sources` / `tool_result`） |
| `POST /api/answer/stream` | SSE：`meta` → `delta`（正文分段）→ `done`（完整结果） |
| `POST /api/route` | 只返回意图与命中词，调试用 |
| `GET/POST /api/sessions` | 会话列表 / 新建 |
| `PATCH/DELETE /api/sessions/{id}` | 重命名 / 删除（删除即物理删除消息） |
| `GET /api/sessions/{id}/messages` | 会话内消息（含来源） |
| `GET /health` | 存活探针 + 生活库状态 |
| `GET /api/stats` | 问答时延统计（avg / P50 / P95 / P99）+ 生活库 / 会话规模 |
| `GET /api/capabilities` | 意图清单、工具清单、生活库规模 |

P95 响应时延目标：基础档 ≤8s、进阶档 ≤5s（见 `docs/requirements.md` 非功能需求）。

## 多意图与生活库（M5）

路由是纯规则（`router.py`），可解释、可单测，**制度永远兜底**：

| 意图 | 触发 | 走向 |
| --- | --- | --- |
| `tool` | 现在几点 / 今天几号 / 天气 / 第几周 / 电话 | `tools/` 实时数据，零幻觉 |
| `policy` | 规定、办法、学籍、学分、补考、奖助学金、借阅… | 原 `service.answer_question`，**一行没改** |
| `life` | 食堂、快递、校园卡、热水、校园网、打印、校医院… | 独立生活库 BM25 |
| `chitchat` | 你好 / 谢谢 / 你能做什么 | 寒暄短路 |

边界：

- 生活库与制度库**物理分离**（不同语料、不同索引、不同检索器），生活内容不会进制度召回，
  `Recall@5` / `refusal_rate` 不受影响；
- 只有「追问且上一轮同意图」时，才把上一轮问题拼进生活库 query；制度链路永远拿原问题；
- 涉及「查别人的手机号 / 学号 / 住址」一律拒绝，工具层也只收录公开办公电话。

## 设计约束

- 引用元数据由代码回填，模型只能挑编号，编造引用在结构上不可能
- 没有依据就拒答，拒答理由是机读 code，不是自由文本
- 非中文提问暂时直接拒答
- 提示词进仓库、带版本号，改动新建 `vN+1.md`
- 新功能只做外挂：制度核心（`service` / `retriever` / `chain`）不为生活库、工具、多轮改一行
- 会话只存问题/回答/来源；渠道侧不落 QQ 号、昵称、头像，日志不打 `user_id`

## 部署

```bash
docker compose up --build      # http://127.0.0.1:8000
```

镜像内置生活库语料；制度库索引与 `data/` 通过 volume 挂载（索引不入 Git，先在本机
`python -m scripts.build_index` 生成）。没有索引时服务仍可启动：制度链路退化为
2 条内置样本的字符匹配，生活/工具/闲聊完全可用。
