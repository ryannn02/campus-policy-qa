# Web / API 契约草案

> 角色：成员 D（前端与接口负责人）
> 目的：为 CLI 之后的 Web/API 层提供统一的 JSON 契约与页面形状参考。

## HTTP JSON API 契约

- 路径：`POST /api/answer`
- 语义：对单个自然语言问题返回一个 `AnswerResult`。
- 编码：请求与响应均为 `application/json`，UTF-8。

### 请求体（Request JSON）

```json
{
  "question": "图书馆能借几本书？"
}
```

- `question`：字符串，必填。
  - 预期为中文问题；非中文问题在服务层按契约拒答（`RefuseReason.UNSUPPORTED_LANGUAGE`）。

（如后续需要，可以扩展可选字段，例如 `max_citations`、`debug` 等，这一版先不预设，按 YAGNI 原则保持最小输入。）

### 响应体（Response JSON）

响应体为 `src/campus_qa/contracts.py` 中 `AnswerResult` 的 JSON 形式，字段含义如下：

```json
{
  "question": "图书馆能借几本书？",
  "answer": "根据《图书馆借阅管理办法》，本科生最多可同时借阅 10 本图书……",
  "citations": [
    {
      "document_id": "policy_library_borrowing_2023",
      "snippet": "本科生借阅数量上限为 10 本，借阅期限为 30 日……",
      "score": 0.92,
      "page": 3,
      "section": "第二章 第三节",
      "url": "https://example.edu/policies/library/borrowing#section-2-3",
      "citation_index": 1
    }
  ],
  "confidence": 0.88,
  "follow_up_questions": ["补考课程冲突怎么处理？", "缓考申请需要提交哪些材料？"],
  "refused": false,
  "refuse_reason": null,
  "top_retrieval_score": 0.92,
  "suggested_contact": null
}
```

- `question`：原始问题字符串（镜像请求的 `question`）。
- `answer`：可选字符串，可回答题时给出自然语言答案；不可回答题为 `null`。
- `citations`：数组，元素为 `Citation`：
  - `document_id`：条款来源文档的唯一 ID 或文件名。
  - `snippet`：与答案直接相关的条款片段。
  - `score`：检索得分（例如余弦相似度）。
  - `page`：可选，文档页码，便于前端跳转到原文。
  - `section`：可选，文档章节标题或编号。
  - `url`：可选，原文在 Web 上的跳转链接。
  - `citation_index`：可选，模型挑选的 1-based 引用编号，前端据此把正文 `[N]` 锚点定位到引用列表。
- `confidence`：可选浮点数，系统对当前答案的整体置信度。
- `follow_up_questions`：可选字符串数组，大模型基于当前问题与回答生成的建议追问（2–4 条）；拒答 / 摘录兜底时为 `null`，由前端回退关键词映射。
- `refused`：布尔值，是否拒答（不可回答题标记为 `true`）。
- `refuse_reason`：可选字符串枚举，拒答原因代码，等价于 `RefuseReason`：
  - `"empty_question"`
  - `"unsupported_language"`
  - `"ambiguous_question"`
  - `"retrieval_empty"`
  - `"retrieval_low_score"`
  - `"invalid_model_output"`
  - `"no_valid_citation"`
  - `"no_evidence_in_corpus"`
- `top_retrieval_score`：可选浮点数，本次检索的 top-1 得分，用于前端展示“有多接近原文”。
- `suggested_contact`：可选字符串，给用户的人工渠道建议，例如“请咨询辅导员 / 教务处”。

#### 拒答示例

```json
{
  "question": "请用英文说明图书馆借阅规则",
  "answer": null,
  "citations": [],
  "confidence": null,
  "refused": true,
  "refuse_reason": "unsupported_language",
  "top_retrieval_score": null,
  "suggested_contact": "请改用中文提问，或咨询辅导员。"
}
```

#### 寒暄回应示例

「你好」「谢谢」「你是谁」等闲聊问题在服务层于检索前直接回应，
`refused=false` 且 `citations` 为空、`confidence=1.0`（未经过检索，无 `top_retrieval_score`）：

```json
{
  "question": "你好",
  "answer": "你好！我是校园制度问答助手，可以帮你查询缓考、休学、选课、奖助学金、图书馆借阅等校内制度问题。请问有什么想了解的？",
  "citations": [],
  "confidence": 1.0,
  "refused": false,
  "refuse_reason": null,
  "top_retrieval_score": null,
  "suggested_contact": null
}
```

> 后端约束：
> - 正常业务场景统一返回 200 + `AnswerResult`，错误场景（服务异常）返回 5xx 与错误信息；不再为合法业务拒答单独定义 HTTP 状态码，以简化前端处理。
> - 请求体不是合法 JSON、或不是 JSON 对象时返回 `400` 与 `{"error": "invalid_json", "message": "..."}`。

### 运维与监控端点（成员 D）

- `GET /health`：存活探针。返回 `200` 与：
  - `{"status": "ok", "service": "campus-policy-qa", "uptime_seconds": 123.456}`
  - 供进程守护 / 健康检查使用。

- `GET /api/stats`：问答时延监控。返回进程内累计的时延统计（最近最多保留 1000 条，进程重启后清零）：

```json
{
  "count": 42,
  "uptime_seconds": 123.456,
  "avg_seconds": 3.21,
  "p50_seconds": 2.90,
  "p95_seconds": 6.10,
  "p99_seconds": 7.80,
  "max_seconds": 8.90,
  "p95_within_base_target": true,
  "p95_within_advanced_target": false
}
```

- `p95_seconds` 对应 `docs/requirements.md` 非功能需求中的 P95 响应时延指标：
  - 基础档目标 ≤ 8s（`p95_within_base_target`）；
  - 进阶档目标 ≤ 5s（`p95_within_advanced_target`）。
- 时延口径为「问题进入 `answer_question` 到返回 `AnswerResult`」的端到端耗时，包含检索与生成，不包含请求体解析。

## Web 页面草图（文本版）

> 目标：最小可用的问答界面，围绕 `AnswerResult` 展示内容。

页面结构（从上到下）：

1. 顶部标题栏
   - 文本：`校园制度问答助手`

2. 问题输入区
   - 单行文本输入框：
     - placeholder：`请输入校园制度相关问题（中文）`
   - 提交按钮：
     - 文本：`提问`

3. 状态 / 错误提示区
   - 加载中：显示 `正在查询制度条款，请稍候……`
   - 网络 / 服务错误：显示例如 `服务暂时不可用，请稍后重试。`

4. 答案展示区
   - 标题：`答复`
   - 正常回答：显示 `AnswerResult.answer`。
   - 拒答时：根据 `refuse_reason` 映射为人类可读文案，例如：
     - `unsupported_language` → `目前仅支持中文提问，请改用中文。`
     - `retrieval_empty` → `暂未找到相关制度条款，请确认问题表述。`
     - `no_evidence_in_corpus` → `现有制度文件无法支撑该问题，请咨询辅导员或相关部门。`

5. 引用列表区
   - 标题：`引用条款`
   - 列表项（每个 `Citation` 一行或一块）：
     - 主文案：`[document_id] snippet`（截断到适当长度）。
     - 侧信息：`score`、`page`、`section`。
     - 如有 `url`：提供“查看原文”超链接，点击跳转学校官网原文位置。

6. 元信息与建议区
   - 展示：
     - `confidence`：例如 `系统置信度：0.88`（仅在非拒答且有值时显示）。
     - `top_retrieval_score`：例如 `检索匹配度：0.92`。
   - 如有 `suggested_contact`：以提示条形式展示，例如：
     - `如需进一步确认，请联系：教务处值班老师。`

7. 页脚（可选，非必需）
   - 简要说明：`本系统回答仅供参考，具体以学校正式制度文件为准。`

> 交互约束（基础版）：
> - 每次提交仅支持单问题同步返回结果，不做多轮对话状态管理；后续如需要可在请求中加入 `session_id` 等扩展字段。
> - 前端不自行生成答案文案，只展示 `AnswerResult` 中的内容或基于 `refuse_reason` 做固定映射。

