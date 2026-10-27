# M2 里程碑任务计划与进阶项契约草案

> 负责人：组长（架构与统筹）
> 状态：M1 已验收通过，本文为 M2 排期与进阶项设计。
> 补充关系：细化 `docs/development_requirements.md` 的里程碑 M2 / M4。

## 一、M2 任务总表

| # | 任务 | 负责人 | 优先级 | 目标 / 验收 |
| --- | --- | --- | --- | --- |
| 1 | 拒答收紧（任务型请求拦截、歧义澄清、`no_evidence_in_corpus` 判定强化） | B + C | P0 | `refuse_failure_rate` ≤ 0.15 |
| 2 | 评测集扩量至 ≥150 条 + 补 `gold` 标注 | C | P0 | 解锁 `grounded_rate` / `Recall@5` |
| 3 | 2 份扫描件 PDF OCR + 24 个遗留文件处置 | A | P1 | `chunks.jsonl` 覆盖 263 份 effective |
| 4 | 答案内可点击引用锚点 | B + C + D | P2 | 正文论断可跳转到对应条款 |
| 5 | 大模型动态生成追问建议 | B + C + D | P2 | 见下文契约草案 |
| 6 | 打 M1 里程碑 git commit | 组长 + E | P0 | 工作区成果入库 |

## 二、任务 5：动态追问契约草案

> 现状：前端用关键词映射生成「你可能还想问」（`scripts/serve.py` 的 `FOLLOWUPS`）。
> 目标：改为由大模型根据当前问题与回答动态生成，前端关键词映射降级为 fallback。

### 2.1 契约（成员 C）——`src/campus_qa/contracts.py`

`AnswerResult` 新增可选字段，默认 `None`，向后兼容：

```python
class AnswerResult(BaseModel):
    ...
    # 大模型基于当前问题与回答生成的建议追问（2–4 条校园制度相关问题）。
    follow_up_questions: Optional[List[str]] = None
```

JSON 形式（`mode="json"`）：

```json
{
  "question": "缓考申请最晚什么时候交？",
  "answer": "……",
  "citations": [ "..." ],
  "follow_up_questions": ["补考课程冲突怎么处理？", "缓考申请需要提交哪些材料？"],
  "refused": false,
  "...": "..."
}
```

- 拒答 / 寒暄场景：`follow_up_questions` 可为 `null`，由前端回退关键词映射。

### 2.2 生成链（成员 B）——`chain.py` + `prompts/answer/v3.md`

- `GeneratedAnswer` 增加 `follow_up_questions: List[str]` 字段（默认空列表）。
- v3 提示词 JSON 输出结构扩展为：

```
{"answer": "...", "citations": [编号], "follow_up_questions": ["追问1", "追问2"]}
```

- 约束：`follow_up_questions` 只给 2–4 条**与当前问题相关、且制度库能回答**的追问；无合适追问给 `[]`。
- `_parse_llm_output` 解析该字段（缺失 / 非字符串列表时取 `[]`，不因此判非法输出）。

### 2.3 服务层（成员 C）——`service.py`

- `answer_question` 把 `generated.follow_up_questions` 透传到 `AnswerResult.follow_up_questions`（仅非拒答路径）。
- 摘录兜底路径（LLM 不可用）：`follow_up_questions` 保持 `None`，前端走关键词 fallback。

### 2.4 前端（成员 D）——`scripts/serve.py`

- `renderFollowups(q, backendQuestions)`：优先用 `data.follow_up_questions`（非空时）；
  为空则回退现有 `FOLLOWUPS` 关键词映射。
- 其余渲染逻辑不变。

### 2.5 测试与验收

- `tests/test_contracts.py`：`follow_up_questions` 默认 `None`。
- `tests/test_chain.py`：`_parse_llm_output` 解析 `follow_up_questions`（含缺失 / 空数组回退）。
- `tests/test_service.py`：非拒答路径透传 `follow_up_questions`；拒答路径为 `None`。
- 验收：浏览器里同一问题追问建议由模型生成、与问题相关；LLM 不可用时仍展示关键词 fallback，界面不空。

## 三、任务 4：引用可点击锚点（简要草案）

- **B**：v3 提示词约定正文锚点标记 `[N]` 仅作内部锚点、由前端消费（区别于已移除的裸编号泄漏问题，改为「正文 `[N]` + 前端渲染成上标链接」）。
- **C**：`Citation` 增加可选 `citation_index`（模型挑选的 1-based 编号），或在 `AnswerResult` 增加 `answer` 到 citation 的锚点映射，保证正文 `[N]` 可定位到 `citations` 列表项。
- **D**：解析正文 `[N]`，渲染为 `<sup>` 可点击上标，滚动 / 高亮到下方对应条款。

> 说明：任务 4 与任务 5 可共用 v3 提示词与契约扩展，建议同期实现，避免重复改两版契约。
