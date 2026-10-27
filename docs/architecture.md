# 架构

## 数据流

```
原始文件（data/raw）
  → 解析（scripts/ingest）→ data/parsed/*.txt
  → 切分（章节 + 页码）→ data/chunks.jsonl
  → 向量化（bge-small-zh-v1.5）→ data/index/
  → 检索 top-k（per_doc=2）
  → 生成链（LangChain）
  → 引用回填 + 拒答判定
  → AnswerResult
```

## LangChain 的边界

- **用**：`ChatPromptTemplate`、`ChatOpenAI`、`StrOutputParser`、`Document`、`BaseRetriever`
- **不用**：文档加载器、切分器、向量库。

原因：现有语料已按「文件 / 章节 / 页码」切好，换切分器会丢页码映射；
4190 条向量的暴力余弦是毫秒级，引入 FAISS/Chroma 只多一份要同步的状态。
检索层用 `BaseRetriever` 适配，检索结果仍然是标准 `Document`。

## 寒暄处理

「你好」「谢谢」「你是谁」等闲聊在 `service.answer_question` 里于检索前直接
短路，返回 `refused=false`、`citations=[]` 的礼貌回应，不进入检索与生成链；
「你好，缓考怎么申请」这类寒暄 + 真实问题不会被误判，仍走正常问答。

## 拒答契约

`refuse_reason` 是机读 code，给用户看的话放 `answer`：

| code | 触发方 | 含义 |
| --- | --- | --- |
| `empty_question` | 代码 | 输入为空或纯空白 |
| `unsupported_language` | 代码 | 提问不含中文，或要求非中文输出 |
| `ambiguous_question` | 代码 | 过短且无具体指向的歧义提问 |
| `retrieval_empty` | 代码 | 没召回任何条款 |
| `retrieval_low_score` | 代码 | top-1 低于 τ |
| `invalid_model_output` | 代码 | 模型输出不是合法 JSON |
| `no_valid_citation` | 代码 | 模型没给出有效编号 |
| `no_evidence_in_corpus` | 模型 | 条款支撑不住问题（含任务型请求等超范围提问） |

## 从旧项目继承

- `data/chunks.jsonl`（4190 块，261 份文档）
- 提示词正文与 6 条回归用例
- 拒答 code 与引用回填规则
- 250 题测试集

旧项目 `p1-rag` 保留不动，作为对照与数据来源。

## 首个里程碑：从 CLI 到 AnswerResult

首轮目标：从命令行输入一个中文问题，到输出一条结构化的 `AnswerResult`端到端跑通；只追求"能跑 + 能演示"检索/生成指标与进阶优化放到第二阶段。

```
scripts.ask
   解析 CLI 参数（问题字符串）
   调用 service.answer_question

service.answer_question
   语言检查（非中文拒答：RefuseReason.UNSUPPORTED_LANGUAGE）
   调用 retriever.retrieve_top_k
   检索空结果或低分 → 按 RefuseReason 标记拒答
   调用 chain.generate_answer_with_citations
   引用编号回填为 Citation 列表
   根据契约构造 AnswerResult

链路输出
   CLI 端打印 AnswerResult.answer 与引用摘要
   保留完整 AnswerResult 作为后续 Web/API 的统一返回结构
```
