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

## 拒答契约

`refuse_reason` 是机读 code，给用户看的话放 `answer`：

| code | 触发方 | 含义 |
| --- | --- | --- |
| `unsupported_language` | 代码 | 提问不含中文 |
| `retrieval_empty` | 代码 | 没召回任何条款 |
| `retrieval_low_score` | 代码 | top-1 低于 τ |
| `invalid_model_output` | 代码 | 模型输出不是合法 JSON |
| `no_valid_citation` | 代码 | 模型没给出有效编号 |
| `no_evidence_in_corpus` | 模型 | 条款支撑不住问题 |

## 从旧项目继承

- `data/chunks.jsonl`（4190 块，261 份文档）
- 提示词正文与 6 条回归用例
- 拒答 code 与引用回填规则
- 250 题测试集

旧项目 `p1-rag` 保留不动，作为对照与数据来源。
