# 向量模型选型评估报告

> 负责人：成员 B（检索与模型链负责人），组长统筹收尾
> 结论：**暂不切换 BGE-M3，保持 bge-small-zh-v1.5**
> 日期：2026-11

## 一、评估背景

- 当前向量模型：`BAAI/bge-small-zh-v1.5`，512 维，sentence-transformers 加载。
- 索引：`data/index/embeddings.npy`（4277×512），暴力余弦，毫秒级。
- 评测集 `eval/questions.jsonl`（162 题，gold 标注 116 条）实测：
  - Recall@5 = 0.733（目标 ≥0.75，差 0.017）
  - grounded_rate = 0.722（目标 ≥0.80）
- 动机：两项检索指标未达标，需验证换更强的 BGE-M3 是否值得。

## 二、评估方法

- 受控对比：同一评测集、同一 `title\ntext` 编码口径、同一 query_prefix，只换向量模型。
- 阶段一（最小改动）：用 sentence-transformers 加载 `BAAI/bge-m3`（1024 维 dense）构建实验索引，不新增依赖。
- 为支持 A/B 切换，落地了可配置化改动（见第四节）。

## 三、实验过程与中止原因

| 模型 | 维度 | 编码 4277 块 | 单 batch 耗时 | 预计总时长 |
| --- | --- | --- | --- | --- |
| bge-small-zh-v1.5 | 512 | 秒级 | — | <1 分钟 |
| BAAI/bge-m3 | 1024 | 极慢 | ~92–112 s/batch | **约 1.8 小时** |

- bge-m3 索引构建在 **4%（3/67 batch）** 时中止，已耗时约 4:48（平均 ~96 s/batch）。
- 原因：bge-m3 参数量更大（约 102M vs bge-small 的 24M）、1024 维，在本机（Mac，MPS/CPU）上编码吞吐过低。
- 实验日志见 `.evallogs/build_m3.log`（构建）与 `.evallogs/eval_small.log`（基线评测，已随收尾清理）。

## 四、结论与建议

**结论：暂不切换 BGE-M3，保持 bge-small-zh-v1.5。**

理由：
1. **构建成本过高**：全量编码约 1.8 小时 vs bge-small 秒级，对 4277 条语料不划算；
2. **收益不确定**：当前暴力余弦已是毫秒级，纯中文短问题场景下 1024 维多语言模型的 Recall@5 提升未必显著；
3. **差距很小**：Recall@5 只差 0.017，优先用更轻量的手段（BM25 多路召回、切分/阈值调优、query 改写）补足，而非直接换重型模型。

**后续方向（M4 优先）**：
1. BM25 + 向量多路召回再重排（`docs/requirements.md` 进阶档，收益最直接）；
2. 补全 gold 标注（71.6% → ≥80%），让 grounded_rate / Recall@5 口径更可信；
3. 若未来在更强硬件（GPU 服务器）上评估，可复用已落地的可配置化能力重跑 bge-m3。

## 五、本次保留的可配置化改动（不破坏默认行为）

| 文件 | 改动 | 作用 |
| --- | --- | --- |
| `config.py` | 新增 `RETRIEVAL_INDEX_DIR` 环境变量（默认 `data/index`） | 支持 A/B 切换索引目录 |
| `config.py` | 别名表加 `bge-m3 → BAAI/bge-m3` | 短名补全 |
| `retriever.py` | 从 `manifest.json` 读取模型名，优先于 `settings.embedding_model` | 索引与查询编码保持一致 |
| `retriever.py` | 检索改用 `settings.index_dir` | 跟随可配置索引目录 |

默认配置（`bge-small-zh-v1.5` + `data/index`）不变，现有检索与评测路径不受影响。

## 六、后续进展（M3 验收后补充）

- 本报告建议的「BM25 多路召回」已在 M3 落地：Recall@5 0.733 → **0.802（达标 ≥0.75）**。
- gold 标注已补全至 100%（162/162）。
- `grounded_rate` 0.776 仍差 0.024（引用挑选环节损耗），与本报告「用轻量手段优化、不换重型模型」的结论一致，转 M4 提示词强化。
- bge-m3 结论维持不变：暂不切换。
