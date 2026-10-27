# 校园制度问答助手评测报告

- 评测集：`eval/questions.jsonl`（20 条）
- 黄金来源标注覆盖率：0.000% （0/20）

## 指标汇总

| 指标 | 数值 | 基础档目标 | 达标 |
| --- | --- | --- | --- |
| 可回答题作答率 answer_rate | 1.000 | — | — |
| 不可回答题拒答率 refusal_rate | 0.500 | ≥0.50 | ✅ |
| 拒答失败率 refuse_failure_rate | 0.200 | ≤0.15 | ❌ |
| 答案有据可依率 grounded_rate | N/A（缺 gold 标注） | ≥0.80 | — |
| 检索 Recall@5 | N/A（缺 gold 标注） | ≥0.75 | — |

## 四类分布与命中

| 类别 | 题数 | 判定正确 | 正确率 |
| --- | --- | --- | --- |
| answerable | 10 | 10 | 1.000 |
| unanswerable | 7 | 4 | 0.571 |
| multi-hop | 2 | 2 | 1.000 |
| ambiguous | 1 | 0 | 0.000 |

### 统计口径说明

- should_answer：12 条；should_refuse：8 条；
- 作答正确：12；拒答正确：4；拒答失败：4。

## 与期望不符的用例

- `[Q14]` 奖学金怎么拿到？ — 期望 refuse，实际 answer（reason=-, top_score=0.667）
- `[Q15]` 能不能帮我写一封向辅导员请假的邮件？ — 期望 refuse，实际 answer（reason=-, top_score=0.579）
- `[Q19]` 学校是否允许学生在宿舍饲养导盲犬等工作犬？如果允许，有哪些额外要求？ — 期望 refuse，实际 answer（reason=-, top_score=0.574）
- `[Q20]` 请用英文详细介绍本校的奖学金种类和评选标准。 — 期望 refuse，实际 answer（reason=-, top_score=0.658）
