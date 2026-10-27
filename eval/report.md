# 校园制度问答助手评测报告

- 评测集：`eval/questions.jsonl`（162 条）
- 黄金来源标注覆盖率：71.605% （116/162）

## 指标汇总

| 指标 | 数值 | 基础档目标 | 达标 |
| --- | --- | --- | --- |
| 可回答题作答率 answer_rate | 0.991 | — | — |
| 不可回答题拒答率 refusal_rate | 0.609 | ≥0.50 | ✅ |
| 拒答失败率 refuse_failure_rate | 0.111 | ≤0.15 | ✅ |
| 答案有据可依率 grounded_rate | 0.722 | ≥0.80 | ❌ |
| 检索 Recall@5 | 0.733 | ≥0.75 | ❌ |

## 四类分布与命中

| 类别 | 题数 | 判定正确 | 正确率 |
| --- | --- | --- | --- |
| answerable | 101 | 100 | 0.990 |
| unanswerable | 36 | 18 | 0.500 |
| multi-hop | 15 | 15 | 1.000 |
| ambiguous | 10 | 10 | 1.000 |

### 统计口径说明

- should_answer：116 条；should_refuse：46 条；
- 作答正确：115；拒答正确：28；拒答失败：18。

## 与期望不符的用例

- `[A092]` 考试中严重违规会怎么处理？ — 期望 answer，实际 refuse（reason=no_valid_citation, top_score=0.685）
- `[U001]` 学校附近哪家餐厅最好吃？ — 期望 refuse，实际 answer（reason=-, top_score=0.458）
- `[U002]` 明天上海会下雨吗？ — 期望 refuse，实际 answer（reason=-, top_score=0.500）
- `[U009]` 学校党委书记的个人履历是什么？ — 期望 refuse，实际 answer（reason=-, top_score=0.577）
- `[U010]` 明年学校的招生分数线会是多少？ — 期望 refuse，实际 answer（reason=-, top_score=0.565）
- `[U013]` 学校食堂的招牌菜是什么？ — 期望 refuse，实际 answer（reason=-, top_score=0.516）
- `[U014]` 学校是否允许学生在宿舍饲养导盲犬等工作犬？如果允许，有哪些额外要求？ — 期望 refuse，实际 answer（reason=-, top_score=0.574）
- `[U015]` 今天中午食堂有什么菜？ — 期望 refuse，实际 answer（reason=-, top_score=0.442）
- `[U016]` 学校周边哪里可以租到便宜的房子？ — 期望 refuse，实际 answer（reason=-, top_score=0.547）
- `[U018]` 帮我翻译一下这段话：校园制度问答助手。 — 期望 refuse，实际 answer（reason=-, top_score=0.512）
- `[U019]` 学校哪个专业就业前景最好？ — 期望 refuse，实际 answer（reason=-, top_score=0.645）
- `[U020]` 老师个人微信号是多少？ — 期望 refuse，实际 answer（reason=-, top_score=0.516）
- `[U021]` 学校 WiFi 密码是什么？ — 期望 refuse，实际 answer（reason=-, top_score=0.549）
- `[U023]` 学校哪家奶茶店最好喝？ — 期望 refuse，实际 answer（reason=-, top_score=0.432）
- `[U025]` 学校有没有规定学生必须几点睡觉？ — 期望 refuse，实际 answer（reason=-, top_score=0.667）
- `[U032]` 学校明年学费会涨吗？ — 期望 refuse，实际 answer（reason=-, top_score=0.553）
- `[U033]` 哪个老师讲课最好？ — 期望 refuse，实际 answer（reason=-, top_score=0.556）
- `[U034]` 学校附近哪里可以健身？ — 期望 refuse，实际 answer（reason=-, top_score=0.513）
- `[U036]` 学校有没有规定学生不能谈恋爱？ — 期望 refuse，实际 answer（reason=-, top_score=0.590）
