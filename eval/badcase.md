# badcase 归因表

> 负责人：成员 C（拒答逻辑与评测脚本）
> 数据来源：`python -m scripts.eval_answer` 的「与期望不符的用例」输出
> 更新约定：每轮评测后追加或更新对应行，错因归类到「系统层改动」。

## 当前 mismatch（M3 验收复跑，共 3 条；随 LLM 运行在 2–3 条间波动）

| ID | 问题 | 类别 | top_score | 错因归因 | 系统层改动 |
| --- | --- | --- | --- | --- | --- |
| U014 | 学校是否允许学生饲养导盲犬等工作犬？ | unanswerable | 0.574 | 含制度词「宿舍」，命中相近「宿舍宠物」条款，但非直接依据 | 模型侧 `no_evidence_in_corpus` 判定强化 |
| U029 | 今年国家公务员考试时间是什么时候？ | unanswerable | 0.556 | 校外考试信息，制度库无依据 | 同上 |
| U032 | 学校明年学费会涨吗？ | unanswerable | 0.553 | 含制度词「学费」，但问的是未来数据，制度库无依据 | 同上 |

> 说明：这三条均「含制度关键词」，领域相关性判定（关键词法）无法拦截，
> 属模型侧「条款支撑不住问题」判定不够坚决，转 M4 在提示词侧强化。

## 归因统计（按系统层）

- 检索层：0 条
- 生成链 / 提示词（no_evidence 判定不坚决）：3 条（U014、U029、U032）—— 成员 B 侧
- 拒答逻辑 / 服务层：0 条（领域相关性判定已拦截 13+ 条生活/常识类误答）
- 意图澄清：0 条

## 指标快照（M3 验收复跑，gold 覆盖 100%）

| 指标 | 数值 | 基础档目标 | 达标 |
| --- | --- | --- | --- |
| 黄金来源标注覆盖率 gold_coverage | 1.000 | ≥0.80 | ✅ |
| 可回答题作答率 answer_rate | 1.000 | ≥0.95 | ✅ |
| 不可回答题拒答率 refusal_rate | 0.935 | ≥0.50 | ✅ |
| 拒答失败率 refuse_failure_rate | 0.019 | ≤0.15 | ✅ |
| 答案有据可依率 grounded_rate | 0.776 | ≥0.80 | ❌ 差 0.024 |
| 检索 Recall@5 | 0.802 | ≥0.75 | ✅ |

### grounded_rate 差距说明（如实记录）

- 口径：snippet 精确匹配（答案引用的 chunk 全文须与 gold 标注 snippet 逐字一致），属较严口径。
- `Recall@5` 0.802 已达标，但模型从 top-5 候选挑选引用时偶未选中 gold 标注的那个 chunk（约 3 条题），导致 `grounded_rate` 0.776 距 0.80 差 0.024。
- 结论：**M3 接受现状**，转 M4 用提示词强化「优先引用与问题最直接相关的条款编号」。

## M4 下一步（已全部落地，见下）

1. **B**：提示词强化 `no_evidence_in_corpus` + 引用精确性 → ✅ v5/v6 已落地，grounded 0.828。
2. **A/B**：版本时效过滤 → ✅ 检索跨文档同主题去重已落地（`_dedup_cross_doc`）。
3. **C**：评测集与 gold → ✅ 162 条 / 100%，一键复跑。

## 错误类型 → 系统层改动文件索引（老师验收要点）

| 错误类型 | 典型 badcase | 对应改动文件（改动点） |
| --- | --- | --- |
| 检索召回不足（Recall@5 低、gold 未进 top-5） | A018 转学 / A023 退学 / A033 作弊情形 | `src/campus_qa/retriever.py`：BM25 字符 2-gram 多路召回 + RRF 融合、`_dedup_cross_doc` 跨文档同主题去重、`_rewrite_query`/`_llm_rewrite_query`/`_rewrite_question` 查询改写；`src/campus_qa/config.py`：per_doc=4 / bm25_top_k=10 / ENABLE_REWRITE 等参数 |
| 应拒答却作答（生活/常识类） | U001 餐厅 / U002 天气 / U009 履历 | `src/campus_qa/service.py`：`_is_domain_relevant` 领域相关性前置判定（_DOMAIN_KEYWORDS + domain_score_threshold=0.68） |
| 应拒答却作答（任务型请求） | U012 写论文 / U007 写邮件 | `src/campus_qa/service.py`：`_is_task_request`（"帮我写/代写/写一封"正则拦截） |
| 应拒答却作答（要求英文输出） | U008 英文介绍 / U017 英文说明 | `src/campus_qa/service.py`：`_requires_non_chinese_output` |
| 应拒答却作答（问未来数据/特殊情形，含制度词） | U032 学费涨 / U014 导盲犬 | `prompts/answer/v4~v6.md`：规则「制度库无法支撑的内容必须置空 answer」「特殊情形不得套用一般规定」 |
| 歧义题未澄清直接答 | B010 处分怎么算 / Q14 奖学金怎么拿 | `src/campus_qa/service.py`：`_is_ambiguous` + `_ambiguous_clarification`（拒答改为澄清追问）；`src/campus_qa/contracts.py`：`AMBIGUOUS_QUESTION` code |
| 闲聊被误判为制度问题 | "你好"/"谢谢" 被拒答 | `src/campus_qa/service.py`：`_greeting_reply` 寒暄识别（问候/感谢/身份/道别） |
| 引用裸编号泄漏到答案正文 | "（[5]）（[1][2]）" | `prompts/answer/v2.md`：规则「编号只写进 citations 字段，正文不写标记」 |
| 编造引用编号 | 模型输出越界编号 | `src/campus_qa/chain.py`：`_normalize_citations`（[1..N] 范围校验 + 去重）；`src/campus_qa/service.py`：`_backfill_citations`（按编号回填真实条款，结构上杜绝编造） |
| 引用过保守漏掉 gold（grounded 下降） | v4 上线后 grounded 0.776→0.681 | `prompts/answer/v5.md`：删除「只引用最贴近一版」规则、恢复「宁多勿漏」；`prompts/answer/v6.md`：靠后编号相关也要挂 |
| 引用条款与 gold 逐字不符（假阴性） | A002 第六/九条版本差 | `eval/questions.jsonl`：gold 多版本标注（162→273 元素）；`eval/_find_gold_variants.py`：difflib ≥0.90 自动找多版本 |
| 评测报告缺逐条明细/图表 | — | `scripts/eval_answer.py`：`_render_case_details` 逐条明细表 + matplotlib 指标柱状图/四类饼图 |
| 扫描件 PDF 无文字层不入库 | 基建处内控办法 20 页扫描件 | `scripts/ingest.py`：`parse_pdf` RapidOCR 回退（文字层 <120 字自动 OCR，页码按物理页保留） |
| ingest 非根目录运行全失败 / chunks 被清空 | IDE 里跑 265 条全 file_missing | `scripts/ingest.py`：local_path 锚定 `ROOT_DIR` 解析 + 临时文件原子替换 + 全失败保留旧文件 + 逐条进度输出 |
| 前端体验（布局/追问/锚点/寒暄展示） | — | `scripts/serve.py`：PAGE_HTML 居中布局、示例问题 chips、`renderFollowups`（后端追问 + 关键词 fallback）、`renderAnswerWithAnchors`（正文 [N] 可点击上标） |
| 推送环境问题 | ghfast 镜像失败 / 代理失效 | `~/.gitconfig`：删除 ghfast `url.insteadOf` 重写、`http.https://github.com.proxy` 指向 NexPath；`.gitignore`：data/raw、output 等不入库 |
