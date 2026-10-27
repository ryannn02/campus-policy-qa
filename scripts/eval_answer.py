"""评测脚本：批量调用问答服务，输出对齐 `docs/requirements.md` 的指标。

指标口径（对应 requirements.md 非功能需求）：

- `refusal_rate`       不可回答题拒答率，目标基础档 ≥0.50、进阶档 ≥0.80；
- `answer_rate`        可回答题作答率（检索 + 生成链路正常出答案的比例）；
- `refuse_failure_rate` 拒答失败率 = 应拒答却作答的比例，是「幻觉率」的
                       自动化上界（真幻觉率需人工判答案是否含无依据编造），
                       目标基础档 ≤15%、进阶档 ≤8%；
- `grounded_rate`      答案有据可依率（需评测集 `gold` 标注），基础档 ≥0.80；
- `recall_at_5`        检索 Recall@5（需评测集 `gold` 标注），基础档 ≥0.75。

评测集 `eval/questions.jsonl` 字段：

- `question`      问题文本；
- `category`      answerable | unanswerable | multi-hop | ambiguous（四类）；
- `should_answer` 布尔，true=可回答、false=不可回答（兼容旧字段）；
- `source_hint`   选题来源说明（供人工核对，不参与计算）；
- `gold`          （可选）黄金来源段落列表，标注后用于 recall_at_5 与
                  grounded_rate，元素含 document_id / snippet / page / section。

当评测集尚未标注 `gold` 时，recall_at_5 与 grounded_rate 输出为 N/A，
并额外报告 `gold_coverage`（已标注比例）作为「评测集就绪度」信号。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

from campus_qa.contracts import AnswerResult
from campus_qa.retriever import retrieve_candidates
from campus_qa.service import answer_question


EVAL_PATH = Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl"
REPORT_PATH = Path(__file__).resolve().parents[1] / "eval" / "report.md"

_CATEGORIES = ("answerable", "unanswerable", "multi-hop", "ambiguous")


def _load_eval_questions(path: Path) -> List[dict]:
    """加载 JSONL 格式的评测集。"""

    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _category_of(item: dict) -> str:
    """返回评测项的类别，优先显式 `category`，缺失时按旧字段粗分。"""

    category = item.get("category")
    if category in _CATEGORIES:
        return category

    if bool(item.get("should_answer", True)):
        return "answerable"
    hint = str(item.get("source_hint", ""))
    if "歧义" in hint:
        return "ambiguous"
    return "unanswerable"


def _hit_gold(gold: list, citations: Iterable[Citation]) -> bool:
    """判断检索/引用结果是否命中任一黄金来源段落（chunk 级 snippet 精确匹配）。"""

    if not gold:
        return False
    gold_snips = {str(g.get("snippet", "")).strip() for g in gold if g.get("snippet")}
    if not gold_snips:
        return False
    return any(c.snippet.strip() in gold_snips for c in citations)


def _compute_metrics(cases: List[Tuple[dict, AnswerResult]]) -> dict:
    """根据评测结果计算指标，并按四类分组统计。"""

    total = len(cases)
    should_answer_total = 0
    should_refuse_total = 0
    answered_and_expected = 0
    refused_and_expected = 0
    refuse_failure = 0

    # Recall@5 与有据可依率的分子分母（仅统计有 gold 标注的题）。
    gold_annotated = 0
    recall_hits = 0
    grounded_hits = 0
    grounded_answered = 0

    per_category = {c: {"total": 0, "correct": 0} for c in _CATEGORIES}

    for item, result in cases:
        expect_answer = bool(item.get("should_answer", True))
        category = _category_of(item)
        per_category[category]["total"] += 1

        gold = item.get("gold") or []
        if gold:
            gold_annotated += 1
            # Recall@5：单独取检索 top-k 候选，判断是否命中黄金来源。
            candidates = retrieve_candidates(str(item.get("question", "")))
            if _hit_gold(gold, candidates[:5]):
                recall_hits += 1
            # 有据可依率：作答且引用命中黄金来源。
            if not result.refused:
                grounded_answered += 1
                if _hit_gold(gold, result.citations):
                    grounded_hits += 1

        if expect_answer:
            should_answer_total += 1
            if not result.refused:
                answered_and_expected += 1
                per_category[category]["correct"] += 1
        else:
            should_refuse_total += 1
            if result.refused:
                refused_and_expected += 1
                per_category[category]["correct"] += 1
            else:
                refuse_failure += 1

    def _rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    return {
        "total": total,
        "gold_annotated": gold_annotated,
        "gold_coverage": _rate(gold_annotated, total),
        "answer_rate": _rate(answered_and_expected, should_answer_total),
        "refusal_rate": _rate(refused_and_expected, should_refuse_total),
        "refuse_failure_rate": _rate(refuse_failure, total),
        "recall_at_5": _rate(recall_hits, gold_annotated),
        "grounded_rate": _rate(grounded_hits, grounded_answered),
        "per_category": per_category,
        "counts": {
            "should_answer": should_answer_total,
            "should_refuse": should_refuse_total,
            "answered_and_expected": answered_and_expected,
            "refused_and_expected": refused_and_expected,
            "refuse_failure": refuse_failure,
        },
    }


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _render_markdown(metrics: dict, mismatches: List[Tuple[dict, AnswerResult]]) -> str:
    """把指标与 mismatch 渲染为 Markdown 报告。"""

    counts = metrics["counts"]
    lines: List[str] = [
        "# 校园制度问答助手评测报告",
        "",
        f"- 评测集：`eval/questions.jsonl`（{metrics['total']} 条）",
        f"- 黄金来源标注覆盖率：{_fmt(metrics['gold_coverage'] * 100)}% "
        f"（{metrics['gold_annotated']}/{metrics['total']}）",
        "",
        "## 指标汇总",
        "",
        "| 指标 | 数值 | 基础档目标 | 达标 |",
        "| --- | --- | --- | --- |",
        f"| 可回答题作答率 answer_rate | {_fmt(metrics['answer_rate'])} | — | — |",
        f"| 不可回答题拒答率 refusal_rate | {_fmt(metrics['refusal_rate'])} | ≥0.50 | "
        f"{'✅' if metrics['refusal_rate'] >= 0.50 else '❌'} |",
        f"| 拒答失败率 refuse_failure_rate | {_fmt(metrics['refuse_failure_rate'])} | ≤0.15 | "
        f"{'✅' if metrics['refuse_failure_rate'] <= 0.15 else '❌'} |",
    ]

    if metrics["gold_annotated"] > 0:
        lines += [
            f"| 答案有据可依率 grounded_rate | {_fmt(metrics['grounded_rate'])} | ≥0.80 | "
            f"{'✅' if metrics['grounded_rate'] >= 0.80 else '❌'} |",
            f"| 检索 Recall@5 | {_fmt(metrics['recall_at_5'])} | ≥0.75 | "
            f"{'✅' if metrics['recall_at_5'] >= 0.75 else '❌'} |",
        ]
    else:
        lines += [
            "| 答案有据可依率 grounded_rate | N/A（缺 gold 标注） | ≥0.80 | — |",
            "| 检索 Recall@5 | N/A（缺 gold 标注） | ≥0.75 | — |",
        ]

    lines += [
        "",
        "## 四类分布与命中",
        "",
        "| 类别 | 题数 | 判定正确 | 正确率 |",
        "| --- | --- | --- | --- |",
    ]
    for category in _CATEGORIES:
        stat = metrics["per_category"][category]
        rate = stat["correct"] / stat["total"] if stat["total"] else 0.0
        lines.append(
            f"| {category} | {stat['total']} | {stat['correct']} | {_fmt(rate)} |"
        )

    lines += [
        "",
        f"### 统计口径说明",
        "",
        f"- should_answer：{counts['should_answer']} 条；should_refuse：{counts['should_refuse']} 条；",
        f"- 作答正确：{counts['answered_and_expected']}；拒答正确：{counts['refused_and_expected']}；拒答失败：{counts['refuse_failure']}。",
        "",
    ]

    if mismatches:
        lines += ["## 与期望不符的用例", ""]
        for item, result in mismatches:
            expected = "answer" if bool(item.get("should_answer", True)) else "refuse"
            actual = "refuse" if result.refused else "answer"
            reason = result.refuse_reason.value if result.refuse_reason else "-"
            score = result.top_retrieval_score
            score_text = _fmt(score) if score is not None else "-"
            lines.append(
                f"- `[{item.get('id')}]` {item.get('question')} — "
                f"期望 {expected}，实际 {actual}（reason={reason}, top_score={score_text}）"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """命令行入口：运行评测并输出指标与 Markdown 报告。"""

    if not EVAL_PATH.exists():
        print(f"eval file not found: {EVAL_PATH}")
        return

    items = _load_eval_questions(EVAL_PATH)
    print(f"loaded {len(items)} eval questions from {EVAL_PATH}")

    cases: List[Tuple[dict, AnswerResult]] = []
    for item in items:
        question = str(item.get("question", ""))
        result = answer_question(question)
        cases.append((item, result))

    metrics = _compute_metrics(cases)
    mismatches = [
        (item, result)
        for item, result in cases
        if bool(item.get("should_answer", True)) == bool(result.refused)
    ]

    report = _render_markdown(metrics, mismatches)
    print(report)

    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(f"\nreport written to {REPORT_PATH}")
    except OSError as exc:  # pragma: no cover - 报告写入失败不影响主流程
        print(f"\n[warning] failed to write report: {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
