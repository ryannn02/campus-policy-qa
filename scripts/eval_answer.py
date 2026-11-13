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
import os
from pathlib import Path
from typing import Iterable, List, Tuple

# matplotlib 缓存目录：沙箱下 ~/.matplotlib 不可写，改到 workspace 内避免每次重建字体缓存。
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，便于脚本化生成图表
import matplotlib.pyplot as plt

from campus_qa.contracts import AnswerResult
from campus_qa.retriever import retrieve_candidates
from campus_qa.service import answer_question


EVAL_PATH = Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl"
REPORT_PATH = Path(__file__).resolve().parents[1] / "eval" / "report.md"
CHART_DIR = REPORT_PATH.parent

_CATEGORIES = ("answerable", "unanswerable", "multi-hop", "ambiguous")

# 图表中文字体（macOS / Linux 常见中文字体，按顺序回退）
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


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

    # Recall@5 与有据可依率的分子分母（仅统计「可回答题」的 gold 命中）。
    # gold_coverage 则统计所有有 gold 字段的题（含不可回答题的干扰 gold 标注）。
    gold_annotated = 0
    recall_eligible = 0
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
            # 仅可回答题的 gold 参与 Recall@5 / 有据可依率；
            # 不可回答题的 gold 是「干扰条款」，只用于标注覆盖率与检索侧分析，
            # 不纳入这两个指标，避免口径污染。
            if expect_answer:
                recall_eligible += 1
                candidates = retrieve_candidates(str(item.get("question", "")))
                if _hit_gold(gold, candidates[:5]):
                    recall_hits += 1
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
        "recall_at_5": _rate(recall_hits, recall_eligible),
        "grounded_rate": _rate(grounded_hits, grounded_answered),
        "per_category": per_category,
        "counts": {
            "should_answer": should_answer_total,
            "should_refuse": should_refuse_total,
            "answered_and_expected": answered_and_expected,
            "refused_and_expected": refused_and_expected,
            "refuse_failure": refuse_failure,
            "recall_eligible": recall_eligible,
        },
    }


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _render_case_details(cases: List[Tuple[dict, AnswerResult]]) -> List[str]:
    """渲染逐条结果明细表：每条一行，含判定与结果摘要说明。"""

    lines: List[str] = [
        "## 逐条结果明细",
        "",
        "| ID | 问题 | 类别 | 期望 | 实际 | 判定 | 拒答原因 | top_score | 引用数 | 命中gold | 结果摘要 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item, result in cases:
        qid = str(item.get("id", "-"))
        question = str(item.get("question", "")).replace("|", "\\|").replace("\n", " ")
        category = _category_of(item)
        expect_answer = bool(item.get("should_answer", True))
        expected = "answer" if expect_answer else "refuse"
        actual = "refuse" if result.refused else "answer"
        verdict = "✅" if expected == actual else "❌"
        reason = result.refuse_reason.value if result.refuse_reason else "-"
        score = result.top_retrieval_score
        score_text = _fmt(score) if score is not None else "-"
        n_cites = len(result.citations)

        gold = item.get("gold") or []
        # 仅可回答题的 gold 参与命中判定；不可回答题的 gold 是干扰标注，显示 —。
        if gold and expect_answer:
            gold_hit = _hit_gold(gold, result.citations)
            gold_text = "✅" if gold_hit else "❌"
        else:
            gold_text = "—"

        if result.refused:
            summary = (result.suggested_contact or reason or "").replace("|", "\\|").replace("\n", " ")
        else:
            summary = (result.answer or "").replace("|", "\\|").replace("\n", " ")
        summary = (summary[:40] + "…") if len(summary) > 40 else summary

        lines.append(
            f"| {qid} | {question} | {category} | {expected} | {actual} | {verdict} | "
            f"{reason} | {score_text} | {n_cites} | {gold_text} | {summary} |"
        )
    return lines + [""]


def _render_charts(metrics: dict) -> List[str]:
    """生成指标对比柱状图与四类分布饼图，保存到 eval/ 下并返回 Markdown 引用行。"""

    lines: List[str] = []

    # 图 1：五项指标 vs 目标柱状图
    labels = ["Recall@5", "有据可依率", "拒答率", "拒答失败率", "作答率"]
    actuals = [
        metrics["recall_at_5"],
        metrics["grounded_rate"],
        metrics["refusal_rate"],
        metrics["refuse_failure_rate"],
        metrics["answer_rate"],
    ]
    # 拒答失败率目标为「≤0.15」，其余为「≥目标」；统一画目标线作参考。
    targets = [0.75, 0.80, 0.50, 0.15, None]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(labels))
    ax.bar(x, actuals, width=0.55, label="实际值", color="#4f81bd")
    for i, t in enumerate(targets):
        if t is not None:
            ax.plot([i - 0.32, i + 0.32], [t, t], color="#c0504d", linewidth=2, linestyle="--")
    for i, v in enumerate(actuals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("指标值")
    ax.set_title("五项指标 vs 目标（虚线为目标值）")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    metrics_png = CHART_DIR / "metrics_chart.png"
    fig.savefig(metrics_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 图 2：四类题量分布饼图
    cat_labels = list(_CATEGORIES)
    cat_sizes = [metrics["per_category"][c]["total"] for c in _CATEGORIES]
    fig2, ax2 = plt.subplots(figsize=(5.5, 5.5))
    ax2.pie(
        cat_sizes,
        labels=cat_labels,
        autopct="%.0f%%",
        startangle=90,
        colors=["#4f81bd", "#c0504d", "#9bbb59", "#f2a900"],
        wedgeprops={"edgecolor": "white"},
    )
    ax2.set_title("四类题量分布")
    pie_png = CHART_DIR / "category_pie.png"
    fig2.savefig(pie_png, dpi=120, bbox_inches="tight")
    plt.close(fig2)

    lines += [
        "",
        "## 图表",
        "",
        f"![五项指标 vs 目标]({metrics_png.name})",
        "",
        f"![四类题量分布]({pie_png.name})",
        "",
    ]
    return lines


def _render_markdown(
    metrics: dict,
    mismatches: List[Tuple[dict, AnswerResult]],
    cases: List[Tuple[dict, AnswerResult]],
) -> str:
    """把指标、mismatch 与逐条结果明细渲染为 Markdown 报告。"""

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
        f"- Recall@5 与有据可依率仅统计可回答题中带 gold 的题（分母 {counts['recall_eligible']} 条）；"
        f"不可回答题的干扰 gold 不参与。",
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

    lines += _render_case_details(cases)
    lines += _render_charts(metrics)

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

    report = _render_markdown(metrics, mismatches, cases)
    print(report)

    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(f"\nreport written to {REPORT_PATH}")
    except OSError as exc:  # pragma: no cover - 报告写入失败不影响主流程
        print(f"\n[warning] failed to write report: {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
