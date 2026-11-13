"""P95 响应时延压测脚本（成员 D：前端与接口负责人）。

对 ``POST /api/answer`` 顺序发送一批真实制度问题（从 ``eval/questions.jsonl``
取 ``should_answer == true`` 的可回答题），逐条记录 HTTP 端到端耗时，计算
avg / P50 / P95 / P99 / max，对照 ``docs/requirements.md`` 的时延目标
（基础档 ≤8s、进阶档 ≤5s），并把结果写入 ``eval/latency_report.md``。

- 默认先做 1 次预热（加载向量模型 / LLM 冷启动），预热耗时单独记录、不计入指标；
- 每条结果区分「llm 生成 / extractive 兜底 / 拒答 / 请求错误」，便于定位瓶颈。

命令行使用：

.. code-block:: bash

    python -m scripts.serve                        # 先起服务
    python -m scripts.bench_latency                # 默认 50 题 + 预热
    python -m scripts.bench_latency -n 60          # 压 60 题
    python -m scripts.bench_latency --base-url http://127.0.0.1:8000
    python -m scripts.bench_latency --no-warmup    # 冷启动计入指标

依赖约束：只用 Python 标准库（urllib + json + time + math），不引入新依赖。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 时延目标（秒），与 docs/requirements.md 非功能需求保持一致。
BASE_TARGET_SECONDS = 8.0
ADVANCED_TARGET_SECONDS = 5.0

# 单请求超时（秒）。LLM 生成链自身 timeout=60，这里留足余量。
_PER_REQUEST_TIMEOUT = 90.0

EVAL_QUESTIONS_PATH = Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl"
REPORT_PATH = Path(__file__).resolve().parents[1] / "eval" / "latency_report.md"


def _load_answerable_questions(path: Path) -> List[str]:
    """加载评测集里 ``should_answer == true`` 的问题文本。"""

    questions: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("should_answer") is True:
                question = str(item.get("question", "")).strip()
                if question:
                    questions.append(question)
    return questions


def _post_question(base_url: str, question: str) -> Tuple[float, int, Dict[str, Any]]:
    """POST 一个问题，返回 ``(耗时秒, http状态码, 响应JSON字典)``。

    网络错误 / 超时 / 5xx 都会返回带 ``error`` 标记的字典，不抛异常，
    保证单条失败不会打断整个压测流程。
    """

    url = base_url.rstrip("/") + "/api/answer"
    payload = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=_PER_REQUEST_TIMEOUT) as resp:
            status = resp.status
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - 网络/超时统一兜底
        elapsed = time.perf_counter() - start
        return elapsed, 0, {"error": type(exc).__name__, "message": str(exc)}
    elapsed = time.perf_counter() - start

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"error": "invalid_json_response", "message": raw[:200]}
    if not isinstance(data, dict):
        data = {"error": "non_object_response", "message": str(data)[:200]}

    return elapsed, status, data


def _detect_mode(data: Dict[str, Any]) -> str:
    """把响应归类为 error / refused / llm / extractive。

    - 带 ``citation_index`` 或非空 ``follow_up_questions`` → LLM 生成；
    - 有 citations 但无 citation_index 且无追问 → LLM 不可用时的摘录兜底。
    """

    if "error" in data:
        return "error"
    if data.get("refused"):
        return "refused"
    if data.get("follow_up_questions"):
        return "llm"
    citations = data.get("citations") or []
    if any(isinstance(c, dict) and c.get("citation_index") is not None for c in citations):
        return "llm"
    return "extractive"


def _percentile(sorted_values: List[float], p: float) -> float:
    """最近秩（nearest-rank）百分位数，与 serve.py 的 /api/stats 口径一致。"""

    if not sorted_values:
        return 0.0
    rank = int(math.ceil(p * len(sorted_values)))
    index = max(0, min(len(sorted_values) - 1, rank - 1))
    return sorted_values[index]


def _seconds(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.3f}"


def _mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="对 /api/answer 做 P95 时延压测")
    parser.add_argument("-n", "--count", type=int, default=50, help="压测问题数（默认 50）")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="服务基地址")
    parser.add_argument("--warmup", dest="warmup", action="store_true", default=True,
                        help="先做 1 次预热（加载向量模型/LLM），不计入指标（默认开启）")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false",
                        help="不做预热，冷启动计入指标")
    args = parser.parse_args(argv)

    if not EVAL_QUESTIONS_PATH.exists():
        print(f"eval 集不存在：{EVAL_QUESTIONS_PATH}", file=sys.stderr)
        return 1

    questions = _load_answerable_questions(EVAL_QUESTIONS_PATH)
    if not questions:
        print("评测集中没有 answerable 问题", file=sys.stderr)
        return 1

    count = min(args.count, len(questions))
    api = args.base_url.rstrip("/") + "/api/answer"
    print(f"加载 {len(questions)} 条 answerable 问题，取前 {count} 条压测")
    print(f"目标服务：{api}")

    warmup_seconds: Optional[float] = None
    warmup_mode: str = "-"
    if args.warmup and count > 0:
        print("预热中（加载向量模型 / LLM，不计入指标）……", flush=True)
        warmup_seconds, _, warmup_data = _post_question(args.base_url, questions[0])
        warmup_mode = _detect_mode(warmup_data)
        print(f"预热完成：{warmup_seconds:.3f}s（mode={warmup_mode}）", flush=True)

    records: List[Dict[str, Any]] = []
    for idx in range(count):
        question = questions[idx]
        elapsed, status, data = _post_question(args.base_url, question)
        mode = _detect_mode(data)
        records.append(
            {
                "index": idx + 1,
                "question": question,
                "elapsed": elapsed,
                "status": status,
                "mode": mode,
            }
        )
        print(f"[{idx + 1:02d}/{count}] {elapsed:6.3f}s {mode:10s} {question[:30]}",
              flush=True)

    latencies = [r["elapsed"] for r in records]
    sorted_latencies = sorted(latencies)
    n = len(latencies)
    avg = sum(latencies) / n if n else 0.0
    p50 = _percentile(sorted_latencies, 0.50)
    p95 = _percentile(sorted_latencies, 0.95)
    p99 = _percentile(sorted_latencies, 0.99)
    max_lat = max(latencies) if latencies else 0.0

    base_ok = p95 <= BASE_TARGET_SECONDS
    adv_ok = p95 <= ADVANCED_TARGET_SECONDS

    from collections import Counter
    mode_counts = Counter(r["mode"] for r in records)

    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")

    lines: List[str] = []
    lines.append("# P95 响应时延压测报告")
    lines.append("")
    lines.append(f"> 生成时间：{now}")
    lines.append(f"> 压测脚本：`scripts/bench_latency.py`")
    lines.append(f"> 目标服务：`{api}`")
    lines.append(f"> 问题来源：`eval/questions.jsonl` 中 answerable 题（共 {n} 条，顺序发送）")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append("| 指标 | 实测 | 基础档（≤8s） | 进阶档（≤5s） |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(f"| 样本数 | {n} | — | — |")
    lines.append(f"| avg | {_seconds(avg)} | — | — |")
    lines.append(f"| P50 | {_seconds(p50)} | — | — |")
    lines.append(f"| P95 | {_seconds(p95)} | {_mark(base_ok)} | {_mark(adv_ok)} |")
    lines.append(f"| P99 | {_seconds(p99)} | — | — |")
    lines.append(f"| max | {_seconds(max_lat)} | — | — |")
    lines.append("")
    lines.append(f"- 基础档（P95 ≤ 8s）：{_mark(base_ok)} {'达标' if base_ok else '未达标'}")
    lines.append(f"- 进阶档（P95 ≤ 5s）：{_mark(adv_ok)} {'达标' if adv_ok else '未达标'}")
    lines.append("")
    if warmup_seconds is not None:
        lines.append(f"- 预热耗时（首次加载向量模型 / LLM 冷启动，未计入指标）：{warmup_seconds:.3f}s（mode={warmup_mode}）")
        lines.append("")
    lines.append("- 结果分布：" + "，".join(f"{k}={v}" for k, v in sorted(mode_counts.items())))
    lines.append("")
    lines.append("## 瓶颈分析（P95 > 5s 时填写）")
    lines.append("")
    if adv_ok:
        lines.append("P95 已落在进阶档（≤5s）内，无额外瓶颈需记录。")
    else:
        lines.append("P95 超过进阶档 5s，与成员 B 的瓶颈定位结论：")
        lines.append("")
        lines.append("- （待补充：首次模型加载 / 向量检索 / LLM 生成 三段各自占比）")
    lines.append("")
    lines.append("## 明细")
    lines.append("")
    lines.append("| # | 问题 | 耗时(s) | HTTP | 结果 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in records:
        question = r["question"].replace("|", "\\|")
        lines.append(
            f"| {r['index']} | {question} | {r['elapsed']:.3f} | {r['status']} | {r['mode']} |"
        )
    lines.append("")

    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"\n报告已写入：{REPORT_PATH}")
    print(f"avg={avg:.3f}s P50={p50:.3f}s P95={p95:.3f}s P99={p99:.3f}s max={max_lat:.3f}s")
    print(f"基础档(≤8s) {'达标' if base_ok else '未达标'}，进阶档(≤5s) {'达标' if adv_ok else '未达标'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
