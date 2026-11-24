"""命令行问答入口脚本。

从命令行参数或标准输入读取问题文本，默认调用
`campus_qa.service.answer_question`（制度链路，保持评测口径）；
加 `--smart` 时改走 `campus_qa.orchestrator.answer`，可命中生活库与实时工具。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

# Windows + 中文路径下 editable 安装的 .pth 可能失效，这里显式兜底
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from campus_qa.contracts import AnswerResult, Citation  # noqa: E402
from campus_qa.service import answer_question  # noqa: E402


def _read_question(argv: List[str]) -> str:
    """从命令行参数或标准输入获取提问。

    优先使用命令行参数；若为空，则尝试从标准输入读取一行。
    """

    if argv:
        return " ".join(argv).strip()

    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        if line:
            return line.strip()

    return ""


def _print_result(result: AnswerResult) -> None:
    """友好打印回答结果与引用摘要。"""

    print("question:")
    print(f"  {result.question}")
    print()

    if result.refused:
        reason = result.refuse_reason.value if result.refuse_reason else "unknown"
        print("refused:")
        print(f"  reason: {reason}")
        if result.top_retrieval_score is not None:
            print(f"  top_retrieval_score: {result.top_retrieval_score:.3f}")
        if result.suggested_contact:
            print(f"  suggested_contact: {result.suggested_contact}")
        return

    print("answer:")
    print(f"  {result.answer}")
    print()

    if not result.citations:
        print("citations: (none)")
        return

    print("citations:")
    for idx, c in enumerate(result.citations[:3], start=1):
        _print_citation(idx, c)


def _print_citation(idx: int, c: Citation) -> None:
    prefix = f"  [{idx}] {c.document_id} (score={c.score:.3f})"
    snippet = c.snippet.strip()
    snippet_preview = snippet[:80] + ("..." if len(snippet) > 80 else "")
    print(prefix)
    print(f"      {snippet_preview}")


def _print_sources(result: AnswerResult) -> None:
    """M5：打印生活库 / 工具来源（制度引用仍走 citations）。"""

    if not result.sources:
        return
    print()
    print("sources:")
    for idx, source in enumerate(result.sources[:3], start=1):
        title = source.title or source.document_id or ""
        snippet = source.snippet.strip().replace("\n", " ")
        preview = snippet[:80] + ("..." if len(snippet) > 80 else "")
        print(f"  [{idx}] {source.kind} · {title}")
        if preview:
            print(f"      {preview}")


def main(argv: List[str] | None = None) -> None:
    """命令行入口：读取问题并打印回答与引用摘要。

    默认输出人类可读摘要；加 `--json` 时输出 `AnswerResult` 的
    JSON 形式（与 `docs/web_api_contract.md` 契约一致）；
    加 `--smart` 时走统一入口（路由 + 生活库 + 工具）。
    """

    if argv is None:
        argv = sys.argv[1:]

    argv = list(argv)
    as_json = "--json" in argv
    smart = "--smart" in argv
    argv = [arg for arg in argv if arg not in {"--json", "--smart"}]

    question = _read_question(argv)
    if not question:
        print("no question provided; pass as arguments or stdin")
        return

    if smart:
        from campus_qa.orchestrator import answer as smart_answer

        result = smart_answer(question)
    else:
        result = answer_question(question)

    if as_json:
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    if result.intent is not None:
        print(f"intent: {result.intent.value}")
    _print_result(result)
    _print_sources(result)


if __name__ == "__main__":  # pragma: no cover
    main()

