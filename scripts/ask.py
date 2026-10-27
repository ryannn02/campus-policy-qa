"""命令行问答入口脚本。

从命令行参数或标准输入读取问题文本，调用
`campus_qa.service.answer_question` 并以人类友好的形式
打印回答摘要和若干引用片段。"""

from __future__ import annotations

import json
import sys
from typing import List

from campus_qa.contracts import AnswerResult, Citation
from campus_qa.service import answer_question


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


def main(argv: List[str] | None = None) -> None:
    """命令行入口：读取问题并打印回答与引用摘要。

    默认输出人类可读摘要；加 `--json` 时输出 `AnswerResult` 的
    JSON 形式（与 `docs/web_api_contract.md` 契约一致）。
    """

    if argv is None:
        argv = sys.argv[1:]

    argv = list(argv)
    as_json = "--json" in argv
    if as_json:
        argv = [arg for arg in argv if arg != "--json"]

    question = _read_question(argv)
    if not question:
        print("no question provided; pass as arguments or stdin")
        return

    result = answer_question(question)

    if as_json:
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    _print_result(result)


if __name__ == "__main__":  # pragma: no cover
    main()

