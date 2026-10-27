"""生成链测试：提示词加载、JSON 解析、引用编号与建议追问校验（不联网、不加载模型）。"""

from __future__ import annotations

import pytest

from campus_qa.chain import (
    GeneratedAnswer,
    InvalidModelOutputError,
    _extract_json_object,
    _format_context,
    _load_prompt,
    _normalize_citations,
    _parse_follow_ups,
    _parse_llm_output,
    generate_answer_with_citations,
)
from campus_qa.contracts import Citation


def _citation(doc_id: str, score: float) -> Citation:
    return Citation(document_id=doc_id, snippet="条款内容", score=score, page=1)


def test_load_prompt_reads_latest_versioned() -> None:
    """提示词应从 prompts/answer 最新版本（v3）读取，且含运行时变量与建议追问。"""

    system, human = _load_prompt()
    assert system
    assert human
    assert "{question}" in human
    assert "{context}" in human
    assert "follow_up_questions" in human


def test_format_context_numbers_from_one() -> None:
    """引用编号必须从 1 开始，供模型引用。"""

    context = _format_context([_citation("a", 0.9), _citation("b", 0.8)])
    assert "[1]" in context
    assert "[2]" in context
    assert "文档 a" in context


def test_extract_json_object_strips_fence_and_prose() -> None:
    """容忍模型在 JSON 前后附带解释文字或代码围栏。"""

    raw = '好的，答案是：```json\n{"answer": "可以", "citations": [1]}\n```'
    assert _extract_json_object(raw) == '{"answer": "可以", "citations": [1]}'


def test_parse_llm_output_valid() -> None:
    """无 follow_up_questions 字段时应解析为空列表，不判非法。"""

    answer, numbers, follow_ups = _parse_llm_output(
        '{"answer": "可以借 10 本", "citations": [1, 3]}', n=5
    )
    assert answer == "可以借 10 本"
    assert numbers == [1, 3]
    assert follow_ups == []


def test_parse_llm_output_follow_ups() -> None:
    """follow_up_questions 为字符串数组时应原样解析。"""

    raw = (
        '{"answer": "可以借 10 本", "citations": [1], '
        '"follow_up_questions": ["借阅期限多久？", "逾期怎么处理？"]}'
    )
    answer, numbers, follow_ups = _parse_llm_output(raw, n=5)
    assert answer == "可以借 10 本"
    assert numbers == [1]
    assert follow_ups == ["借阅期限多久？", "逾期怎么处理？"]


def test_parse_follow_ups_missing_or_non_list_defaults_empty() -> None:
    """缺失 / 非列表的 follow_up_questions 一律回退为空列表。"""

    assert _parse_follow_ups(None) == []
    assert _parse_follow_ups("不是列表") == []
    assert _parse_follow_ups({"a": 1}) == []


def test_parse_follow_ups_filters_non_strings_and_blanks() -> None:
    """列表内非字符串项跳过、空白字符串剔除。"""

    assert _parse_follow_ups([1, 2, 3]) == []
    assert _parse_follow_ups(["  ", "有效追问", 3, "第二条"]) == ["有效追问", "第二条"]


def test_parse_llm_output_rejects_invalid_json() -> None:
    with pytest.raises(InvalidModelOutputError):
        _parse_llm_output("不是 JSON", n=5)


def test_parse_llm_output_rejects_non_object() -> None:
    with pytest.raises(InvalidModelOutputError):
        _parse_llm_output("[1, 2, 3]", n=5)


def test_normalize_citations_drops_out_of_range_and_dupes() -> None:
    """越界、重复、非整数、布尔都应被丢弃。"""

    assert _normalize_citations([1, 3, 3, 9, -1, "2", "x", True], n=3) == [1, 3, 2]


def test_generate_returns_empty_without_candidates() -> None:
    """无候选条款时不发 LLM 调用，直接返回空结果。"""

    result = generate_answer_with_citations("问题", [])
    assert result == GeneratedAnswer(
        answer="", citation_numbers=[], confidence=0.0, follow_up_questions=[]
    )


def test_generated_answer_follow_ups_default_empty() -> None:
    """GeneratedAnswer 的 follow_up_questions 默认应为空列表。"""

    result = GeneratedAnswer(answer="答案", citation_numbers=[1], confidence=0.8)
    assert result.follow_up_questions == []
