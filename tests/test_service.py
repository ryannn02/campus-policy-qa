"""服务编排层测试：拒答分支与生成兜底（不联网、不加载模型）。"""

from __future__ import annotations

import dataclasses
from typing import List

import campus_qa.service as service
from campus_qa.chain import GeneratedAnswer, InvalidModelOutputError
from campus_qa.config import settings
from campus_qa.contracts import Citation, RefuseReason


def _citation(score: float) -> Citation:
    return Citation(document_id="doc-1", snippet="第十条 借阅期限为 60 天。", score=score)


def test_greets_hello_without_retrieval(monkeypatch) -> None:
    def _fail(_question: str) -> List[Citation]:  # pragma: no cover - 不应被调用
        raise AssertionError("寒暄不应进入检索")

    monkeypatch.setattr(service, "retrieve_candidates", _fail)

    result = service.answer_question("你好")

    assert result.refused is False
    assert result.citations == []
    assert "助手" in result.answer


def test_greets_thanks_without_retrieval(monkeypatch) -> None:
    def _fail(_question: str) -> List[Citation]:  # pragma: no cover - 不应被调用
        raise AssertionError("寒暄不应进入检索")

    monkeypatch.setattr(service, "retrieve_candidates", _fail)

    result = service.answer_question("谢谢")

    assert result.refused is False
    assert "不客气" in result.answer


def test_english_greeting_replies_politely(monkeypatch) -> None:
    def _fail(_question: str) -> List[Citation]:  # pragma: no cover - 不应被调用
        raise AssertionError("寒暄不应进入检索")

    monkeypatch.setattr(service, "retrieve_candidates", _fail)

    result = service.answer_question("hello")

    assert result.refused is False
    assert result.answer


def test_mixed_greeting_and_question_still_answers(monkeypatch) -> None:
    """「你好 + 真实问题」不应被当作纯寒暄，仍需走检索与生成链。"""

    monkeypatch.setattr(service, "retrieve_candidates", lambda _q: [_citation(0.66)])
    monkeypatch.setattr(
        service,
        "_generate_answer",
        lambda _q, _c: GeneratedAnswer("缓考申请需在规定时间内提交。", [1], 0.66),
    )

    result = service.answer_question("你好，缓考申请最晚什么时候交？")

    assert result.refused is False
    assert result.citations
    assert "缓考" in result.answer


def test_rejects_non_chinese_question(monkeypatch) -> None:
    def _fail(_question: str) -> List[Citation]:  # pragma: no cover - 不应被调用
        raise AssertionError("非中文问题不应进入检索")

    monkeypatch.setattr(service, "retrieve_candidates", _fail)

    result = service.answer_question("How many books can I borrow?")

    assert result.refused is True
    assert result.refuse_reason is RefuseReason.UNSUPPORTED_LANGUAGE
    assert result.suggested_contact


def test_rejects_non_chinese_output_requirement(monkeypatch) -> None:
    def _fail(_question: str) -> List[Citation]:  # pragma: no cover - 不应被调用
        raise AssertionError("要求非中文输出的问题不应进入检索")

    monkeypatch.setattr(service, "retrieve_candidates", _fail)

    result = service.answer_question("请用英文详细介绍本校的奖学金种类和评选标准。")

    assert result.refused is True
    assert result.refuse_reason is RefuseReason.UNSUPPORTED_LANGUAGE


def test_rejects_task_request(monkeypatch) -> None:
    def _fail(_question: str) -> List[Citation]:  # pragma: no cover - 不应被调用
        raise AssertionError("任务型请求不应进入检索")

    monkeypatch.setattr(service, "retrieve_candidates", _fail)

    result = service.answer_question("能不能帮我写一封向辅导员请假的邮件？")

    assert result.refused is True
    assert result.refuse_reason is RefuseReason.NO_EVIDENCE_IN_CORPUS


def test_rejects_ambiguous_question(monkeypatch) -> None:
    def _fail(_question: str) -> List[Citation]:  # pragma: no cover - 不应被调用
        raise AssertionError("歧义问题不应进入检索")

    monkeypatch.setattr(service, "retrieve_candidates", _fail)

    result = service.answer_question("奖学金怎么拿到？")

    assert result.refused is True
    assert result.refuse_reason is RefuseReason.AMBIGUOUS_QUESTION


def test_specific_question_is_not_ambiguous(monkeypatch) -> None:
    """含具体限定的短问题（如「国家助学金」）不应被误判为歧义。"""

    monkeypatch.setattr(service, "retrieve_candidates", lambda _q: [_citation(0.66)])
    monkeypatch.setattr(
        service,
        "_generate_answer",
        lambda _q, _c: GeneratedAnswer("国家助学金需满足家庭经济困难等条件。", [1], 0.66),
    )

    result = service.answer_question("国家助学金怎么申请？")

    assert result.refused is False
    assert result.citations


def test_rejects_ambiguous_credit_question(monkeypatch) -> None:
    """「学分不够怎么办」这类过短且无具体指向的问题应判为歧义。"""

    def _fail(_question: str) -> List[Citation]:  # pragma: no cover - 不应被调用
        raise AssertionError("歧义问题不应进入检索")

    monkeypatch.setattr(service, "retrieve_candidates", _fail)

    result = service.answer_question("学分不够怎么办？")

    assert result.refused is True
    assert result.refuse_reason is RefuseReason.AMBIGUOUS_QUESTION


def test_rejects_empty_question() -> None:
    result = service.answer_question("   ")

    assert result.refused is True
    assert result.refuse_reason is RefuseReason.EMPTY_QUESTION


def test_rejects_when_retrieval_is_empty(monkeypatch) -> None:
    monkeypatch.setattr(service, "retrieve_candidates", lambda _q: [])

    result = service.answer_question("图书馆能借几本书？")

    assert result.refused is True
    assert result.refuse_reason is RefuseReason.RETRIEVAL_EMPTY


def test_rejects_below_threshold(monkeypatch) -> None:
    monkeypatch.setattr(service, "retrieve_candidates", lambda _q: [_citation(0.10)])

    result = service.answer_question("图书馆能借几本书？")

    assert result.refused is True
    assert result.refuse_reason is RefuseReason.RETRIEVAL_LOW_SCORE
    assert result.top_retrieval_score == 0.10


def test_answers_when_score_passes_gate(monkeypatch) -> None:
    monkeypatch.setattr(service, "retrieve_candidates", lambda _q: [_citation(0.66)])
    monkeypatch.setattr(
        service,
        "_generate_answer",
        lambda _q, _c: GeneratedAnswer("本科生可同时借阅 30 册。", [1], 0.66),
    )

    result = service.answer_question("图书馆能借几本书？")

    assert result.refused is False
    assert result.refuse_reason is None
    assert result.answer == "本科生可同时借阅 30 册。"
    assert len(result.citations) == 1
    assert result.citations[0].score == 0.66
    assert result.confidence == 0.66


def test_backfills_citations_by_model_numbers(monkeypatch) -> None:
    """引用按模型挑选的编号回填，只含被引用的条款，且保持编号顺序。"""

    candidates = [_citation(0.9), _citation(0.8), _citation(0.7)]
    monkeypatch.setattr(service, "retrieve_candidates", lambda _q: candidates)
    monkeypatch.setattr(
        service,
        "_generate_answer",
        lambda _q, _c: GeneratedAnswer("答案", [3, 1], 0.9),
    )

    result = service.answer_question("图书馆能借几本书？")

    assert result.refused is False
    assert [c.score for c in result.citations] == [0.7, 0.9]
    assert result.top_retrieval_score == 0.9


def test_refuses_on_invalid_model_output(monkeypatch) -> None:
    monkeypatch.setattr(service, "retrieve_candidates", lambda _q: [_citation(0.66)])

    def _boom(_q: str, _c: list[Citation]) -> GeneratedAnswer:
        raise InvalidModelOutputError("bad json")

    monkeypatch.setattr(service, "_generate_answer", _boom)

    result = service.answer_question("图书馆能借几本书？")

    assert result.refused is True
    assert result.refuse_reason is RefuseReason.INVALID_MODEL_OUTPUT
    assert result.top_retrieval_score == 0.66


def test_refuses_on_no_valid_citation(monkeypatch) -> None:
    monkeypatch.setattr(service, "retrieve_candidates", lambda _q: [_citation(0.66)])
    monkeypatch.setattr(
        service,
        "_generate_answer",
        lambda _q, _c: GeneratedAnswer("有答案但没引用", [], 0.66),
    )

    result = service.answer_question("图书馆能借几本书？")

    assert result.refused is True
    assert result.refuse_reason is RefuseReason.NO_VALID_CITATION


def test_refuses_on_no_evidence_in_corpus(monkeypatch) -> None:
    monkeypatch.setattr(service, "retrieve_candidates", lambda _q: [_citation(0.66)])
    monkeypatch.setattr(
        service,
        "_generate_answer",
        lambda _q, _c: GeneratedAnswer("", [], 0.66),
    )

    result = service.answer_question("图书馆能借几本书？")

    assert result.refused is True
    assert result.refuse_reason is RefuseReason.NO_EVIDENCE_IN_CORPUS


def test_falls_back_to_extractive_answer_when_llm_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(service, "retrieve_candidates", lambda _q: [_citation(0.66)])
    monkeypatch.setattr(service, "_generate_answer", lambda _q, _c: None)

    result = service.answer_question("图书馆能借几本书？")

    assert result.refused is False
    assert "第十条 借阅期限为 60 天。" in result.answer
    assert "以原文为依据" in result.answer


def test_threshold_is_configurable(monkeypatch) -> None:
    monkeypatch.setattr(
        service,
        "settings",
        dataclasses.replace(settings, retrieval_score_threshold=0.80),
    )
    monkeypatch.setattr(service, "retrieve_candidates", lambda _q: [_citation(0.66)])

    result = service.answer_question("图书馆能借几本书？")

    assert result.refused is True
    assert result.refuse_reason is RefuseReason.RETRIEVAL_LOW_SCORE


def test_backfill_sets_citation_index(monkeypatch) -> None:
    """回填引用时应写入模型挑选的 1-based 编号，供前端正文锚点定位。"""

    candidates = [_citation(0.9), _citation(0.8), _citation(0.7)]
    citations = service._backfill_citations(candidates, [3, 1])

    assert [c.citation_index for c in citations] == [3, 1]
    assert [c.score for c in citations] == [0.7, 0.9]


def test_passes_follow_up_questions_when_present(monkeypatch) -> None:
    generated = GeneratedAnswer("本科生可同时借阅 30 册。", [1], 0.66)
    generated.follow_up_questions = ["补考课程冲突怎么处理？", "缓考申请需要哪些材料？"]
    monkeypatch.setattr(service, "retrieve_candidates", lambda _q: [_citation(0.66)])
    monkeypatch.setattr(service, "_generate_answer", lambda _q, _c: generated)

    result = service.answer_question("图书馆能借几本书？")

    assert result.refused is False
    assert result.follow_up_questions == ["补考课程冲突怎么处理？", "缓考申请需要哪些材料？"]


def test_follow_up_none_when_generated_lacks_field(monkeypatch) -> None:
    """生成链尚未返回 follow_up_questions 时，透传结果应为 None。"""

    generated = GeneratedAnswer("本科生可同时借阅 30 册。", [1], 0.66)
    monkeypatch.setattr(service, "retrieve_candidates", lambda _q: [_citation(0.66)])
    monkeypatch.setattr(service, "_generate_answer", lambda _q, _c: generated)

    result = service.answer_question("图书馆能借几本书？")

    assert result.refused is False
    assert result.follow_up_questions is None


def test_follow_up_none_on_refusal(monkeypatch) -> None:
    """拒答路径不产生追问建议。"""

    monkeypatch.setattr(service, "retrieve_candidates", lambda _q: [])

    result = service.answer_question("图书馆能借几本书？")

    assert result.refused is True
    assert result.follow_up_questions is None
