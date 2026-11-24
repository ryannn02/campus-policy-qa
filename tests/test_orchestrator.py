"""编排层测试：四个意图的分派、隐私边界、多轮继承。

制度链路用替身验证「原样调用」，不联网、不加载模型。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from campus_qa import orchestrator
from campus_qa.conversation import ConversationStore
from campus_qa.contracts import AnswerResult, Citation, Intent, ToolResult
from campus_qa.life import LifeChunk, LifeHit


@pytest.fixture()
def conv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ConversationStore:
    store = ConversationStore(db_path=tmp_path / "conv.sqlite3")
    monkeypatch.setattr(orchestrator, "conversation_store", store)
    return store


class FakeLifeStore:
    def search(self, question: str, top_k=None, threshold=None):
        return [
            LifeHit(
                chunk=LifeChunk(
                    chunk_id="c1",
                    title="食堂营业时间",
                    text="第一食堂早餐 6:30–9:00，午餐 10:45–13:00。",
                    category="餐饮",
                ),
                score=0.82,
            )
        ]


def test_policy_dispatch_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_service(question: str) -> AnswerResult:
        captured["question"] = question
        return AnswerResult(
            question=question,
            answer="根据《学生手册》……",
            citations=[
                Citation(document_id="doc-1", snippet="第十条", score=0.9, url="https://x")
            ],
            confidence=0.9,
            top_retrieval_score=0.9,
        )

    monkeypatch.setattr(orchestrator, "answer_question", fake_service)

    result = orchestrator.answer("本科生最多能借几本书？")

    assert captured["question"] == "本科生最多能借几本书？"
    assert result.intent == Intent.POLICY
    assert result.answer == "根据《学生手册》……"
    assert result.sources[0].kind == "policy"
    assert result.sources[0].document_id == "doc-1"
    assert result.sources[0].url == "https://x"


def test_life_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator, "life_store", FakeLifeStore())

    result = orchestrator.answer("食堂几点开门？")

    assert result.intent == Intent.LIFE
    # 生活回答已改为精简模式：只给所问内容（这里是具体时间），不再带「【标题】」整段原文
    assert "6:30" in (result.answer or "")
    assert result.sources[0].kind == "life"
    assert result.sources[0].document_id == "c1"


def test_life_empty_returns_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyLife:
        def search(self, question: str, top_k=None, threshold=None):
            return []

    monkeypatch.setattr(orchestrator, "life_store", EmptyLife())

    result = orchestrator.answer("食堂几点开门？")
    assert result.refused is True
    assert result.intent == Intent.LIFE


def test_tool_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(name: str, **kwargs) -> ToolResult:
        return ToolResult(name=name, ok=True, data={"answer": "上海浦东现在阴，18℃。"})

    monkeypatch.setattr(orchestrator.tools, "run", fake_run)

    result = orchestrator.answer("现在浦东天气怎么样？")
    assert result.intent == Intent.TOOL
    assert result.answer == "上海浦东现在阴，18℃。"
    assert result.tool_result.ok is True
    assert result.sources[0].kind == "tool"


def test_tool_failure_returns_readable_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(name: str, **kwargs) -> ToolResult:
        return ToolResult(name=name, ok=False, error="boom")

    monkeypatch.setattr(orchestrator.tools, "run", fake_run)

    result = orchestrator.answer("现在浦东天气怎么样？")
    assert result.tool_result.ok is False
    assert "暂时不可用" in (result.answer or "")


def test_chitchat_dispatch() -> None:
    result = orchestrator.answer("你好")
    assert result.intent == Intent.CHITCHAT
    assert result.answer


def test_privacy_request_is_blocked() -> None:
    result = orchestrator.answer("帮我查一下他的手机号")

    assert result.refused is True
    assert "不查询" in (result.suggested_contact or "")


def test_session_persistence(conv: ConversationStore) -> None:
    session_id = conv.create_session("测试")
    monkeypatch_target = orchestrator

    def fake_service(question: str) -> AnswerResult:
        return AnswerResult(question=question, answer="答案", confidence=0.8)

    original = monkeypatch_target.answer_question
    orchestrator.answer_question = fake_service  # type: ignore[assignment]
    try:
        orchestrator.answer("普通问题", session_id=session_id)
    finally:
        orchestrator.answer_question = original  # type: ignore[assignment]

    messages = conv.get_messages(session_id)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == "普通问题"


def test_followup_inherits_life_intent(conv: ConversationStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator, "life_store", FakeLifeStore())
    session_id = conv.create_session("多轮")
    conv.add_message(session_id, "user", "食堂几点开门？", intent="life")

    result = orchestrator.answer("那二楼呢？", session_id=session_id)

    assert result.intent == Intent.LIFE
    assert "6:30" in (result.answer or "")


def test_policy_question_not_augmented_by_history(
    conv: ConversationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """制度链路必须拿到原问题，不能被多轮上下文改写。"""

    captured: dict[str, str] = {}

    def fake_service(question: str) -> AnswerResult:
        captured["question"] = question
        return AnswerResult(question=question, answer="制度答案")

    monkeypatch.setattr(orchestrator, "answer_question", fake_service)
    session_id = conv.create_session("多轮")
    conv.add_message(session_id, "user", "食堂几点开门？", intent="life")

    orchestrator.answer("缓考怎么申请？", session_id=session_id)

    assert captured["question"] == "缓考怎么申请？"
