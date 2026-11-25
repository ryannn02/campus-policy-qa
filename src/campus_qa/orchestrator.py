"""统一问答入口：路由 + 制度链路 + 生活库 + 工具 + 会话。

分层原则（重要）：

- **制度问答一行不改**。`INTENT=POLICY` 时原样调用
  `service.answer_question(question)`，不改检索、不改提示词、不改拒答，
  保证 `refusal_rate=1.000` / `Recall@5=0.862` 不被新功能影响。
- 新能力全部是「旁边挂」：生活库自己的语料与 BM25、工具自己的确定性
  答案、会话层自己存的多轮上下文。
- 只有「追问且路由器判定为继承」时，才把上一轮问题拼进检索 query；
  单轮问题（也就是评测集里的题）走的是完全不变的路径。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .contracts import AnswerResult, Intent, RefuseReason, Source, ToolResult
from .conversation import store as conversation_store
from .life import LifeHit, store as life_store
from . import life_summary
from .router import RouteDecision, classify, is_followup
from .safety import privacy_request_reason
from .service import answer_question
from . import tools


_PRIVACY_MESSAGE = (
    "本系统不查询、不收集任何人的个人信息（手机号、学号、住址、社交账号等）。"
    "如果你需要找人，建议通过辅导员或班级群；如果是校园事务，我可以帮你查制度或生活服务。"
)


def _policy_sources(result: AnswerResult) -> List[Source]:
    return [
        Source(
            kind="policy",
            title=citation.document_id,
            snippet=citation.snippet,
            score=citation.score,
            document_id=citation.document_id,
            url=citation.url,
        )
        for citation in result.citations
    ]


def _fallback_body(hits: List[LifeHit]) -> str:
    """精简失败时的最后兜底：首条命中的前两句，而不是整段。"""

    first = hits[0].chunk.text.strip().replace("\n", " ")
    sentences = [part.strip() for part in re.split(r"(?<=[。！？；])", first) if part.strip()]
    return " ".join(sentences[:2]) if sentences else first[:120]


def _life_answer(question: str, hits: List[LifeHit], focus: Optional[str] = None) -> AnswerResult:
    """`focus` 是用来做精简的问题：追问时带上上一轮，模型才知道「那二楼」在问什么。"""

    if not hits:
        return AnswerResult(
            question=question,
            refused=True,
            refuse_reason=RefuseReason.RETRIEVAL_EMPTY,
            suggested_contact="生活库里暂时没有这条信息，可以问辅导员或对应部门。",
            intent=Intent.LIFE,
        )

    body = life_summary.condense(focus or question, hits) or _fallback_body(hits)
    answer = f"{body}{life_summary.NOTE}"

    sources = [
        Source(
            kind="life",
            title=hit.chunk.title,
            snippet=hit.chunk.text.strip(),
            score=hit.score,
            document_id=hit.chunk.chunk_id,
        )
        for hit in hits
    ]
    return AnswerResult(
        question=question,
        answer=answer,
        sources=sources,
        confidence=hits[0].score,
        top_retrieval_score=hits[0].score,
        intent=Intent.LIFE,
    )


def _tool_answer(question: str, tool_name: str) -> AnswerResult:
    result: ToolResult = tools.run(tool_name)
    label = next(
        (item["label"] for item in tools.describe() if item["name"] == tool_name),
        tool_name,
    )

    if result.ok:
        answer = str(result.data.get("answer") or "工具已返回结果。")
        sources = [
            Source(
                kind="tool",
                title=f"{label}（实时数据）",
                snippet=answer,
                score=1.0,
            )
        ]
    else:
        answer = {
            "weather": "天气服务暂时不可用，请稍后再试，或直接看手机天气。",
        }.get(tool_name, "这个工具暂时不可用，请稍后再试。")
        sources = []

    return AnswerResult(
        question=question,
        answer=answer,
        sources=sources,
        confidence=1.0 if result.ok else None,
        intent=Intent.TOOL,
        tool_result=result,
    )


def _chitchat_answer(question: str) -> AnswerResult:
    # 寒暄交给原 service 处理（它的措辞已经过评测与打磨）
    result = answer_question(question)
    if result.answer and not result.refused:
        result.intent = Intent.CHITCHAT
        return result

    return AnswerResult(
        question=question,
        answer=(
            "我是校园问答助手，可以帮你做三类事：\n"
            "1. 查制度：学分、选课、补考、休学、奖助学金、图书馆借阅等；\n"
            "2. 查生活服务：食堂、快递、校园卡、热水、网络、打印、校医院等；\n"
            "3. 查实时信息：天气、现在第几周、常用电话。"
        ),
        confidence=1.0,
        intent=Intent.CHITCHAT,
    )


def _augment_for_followup(question: str, last_question: Optional[str]) -> str:
    """追问时把上一轮问题拼进检索 query。

    只影响 `reason == "inherit"` 的追问；单轮问题完全不走这段逻辑。
    """

    if not last_question:
        return question
    last = last_question.strip()
    if not last or last == question.strip():
        return question
    return f"{last} {question}"


def route_question(
    question: str,
    session_id: Optional[str] = None,
) -> RouteDecision:
    """对外暴露路由结果，便于调试与前端展示。"""

    previous = conversation_store.last_intent(session_id) if session_id else None
    last_question = conversation_store.last_user_question(session_id) if session_id else None
    previous_intent = Intent(previous) if previous in {item.value for item in Intent} else None
    return classify(question, previous_intent=previous_intent, last_question=last_question)


def answer(question: str, session_id: Optional[str] = None) -> AnswerResult:
    """从问题字符串生成结构化回答（Web / CLI / 渠道共用）。"""

    text = (question or "").strip()

    # 0) 隐私边界：任何「查别人信息」的请求直接拒绝，不进入检索
    if text and privacy_request_reason(text):
        result = AnswerResult(
            question=text,
            refused=True,
            refuse_reason=RefuseReason.NO_EVIDENCE_IN_CORPUS,
            suggested_contact=_PRIVACY_MESSAGE,
        )
        _persist(session_id, text, result, intent="boundary")
        return result

    previous_intent: Optional[Intent] = None
    last_question: Optional[str] = None
    if session_id:
        previous = conversation_store.last_intent(session_id)
        if previous in {item.value for item in Intent}:
            previous_intent = Intent(previous)
        last_question = conversation_store.last_user_question(session_id)

    decision = classify(text, previous_intent=previous_intent, last_question=last_question)

    if decision.intent == Intent.LIFE:
        query = text
        if decision.reason == "inherit" and last_question:
            query = _augment_for_followup(text, last_question)
        result = _life_answer(text, life_store.search(query), focus=query)
    elif decision.intent == Intent.TOOL:
        result = _tool_answer(text, decision.tool or "campus_time")
    elif decision.intent == Intent.CHITCHAT:
        result = _chitchat_answer(text)
    else:
        # 制度链路：原样调用，不加上下文、不改参数
        result = answer_question(text)
        result.intent = Intent.POLICY
        result.sources = _policy_sources(result)

    _persist(session_id, text, result, intent=decision.intent.value)
    return result


def _persist(
    session_id: Optional[str],
    question: str,
    result: AnswerResult,
    intent: str,
) -> None:
    """落库多轮上下文；没有 session_id 时纯无状态，不写任何东西。"""

    if not session_id:
        return
    try:
        content = result.answer or result.suggested_contact or ""
        conversation_store.add_message(
            session_id, "user", question, intent=intent
        )
        conversation_store.add_message(
            session_id,
            "assistant",
            content,
            intent=intent,
            sources=[source.model_dump(mode="json") for source in result.sources],
        )
        conversation_store.touch_session(session_id, intent)
    except Exception:  # noqa: BLE001 - 会话写入失败不影响回答本身
        pass


def capabilities() -> Dict[str, object]:
    """服务能力清单，`/api/capabilities` 与前端「能做什么」共用。"""

    return {
        "intents": [item.value for item in Intent],
        "tools": tools.describe(),
        "life_library": life_store.stats(),
    }


def health_snapshot() -> Tuple[bool, Dict[str, object]]:
    """健康检查：索引存在性与库规模。"""

    life_stats = life_store.stats()
    ok = life_stats.get("load_error") is None
    return ok, {"life": life_stats, "tools": [item["name"] for item in tools.describe()]}
