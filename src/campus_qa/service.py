"""问答服务编排层：从问题字符串到 `AnswerResult`。

流程：

- 简单语言检查：不含中文字符则直接拒答；
- 使用 `retriever.retrieve_candidates` 返回若干候选条款；
- 基于检索得分做拒答判定；
- 调用 `chain.generate_answer_with_citations` 生成答案；
- 将候选条款作为引用挂到 `AnswerResult.citations`；
- 生成链不可用（无密钥 / 网络异常）时退化为摘录式回答，
  保证端到端流程不会因外部依赖中断。
"""

from __future__ import annotations

import re
from typing import List, Optional

from .chain import GeneratedAnswer, InvalidModelOutputError
from .config import settings
from .contracts import AnswerResult, Citation, RefuseReason
from .retriever import retrieve_candidates


_SUGGESTED_CONTACT = "请咨询辅导员或相关职能部门（教务处 / 学生处）。"


def _normalize(text: str) -> str:
    """去掉空白与常见中英文标点，并转小写，供寒暄匹配使用。"""

    return re.sub(r"[\s，。！？、,.!?~～]+", "", text).lower()


def _greeting_reply(question: str) -> Optional[str]:
    """识别寒暄/闲聊，返回礼貌回应；非闲聊返回 None。

    命中的问题不进入检索与生成链，直接给出轻量回应，避免
    「你好」「谢谢」这类闲聊被误判为无法回答的制度问题。
    只做精确匹配，因此「你好，缓考怎么申请」这类寒暄 + 真实问题
    不会命中，仍会正常走检索回答。
    """

    q = _normalize(question)
    if not q:
        return None

    # 问候
    if q in {
        "你好", "您好", "你好啊", "您好啊", "你好呀", "嗨", "嗨喽",
        "哈喽", "在吗", "在不在", "在么", "hi", "hello",
    }:
        return (
            "你好！我是校园制度问答助手，可以帮你查询缓考、休学、选课、"
            "奖助学金、图书馆借阅等校内制度问题。请问有什么想了解的？"
        )

    # 感谢
    if q in {"谢谢", "谢谢你", "谢谢啦", "感谢", "多谢", "辛苦了", "thanks", "thankyou"}:
        return "不客气！还有什么校园制度方面的问题，随时可以问我。"

    # 身份 / 能力
    if q in {
        "你是谁", "你叫什么", "你能做什么", "你能干什么", "你有什么用",
        "你会什么", "你能帮我做什么", "自我介绍",
    } or "你是谁" in q:
        return (
            "我是校园制度问答助手，基于学校公开的制度文件回答问题，覆盖缓考、"
            "休学、选课、奖助学金、借阅、奖惩等校内规定。你可以直接问我，比如"
            "「缓考申请最晚什么时候交？」"
        )

    # 道别
    if q in {"再见", "拜拜", "下次见", "回头聊", "bye", "byebye"}:
        return "好的，再见！有制度问题随时来找我。"

    return None


def _contains_chinese(text: str) -> bool:
    """判断字符串中是否含有中文字符。"""

    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


# 任务型请求 / 输出语言要求 / 歧义检测的启发式规则。
_TASK_REQUEST_RE = re.compile(
    r"帮我写|帮我做|写一封|代写|替我写|帮我撰写|帮我起草|请帮我写|帮我写个|帮我做一下|帮我查一下"
)

_OUTPUT_LANGUAGE_RE = re.compile(
    r"用英文|用英语|英文介绍|英文回答|英文说明|英文写|翻译成英文|翻译成英语|请用英文|请用英语"
)

# 泛化大类词：单独出现且无具体限定时，短问题常因信息不足而歧义。
_AMBIGUOUS_TOPICS = ("奖学金", "助学金", "补助", "资助", "评优", "评奖", "荣誉", "处分", "加分", "补贴", "学分")
_AMBIGUOUS_SPECIFIERS = ("国家", "励志", "学业", "研究生", "本科", "综合", "单项", "上海市", "优秀")


def _requires_non_chinese_output(question: str) -> bool:
    """识别「用英文 / 翻译」等要求非中文输出的提问。"""

    return bool(_OUTPUT_LANGUAGE_RE.search(question))


def _is_task_request(question: str) -> bool:
    """识别「帮我写 / 代写 / 写一封」等任务型请求。"""

    return bool(_TASK_REQUEST_RE.search(question))


def _is_ambiguous(question: str) -> bool:
    """识别过短且无具体指向的歧义提问。

    仅当问题足够短（去标点 ≤10 字）、含泛化大类词、且不含具体限定词
    时判为歧义，避免误伤「国家助学金怎么申请」等具体问题。
    """

    cleaned = re.sub(r"[\s，。！？、；：,.!?;:]+", "", question)
    if len(cleaned) > 10:
        return False
    if not any(topic in cleaned for topic in _AMBIGUOUS_TOPICS):
        return False
    if any(spec in cleaned for spec in _AMBIGUOUS_SPECIFIERS):
        return False
    return True


def _build_answer_from_citations(question: str, citations: List[Citation]) -> str:
    """生成摘录式兜底回答。

    当生成链不可用时，直接把最相关条款原文交给用户，
    并提示以原文为准，避免给出无依据的结论。
    """

    if not citations:
        return "暂无匹配条款，请参考原始制度文档。"

    first = citations[0]
    source = first.section or first.document_id
    return (
        f"根据《{source}》相关条款，摘录如下：\n"
        f"{first.snippet}\n"
        "具体执行细则以学校最新正式文件为准，请以原文为依据。"
    )


def _backfill_citations(candidates: List[Citation], numbers: List[int]) -> List[Citation]:
    """把模型挑选的 1-based 编号回填为 Citation 列表。

    编号已在生成链内做范围与去重校验，这里只做防御性下标检查，
    保证回填出的引用一定来自本次检索结果，结构上杜绝编造引用。
    同时把编号写入 ``citation_index``，供前端把正文 [N] 锚点定位到引用。
    """

    citations: List[Citation] = []
    for num in numbers:
        idx = num - 1
        if 0 <= idx < len(candidates):
            citations.append(candidates[idx].model_copy(update={"citation_index": num}))
    return citations


def _generate_answer(question: str, candidates: List[Citation]) -> Optional[GeneratedAnswer]:
    """调用生成链。

    - LLM 不可用（无密钥 / 网络异常 / 超时）→ 返回 None，由调用方摘录兜底；
    - 模型输出非法 JSON → 向上抛出 InvalidModelOutputError，映射为拒答；
    - 成功 → 返回 GeneratedAnswer（answer / citation_numbers / confidence）。
    """

    if not settings.llm_api_key:
        return None

    try:
        from .chain import generate_answer_with_citations

        return generate_answer_with_citations(question, candidates)
    except InvalidModelOutputError:
        raise
    except Exception:
        return None


def answer_question(question: str) -> AnswerResult:
    """从问题字符串生成结构化回答结果。"""

    question = question.strip()
    if not question:
        return AnswerResult(
            question=question,
            refused=True,
            refuse_reason=RefuseReason.EMPTY_QUESTION,
            suggested_contact="请输入具体的校园制度问题。",
        )

    greeting = _greeting_reply(question)
    if greeting is not None:
        return AnswerResult(
            question=question,
            answer=greeting,
            citations=[],
            confidence=1.0,
        )

    if not _contains_chinese(question):
        return AnswerResult(
            question=question,
            refused=True,
            refuse_reason=RefuseReason.UNSUPPORTED_LANGUAGE,
            suggested_contact="请改用中文提问，或咨询辅导员。",
        )

    if _requires_non_chinese_output(question):
        return AnswerResult(
            question=question,
            refused=True,
            refuse_reason=RefuseReason.UNSUPPORTED_LANGUAGE,
            suggested_contact="目前仅支持中文回答，请改用中文提问。",
        )

    if _is_task_request(question):
        return AnswerResult(
            question=question,
            refused=True,
            refuse_reason=RefuseReason.NO_EVIDENCE_IN_CORPUS,
            suggested_contact="该请求超出校园制度问答范围，请咨询辅导员或相关部门。",
        )

    if _is_ambiguous(question):
        return AnswerResult(
            question=question,
            refused=True,
            refuse_reason=RefuseReason.AMBIGUOUS_QUESTION,
            suggested_contact="问题信息不足，请补充具体制度名称或场景后重新提问。",
        )

    candidates = retrieve_candidates(question)
    if not candidates:
        return AnswerResult(
            question=question,
            refused=True,
            refuse_reason=RefuseReason.RETRIEVAL_EMPTY,
            suggested_contact=_SUGGESTED_CONTACT,
        )

    top_score = max(c.score for c in candidates)
    if top_score < settings.retrieval_score_threshold:
        return AnswerResult(
            question=question,
            refused=True,
            refuse_reason=RefuseReason.RETRIEVAL_LOW_SCORE,
            top_retrieval_score=top_score,
            suggested_contact=_SUGGESTED_CONTACT,
        )

    try:
        generated = _generate_answer(question, candidates)
    except InvalidModelOutputError:
        return AnswerResult(
            question=question,
            refused=True,
            refuse_reason=RefuseReason.INVALID_MODEL_OUTPUT,
            top_retrieval_score=top_score,
            suggested_contact=_SUGGESTED_CONTACT,
        )

    if generated is None:
        # 生成链不可用 → 摘录兜底，引用全量挂载。
        answer = _build_answer_from_citations(question, candidates)
        return AnswerResult(
            question=question,
            answer=answer,
            citations=candidates,
            confidence=top_score,
            top_retrieval_score=top_score,
        )

    answer = generated.answer.strip()
    citations = _backfill_citations(candidates, generated.citation_numbers)

    if not answer:
        # 模型明确判定条款支撑不住问题。
        return AnswerResult(
            question=question,
            refused=True,
            refuse_reason=RefuseReason.NO_EVIDENCE_IN_CORPUS,
            top_retrieval_score=top_score,
            suggested_contact=_SUGGESTED_CONTACT,
        )

    if not citations:
        # 有答案但没有任何有效引用编号。
        return AnswerResult(
            question=question,
            refused=True,
            refuse_reason=RefuseReason.NO_VALID_CITATION,
            top_retrieval_score=top_score,
            suggested_contact=_SUGGESTED_CONTACT,
        )

    follow_ups = getattr(generated, "follow_up_questions", None)
    return AnswerResult(
        question=question,
        answer=answer,
        citations=citations,
        confidence=generated.confidence,
        follow_up_questions=list(follow_ups) if follow_ups else None,
        top_retrieval_score=top_score,
    )


if __name__ == "__main__":  # pragma: no cover
    # 简单自检：问一个示例问题，打印 answer 与引用摘要。
    demo_q = "图书馆能借几本书？"
    result = answer_question(demo_q)
    print("demo question:", demo_q)
    if result.refused:
        print("refused:", result.refuse_reason)
    else:
        print("answer:\n", result.answer)
        for c in result.citations[:3]:
            print(f"- [{c.document_id}] {c.snippet[:40]}... (score={c.score:.3f})")
