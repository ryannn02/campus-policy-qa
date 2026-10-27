"""生成链实现：版本化提示词 + JSON 结构化输出 + 引用编号回填。

职责（成员 B，RAG 核心）：
- 从 ``prompts/answer/v1.md`` 读取版本化提示词模板；
- 使用 LangChain 组合 ``ChatPromptTemplate → ChatOpenAI → StrOutputParser``；
- 要求模型输出合法 JSON：``{"answer": ..., "citations": [编号]}``；
- 校验引用编号必须落在候选条款 ``[1..N]`` 范围内，杜绝编造引用；
- 返回 ``GeneratedAnswer``（answer / citation_numbers / confidence），
  由 service 层按编号回填为 ``Citation`` 列表。

当 v1.md 缺失时回退到内置默认模板；模型输出非法 JSON 时抛
``InvalidModelOutputError``，交由 service 层映射为拒答。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from .config import settings
from .contracts import Citation


PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts" / "answer"
# 提示词按版本演进：优先最新版本，vN 缺失或损坏时回退上一版本。
# v1 永不删，作为基线；v2 提升回答的自然与亲和度。
_PROMPT_FILES = [PROMPT_DIR / "v2.md", PROMPT_DIR / "v1.md"]


@dataclass
class GeneratedAnswer:
    """生成链的结构化返回：答案 + 引用编号 + 置信度。"""

    answer: str
    citation_numbers: List[int]
    confidence: float


class InvalidModelOutputError(Exception):
    """模型输出无法解析为合法 JSON，或缺少必要字段。"""


# v1.md 缺失或损坏时的内置兜底模板。
_FALLBACK_SYSTEM = (
    "你是校园制度问答助手，只根据下面提供的制度条款片段回答问题。"
    "不要编造不存在于片段中的结论；如果片段无法支撑完整答案，"
    "请尽量说明已有条款能覆盖的部分，并提醒用户咨询教务处或辅导员。"
)

_FALLBACK_HUMAN = (
    "用户问题：\n{question}\n\n"
    "相关制度条款片段：\n{context}\n\n"
    "请用简洁的中文回答用户问题，并引用关键制度条款的要点。"
)

_prompt_cache: Optional[Tuple[str, str]] = None


def _extract_section(text: str, tag: str) -> str:
    """从 v1.md 提取 ``<!-- SYSTEM -->`` / ``<!-- HUMAN -->`` 段。"""

    pattern = re.compile(rf"<!--\s*{tag}\s*-->(.*?)(?=<!--|\Z)", re.DOTALL)
    match = pattern.search(text)
    if not match:
        return ""
    return match.group(1).strip()


def _load_prompt() -> Tuple[str, str]:
    """读取版本化提示词，失败时回退内置默认模板。"""

    global _prompt_cache
    if _prompt_cache is not None:
        return _prompt_cache

    system = _FALLBACK_SYSTEM
    human = _FALLBACK_HUMAN
    for prompt_file in _PROMPT_FILES:
        try:
            text = prompt_file.read_text(encoding="utf-8")
        except OSError:
            continue
        loaded_system = _extract_section(text, "SYSTEM")
        loaded_human = _extract_section(text, "HUMAN")
        if loaded_system and loaded_human:
            system, human = loaded_system, loaded_human
            break

    _prompt_cache = (system, human)
    return _prompt_cache


def _build_prompt() -> ChatPromptTemplate:
    """基于版本化模板构建 LangChain 提示词。"""

    system, human = _load_prompt()
    return ChatPromptTemplate.from_messages([("system", system), ("human", human)])


def _format_context(candidates: List[Citation]) -> str:
    """将候选条款列表拼接为带编号的上下文字符串。"""

    parts = []
    for idx, c in enumerate(candidates, start=1):
        parts.append(
            f"[{idx}] 文档 {c.document_id} 页 {c.page or '-'} 段 {c.section or '-'}:\n"
            f"{c.snippet}\n"
        )
    return "\n".join(parts)


def _extract_json_object(text: str) -> str:
    """从模型输出中截取最外层 JSON 对象。

    容忍模型在 JSON 前后附带解释性文字或 markdown 代码围栏，
    只取第一个 ``{`` 到最后一个 ``}`` 之间的内容。
    """

    text = (text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _normalize_citations(raw: object, n: int) -> List[int]:
    """把模型给出的 citations 规整为去重、落在 [1..n] 的编号列表。

    非法项（非整数、越界、布尔）直接丢弃，保证回填时只会命中
    本次检索真正返回的条款，从结构上杜绝编造引用。
    """

    if not isinstance(raw, list):
        return []

    numbers: List[int] = []
    seen: set[int] = set()
    for item in raw:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            num = item
        elif isinstance(item, str) and item.strip().isdigit():
            num = int(item.strip())
        else:
            continue
        if 1 <= num <= n and num not in seen:
            seen.add(num)
            numbers.append(num)
    return numbers


def _parse_llm_output(raw: str, n: int) -> Tuple[str, List[int]]:
    """解析模型输出为 (answer, citation_numbers)。

    JSON 解析失败或不是对象时抛 ``InvalidModelOutputError``。
    """

    text = _extract_json_object(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidModelOutputError("模型输出不是合法 JSON") from exc

    if not isinstance(data, dict):
        raise InvalidModelOutputError("模型输出 JSON 不是对象")

    answer = str(data.get("answer", "")).strip()
    numbers = _normalize_citations(data.get("citations"), n)
    return answer, numbers


def generate_answer_with_citations(
    question: str,
    candidates: List[Citation],
) -> GeneratedAnswer:
    """调用单一 LLM 生成带引用的结构化回答。

    - 输入：原始问题与检索得出的候选条款；
    - 输出：``GeneratedAnswer``，含 answer、citation_numbers、confidence；
    - 无候选条款时直接返回空结果，不发起 LLM 调用。

    模型输出非法 JSON 时抛 ``InvalidModelOutputError``，
    由 service 层捕获并映射为 ``invalid_model_output`` 拒答。
    """

    if not candidates:
        return GeneratedAnswer(answer="", citation_numbers=[], confidence=0.0)

    prompt = _build_prompt()
    llm = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
        timeout=60,
        max_retries=1,
    )
    chain = prompt | llm | StrOutputParser()

    context = _format_context(candidates)
    raw = chain.invoke({"question": question, "context": context})

    answer, numbers = _parse_llm_output(raw, len(candidates))

    # 置信度：检索 top-1 分按有效引用率折减，避免"引用多但检索分低"虚高。
    top_score = max(c.score for c in candidates)
    citation_ratio = len(numbers) / max(len(candidates), 1)
    confidence = float(top_score * (0.5 + 0.5 * citation_ratio))

    return GeneratedAnswer(answer=answer, citation_numbers=numbers, confidence=confidence)


__all__ = [
    "GeneratedAnswer",
    "InvalidModelOutputError",
    "generate_answer_with_citations",
]
