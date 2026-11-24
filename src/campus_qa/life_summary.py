"""生活库回答精简：只保留与问题相关的内容，不整段贴资料。

生活库本身是 BM25 检索，命中段落往往是一整块服务说明（营业时间、位置、
流程、反馈渠道全在一起）。直接拼起来会把无关内容也塞给学生。

本模块做两件事（按顺序兜底）：

1. **模型精简**（配置了 LLM 时）：把问题 + 命中段落交给模型，要求只回答
   所问内容、控制在几句以内、不添加资料里没有的信息；
2. **句级抽取**（无密钥 / 调用失败 / 输出为空时）：把段落切成句子，按与
   问题的字符二元组重合度打分，只取最相关的少数几句（保持原文顺序）。

两种模式都保留原始出处（`sources`），只是不再把整段原文当答案。
"""

from __future__ import annotations

import re
from typing import List, Sequence

from .config import settings
from .life import LifeHit

NOTE = "（以现场公告为准）"

_SYSTEM_PROMPT = (
    "你是校园生活服务问答助手。根据给定的资料回答学生的问题，严格遵守：\n"
    "1. 只回答问题所问的内容，资料里与问题无关的部分一律不要提；\n"
    "2. 直接给结论，用 1–3 句话或最多 3 条短要点，不要开场白、不要复述问题；\n"
    "3. 保留关键信息：时间、地点、流程、费用、电话、注意事项；\n"
    "4. 只用资料里的信息，不要推测或补充资料里没有的内容；\n"
    "5. 不要加免责声明，不要输出 Markdown 标题。"
)

_USER_TEMPLATE = "问题：{question}\n\n资料：\n{context}\n\n请直接给出简短回答。"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？])")
# 问题里的疑问词 / 泛词，抽句时不参与打分
_STOPWORDS = set(
    "的 了 吗 呢 啊 吧 是 在 有 和 与 请问 一下 怎么 怎样 如何 什么 哪里 哪儿 几点 多少 "
    "可以 能 不能 是否 需要 应该 我们 我 你 他 学校 校内 校园 请问 现在 今天".split()
)
_TIME_RE = re.compile(r"\d{1,2}[:：]\d{2}|\d{1,2}\s*[点时]\b|\d{1,2}[：:]\d{2}\s*[-–—~至到]\s*\d{1,2}[：:]\d{2}")
_LOCATION_RE = re.compile(r"位于|地址|路\s*\d+号|东南角|西北角|东北角|西南角|栋|楼|层|室|驿站|窗口|(?:充值|取件|服务|回收|投放)点")
_PROCESS_RE = re.compile(r"登录|点击|选择|申请|办理|步骤|流程|扫码|预约|填写|提交|携带|凭|需|先|再")
# 问题类型 → (该类型句子的加权, 匹配该类型句子的正则)
_QUESTION_TYPE_MARKERS = (
    (("几点", "时间", "时段", "什么时候", "开门", "营业", "放假"), _TIME_RE, 3),
    (("哪里", "哪儿", "地址", "位置", "怎么去", "在哪个", "哪个楼"), _LOCATION_RE, 3),
    (("怎么", "如何", "流程", "步骤", "办理", "申请", "充值", "补办", "挂失"), _PROCESS_RE, 2),
)


def _context(hits: Sequence[LifeHit], limit: int = 4) -> str:
    blocks = []
    for index, hit in enumerate(hits[:limit], start=1):
        body = hit.chunk.text.strip().replace("\n", " ")
        blocks.append(f"[{index}] {hit.chunk.title}：{body}")
    return "\n".join(blocks)


def _llm_condense(question: str, hits: Sequence[LifeHit]) -> str:
    """调用 LLM 做精简；任何失败都抛给上层走抽取兜底。"""

    if not getattr(settings, "llm_api_key", None):
        raise RuntimeError("no llm api key")

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
        timeout=20,
        max_retries=0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("user", _USER_TEMPLATE)]
    )
    chain = prompt | llm | StrOutputParser()
    text = str(chain.invoke({"question": question, "context": _context(hits)}) or "")
    text = text.strip()
    if not text or len(text) < 2:
        raise RuntimeError("empty llm output")
    return text


def _bigrams(text: str) -> set[str]:
    cleaned = re.sub(r"[\s，。！？、；：“”‘’（）()《》\-—,.!?;:\[\]【】]", "", text)
    for word in _STOPWORDS:
        cleaned = cleaned.replace(word, " ")
    cleaned = cleaned.replace(" ", "")
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)} if len(cleaned) > 1 else set()


def _extractive(question: str, hits: Sequence[LifeHit], max_sentences: int = 3) -> str:
    """句级抽取：按与问题的二元组重合度挑最相关的几句，保持原文顺序。"""

    query_grams = _bigrams(question)
    boosts = [
        (pattern, weight)
        for markers, pattern, weight in _QUESTION_TYPE_MARKERS
        if any(m in question for m in markers)
    ]
    scored: List[tuple[int, int, int, str]] = []  # (hit_index, sent_index, score, sentence)

    for hit_index, hit in enumerate(hits[:3]):
        body = hit.chunk.text.strip().replace("\n", " ")
        sentences = [s.strip(" -•·\t") for s in _SENTENCE_SPLIT_RE.split(body)]
        sentences = [s for s in sentences if s]
        for sent_index, sentence in enumerate(sentences):
            if len(sentence) < 4:
                continue
            score = len(query_grams & _bigrams(sentence)) if query_grams else 0
            score += sum(weight for pattern, weight in boosts if pattern.search(sentence))
            # 纯标题行（以冒号结尾且很短）信息量低
            if sentence.endswith(("：", ":")) and len(sentence) <= 20:
                score -= 2
            scored.append((hit_index, sent_index, score, sentence))

    if not scored:
        return ""

    picked = sorted(scored, key=lambda item: item[2], reverse=True)
    if picked[0][2] == 0:
        # 完全没有词汇重合：退到首条命中段的前两句
        top_hit = sorted(item for item in scored if item[0] == 0)[:2]
        picked = top_hit or picked[:max_sentences]
    else:
        # 相对阈值：只保留得分不低于最高分 60% 的句子，避免凑数
        threshold = max(1, picked[0][2] * 0.6)
        picked = [item for item in picked if item[2] >= threshold][:max_sentences]
    picked.sort(key=lambda item: (item[0], item[1]))

    # 总长度控制：超过约 260 字就从相关度最低的句子开始丢弃
    while len(picked) > 1 and sum(len(item[3]) for item in picked) > 260:
        picked.remove(min(picked, key=lambda item: item[2]))

    return " ".join(item[3] for item in picked)


def condense(question: str, hits: Sequence[LifeHit]) -> str:
    """返回精简后的生活类回答；无命中返回空串。"""

    if not hits:
        return ""
    try:
        return _llm_condense(question, hits)
    except Exception:  # noqa: BLE001 - 无密钥 / 超时 / 输出异常都退到抽取
        return _extractive(question, hits)


__all__ = ["condense", "NOTE"]
