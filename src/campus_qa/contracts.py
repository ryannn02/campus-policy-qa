"""基础契约：回答结果与引用结构，以及拒答原因枚举。"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Intent(str, Enum):
    """问题意图分类（M5 路由器输出）。

    制度问答仍然走原有 service 链路；其余三类由 orchestrator 分派，
    互不影响，保证制度库的拒答与召回指标不被稀释。
    """

    POLICY = "policy"
    LIFE = "life"
    TOOL = "tool"
    CHITCHAT = "chitchat"


class Source(BaseModel):
    """跨库统一的来源条目（前端引用卡片使用）。

    与 `Citation` 并存：`Citation` 是制度条款契约（字段不能动），
    `Source` 是新链路（生活库 / 工具 / 渠道）的通用来源描述。
    """

    kind: str  # policy / life / tool
    title: str
    snippet: str = ""
    score: float = 0.0
    document_id: Optional[str] = None
    url: Optional[str] = None


class ToolResult(BaseModel):
    """工具调用结果（天气、时间、校历、常用电话）。"""

    name: str
    ok: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class RefuseReason(str, Enum):
    """不可回答 / 拒答原因枚举。

    code 取值与 docs/architecture.md 及项目说明文档保持一致。
    """

    EMPTY_QUESTION = "empty_question"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    AMBIGUOUS_QUESTION = "ambiguous_question"
    RETRIEVAL_EMPTY = "retrieval_empty"
    RETRIEVAL_LOW_SCORE = "retrieval_low_score"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    NO_VALID_CITATION = "no_valid_citation"
    NO_EVIDENCE_IN_CORPUS = "no_evidence_in_corpus"


class Citation(BaseModel):
    """单条引用条款的信息。"""

    document_id: str
    snippet: str
    score: float

    # 方便前端跳转原文的定位信息
    page: Optional[int] = None
    section: Optional[str] = None
    url: Optional[str] = None

    # 模型挑选的 1-based 引用编号，供前端把正文 [N] 锚点定位到引用列表。
    citation_index: Optional[int] = None


class AnswerResult(BaseModel):
    """系统对单个提问的回答结果。"""

    question: str

    # 可回答题：给出 answer + citations
    answer: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    confidence: Optional[float] = None

    # 大模型基于当前问题与回答生成的建议追问（2–4 条）；拒答 / 摘录兜底为 None。
    follow_up_questions: Optional[List[str]] = None

    # 不可回答题：标记 refused + refuse_reason
    refused: bool = False
    refuse_reason: Optional[RefuseReason] = None
    top_retrieval_score: Optional[float] = None
    suggested_contact: Optional[str] = None

    # ---- M5 新增（全部带默认值，旧调用方与旧契约零影响）----
    # 路由命中的意图；None 表示未经路由器（例如直接调用 service）。
    intent: Optional[Intent] = None
    # 生活库 / 工具链路的来源卡片；制度问答仍以 citations 为准。
    sources: List[Source] = Field(default_factory=list)
    # 工具调用的结构化结果：前端可直接渲染，不必解析自然语言。
    tool_result: Optional[ToolResult] = None

