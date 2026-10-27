"""基础契约：回答结果与引用结构，以及拒答原因枚举。"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class RefuseReason(str, Enum):
    """不可回答 / 拒答原因枚举。

    code 取值与 docs/architecture.md 及项目说明文档保持一致。
    """

    EMPTY_QUESTION = "empty_question"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
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


class AnswerResult(BaseModel):
    """系统对单个提问的回答结果。"""

    question: str

    # 可回答题：给出 answer + citations
    answer: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    confidence: Optional[float] = None

    # 不可回答题：标记 refused + refuse_reason
    refused: bool = False
    refuse_reason: Optional[RefuseReason] = None
    top_retrieval_score: Optional[float] = None
    suggested_contact: Optional[str] = None

