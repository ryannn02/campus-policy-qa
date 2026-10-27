"""检索层的纯逻辑测试（不加载 bge 模型、不读真实索引）。"""

from __future__ import annotations

import numpy as np
import pytest

from campus_qa.contracts import Citation
from campus_qa.retriever import _coerce_page


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        (3, 3),
        ("42", 42),
        ("p.12", 12),
        (7.0, 7),
    ],
)
def test_coerce_page(raw: object, expected: object) -> None:
    """chunks.jsonl 里 1588 条记录的 page 是空字符串，必须清洗。"""

    assert _coerce_page(raw) == expected


def test_citation_accepts_coerced_page() -> None:
    """清洗后的 page 能通过 pydantic 校验。"""

    citation = Citation(
        document_id="doc-1",
        snippet="第十条 借阅期限为 60 天。",
        score=0.72,
        page=_coerce_page(""),
    )
    assert citation.page is None


def test_cosine_ranking_is_deterministic() -> None:
    """暴力余弦检索的核心运算：归一化后点积即余弦相似度。"""

    index = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype="float32")
    query = np.array([1.0, 0.0], dtype="float32")

    normalized_index = index / np.linalg.norm(index, axis=1, keepdims=True)
    scores = normalized_index @ (query / np.linalg.norm(query))

    assert scores.argmax() == 0
    assert scores[0] == pytest.approx(1.0)
    assert scores[2] == pytest.approx(0.70710678, rel=1e-6)
