"""契约与配置的快速回归测试（不联网、不加载向量模型）。"""

from __future__ import annotations

from pathlib import Path

from campus_qa.config import ROOT_DIR, settings
from campus_qa.contracts import AnswerResult, Citation, RefuseReason


def test_refuse_reason_values_match_api_contract() -> None:
    """拒答 code 是机读契约，取值不能被随意改动。"""

    expected = {
        "empty_question",
        "unsupported_language",
        "retrieval_empty",
        "retrieval_low_score",
        "invalid_model_output",
        "no_valid_citation",
        "no_evidence_in_corpus",
    }
    assert {reason.value for reason in RefuseReason} == expected


def test_answer_result_defaults() -> None:
    result = AnswerResult(question="图书馆开放时间？")
    assert result.refused is False
    assert result.refuse_reason is None
    assert result.citations == []
    assert result.answer is None


def test_citation_requires_document_id_and_snippet() -> None:
    citation = Citation(document_id="doc-1", snippet="第十条 借阅期限为 60 天。", score=0.8)
    assert citation.page is None
    assert citation.section is None


def test_settings_paths_point_into_repository() -> None:
    """路径必须指向仓库内的 data 目录，而不是仓库的上一级。"""

    assert ROOT_DIR == Path(__file__).resolve().parents[1]
    assert settings.data_dir == ROOT_DIR / "data"
    assert settings.chunks_path == ROOT_DIR / "data" / "chunks.jsonl"
    assert settings.index_dir == ROOT_DIR / "data" / "index"


def test_embedding_model_name_is_fully_qualified() -> None:
    """短名会在配置层补全，避免 SentenceTransformer 找不到模型。"""

    assert "/" in settings.embedding_model
