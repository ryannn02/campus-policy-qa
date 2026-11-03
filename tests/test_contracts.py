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
        "ambiguous_question",
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
    assert result.follow_up_questions is None


def test_citation_requires_document_id_and_snippet() -> None:
    citation = Citation(document_id="doc-1", snippet="第十条 借阅期限为 60 天。", score=0.8)
    assert citation.page is None
    assert citation.section is None
    assert citation.citation_index is None


def test_citation_index_round_trips() -> None:
    citation = Citation(
        document_id="doc-1", snippet="第十条 借阅期限为 60 天。", score=0.8, citation_index=3
    )
    assert citation.citation_index == 3


def test_settings_paths_point_into_repository() -> None:
    """路径必须指向仓库内的 data 目录，而不是仓库的上一级。"""

    assert ROOT_DIR == Path(__file__).resolve().parents[1]
    assert settings.data_dir == ROOT_DIR / "data"
    assert settings.chunks_path == ROOT_DIR / "data" / "chunks.jsonl"
    assert settings.index_dir == ROOT_DIR / "data" / "index"


def test_embedding_model_name_is_fully_qualified() -> None:
    """短名会在配置层补全，避免 SentenceTransformer 找不到模型。"""

    assert "/" in settings.embedding_model


def test_index_dir_configurable_via_env(monkeypatch) -> None:
    """RETRIEVAL_INDEX_DIR 环境变量应能切换向量索引目录。"""

    from campus_qa.config import load_settings

    monkeypatch.setenv("RETRIEVAL_INDEX_DIR", "/tmp/test_index")
    assert load_settings().index_dir == Path("/tmp/test_index")


def test_index_dir_defaults_to_data_index(monkeypatch) -> None:
    """未设置 RETRIEVAL_INDEX_DIR 时默认 data/index，保持既有行为。"""

    from campus_qa.config import load_settings

    monkeypatch.delenv("RETRIEVAL_INDEX_DIR", raising=False)
    assert load_settings().index_dir == ROOT_DIR / "data" / "index"


def test_bge_m3_alias_resolves_to_full_name(monkeypatch) -> None:
    """短名 bge-m3 应补全为 BAAI/bge-m3。"""

    from campus_qa.config import load_settings

    monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3")
    assert load_settings().embedding_model == "BAAI/bge-m3"
