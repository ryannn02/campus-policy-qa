"""生活库检索测试（临时语料 + 内存索引，不联网）。"""

from __future__ import annotations

import json
from pathlib import Path

from campus_qa.life import LifeChunk, LifeStore, build_index, load_chunks


def _write_chunks(path: Path) -> None:
    rows = [
        {
            "chunk_id": "c1",
            "title": "食堂营业时间",
            "text": "第一食堂早餐 6:30–9:00，午餐 10:45–13:00，晚餐 16:30–19:00。",
            "category": "餐饮",
        },
        {
            "chunk_id": "c2",
            "title": "快递驿站位置",
            "text": "快递驿站设在学生生活区，凭取件码取件，包裹保留 3 天。",
            "category": "快递",
        },
        {
            "chunk_id": "c3",
            "title": "校园卡挂失",
            "text": "校园卡丢失先挂失，24 小时后到卡务中心补办。",
            "category": "校园卡",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_load_chunks_and_build_index(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)

    chunks = load_chunks(chunks_path)
    assert [chunk.chunk_id for chunk in chunks] == ["c1", "c2", "c3"]

    index = build_index(chunks)
    assert index["doc_count"] == 3
    assert index["tokenizer"] == "char-bigram"
    assert len(index["doc_lengths"]) == 3
    assert "食堂" in index["postings"]


def test_search_hits_expected_category(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)
    store = LifeStore(chunks_path=chunks_path, index_path=tmp_path / "index.json")

    hits = store.search("食堂几点开门")
    assert hits
    assert hits[0].chunk.chunk_id == "c1"
    assert hits[0].score > 0


def test_search_keeps_same_category_only(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)
    store = LifeStore(chunks_path=chunks_path, index_path=tmp_path / "index.json")

    hits = store.search("校园卡丢了怎么办")
    assert hits
    assert hits[0].chunk.chunk_id == "c3"
    assert {hit.chunk.category for hit in hits} == {"校园卡"}


def test_search_unrelated_question_returns_empty(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)
    store = LifeStore(chunks_path=chunks_path, index_path=tmp_path / "index.json")

    assert store.search("量子退相干的特征时间") == []


def test_index_is_rebuilt_when_doc_count_mismatches(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)
    index_path = tmp_path / "index.json"
    # 故意写一个过期的索引（只记了 1 条）
    index_path.write_text(
        json.dumps({"doc_count": 1, "doc_lengths": [10], "avg_length": 10, "postings": {}}),
        encoding="utf-8",
    )

    store = LifeStore(chunks_path=chunks_path, index_path=index_path)
    hits = store.search("快递驿站")
    assert hits and hits[0].chunk.chunk_id == "c2"


def test_missing_chunks_is_graceful(tmp_path: Path) -> None:
    store = LifeStore(chunks_path=tmp_path / "nope.jsonl", index_path=tmp_path / "index.json")
    assert store.search("食堂") == []
    assert store.doc_count == 0
    assert store.load_error is None


def test_blank_question_returns_empty(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)
    store = LifeStore(chunks_path=chunks_path, index_path=tmp_path / "index.json")
    assert store.search("   ") == []


def test_stats_reports_categories(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)
    store = LifeStore(chunks_path=chunks_path, index_path=tmp_path / "index.json")
    stats = store.stats()
    assert stats["doc_count"] == 3
    assert stats["categories"] == {"餐饮": 1, "快递": 1, "校园卡": 1}


def test_life_chunk_searchable_text_weights_title() -> None:
    chunk = LifeChunk(chunk_id="x", title="食堂", text="营业时间")
    assert chunk.searchable_text.startswith("食堂 食堂")
