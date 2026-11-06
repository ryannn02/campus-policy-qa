"""BM25 多路召回与 RRF 融合的纯逻辑测试（不加载模型、不读真实索引）。"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import campus_qa.retriever as retriever


def test_bigrams_chinese_2gram() -> None:
    """字符级 2-gram：连续滑窗、去空白；短文本无 2-gram 返回空。"""

    assert retriever._bigrams("图书馆借书") == ["图书", "书馆", "馆借", "借书"]
    assert retriever._bigrams("缓考 申请") == ["缓考", "考申", "申请"]
    assert retriever._bigrams("单") == []
    assert retriever._bigrams("") == []


def test_bm25_scores_prefer_doc_with_query_terms() -> None:
    """包含查询 2-gram 的文档应得更高 BM25 分，无关文档为 0。"""

    docs = ["图书馆借阅规则", "足球比赛规则"]
    postings: dict[str, list[tuple[int, int]]] = {}
    lengths: list[int] = []
    for i, doc in enumerate(docs):
        grams = retriever._bigrams(doc)
        lengths.append(len(grams))
        tf: dict[str, int] = {}
        for gram in grams:
            tf[gram] = tf.get(gram, 0) + 1
        for gram, count in tf.items():
            postings.setdefault(gram, []).append((i, count))

    avg = sum(lengths) / len(lengths)
    scores = retriever._compute_bm25_scores(
        retriever._bigrams("图书馆"), postings, lengths, avg, len(docs)
    )

    assert scores.get(0, 0.0) > 0.0
    assert scores.get(1, 0.0) == 0.0


def test_bm25_longer_doc_penalty_and_tf_saturation() -> None:
    """相同词频下短文档得分更高（长度归一化生效）。"""

    long_doc = "图书" + "。" * 60 + "规则"
    short_doc = "图书规则"
    docs = [long_doc, short_doc]
    postings: dict[str, list[tuple[int, int]]] = {}
    lengths: list[int] = []
    for i, doc in enumerate(docs):
        grams = retriever._bigrams(doc)
        lengths.append(len(grams))
        tf: dict[str, int] = {}
        for gram in grams:
            tf[gram] = tf.get(gram, 0) + 1
        for gram, count in tf.items():
            postings.setdefault(gram, []).append((i, count))

    avg = sum(lengths) / len(lengths)
    scores = retriever._compute_bm25_scores(
        retriever._bigrams("图书规则"), postings, lengths, avg, len(docs)
    )

    assert scores[1] > scores[0]


def test_rrf_fuse_ranks_consensus_docs_first() -> None:
    """两路都命中的文档应排前；向量路 rank1 高于 BM25 路 rank2。"""

    vec = np.array([0.6, 0.4, 0.9, 0.1])  # 向量 rank: 2, 0, 1, 3
    bm25 = {0: 5.0, 3: 4.0}  # BM25 rank: 0, 3

    fused = retriever._rrf_fuse(vec, bm25, vector_k=4, bm25_k=2)
    indices = [idx for idx, _ in fused]

    # doc0 = 1/62+1/61 ≈ 0.0325（两路都进前二）
    # doc3 = 1/64+1/62 ≈ 0.0318
    # doc2 = 1/61 ≈ 0.0164；doc1 = 1/63 ≈ 0.0159
    assert indices == [0, 3, 2, 1]


def test_rrf_fuse_excludes_out_of_window() -> None:
    """窗口外的文档不参与融合，不出现平白加分。"""

    vec = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    bm25 = {4: 3.0}  # 只在 BM25 路出现

    fused = retriever._rrf_fuse(vec, bm25, vector_k=2, bm25_k=1)
    indices = {idx for idx, _ in fused}

    # doc0（向量 rank1）与 doc4（BM25 rank1）同分，doc1 为向量 rank2；
    # 向量窗口外的 doc2/doc3 不出现。
    assert indices == {0, 1, 4}
    assert len(fused) == 3
    assert fused[0][0] == 0


def test_rank_and_build_applies_per_doc_limit(monkeypatch) -> None:
    """融合后组装仍应套用 top_k 与 per_doc 限额，score 沿用向量分。"""

    chunks = [
        retriever._IndexedChunk(doc_id="a", text="t1", page=None, section=None, url=None),
        retriever._IndexedChunk(doc_id="a", text="t2", page=None, section=None, url=None),
        retriever._IndexedChunk(doc_id="b", text="t3", page=None, section=None, url=None),
    ]
    monkeypatch.setattr(retriever, "_indexed_chunks", chunks)

    scores = np.array([0.9, 0.8, 0.7], dtype="float32")
    ranked = [0, 1, 2]

    new_settings = dataclasses.replace(
        retriever.settings, retrieval_top_k=3, retrieval_per_doc=2
    )
    monkeypatch.setattr(retriever, "settings", new_settings)
    cites = retriever._rank_and_build_citations(ranked, scores)
    assert [c.document_id for c in cites] == ["a", "a", "b"]
    assert [c.score for c in cites] == pytest.approx([0.9, 0.8, 0.7])

    strict_settings = dataclasses.replace(
        retriever.settings, retrieval_top_k=3, retrieval_per_doc=1
    )
    monkeypatch.setattr(retriever, "settings", strict_settings)
    cites = retriever._rank_and_build_citations(ranked, scores)
    assert [c.document_id for c in cites] == ["a", "b"]


def _mk_chunk(doc_id: str, text: str) -> retriever._IndexedChunk:
    return retriever._IndexedChunk(doc_id=doc_id, text=text, page=None, section=None, url=None)


def test_dedup_cross_doc_removes_near_duplicates(monkeypatch) -> None:
    """跨文档且文本高度重合（≥0.85）的候选只保留先到（分数更高）的代表。"""

    chunks = [
        _mk_chunk("A", "第三十九条 转专业条件：学生确有专长可以申请转专业。"),
        _mk_chunk("B", "第三十九条 转专业条件：学生确有专长可以申请转专业。"),
        _mk_chunk("C", "图书馆借阅规则：每人可借三十册图书。"),
    ]
    monkeypatch.setattr(retriever, "_indexed_chunks", chunks)
    kept = retriever._dedup_cross_doc([0, 1, 2])
    assert kept == [0, 2]


def test_dedup_keeps_same_doc_chunks(monkeypatch) -> None:
    """同一文档内的条款不去重（多跳题需要同文档多处条款）。"""

    chunks = [
        _mk_chunk("A", "第一条 甲事项。"),
        _mk_chunk("A", "第一条 甲事项。"),
    ]
    monkeypatch.setattr(retriever, "_indexed_chunks", chunks)
    kept = retriever._dedup_cross_doc([0, 1])
    assert kept == [0, 1]


def test_dedup_keeps_distinct_topics(monkeypatch) -> None:
    """主题不同（重合率低）的候选全部保留。"""

    chunks = [
        _mk_chunk("A", "缓考申请须在考试前提交申请表。"),
        _mk_chunk("B", "国家助学金面向家庭经济困难学生。"),
        _mk_chunk("C", "图书馆每人可借三十册图书。"),
    ]
    monkeypatch.setattr(retriever, "_indexed_chunks", chunks)
    kept = retriever._dedup_cross_doc([0, 1, 2])
    assert kept == [0, 1, 2]


def test_dedup_partial_overlap_below_threshold_kept(monkeypatch) -> None:
    """部分相似但重合率 <0.85 的候选不去（不同版本但信息不同）。"""

    chunks = [
        _mk_chunk("A", "缓考申请须在考试前提交。"),
        _mk_chunk("B", "缓考申请须在考试后三个工作日内补交证明材料。"),
    ]
    monkeypatch.setattr(retriever, "_indexed_chunks", chunks)
    kept = retriever._dedup_cross_doc([0, 1])
    assert kept == [0, 1]


def test_dedup_preserves_order_and_handles_empty(monkeypatch) -> None:
    """先到先得（分数降序），空输入返回空列表。"""

    chunks = [
        _mk_chunk("A", "相同内容条款。"),
        _mk_chunk("B", "相同内容条款。"),
    ]
    monkeypatch.setattr(retriever, "_indexed_chunks", chunks)
    assert retriever._dedup_cross_doc([]) == []
    assert retriever._dedup_cross_doc([1, 0]) == [1]
