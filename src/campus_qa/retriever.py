"""最简检索接口：从问题字符串返回若干候选条款。

首个里程碑要求：在保持
`retriever.retrieve_candidates(question: str) -> List[Citation]`
接口不变的前提下，优先使用真实向量索引
`data/index/embeddings.npy` 做检索，并填好
`Citation.score` 便于拒答阈值使用；在索引或
向量模型不可用时，退回内置样本文本的字符级
Jaccard 检索，占位演示。

接口设计目标：
- 输入：问题字符串 `question: str`；
- 输出：若干候选条款 `List[Citation]`，字段与
  `AnswerResult.citations` 完全一致，方便后续直接
  映射与回填。

实现约束：不引入新的向量库或复杂框架，仅使用
现有 `embeddings.npy` + `numpy` + `sentence-transformers`
完成暴力余弦检索。索引加载失败时不抛异常，
直接回退到样本检索。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import settings
from .contracts import Citation


_PAGE_NUMBER_RE = re.compile(r"\d+")


def _coerce_page(value: object) -> Optional[int]:
    """把 chunk 里的 page 字段规整成 int 或 None。

    `chunks.jsonl` 中约四成记录的 page 是空字符串（无页码来源），
    直接传给 `Citation.page: Optional[int]` 会触发 pydantic
    校验错误，必须在检索层统一做一次清洗。
    """

    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        match = _PAGE_NUMBER_RE.search(value)
        if match:
            return int(match.group())
    return None


@dataclass
class _IndexedChunk:
    """用于从 chunks.jsonl 构建的简化索引记录。"""

    doc_id: str
    text: str
    page: Optional[int]
    section: Optional[str]
    url: Optional[str]
    title: str = ""


_SAMPLE_CLAUSES: List[Citation] = [
    Citation(
        document_id="student_handbook",
        snippet=(
            "学生如因特殊原因无法参加正常考试，可按照教务处发布的缓考"
            "管理办法，在规定时间内提交缓考申请表，由任课教师与所在学院"
            "审核后报教务处备案。"
        ),
        score=1.0,
        page=42,
        section="缓考申请",
        url=None,
    ),
    Citation(
        document_id="library_policy",
        snippet=(
            "本科学生在校期间一般可同时借阅不超过十册图书，具体数量以"
            "图书馆最新借阅规则为准；逾期未还按图书馆相关规定处理。"
        ),
        score=1.0,
        page=7,
        section="借阅数量与逾期处理",
        url=None,
    ),
]


_embeddings_matrix: Optional[np.ndarray] = None
_indexed_chunks: List[_IndexedChunk] = []
_query_prefix: str = ""
_index_loaded: bool = False
_embedding_model = None  # 懒加载 SentenceTransformer 模型
_embedding_failed: bool = False  # 模型加载失败后不再重试


def _load_index() -> None:
    """惰性加载向量索引和对应的 chunk 元数据。

    - embeddings.npy：形状约为 (num_chunks, dim)；
    - manifest.json：提供 dim、chunks_file、query_prefix 等元信息；
    - chunks_file：JSONL，每行一个 chunk，字段包括 doc_id、text、page 等。

    加载失败时保持索引为空，让检索逻辑自动回退
    到 `_SAMPLE_CLAUSES` 的字符级检索。"""

    global _embeddings_matrix, _indexed_chunks, _query_prefix, _index_loaded

    if _index_loaded:
        return

    _index_loaded = True

    embeddings_path = settings.index_dir / "embeddings.npy"
    manifest_path = settings.index_dir / "manifest.json"

    if not embeddings_path.exists() or not manifest_path.exists():
        return

    try:
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError):
        return

    try:
        embeddings = np.load(embeddings_path)
    except OSError:
        return

    _embeddings_matrix = embeddings
    _query_prefix = str(manifest.get("query_prefix", ""))

    chunks_file_relative = manifest.get("chunks_file")
    if isinstance(chunks_file_relative, str) and chunks_file_relative:
        chunks_path = settings.data_dir.parent / chunks_file_relative
    else:
        chunks_path = settings.chunks_path

    if not chunks_path.exists():
        return

    indexed_chunks: List[_IndexedChunk] = []
    try:
        with chunks_path.open("r", encoding="utf-8") as chunks_file:
            for line in chunks_file:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                try:
                    record = json.loads(line_stripped)
                except json.JSONDecodeError:
                    continue

                indexed_chunks.append(
                    _IndexedChunk(
                        doc_id=str(record.get("doc_id", "")),
                        text=str(record.get("text", "")),
                        page=_coerce_page(record.get("page")),
                        section=record.get("section") or None,
                        url=record.get("source_url"),
                        title=str(record.get("title", "")).strip(),
                    )
                )
    except OSError:
        return

    if not indexed_chunks:
        return

    # 对齐索引长度，避免 embeddings 与 chunks 条数不匹配。
    chunk_count = min(len(indexed_chunks), _embeddings_matrix.shape[0])
    _indexed_chunks = indexed_chunks[:chunk_count]
    _embeddings_matrix = _embeddings_matrix[:chunk_count]


def _encode_question(question: str) -> Optional[np.ndarray]:
    """使用 bge-small-zh-v1.5 对问题编码，返回向量表示。

    为与索引构建保持一致，前置 `manifest.query_prefix`。
    当模型或索引不可用、或问题为空时返回 None。"""

    global _embedding_model, _embedding_failed

    cleaned_question = question.strip()
    if not cleaned_question or _embeddings_matrix is None:
        return None

    if _embedding_failed:
        return None

    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _embedding_model = SentenceTransformer(settings.embedding_model)
        except Exception:
            # 模型不可用（未下载、名称写错、无网络）时只尝试一次，
            # 之后统一走样本检索，避免每个问题都卡在加载上。
            _embedding_failed = True
            return None

    query_text = f"{_query_prefix}{cleaned_question}" if _query_prefix else cleaned_question
    # 返回一维向量，便于后续余弦相似度计算。
    try:
        vector = _embedding_model.encode(query_text)
    except Exception:
        return None
    return np.asarray(vector, dtype="float32")


def _retrieve_with_embeddings(question: str) -> List[Citation]:
    """基于真实向量索引的检索实现。

    - 使用 bge 模型对问题编码；
    - 计算与所有 chunk 向量的余弦相似度；
    - 按得分排序，应用 `top_k` 和 `per_doc` 限额；
    - 返回填好 `score` 的 `Citation` 列表。"""

    if _embeddings_matrix is None or not _indexed_chunks:
        return []

    query_vector = _encode_question(question)
    if query_vector is None:
        return []

    if query_vector.shape[-1] != _embeddings_matrix.shape[1]:
        return []

    query_norm = float(np.linalg.norm(query_vector))
    if query_norm == 0.0:
        return []

    normalized_query = query_vector / query_norm
    row_norms = np.linalg.norm(_embeddings_matrix, axis=1)
    safe_row_norms = np.where(row_norms == 0.0, 1e-8, row_norms)
    normalized_index = _embeddings_matrix / safe_row_norms[:, None]

    scores = normalized_index @ normalized_query

    top_k = min(settings.retrieval_top_k, scores.shape[0])
    if top_k <= 0:
        return []

    sorted_indices = scores.argsort()[::-1]
    per_doc_limit = max(settings.retrieval_per_doc, 1)

    citations: List[Citation] = []
    doc_counts: dict[str, int] = {}

    for index_position in sorted_indices:
        if len(citations) >= top_k:
            break

        chunk = _indexed_chunks[int(index_position)]
        doc_id = chunk.doc_id
        current_count = doc_counts.get(doc_id, 0)
        if current_count >= per_doc_limit:
            continue

        doc_counts[doc_id] = current_count + 1
        score = float(scores[int(index_position)])

        citations.append(
            Citation(
                document_id=chunk.doc_id,
                snippet=chunk.text,
                score=score,
                page=chunk.page,
                section=chunk.section or chunk.title or None,
                url=chunk.url,
            )
        )

    return citations


def _simple_overlap_score(question: str, text: str) -> float:
    """用极简字符重合度作为候选条款的相似度得分。

    - 输入和输出范围都在 [0.0, 1.0]，便于后续与
      `settings.retrieval_score_threshold` 对齐；
    - 当前实现只看非空字符集合的 Jaccard 重合，作为
      将来向量相似度的占位。"""

    cleaned_question = question.strip()
    cleaned_text = text.strip()
    if not cleaned_question or not cleaned_text:
        return 0.0

    question_chars = set(cleaned_question)
    text_chars = set(cleaned_text)
    intersection_size = len(question_chars & text_chars)
    if intersection_size == 0:
        return 0.0

    union_size = len(question_chars | text_chars)
    return intersection_size / union_size


def _retrieve_with_samples(question: str) -> List[Citation]:
    """基于内置样本文本的退化检索实现。

    当真实索引不可用时，仍然提供一个可演示的
    `retrieve_candidates` 实现，方便首个里程碑跑通。"""

    scored_clauses = []
    for clause in _SAMPLE_CLAUSES:
        score = _simple_overlap_score(question, clause.snippet)
        scored_clauses.append((score, clause))

    scored_clauses.sort(key=lambda item: item[0], reverse=True)

    top_k = settings.retrieval_top_k
    citations: List[Citation] = []
    for score, clause in scored_clauses[:top_k]:
        citations.append(
            Citation(
                document_id=clause.document_id,
                snippet=clause.snippet,
                score=score,
                page=clause.page,
                section=clause.section,
                url=clause.url,
            )
        )

    return citations


def retrieve_candidates(question: str) -> List[Citation]:
    """统一检索入口：输入问题字符串，返回若干候选条款。

    优先使用真实向量索引；索引或模型不可用时
    自动退回样本检索，不对外暴露异常。返回值
    始终为 `Citation` 列表，字段与 `AnswerResult.citations`
    对齐。"""

    _load_index()

    if _embeddings_matrix is not None and _indexed_chunks:
        try:
            citations = _retrieve_with_embeddings(question)
        except Exception:
            # 检索层任何异常都不应让整个问答流程崩掉，
            # 退化为样本检索，由上层按低分拒答处理。
            citations = []
        if citations:
            return citations

    return _retrieve_with_samples(question)


__all__ = ["retrieve_candidates"]


