"""检索接口：多路召回（向量 + BM25，RRF 融合）返回候选条款。

在保持 `retriever.retrieve_candidates(question: str) -> List[Citation]`
接口不变的前提下：

- 向量路：`data/index/embeddings.npy` + bge 编码，暴力余弦；
- BM25 路：字符级 2-gram（中文无需分词器，不新增依赖），
  可通过 `settings.retrieval_enable_bm25` 开关；
- 两路 RRF 融合排序后，套用 `top_k` 与 `per_doc` 限额；
- `Citation.score` 保留向量余弦分，拒答阈值闸门语义不变。

索引或向量模型不可用时，退回内置样本文本的字符级
Jaccard 检索，占位演示；索引加载失败时不抛异常。

接口设计目标：
- 输入：问题字符串 `question: str`；
- 输出：若干候选条款 `List[Citation]`，字段与
  `AnswerResult.citations` 完全一致，方便后续直接
  映射与回填。"""

from __future__ import annotations

import difflib
import json
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import settings
from .contracts import Citation


_PAGE_NUMBER_RE = re.compile(r"\d+")

# BM25 与 RRF 融合参数（字符级 2-gram，中文无需分词器）。
_BM25_K1 = 1.5
_BM25_B = 0.75
_RRF_K = 60.0

# 跨文档同主题去重：字符重合率 ≥ 该值视为同一主题多版本。
_DEDUP_RATIO_THRESHOLD = 0.85
# SequenceMatcher.ratio ≥ 0.85 的必要条件：max/min 长度比 ≤ 2/0.85 - 1 ≈ 1.353。
_DEDUP_LEN_RATIO_UPPER = 2.0 / _DEDUP_RATIO_THRESHOLD - 1.0


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
_embedding_model_name: str = ""  # 从 manifest 读取的模型名，优先于 settings.embedding_model
_index_loaded: bool = False
_embedding_model = None  # 懒加载 SentenceTransformer 模型
_embedding_failed: bool = False  # 模型加载失败后不再重试

# BM25 字符级 2-gram 倒排索引（随向量索引一同构建）。
_bm25_postings: Dict[str, List[Tuple[int, int]]] = {}  # term -> [(chunk_idx, tf)]
_bm25_doc_lengths: List[int] = []  # 每个 chunk 的 2-gram 数量
_bm25_avg_length: float = 0.0
_bm25_ready: bool = False


def _load_index() -> None:
    """惰性加载向量索引和对应的 chunk 元数据。

    - embeddings.npy：形状约为 (num_chunks, dim)；
    - manifest.json：提供 dim、chunks_file、query_prefix 等元信息；
    - chunks_file：JSONL，每行一个 chunk，字段包括 doc_id、text、page 等。

    加载失败时保持索引为空，让检索逻辑自动回退
    到 `_SAMPLE_CLAUSES` 的字符级检索。"""

    global _embeddings_matrix, _indexed_chunks, _query_prefix, _embedding_model_name, _index_loaded

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
    _embedding_model_name = str(manifest.get("model", "")).strip()

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

    _build_bm25_index()


def _encode_question(question: str) -> Optional[np.ndarray]:
    """使用 manifest 指定的向量模型对问题编码，返回向量表示。

    模型名优先取 ``manifest.model``（与索引构建时一致），
    缺省回退 ``settings.embedding_model``；前置 ``manifest.query_prefix``。
    当模型或索引不可用、或问题为空时返回 None。"""

    global _embedding_model, _embedding_failed

    cleaned_question = question.strip()
    if not cleaned_question or _embeddings_matrix is None:
        return None

    if _embedding_failed:
        return None

    if _embedding_model is None:
        model_name = _embedding_model_name or settings.embedding_model
        try:
            from sentence_transformers import SentenceTransformer

            _embedding_model = SentenceTransformer(model_name)
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


def _compute_vector_scores(query_vector: np.ndarray) -> Optional[np.ndarray]:
    """计算问题向量与全部 chunk 的余弦相似度，返回一维分数数组。"""

    if _embeddings_matrix is None or not _indexed_chunks:
        return None
    if query_vector.shape[-1] != _embeddings_matrix.shape[1]:
        return None

    query_norm = float(np.linalg.norm(query_vector))
    if query_norm == 0.0:
        return None

    normalized_query = query_vector / query_norm
    row_norms = np.linalg.norm(_embeddings_matrix, axis=1)
    safe_row_norms = np.where(row_norms == 0.0, 1e-8, row_norms)
    normalized_index = _embeddings_matrix / safe_row_norms[:, None]
    return normalized_index @ normalized_query


def _bigrams(text: str) -> List[str]:
    """字符级 2-gram 切分：去空白后连续滑窗，中文无需分词器。"""

    cleaned = re.sub(r"\s+", "", text)
    return [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]


def _compute_bm25_scores(
    query_grams: List[str],
    postings: Dict[str, List[Tuple[int, int]]],
    doc_lengths: List[int],
    avg_length: float,
    n_docs: int,
) -> Dict[int, float]:
    """按 BM25 公式对全部 chunk 打分，返回 {chunk_idx: score}。

    查询侧 2-gram 去重（重复项不重复计入），与常见 BM25 简化一致。
    """

    scores: Dict[int, float] = {}
    for gram in set(query_grams):
        posting = postings.get(gram)
        if not posting:
            continue
        df = len(posting)
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        for doc_idx, tf in posting:
            doc_len = doc_lengths[doc_idx]
            denominator = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * doc_len / avg_length)
            gain = idf * tf * (_BM25_K1 + 1) / denominator
            scores[doc_idx] = scores.get(doc_idx, 0.0) + gain
    return scores


def _build_bm25_index() -> None:
    """基于 `_indexed_chunks` 构建字符级 2-gram 倒排索引。

    与向量索引同生命周期，只在 `_load_index` 成功后执行一次；
    样本回退路径没有语料，因此不构建。
    """

    global _bm25_postings, _bm25_doc_lengths, _bm25_avg_length, _bm25_ready

    _bm25_postings = {}
    doc_lengths: List[int] = []
    for doc_idx, chunk in enumerate(_indexed_chunks):
        grams = _bigrams(chunk.text)
        doc_lengths.append(len(grams))
        tf: Dict[str, int] = {}
        for gram in grams:
            tf[gram] = tf.get(gram, 0) + 1
        for gram, count in tf.items():
            _bm25_postings.setdefault(gram, []).append((doc_idx, count))

    _bm25_doc_lengths = doc_lengths
    _bm25_avg_length = (sum(doc_lengths) / len(doc_lengths)) if doc_lengths else 0.0
    _bm25_ready = True


def _bm25_scores_for_question(question: str) -> Dict[int, float]:
    """对问题做 BM25 打分；索引未就绪或问题无有效 2-gram 时返回空。"""

    if not _bm25_ready:
        return {}
    grams = _bigrams(question)
    if not grams:
        return {}
    return _compute_bm25_scores(
        grams, _bm25_postings, _bm25_doc_lengths, _bm25_avg_length, len(_bm25_doc_lengths)
    )


def _rrf_fuse(
    vector_scores: np.ndarray,
    bm25_scores: Dict[int, float],
    vector_k: int = 5,
    bm25_k: int = 5,
    rrf_k: float = _RRF_K,
) -> List[Tuple[int, float]]:
    """RRF 融合向量路与 BM25 路，返回 [(chunk_idx, rrf_score)] 按分数降序。"""

    rrf: Dict[int, float] = {}
    for rank, idx in enumerate(vector_scores.argsort()[::-1][:vector_k]):
        i = int(idx)
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, idx in enumerate(sorted(bm25_scores, key=bm25_scores.get, reverse=True)[:bm25_k]):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(rrf.items(), key=lambda kv: (-kv[1], kv[0]))


def _dedup_cross_doc(
    ranked_indices: List[int],
    threshold: float = _DEDUP_RATIO_THRESHOLD,
) -> List[int]:
    """跨文档同主题去重（融合排序之后、per_doc/top_k 限额之前）。

    按给定顺序遍历（分数已降序），若当前候选与已保留的某条候选
    ``document_id`` 不同、且文本字符重合率（``difflib.SequenceMatcher.ratio``）
    ≥ ``threshold``，视为同一主题的多版本，跳过；否则保留。
    先到先得 ⇒ 保留分数更高的代表。

    同一文档内的候选不参与去重（多跳题需要同文档多处条款）。
    长度比过大的候选对 ratio 必低于阈值，直接跳过精确计算。
    """

    if not ranked_indices:
        return []

    kept: List[int] = []
    kept_meta: List[Tuple[str, str]] = []  # 已保留代表的 (doc_id, text)

    for idx in ranked_indices:
        chunk = _indexed_chunks[int(idx)]
        duplicated = False
        for kept_doc_id, kept_text in kept_meta:
            if kept_doc_id == chunk.doc_id:
                continue
            len_ratio = len(kept_text) / max(len(chunk.text), 1)
            if len_ratio > _DEDUP_LEN_RATIO_UPPER or len_ratio < (1.0 / _DEDUP_LEN_RATIO_UPPER):
                continue
            if difflib.SequenceMatcher(None, kept_text, chunk.text).ratio() >= threshold:
                duplicated = True
                break
        if duplicated:
            continue
        kept.append(idx)
        kept_meta.append((chunk.doc_id, chunk.text))

    return kept


def _rank_and_build_citations(
    ranked_indices: List[int],
    scores: np.ndarray,
) -> List[Citation]:
    """按给定索引顺序组装 Citation，套用 top_k 与 per_doc 限额。

    `scores` 为向量余弦分数组，`Citation.score` 沿用向量分，
    保证拒答阈值闸门语义不变（融合只改排序与取舍）。
    """

    top_k = min(settings.retrieval_top_k, len(ranked_indices))
    if top_k <= 0:
        return []

    per_doc_limit = max(settings.retrieval_per_doc, 1)
    citations: List[Citation] = []
    doc_counts: Dict[str, int] = {}

    for index_position in ranked_indices:
        if len(citations) >= top_k:
            break
        idx = int(index_position)
        chunk = _indexed_chunks[idx]
        doc_id = chunk.doc_id
        if doc_counts.get(doc_id, 0) >= per_doc_limit:
            continue
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        citations.append(
            Citation(
                document_id=chunk.doc_id,
                snippet=chunk.text,
                score=float(scores[idx]),
                page=chunk.page,
                section=chunk.section or chunk.title or None,
                url=chunk.url,
            )
        )
    return citations


def _retrieve_with_embeddings(question: str) -> List[Citation]:
    """基于真实索引的多路检索实现。

    - 向量路：bge 编码 + 暴力余弦；
    - BM25 路（开关 `settings.retrieval_enable_bm25`）：字符级 2-gram 打分；
    - 两路 RRF 融合排序 → 跨文档同主题去重 → per_doc → top_k；
    - `Citation.score` 保留向量余弦分，拒答阈值语义不变。
    """

    if _embeddings_matrix is None or not _indexed_chunks:
        return []

    query_vector = _encode_question(question)
    if query_vector is None:
        return []

    vector_scores = _compute_vector_scores(query_vector)
    if vector_scores is None:
        return []

    if settings.retrieval_enable_bm25:
        bm25_scores = _bm25_scores_for_question(question)
        if bm25_scores:
            fused = _rrf_fuse(
                vector_scores,
                bm25_scores,
                vector_k=settings.retrieval_top_k,
                bm25_k=settings.retrieval_bm25_top_k,
            )
            ranked = [idx for idx, _ in fused]
        else:
            ranked = [int(i) for i in vector_scores.argsort()[::-1]]
    else:
        ranked = [int(i) for i in vector_scores.argsort()[::-1]]

    # 候选窗口：去重后需仍能凑满 top_k（含 per_doc），
    # 取 top_k + bm25_top_k + 缓冲；只对窗口内候选做两两比较。
    window = settings.retrieval_top_k + settings.retrieval_bm25_top_k + 5
    ranked = _dedup_cross_doc(ranked[:window])
    return _rank_and_build_citations(ranked, vector_scores)


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


