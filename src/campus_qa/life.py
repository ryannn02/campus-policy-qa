"""校园生活库检索（BM25，与制度库物理分离）。

为什么单独一套：

- 制度库的召回/拒答指标是硬约束，混入食堂、快递、校园卡内容会直接
  稀释 `Recall@5` 与 `refusal_rate`；
- 生活问答的颗粒度不同（「几点开门」「快递在哪取」），用与制度库同款的
  字符 2-gram BM25 就够，不需要再挂一套向量模型；
- 生活库更新频繁（每学期都变），重建索引必须是秒级的。

索引结构（`data/life/index.json`）：

```
{
  "version": 1,
  "tokenizer": "char-bigram",
  "doc_count": 42,
  "avg_length": 118.4,
  "doc_lengths": [ ... ],
  "postings": {"食堂": [[0, 3], [5, 1]], ...}
}
```

语料本身仍在 `data/life/chunks.jsonl`，索引只存统计量，改语料必须重建。
"""

from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import settings


_BM25_K1 = 1.5
_BM25_B = 0.75
# 把无上界的 BM25 原始分压到 [0,1)，便于用统一阈值判定「库里有依据」
_SCORE_SMOOTHING = 2.5
# 相对保留比例：低于 top-1 的 60% 视为弱相关，不进答案
_RELATIVE_KEEP_RATIO = 0.7

# 疑问/口语停用 2-gram：不参与 BM25，否则「怎么办」「怎么写」会把
# 快递地址之类的无关条目拉进校园卡/空调问题里。
_QUERY_STOP_GRAMS = {
    "怎么", "么办", "什么", "么时", "多少", "几点", "哪里", "在哪",
    "如何", "可以", "的话", "请问", "一下", "怎样", "么样", "哪个",
}


@dataclass(frozen=True)
class LifeChunk:
    """生活库单条语料。"""

    chunk_id: str
    title: str
    text: str
    category: str = "通用"
    source: str = ""
    updated_at: str = ""

    @property
    def searchable_text(self) -> str:
        # 标题权重靠拼接实现：标题里的词在 BM25 里被重复计入
        return f"{self.title} {self.title} {self.text}"


@dataclass(frozen=True)
class LifeHit:
    """一次生活检索命中的结果。"""

    chunk: LifeChunk
    score: float

    @property
    def normalized_score(self) -> float:
        return _smooth(self.score)


def _smooth(raw_score: float) -> float:
    if raw_score <= 0:
        return 0.0
    return raw_score / (raw_score + _SCORE_SMOOTHING)


def _bigrams(text: str) -> List[str]:
    """字符级 2-gram，与制度库 BM25 同口径（中文无需分词器）。"""

    cleaned = re.sub(r"\s+", "", text)
    return [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]


def load_chunks(path: Optional[Path] = None) -> List[LifeChunk]:
    """读取生活库语料；文件缺失时返回空列表而不是抛异常。"""

    target = Path(path or settings.life_chunks_path)
    if not target.exists():
        return []

    chunks: List[LifeChunk] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:  # 单行坏数据不拖垮整库
                raise ValueError(f"{target} 第 {line_no} 行不是合法 JSON") from exc
            chunks.append(
                LifeChunk(
                    chunk_id=str(payload.get("chunk_id") or f"life_{line_no:04d}"),
                    title=str(payload.get("title") or ""),
                    text=str(payload.get("text") or ""),
                    category=str(payload.get("category") or "通用"),
                    source=str(payload.get("source") or ""),
                    updated_at=str(payload.get("updated_at") or ""),
                )
            )
    return chunks


def build_index(chunks: Sequence[LifeChunk]) -> Dict[str, object]:
    """由语料计算 BM25 统计量，返回可直接 json.dump 的字典。"""

    postings: Dict[str, List[List[int]]] = {}
    doc_lengths: List[int] = []

    for doc_idx, chunk in enumerate(chunks):
        grams = _bigrams(chunk.searchable_text)
        doc_lengths.append(len(grams))
        term_freq: Dict[str, int] = {}
        for gram in grams:
            term_freq[gram] = term_freq.get(gram, 0) + 1
        for gram, tf in term_freq.items():
            postings.setdefault(gram, []).append([doc_idx, tf])

    avg_length = (sum(doc_lengths) / len(doc_lengths)) if doc_lengths else 0.0
    return {
        "version": 1,
        "tokenizer": "char-bigram",
        "doc_count": len(chunks),
        "avg_length": round(avg_length, 4),
        "doc_lengths": doc_lengths,
        "postings": postings,
    }


class LifeStore:
    """生活库检索器：惰性加载语料与索引，索引缺失时就地重建。

    线程安全（FastAPI 线程池会并发调用），加载失败不抛异常，
    退化为空结果，避免把生活库故障传导到制度问答主链路。
    """

    def __init__(
        self,
        chunks_path: Optional[Path] = None,
        index_path: Optional[Path] = None,
    ) -> None:
        self._chunks_path = Path(chunks_path or settings.life_chunks_path)
        self._index_path = Path(index_path or settings.life_index_path)
        self._lock = threading.Lock()
        self._chunks: List[LifeChunk] = []
        self._postings: Dict[str, List[Tuple[int, int]]] = {}
        self._doc_lengths: List[int] = []
        self._avg_length: float = 0.0
        self._loaded = False
        self._load_error: Optional[str] = None

    # ---- 加载 ----

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                self._chunks = load_chunks(self._chunks_path)
                index = self._read_index()
                if index is None:
                    index = build_index(self._chunks)
                self._apply_index(index)
            except Exception as exc:  # noqa: BLE001 - 生活库故障不应拖垮主链路
                self._load_error = str(exc)
                self._chunks = []
                self._postings = {}
                self._doc_lengths = []
                self._avg_length = 0.0
            finally:
                self._loaded = True

    def _read_index(self) -> Optional[Dict[str, object]]:
        if not self._index_path.exists():
            return None
        try:
            with self._index_path.open("r", encoding="utf-8") as handle:
                index = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return None

        # 语料变了但索引没重建 → 索引失效，回退到内存重建
        if int(index.get("doc_count", -1)) != len(self._chunks):
            return None
        return index

    def _apply_index(self, index: Dict[str, object]) -> None:
        self._doc_lengths = [int(value) for value in index.get("doc_lengths", [])]
        self._avg_length = float(index.get("avg_length", 0.0))
        raw_postings = index.get("postings", {}) or {}
        postings: Dict[str, List[Tuple[int, int]]] = {}
        for gram, pairs in raw_postings.items():  # type: ignore[union-attr]
            postings[str(gram)] = [(int(pair[0]), int(pair[1])) for pair in pairs]
        self._postings = postings

    # ---- 检索 ----

    def search(
        self,
        question: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[LifeHit]:
        """返回按分数降序的生活库命中；低于阈值的不返回。"""

        self._ensure_loaded()
        if not self._chunks:
            return []

        grams = [gram for gram in _bigrams(question) if gram not in _QUERY_STOP_GRAMS]
        if not grams:
            return []

        limit = top_k if top_k is not None else settings.life_top_k
        floor = threshold if threshold is not None else settings.life_score_threshold

        scores: Dict[int, float] = {}
        n_docs = len(self._doc_lengths)
        for gram in set(grams):  # 查询侧去重，与制度库 BM25 一致
            posting = self._postings.get(gram)
            if not posting:
                continue
            df = len(posting)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            for doc_idx, tf in posting:
                doc_len = self._doc_lengths[doc_idx]
                denominator = tf + _BM25_K1 * (
                    1 - _BM25_B + _BM25_B * doc_len / (self._avg_length or 1.0)
                )
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * tf * (_BM25_K1 + 1) / denominator

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            return []

        # 相对阈值：只保留与 top-1 同一量级的命中，避免 BM25 的弱相关噪音
        # （例如问「校园卡丢了」却把「快递地址怎么写」排到第二）。
        top_smoothed = _smooth(ranked[0][1])
        relative_floor = top_smoothed * _RELATIVE_KEEP_RATIO
        effective_floor = max(floor, relative_floor)

        hits: List[LifeHit] = []
        top_category = self._chunks[ranked[0][0]].category if ranked else ""
        for doc_idx, raw_score in ranked:
            smoothed = _smooth(raw_score)
            if smoothed < effective_floor:
                continue
            chunk = self._chunks[doc_idx]
            # 只保留与首条同主题的结果，避免“校园卡”问题混进快递地址、
            # “校医院”问题混进食堂营业时间。
            if hits and top_category and chunk.category != top_category:
                continue
            hits.append(LifeHit(chunk=chunk, score=round(smoothed, 4)))
            if len(hits) >= limit:
                break
        return hits

    # ---- 运维信息 ----

    @property
    def doc_count(self) -> int:
        self._ensure_loaded()
        return len(self._chunks)

    @property
    def load_error(self) -> Optional[str]:
        self._ensure_loaded()
        return self._load_error

    def stats(self) -> Dict[str, object]:
        self._ensure_loaded()
        categories: Dict[str, int] = {}
        for chunk in self._chunks:
            categories[chunk.category] = categories.get(chunk.category, 0) + 1
        return {
            "doc_count": len(self._chunks),
            "categories": categories,
            "chunks_path": str(self._chunks_path),
            "index_path": str(self._index_path),
            "index_exists": self._index_path.exists(),
            "load_error": self._load_error,
        }


# 进程级单例：语料与索引只加载一次
store = LifeStore()
