"""一键构建向量索引的最小可行脚本。

设计原则：
- 不重新发明解析与切分，只复用现有的 ``data/chunks.jsonl``；
- 用本地向量模型 `sentence-transformers` 直接编码全部 chunk；
- 输出向量矩阵 ``embeddings.npy`` 和简单的 ``manifest.json``，
  后续检索逻辑可以直接读取使用。

命令行使用：

.. code-block:: bash

    python -m scripts.build_index                 # 使用默认模型
    python -m scripts.build_index --model NAME    # 自选 embedding 模型
    python -m scripts.build_index --batch-size 64 # 调整批大小
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from campus_qa.config import settings


def _load_chunks(path: Path) -> List[Dict[str, Any]]:
    """读取 ``chunks.jsonl`` 并返回字典列表。

    当前项目已有可复用的切分结果，不在此处重跑解析流水线。
    """

    items: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _chunk_text(chunk: Dict[str, Any]) -> str:
    """拼接标题与正文，作为向量化输入文本。

    规则：``title\ntext``，与旧项目保持一致，便于复用经验。
    """

    title = str(chunk.get("title", "")).strip()
    text = str(chunk.get("text", "")).strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def _load_model(name: str | None) -> Any:
    """加载 sentence-transformers 模型。

    优先使用 ``settings.embedding_model``，允许通过命令行覆盖。
    """

    from sentence_transformers import SentenceTransformer
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_name = name or settings.embedding_model
    return SentenceTransformer(model_name, device=device)


def _encode_texts(model: Any, texts: List[str], batch_size: int) -> np.ndarray:
    """将所有文本编码为归一化向量矩阵。"""

    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 200,
    ).astype("float32")


def build_index(model_name: str | None = None, batch_size: int = 64) -> tuple[int, int]:
    """构建向量索引并写入 ``data/index`` 目录。

    - 输入：``settings.chunks_path`` 指向的 chunks.jsonl；
    - 输出：``settings.index_dir`` 下的 ``embeddings.npy`` 和 ``manifest.json``。
    """

    chunks_path = settings.chunks_path
    index_dir = settings.index_dir

    chunks = _load_chunks(chunks_path)
    texts = [_chunk_text(c) for c in chunks]

    model = _load_model(model_name)
    vecs = _encode_texts(model, texts, batch_size=batch_size)

    index_dir.mkdir(parents=True, exist_ok=True)
    emb_path = index_dir / "embeddings.npy"
    np.save(emb_path, vecs)

    manifest = {
        "model": model_name or settings.embedding_model,
        "chunks": len(chunks),
        "dim": int(vecs.shape[1]),
        "batch_size": batch_size,
        "chunks_file": str(chunks_path),
        "query_prefix": "为这个句子生成表示以用于检索相关文章：",
    }
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return vecs.shape


def main() -> None:
    parser = argparse.ArgumentParser(description="构建本地向量索引")
    parser.add_argument("--model", default=None, help="embedding 模型名称")
    parser.add_argument("--batch-size", type=int, default=64, help="编码批大小")
    args = parser.parse_args()

    shape = build_index(model_name=args.model, batch_size=args.batch_size)
    print(f"索引完成：{shape[0]} 条 × {shape[1]} 维")
    print("模型与语料指纹记录在 data/index/manifest.json，重建索引请再次运行该脚本。")


if __name__ == "__main__":  # pragma: no cover
    main()


