"""重建校园生活库 BM25 索引。

用法：

```
python -m scripts.build_life_index            # 重建并写 data/life/index.json
python -m scripts.build_life_index --check    # 只检查索引与语料是否一致
```

索引只存统计量（doc_count / doc_lengths / postings），语料仍在
`data/life/chunks.jsonl`；两者不一致时 `LifeStore` 会就地重建，
所以这个脚本主要是「部署时预热」与「CI 校验」用。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for path in (str(_ROOT), str(_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from campus_qa.config import settings  # noqa: E402
from campus_qa.life import build_index, load_chunks  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="重建生活库索引")
    parser.add_argument("--chunks", type=Path, default=settings.life_chunks_path)
    parser.add_argument("--out", type=Path, default=settings.life_index_path)
    parser.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = parser.parse_args(argv)

    chunks = load_chunks(args.chunks)
    if not chunks:
        print(f"[life] 语料为空：{args.chunks}", file=sys.stderr)
        return 2

    index = build_index(chunks)
    payload = json.dumps(index, ensure_ascii=False, separators=(",", ":"))

    if args.check:
        if not args.out.exists():
            print(f"[life] 索引不存在：{args.out}", file=sys.stderr)
            return 1
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        if int(existing.get("doc_count", -1)) != len(chunks):
            print(
                f"[life] 索引与语料不一致：index={existing.get('doc_count')} chunks={len(chunks)}",
                file=sys.stderr,
            )
            return 1
        print(f"[life] 索引一致：{len(chunks)} 条")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="utf-8")

    total_grams = len(index["postings"])
    print(f"[life] 已写入 {args.out}")
    print(f"[life] 语料 {len(chunks)} 条 / 平均长度 {index['avg_length']:.1f} / 2-gram {total_grams} 个")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
