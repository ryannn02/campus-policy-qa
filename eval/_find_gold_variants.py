"""为可回答题 gold 补充「实质相同」的多版本条款（成员 C，M2 收尾）。

诊断：同一制度条款在多份文件里重复收录、条款号不同（如「第六条/第九条」），
gold 只标了一个版本，系统命中另一个版本被判 ❌。本脚本：

- 对每条可回答题的每个 gold snippet，在 data/chunks.jsonl 里用
  difflib.SequenceMatcher 找字符重合率 ≥0.90 的其它 chunk；
- 命中且 document_id 不同 / text 不同的，把这些 chunk 作为新 gold 元素追加；
- 只追加、不删除；不改 question/category/should_answer。

用法：
  python eval/_find_gold_variants.py            # dry-run，打印候选变体对
  python eval/_find_gold_variants.py --apply    # 应用，写回 eval/questions.jsonl
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"

RATIO_THRESHOLD = 0.92

# 主体 / 奖项限定词冲突：国家 vs 上海市、研究生 vs 非研究生（学生/本专科）。
# 这类「不同制度但条款相似」不算实质相同的多版本，需剔除。
_ENTITY_CONFLICTS = (("国家", "上海市"),)


def _conflicting_entities(a: str, b: str) -> bool:
    """判断两个文本是否含冲突的主体/奖项限定词（不同制度却条款相似）。"""

    for x, y in _ENTITY_CONFLICTS:
        if (x in a and y in b) or (y in a and x in b):
            return True
    a_grad = "研究生" in a
    b_grad = "研究生" in b
    if a_grad != b_grad:
        return True
    return False


def _normalize(text: str) -> str:
    """归一化：去掉开头条款号（第X条）与所有空白，保留正文。

    使「第六条/第九条」「第七条/第四条」这类仅条款号不同的版本
    归一化后高度一致；而主题巧合（共享一句引用但正文不同）不受影响。
    """

    t = re.sub(r"^第[一二三四五六七八九十百\d]+条\s*", "", text)
    t = re.sub(r"\s+", "", t)
    return t


def _coerce_page(value: object):
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        m = re.search(r"\d+", value)
        return int(m.group()) if m else None
    return None


def load_chunks() -> List[dict]:
    chunks: List[dict] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            chunks.append({
                "document_id": r.get("doc_id"),
                "snippet": (r.get("text") or "").strip(),
                "page": _coerce_page(r.get("page")),
                "section": (r.get("section") or "").strip() or (r.get("title") or "").strip(),
            })
    return chunks


def load_questions() -> List[dict]:
    return [json.loads(l) for l in QUESTIONS_PATH.open("r", encoding="utf-8") if l.strip()]


def find_variants(snippet: str, chunks: List[dict], threshold: float) -> List[Tuple[dict, float]]:
    """在 chunks 中找与 snippet 字符重合率 ≥ threshold 的其它 chunk。

    用归一化（去条款号/空白）后的文本比较，避免仅条款号不同被判低分；
    主题巧合（共享一句引用但正文不同）则因归一化后仍不同而被排除。
    """

    variants: List[Tuple[dict, float]] = []
    n = len(snippet)
    if n == 0:
        return variants
    norm_snippet = _normalize(snippet)
    for c in chunks:
        c_text = c["snippet"]
        if c_text == snippet:
            continue
        # 长度预筛：重合率 ≥0.9 时长度差不会超过约 25%。
        if abs(len(c_text) - n) / n > 0.25:
            continue
        if _conflicting_entities(snippet, c_text):
            continue
        ratio = difflib.SequenceMatcher(None, norm_snippet, _normalize(c_text)).ratio()
        if ratio >= threshold:
            variants.append((c, ratio))
    return variants


def main() -> None:
    apply = "--apply" in __import__("sys").argv
    chunks = load_chunks()
    items = load_questions()

    total_added = 0
    affected_questions = 0
    report: List[str] = []

    for item in items:
        if not item.get("should_answer"):
            continue
        gold = item.get("gold") or []
        if not gold:
            continue

        existing_snips = {g.get("snippet", "") for g in gold}
        new_gold = list(gold)
        added_for_this = 0

        for g in gold:
            snippet = g.get("snippet", "")
            for c, ratio in find_variants(snippet, chunks, RATIO_THRESHOLD):
                c_text = c["snippet"]
                if c_text in existing_snips:
                    continue
                existing_snips.add(c_text)
                new_gold.append({
                    "document_id": c["document_id"],
                    "snippet": c_text,
                    "page": c["page"],
                    "section": c["section"],
                })
                total_added += 1
                added_for_this += 1
                report.append(
                    f"[{item['id']}] ratio={ratio:.3f}\n"
                    f"  原: {snippet[:60].replace(chr(10), ' ')}\n"
                    f"  变: {c_text[:60].replace(chr(10), ' ')}\n"
                    f"  变 doc={c['document_id']} page={c['page']}"
                )

        if added_for_this:
            affected_questions += 1
            if apply:
                item["gold"] = new_gold

    print(f"候选变体总数: {total_added}（涉及 {affected_questions} 题）")
    print("=" * 70)
    for line in report:
        print(line)

    if apply:
        with QUESTIONS_PATH.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\n[applied] 已写回 {QUESTIONS_PATH}")
    else:
        print("\n[dry-run] 未写回。确认无误后加 --apply 重跑。")


if __name__ == "__main__":
    main()
