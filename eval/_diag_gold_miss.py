"""诊断可回答题中「引用未命中 gold」的原因分类（复用于历轮验收）。"""
import difflib
import json
from collections import Counter

from campus_qa.retriever import retrieve_candidates
from campus_qa.service import answer_question


def main() -> None:
    items = [json.loads(l) for l in open("eval/questions.jsonl", encoding="utf-8") if l.strip()]
    targets = [i for i in items if i.get("should_answer") and i.get("gold")]

    misses = []
    for item in targets:
        q = item["question"]
        gold_snips = [g["snippet"].strip() for g in item["gold"]]

        cands = retrieve_candidates(q)
        gold_in_top5 = any(
            any(c.snippet.strip() == gs for gs in gold_snips) for c in cands[:5]
        )

        r = answer_question(q)
        cites = r.citations
        if any(c.snippet.strip() in gold_snips for c in cites):
            continue

        best_ratio = 0.0
        for c in cites:
            for gs in gold_snips:
                best_ratio = max(
                    best_ratio,
                    difflib.SequenceMatcher(None, c.snippet.strip(), gs).ratio(),
                )

        misses.append(
            {
                "id": item["id"],
                "question": q,
                "gold_in_top5": gold_in_top5,
                "best_ratio": round(best_ratio, 3),
                "n_cites": len(cites),
            }
        )

    cats = Counter()
    for m in misses:
        if not m["gold_in_top5"]:
            cats["B_检索未召回gold"] += 1
        elif m["best_ratio"] >= 0.9:
            cats["A_条款重复版本(假阴性)"] += 1
        else:
            cats["C_检索到了但模型挑偏"] += 1

    print(f"可回答题带 gold：{len(targets)} 条；引用未命中：{len(misses)} 条")
    print("分类统计:", dict(cats))
    print()
    for m in misses:
        if not m["gold_in_top5"]:
            cat = "B"
        elif m["best_ratio"] >= 0.9:
            cat = "A"
        else:
            cat = "C"
        print(f"[{m['id']}] {m['question'][:36]} | {cat} | gold在top5={m['gold_in_top5']} | 最高相似度={m['best_ratio']} | 引用数={m['n_cites']}")


if __name__ == "__main__":
    main()
