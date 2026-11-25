"""校园墙 → 问题集：抓取、提炼、脱敏、去重。

用户要的不是「导入一份别人整理好的 CSV」，而是**直接从校园墙拿原始内容，
再从中提炼出学生真正在问的问题**。所以这个脚本的主路径是「抓」，
提炼与脱敏是内置步骤，CSV/JSONL 只作为离线调试的备用输入。

用法：

```
# 1) 校园墙有公开/半公开接口时
python -m scripts.crawl_wall --source "https://example.edu/wall/api" --limit 300

# 2) 已把墙的原始内容落成 JSONL（每行一个 {"text": "..."}）时，离线验证提炼质量
python -m scripts.crawl_wall --raw-jsonl data/wall/raw_sample.jsonl

# 3) 只看提炼结果，不写文件
python -m scripts.crawl_wall --raw-jsonl data/wall/raw_sample.jsonl --dry-run
```

输出：

- `data/wall/questions.jsonl`：提炼后的问题（已脱敏、已去重），可直接作为评测集；
- `data/wall/report.json`：抓了多少条、提炼出多少问、各类目分布、丢弃原因。

**边界**：只提炼「问题」与「回答所依据的信息」，不采集发帖人身份；
所有文本先过 `campus_qa.safety.redact`，命中个人信息即替换为占位符再落盘。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for path in (str(_ROOT), str(_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from campus_qa.config import settings  # noqa: E402
from campus_qa.safety import redact  # noqa: E402


# 问句信号：有这些词/标点才可能是「学生在问什么」
_QUESTION_SIGNALS = re.compile(
    r"[？?]|怎么|怎样|如何|哪里|在哪|几点|多少|什么|能不能|可不可以|是不是|有没有|"
    r"吗|呢|求问|请问|想问|有人知道|谁知道|求助|求告知|想问一下|"
    # 选择问与正反问没有「吗」，靠这些信号识别：A 还是 B / 用不用 / 应不应该
    r"还是|哪|用不用|应不应该|要不要|能否|([\u4e00-\u9fff])不\1"
)

# 帖子里的噪音：墙号自称、投稿格式、@、话题、链接、表情
_NOISE_PATTERNS = [
    re.compile(r"\[CQ:[^\]]+\]"),          # OneBot CQ 码
    re.compile(r"@[^\s，。！？]{1,20}"),    # @某人
    re.compile(r"https?://\S+"),           # 链接
    re.compile(r"#[^#]{1,30}#"),           # 话题
    re.compile(r"^(墙墙|表白墙|校园墙|投稿|匿名|代发|帮转|求扩散)[：:\s]*"),
]

# 投稿常见的开场白，提炼时去掉，不动语义
_PREFIX_PATTERNS = [
    re.compile(r"^(求问|请问|想问一下|想问|有人知道|谁知道|求助)[：:，,\s]*"),
    re.compile(r"^(万能的朋友圈|万能的墙|在线等|急)[：:，,\s]*"),
]

_SENTENCE_SPLIT = re.compile(r"[。！？!?\n；;]+")

# 句尾常见投稿口头禅，提炼时剥掉
_TRAILING_NOISE = re.compile(r"[，,；;]?\s*(在线等|急|求回复|谢谢|拜托|求告知)\s*[。！!？?]*$")

# 去重时忽略的填充词：让「应该怎么办」和「怎么办」落到同一个 key
_DEDUPE_FILLERS = re.compile(r"应该|可以|一下|请问|想问|想问问|大概|具体|有人知道|谁知道|求助")

_CATEGORY_RULES: List[Tuple[str, re.Pattern[str]]] = [
    ("餐饮", re.compile(r"食堂|吃饭|早餐|午餐|晚餐|夜宵|饭菜|打饭|餐厅")),
    ("快递", re.compile(r"快递|驿站|取件|寄件|包裹|外卖柜")),
    ("校园卡", re.compile(r"校园卡|一卡通|校园码|充值|挂失|补卡|余额")),
    ("宿舍", re.compile(r"宿舍|寝室|热水|洗澡|洗衣|空调|报修|开水")),
    ("图书馆", re.compile(r"图书馆|自习|座位|借书|还书|闭馆|开放时间")),
    ("网络", re.compile(r"校园网|wifi|无线|宽带|上网|认证")),
    ("医疗", re.compile(r"校医院|医务室|看病|医保|报销|开药")),
    ("交通", re.compile(r"班车|校车|地铁|停车|充电|电动车|通行证")),
    ("学习", re.compile(r"选课|学分|补考|重修|缓考|成绩|考试|教务")),
]


@dataclass
class WallPost:
    """一条墙上的原始内容（不含任何作者字段）。

    `replies` 是这条帖子下的回复/评论：只用来提炼「回答」，
    同样会先脱敏，且不保留回复者身份。
    """

    text: str
    created_at: str = ""
    replies: List[str] = field(default_factory=list)


@dataclass
class QuestionItem:
    question: str
    category: str
    source: str
    created_at: str = ""
    answer: str = ""

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "question": self.question,
            "category": self.category,
            "source": self.source,
            "created_at": self.created_at,
        }
        if self.answer:
            payload["answer"] = self.answer
        return payload


@dataclass
class ExtractStats:
    posts: int = 0
    candidates: int = 0
    kept: int = 0
    duplicates: int = 0
    too_short: int = 0
    too_long: int = 0
    redacted: int = 0
    dropped_by_rule: Dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.dropped_by_rule[reason] = self.dropped_by_rule.get(reason, 0) + 1


# ---------------------------------------------------------------- 抓取


def _collect_replies(row: Dict[str, Any]) -> List[str]:
    """从一条记录里收集回复文本（支持 replies 数组 / answer / reply 字段）。"""

    replies: List[str] = []
    raw_replies = row.get("replies")
    if isinstance(raw_replies, list):
        for item in raw_replies:
            if isinstance(item, str) and item.strip():
                replies.append(item.strip())
            elif isinstance(item, dict):
                for key in ("text", "content", "message"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        replies.append(value.strip())
                        break
    for key in ("answer", "reply"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            replies.append(value.strip())
    return replies


def fetch_http(source: str, limit: int, timeout: float) -> List[WallPost]:
    """从校园墙的 HTTP 接口抓原始帖子。

    兼容三种返回：`{"posts": [...]}`、`{"data": {"posts": [...]}}`、顶层数组。
    每个元素取 `text` / `content` / `message` / `raw_message` 中第一个字符串。
    """

    import httpx

    headers = {"User-Agent": "campus-qa-wall-crawler/0.1"}
    if settings.wall_source and settings.wall_source.startswith("http"):
        headers.setdefault("Referer", settings.wall_source)

    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        response = client.get(source, params={"limit": limit})
        response.raise_for_status()
        payload = response.json()

    rows: Iterable[Any]
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("posts") or (payload.get("data") or {}).get("posts") or []
    else:
        rows = []

    posts: List[WallPost] = []
    for row in rows:
        if isinstance(row, str):
            posts.append(WallPost(text=row))
            continue
        if not isinstance(row, dict):
            continue
        for key in ("text", "content", "message", "raw_message"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                posts.append(
                    WallPost(
                        text=value,
                        created_at=str(row.get("created_at") or ""),
                        replies=_collect_replies(row),
                    )
                )
                break
        if len(posts) >= limit:
            break
    return posts


def load_raw_jsonl(path: Path, limit: int) -> List[WallPost]:
    """离线调试用：读取已落盘的原始帖子（每行一个对象或裸字符串）。"""

    posts: List[WallPost] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                payload = line
            if isinstance(payload, str):
                posts.append(WallPost(text=payload))
            elif isinstance(payload, dict):
                text = payload.get("text") or payload.get("content") or payload.get("message")
                if isinstance(text, str):
                    posts.append(
                        WallPost(
                            text=text,
                            created_at=str(payload.get("created_at") or ""),
                            replies=_collect_replies(payload),
                        )
                    )
            if len(posts) >= limit:
                break
    return posts


# ---------------------------------------------------------------- 提炼


def clean_post(text: str) -> str:
    cleaned = text or ""
    for pattern in _NOISE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    for pattern in _PREFIX_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_question(text: str) -> str:
    """统一问句形态：去尾随口头禅与语气词，补问号。"""

    body = _TRAILING_NOISE.sub("", text.strip()).strip()
    body = body.strip("，,、 ")
    for pattern in _PREFIX_PATTERNS:
        body = pattern.sub("", body).strip()
    # 只去「呢/啊/呀/么」这类语气词；「吗」是疑问标记，「什么」的「么」要保留
    body = re.sub(r"(?<!什)(呢|啊|呀|么)$", "", body).strip()
    body = body.rstrip("。！!?？；; ")
    if not body:
        return ""
    return f"{body}？"


def guess_category(text: str) -> str:
    for name, pattern in _CATEGORY_RULES:
        if pattern.search(text):
            return name
    return "其他"


def extract_questions(post: WallPost, stats: ExtractStats) -> List[QuestionItem]:
    """从单条帖子里抽出问句。"""

    cleaned = clean_post(post.text)
    if not cleaned:
        stats.drop("empty")
        return []

    # 回复只作为「回答」素材：先清洗、再脱敏，不保留回复者身份
    answer_text = ""
    for raw_reply in post.replies:
        reply = clean_post(raw_reply)
        if not reply:
            continue
        reply_redacted = redact(reply)
        if reply_redacted.changed:
            stats.redacted += 1
        answer_text = f"{answer_text}；{reply_redacted.text}" if answer_text else reply_redacted.text
    answer_text = answer_text[:200]

    items: List[QuestionItem] = []
    for raw_sentence in _SENTENCE_SPLIT.split(cleaned):
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        stats.candidates += 1

        if len(sentence) < 6:
            stats.too_short += 1
            stats.drop("too_short")
            continue
        if len(sentence) > 80:
            stats.too_long += 1
            stats.drop("too_long")
            continue

        # 先脱敏再判问句：统计准确，且个人信息不会进入任何下游判断
        redacted = redact(sentence)
        if redacted.changed:
            stats.redacted += 1

        if not _QUESTION_SIGNALS.search(redacted.text):
            stats.drop("not_a_question")
            continue

        question = normalize_question(redacted.text)
        if not question:
            stats.drop("normalize_empty")
            continue

        items.append(
            QuestionItem(
                question=question,
                category=guess_category(question),
                source="校园墙",
                created_at=post.created_at,
                answer=answer_text,
            )
        )
    return items


def _shingles(text: str) -> set[str]:
    compact = re.sub(r"[\s，。！？、,.!?；;：:（）()\"'“”]", "", text)
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def _jaccard(a: str, b: str) -> float:
    left, right = _shingles(a), _shingles(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _dedupe_key(text: str) -> str:
    """去重键：去标点 + 去填充词。"""

    compact = re.sub(r"[\s，。！？、,.!?；;：:（）()\"'“”]", "", text)
    return _DEDUPE_FILLERS.sub("", compact)


def dedupe(items: List[QuestionItem], threshold: float = 0.82) -> Tuple[List[QuestionItem], int]:
    """去重：先精确（去标点与填充词），再 2-gram Jaccard 模糊。"""

    kept: List[QuestionItem] = []
    seen: List[str] = []
    duplicates = 0

    for item in items:
        key = _dedupe_key(item.question)
        if key in seen:
            duplicates += 1
            continue
        if any(_jaccard(item.question, existing) >= threshold for existing in seen):
            duplicates += 1
            continue
        seen.append(key)
        kept.append(item)
    return kept, duplicates


# ---------------------------------------------------------------- 主流程


def build_question_set(
    posts: List[WallPost],
    threshold: float = 0.82,
) -> Tuple[List[QuestionItem], ExtractStats]:
    stats = ExtractStats(posts=len(posts))
    candidates: List[QuestionItem] = []
    for post in posts:
        candidates.extend(extract_questions(post, stats))

    kept, duplicates = dedupe(candidates, threshold=threshold)
    stats.duplicates += duplicates
    stats.kept = len(kept)
    return kept, stats


def _report(stats: ExtractStats, kept: List[QuestionItem], source: str) -> Dict[str, Any]:
    categories: Dict[str, int] = {}
    for item in kept:
        categories[item.category] = categories.get(item.category, 0) + 1
    return {
        "source": source,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "posts": stats.posts,
        "candidates": stats.candidates,
        "kept": stats.kept,
        "duplicates": stats.duplicates,
        "too_short": stats.too_short,
        "too_long": stats.too_long,
        "redacted": stats.redacted,
        "dropped_by_rule": stats.dropped_by_rule,
        "categories": categories,
        "note": "仅提炼问题，未保留任何发帖人信息；文本已脱敏",
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="从校园墙提炼问题集")
    parser.add_argument("--source", help="校园墙 HTTP 接口（或 browser，需另行接入）")
    parser.add_argument("--raw-jsonl", type=Path, help="离线调试：已落盘的原始帖子")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--out", type=Path, default=_ROOT / "data" / "wall" / "questions.jsonl")
    parser.add_argument("--report", type=Path, default=_ROOT / "data" / "wall" / "report.json")
    parser.add_argument("--threshold", type=float, default=0.82, help="模糊去重阈值")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    args = parser.parse_args(argv)

    if args.raw_jsonl:
        posts = load_raw_jsonl(args.raw_jsonl, args.limit)
        source_label = f"file:{args.raw_jsonl}"
    elif args.source and args.source != "browser":
        posts = fetch_http(args.source, args.limit, settings.tool_timeout_seconds)
        source_label = args.source
    elif args.source == "browser":
        print(
            "浏览器模式需要校园墙的具体入口（QQ 空间号 / 群号 / 小程序 / 网页），"
            "确认入口后再补适配。",
            file=sys.stderr,
        )
        return 2
    else:
        print(
            "未提供校园墙入口。请用 --source <接口地址> 或设置 .env 的 WALL_SOURCE；\n"
            "如果只有排序后的导出文件，可用 --raw-jsonl 做离线提炼验证。",
            file=sys.stderr,
        )
        return 2

    kept, stats = build_question_set(posts, threshold=args.threshold)
    report = _report(stats, kept, source_label)

    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        for item in kept[:20]:
            print(f"- [{item.category}] {item.question}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(kept, start=1):
            payload = item.as_dict()
            payload["id"] = f"W{index:04d}"
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[wall] 抓取 {stats.posts} 条 → 候选 {stats.candidates} → 保留 {stats.kept}")
    print(f"[wall] 已写入 {args.out}")
    print(f"[wall] 报告 {args.report}")
    return 0 if kept else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
