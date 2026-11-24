"""校园墙提炼脚本测试：脱敏、去噪、去重、类目。"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.crawl_wall import (
    build_question_set,
    clean_post,
    dedupe,
    guess_category,
    load_raw_jsonl,
    normalize_question,
    QuestionItem,
)


_FIXTURE = Path(__file__).parent / "fixtures" / "wall_posts.jsonl"


def test_clean_post_removes_cq_and_mentions() -> None:
    text = "[CQ:face,id=1] @小明 请问宿舍热水几点有？"
    cleaned = clean_post(text)
    assert "[CQ:" not in cleaned
    assert "@小明" not in cleaned
    assert cleaned.startswith("请问宿舍热水")


def test_normalize_question_adds_question_mark() -> None:
    assert normalize_question("请问食堂几点开门呢") == "食堂几点开门？"
    assert normalize_question("   ") == ""


def test_guess_category() -> None:
    assert guess_category("快递驿站几点关门") == "快递"
    assert guess_category("校园卡丢了怎么办") == "校园卡"
    assert guess_category("量子力学作业") == "其他"


def test_build_question_set_from_fixture() -> None:
    posts = load_raw_jsonl(_FIXTURE, limit=100)
    assert len(posts) == 7

    kept, stats = build_question_set(posts)

    assert stats.posts == 7
    assert stats.kept == len(kept)
    assert len(kept) >= 4
    assert stats.redacted >= 1

    questions = [item.question for item in kept]
    joined = "".join(questions)
    # 个人信息不落盘
    assert "13812345678" not in joined
    assert "zhangsan" not in joined
    assert "小明" not in joined
    # 二手交易帖不是「问题」
    assert not any("二手自行车" in question for question in questions)
    # 类目覆盖
    assert {item.category for item in kept} >= {"快递", "校园卡", "餐饮", "图书馆"}


def test_empty_question_mark_does_not_create_noise() -> None:
    posts = load_raw_jsonl(_FIXTURE, limit=100)
    kept, _ = build_question_set(posts)
    assert all(question.question.endswith("？") for question in kept)


def test_replies_become_redacted_answer() -> None:
    posts = load_raw_jsonl(_FIXTURE, limit=100)
    kept, _ = build_question_set(posts)

    canteen = next(item for item in kept if item.category == "餐饮")
    assert "麻辣香锅" in canteen.answer
    # 回复里的手机号同样脱敏，不落盘
    assert "13900001111" not in canteen.answer
    assert "[手机号]" in canteen.answer
    assert canteen.as_dict()["answer"] == canteen.answer


def test_dedupe_exact_and_fuzzy() -> None:
    items = [
        QuestionItem(question="校园卡丢了怎么办？", category="校园卡", source="校园墙"),
        QuestionItem(question="校园卡丢了怎么办?", category="校园卡", source="校园墙"),
        QuestionItem(question="校园卡丢了应该怎么办？", category="校园卡", source="校园墙"),
        QuestionItem(question="食堂几点开门？", category="餐饮", source="校园墙"),
    ]
    kept, duplicates = dedupe(items, threshold=0.82)
    assert duplicates >= 1
    assert len(kept) <= 3
    assert any(item.category == "餐饮" for item in kept)


def test_report_fields(tmp_path: Path) -> None:
    from scripts.crawl_wall import _report

    posts = load_raw_jsonl(_FIXTURE, limit=100)
    kept, stats = build_question_set(posts)
    report = _report(stats, kept, "fixture")
    payload = json.dumps(report, ensure_ascii=False)

    assert "categories" in report
    assert report["kept"] == len(kept)
    assert "未保留任何发帖人信息" in report["note"]
    assert "13812345678" not in payload
