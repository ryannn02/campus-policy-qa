"""生活库回答精简：只保留与问题相关的内容，不整段贴资料。"""

from __future__ import annotations

from campus_qa import life_summary
from campus_qa.life import LifeChunk, LifeHit


def _hit(chunk_id: str, title: str, text: str, score: float = 1.0) -> LifeHit:
    return LifeHit(chunk=LifeChunk(chunk_id=chunk_id, title=title, text=text), score=score)


_CANTEEN = (
    "食堂营业时间（几点开门/供餐时段）：校内共 4 个食堂：东食堂、西食堂、金海苑、南溪苑。"
    "供餐时间：东一 6:30–21:00；东二早餐 6:30–9:30、午餐 10:30–13:00；西一/西二 6:30–20:30。"
    "食堂由后勤服务中心统筹管理，对饭菜质量有意见可通过“上海二工大后勤”反馈。"
)
_EXPRESS = (
    "校园快递中心位于体育场东南角，服务时间 9:00–21:00；凭取件码取件。"
    "寄件需要实名认证，包装请尽量简单，胶带不要缠太多。"
    "校园快递由后勤服务中心统筹管理，有问题可联系快递中心前台。"
)


def test_extractive_keeps_only_time_sentence_for_when_question() -> None:
    hits = [_hit("c1", "食堂营业时间", _CANTEEN)]

    answer = life_summary._extractive("食堂几点开饭？", hits)

    assert "6:30" in answer
    assert "反馈" not in answer
    assert "统筹管理" not in answer


def test_extractive_prefers_location_sentence_for_where_question() -> None:
    hits = [_hit("c2", "快递驿站位置", _EXPRESS)]

    answer = life_summary._extractive("快递驿站在哪里？", hits)

    assert "体育场东南角" in answer
    assert "寄件" not in answer
    assert "统筹管理" not in answer


def test_extractive_caps_total_length() -> None:
    long_text = "".join(f"第{i}条服务说明的时间是 {i}:00–{i + 1}:00，地点在教学楼{i}层。" for i in range(1, 9))
    hits = [_hit("c3", "长段落", long_text)]

    answer = life_summary._extractive("几点开门", hits)

    assert len(answer) <= 260 + 40  # 单句不会被截断，允许一句的余量


def test_condense_falls_back_to_extractive_without_llm(monkeypatch) -> None:
    hits = [_hit("c4", "快递驿站位置", _EXPRESS)]

    def _boom(*_args, **_kwargs):
        raise RuntimeError("no llm")

    monkeypatch.setattr(life_summary, "_llm_condense", _boom)
    answer = life_summary.condense("快递驿站在哪里？", hits)

    assert "体育场东南角" in answer


def test_condense_empty_without_hits(monkeypatch) -> None:
    monkeypatch.setattr(life_summary, "_llm_condense", lambda *_a, **_k: "不应被调用")
    assert life_summary.condense("任何问题", []) == ""
