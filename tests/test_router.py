"""意图路由测试（纯规则，不联网、不加载模型）。"""

from __future__ import annotations

from campus_qa.contracts import Intent
from campus_qa.router import classify, is_followup


def test_policy_intent_for_procurement_and_repair_projects() -> None:
    assert classify("货物与服务类采购合同什么金额要签合同？").intent == Intent.POLICY
    assert classify("房屋、设施维修项目总投资50万元以上的由哪个部门归口管理？").intent == Intent.POLICY
    assert classify("基本建设项目立项由哪个部门负责？").intent == Intent.POLICY


def test_dorm_repair_still_routes_to_life() -> None:
    assert classify("宿舍空调坏了怎么报修？").intent == Intent.LIFE


def test_dorm_leader_and_garbage_sorting_routes() -> None:
    assert classify("寝室长的主要职责是什么？").intent == Intent.POLICY
    assert classify("垃圾分类投放时间是什么时候？").intent == Intent.LIFE


def test_sports_venue_intent_is_life() -> None:
    assert classify("体育场馆怎么开放？").intent == Intent.LIFE


def test_life_intent_for_canteen() -> None:
    decision = classify("食堂几点开门？")
    assert decision.intent == Intent.LIFE
    assert decision.reason == "life_keyword"


def test_policy_intent_for_leave_rules() -> None:
    decision = classify("缓考怎么申请？")
    assert decision.intent == Intent.POLICY
    assert "缓考" in decision.matched


def test_policy_intent_for_regulation_words() -> None:
    assert classify("学生宿舍管理规定有哪些？").intent == Intent.POLICY


def test_tool_weather() -> None:
    decision = classify("现在浦东天气怎么样？")
    assert decision.intent == Intent.TOOL
    assert decision.tool == "weather"


def test_tool_time_requires_explicit_phrase() -> None:
    assert classify("现在几点了？").tool == "campus_time"
    # 「图书馆几点开门」是生活问题，不能被时间工具抢走
    assert classify("图书馆几点开门？").intent == Intent.LIFE


def test_tool_calendar_and_phones() -> None:
    assert classify("本学期第几周了？").tool == "campus_calendar"
    assert classify("保卫处电话是多少？").tool == "campus_phones"


def test_chitchat() -> None:
    assert classify("你好").intent == Intent.CHITCHAT
    assert classify("谢谢你").intent == Intent.CHITCHAT


def test_unknown_question_falls_back_to_policy() -> None:
    """兜底必须回制度链路，保证拒答行为与 M3 一致。"""

    decision = classify("量子纠缠的退相干时间是多少？")
    assert decision.intent == Intent.POLICY
    assert decision.reason == "fallback"


def test_empty_question() -> None:
    decision = classify("   ")
    assert decision.intent == Intent.POLICY
    assert decision.reason == "empty"


def test_followup_inherits_previous_intent() -> None:
    decision = classify("那二楼呢？", previous_intent=Intent.LIFE, last_question="食堂几点开门？")
    assert decision.intent == Intent.LIFE
    assert decision.reason == "inherit"


def test_followup_without_history_falls_back() -> None:
    assert classify("那呢？").reason == "fallback"


def test_is_followup() -> None:
    assert is_followup("那快递呢？")
    assert is_followup("几点关门？", last_question="快递驿站在哪里？")
    assert not is_followup("请说明学生手册里关于转专业的规定")
