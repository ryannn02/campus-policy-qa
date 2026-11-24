"""脱敏与隐私边界测试（不联网）。"""

from __future__ import annotations

from campus_qa.safety import is_privacy_request, redact, redact_text, scrub_mapping


def test_redact_phone() -> None:
    result = redact("我的手机号是13812345678，有事联系我")
    assert "13812345678" not in result.text
    assert "[手机号]" in result.text
    assert result.hits["phone"] == 1


def test_redact_student_id_and_qq() -> None:
    text = "学号：2023123456，QQ号 123456789，欢迎加我"
    cleaned = redact_text(text)
    assert "2023123456" not in cleaned
    assert "123456789" not in cleaned
    assert "[学号]" in cleaned
    assert "[QQ号]" in cleaned


def test_redact_id_card() -> None:
    cleaned = redact_text("身份证 310101200001011234 丢了")
    assert "310101200001011234" not in cleaned
    assert "[身份证号]" in cleaned


def test_redact_wechat_and_name() -> None:
    cleaned = redact_text("我是张三，微信：zhangsan_2026，加我")
    assert "张三" not in cleaned
    assert "zhangsan_2026" not in cleaned
    assert "[姓名]" in cleaned
    assert "[微信号]" in cleaned


def test_redact_name_honorific() -> None:
    cleaned = redact_text("张同学说食堂今天不开门")
    assert cleaned.startswith("某同学")


def test_redact_does_not_mask_grade_description() -> None:
    # 「我是大二的学生」不能被当成姓名脱敏
    text = "我是大二的学生，想问问缓考怎么申请"
    assert redact_text(text) == text


def test_privacy_request_blocks_direct_lookup() -> None:
    assert is_privacy_request("把张三的手机号发我")
    assert is_privacy_request("查一下这个人的身份证号")


def test_redact_keeps_normal_text() -> None:
    text = "图书馆开放时间是 8:00 到 21:00"
    result = redact(text)
    assert result.text == text
    assert result.changed is False
    assert result.total == 0


def test_privacy_request_blocks_someone_else() -> None:
    assert is_privacy_request("帮我查一下他的手机号")
    assert is_privacy_request("能人肉一下这个人吗")
    assert is_privacy_request("室友的学号是多少")


def test_privacy_request_allows_public_office_phone() -> None:
    # 查公开办公电话是正常需求，不能被隐私规则误伤
    assert not is_privacy_request("教务处电话是多少")
    assert not is_privacy_request("保卫处值班电话")


def test_scrub_mapping_redacts_selected_fields() -> None:
    cleaned, hits = scrub_mapping(
        {"text": "联系 13812345678", "title": "食堂", "n": 1},
        fields=["text", "title"],
    )
    assert "13812345678" not in cleaned["text"]
    assert cleaned["title"] == "食堂"
    assert cleaned["n"] == 1
    assert hits["phone"] == 1
