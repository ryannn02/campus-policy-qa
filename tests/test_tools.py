"""工具层测试：天气（注入假响应）、时间、校历、电话。"""

from __future__ import annotations

from datetime import date

import pytest

from campus_qa.tools import campus, weather


_FAKE_PAYLOAD = {
    "current": {
        "temperature_2m": 18.4,
        "apparent_temperature": 17.9,
        "relative_humidity_2m": 62,
        "weather_code": 3,
        "wind_speed_10m": 12.5,
        "precipitation": 0.0,
    },
    "daily": {
        "time": ["2026-11-24", "2026-11-25", "2026-11-26"],
        "weather_code": [3, 61, 2],
        "temperature_2m_max": [21.0, 19.0, 20.0],
        "temperature_2m_min": [14.0, 13.0, 13.5],
        "precipitation_probability_max": [10, 70, 20],
    },
}


def test_weather_query_with_injected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(weather, "_get_json", lambda *args, **kwargs: _FAKE_PAYLOAD)

    result = weather.query()

    assert result.ok is True
    assert result.name == "weather"
    assert result.data["condition"] == "阴"
    assert result.data["temperature"] == 18.4
    assert "上海浦东" in result.data["answer"]
    assert "℃" in result.data["answer"]
    assert len(result.data["forecast"]) == 3
    assert result.data["forecast"][0]["weekday"] == "周二"


def test_weather_query_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(weather, "_get_json", _boom)

    result = weather.query()
    assert result.ok is False
    assert "天气服务暂时不可用" in (result.error or "")


def test_dress_tip_covers_rain() -> None:
    assert "带伞" in weather._dress_tip(20.0, 80)
    assert "羽绒服" in weather._dress_tip(2.0, 10)


def test_current_time_returns_iso_and_answer() -> None:
    result = campus.current_time()

    assert result.ok is True
    assert result.data["datetime"].endswith("+08:00")
    assert result.data["timezone"] == "Asia/Shanghai"
    assert "星期" in result.data["answer"]


def test_calendar_payload_week_number() -> None:
    payload = campus._calendar_payload(date(2026, 11, 24))
    # 校历文件：2026 秋季学期第 1 周周一为 9月21日，11-24 是第 10 周（用户确认）
    assert payload["in_term"] is True
    assert payload["week"] == 10
    assert payload["term"] == "2026-2027 学年秋季学期"


def test_calendar_payload_before_term() -> None:
    payload = campus._calendar_payload(date(2026, 8, 1))
    assert payload["in_term"] is False
    assert payload["week"] is None
    assert payload["days_until_start"] == 51


def test_calendar_counts_each_term_separately(tmp_path, monkeypatch) -> None:
    """夏季学期与秋季学期各自从第 1 周起算，不跨学期累加。"""

    import dataclasses
    import json

    calendar = tmp_path / "academic_calendar.json"
    calendar.write_text(
        json.dumps(
            {
                "terms": [
                    {"name": "夏季学期", "start": "2026-06-22", "weeks": 12},
                    {"name": "秋季学期", "start": "2026-09-21", "weeks": 20},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        campus, "settings", dataclasses.replace(campus.settings, academic_calendar_path=calendar)
    )

    summer = campus._calendar_payload(date(2026, 7, 13))  # 夏季学期第 4 周周一
    assert (summer["in_term"], summer["week"], summer["term"]) == (True, 4, "夏季学期")

    between = campus._calendar_payload(date(2026, 9, 15))  # 两学期之间：假期
    assert between["in_term"] is False and between["week"] is None

    autumn = campus._calendar_payload(date(2026, 9, 22))  # 秋季开学第 2 天，重新从第 1 周算
    assert (autumn["in_term"], autumn["week"], autumn["term"]) == (True, 1, "秋季学期")


def test_phones_returns_public_numbers_only() -> None:
    result = campus.phones()
    numbers = [item["phone"] for item in result.data["items"]]

    assert "50214974" in numbers
    # 不允许出现 11 位个人手机号
    assert all(not (number.isdigit() and len(number) == 11) for number in numbers)
    assert "个人手机号不在本系统收录范围" in result.data["note"]


def test_phones_keyword_filter() -> None:
    result = campus.phones("保卫")
    assert len(result.data["items"]) == 1
    assert result.data["items"][0]["name"] == "保卫处值班"
