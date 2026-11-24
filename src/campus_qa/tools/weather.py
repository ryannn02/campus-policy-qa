"""天气工具：Open-Meteo（免 API key）。

默认坐标是金海路校区（浦东），可在 `.env` 用 WEATHER_CITY / WEATHER_LATITUDE /
WEATHER_LONGITUDE 改。工具只做「查天气」，不猜气候趋势，也不做穿衣建议的
自由发挥——穿衣提示由固定规则生成，避免模型编造。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import httpx

from ..config import settings
from ..contracts import ToolResult


_API_URL = "https://api.open-meteo.com/v1/forecast"

# WMO 天气代码 → 中文描述
_WMO_CODES: Dict[int, str] = {
    0: "晴",
    1: "大部晴朗",
    2: "多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "毛毛雨",
    53: "毛毛雨",
    55: "毛毛雨",
    56: "冻毛毛雨",
    57: "冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "阵雨",
    81: "阵雨",
    82: "强阵雨",
    85: "阵雪",
    86: "阵雪",
    95: "雷阵雨",
    96: "雷阵雨伴冰雹",
    99: "雷阵雨伴冰雹",
}


def _describe(code: Any) -> str:
    try:
        return _WMO_CODES.get(int(code), "未知天气")
    except (TypeError, ValueError):
        return "未知天气"


def _get_json(url: str, params: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """单次 HTTP GET，独立成函数便于测试替换（不联网）。"""

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.json()


def _dress_tip(temperature: float, rain_probability: float) -> str:
    if temperature >= 30:
        tip = "短袖短裤，注意防晒补水"
    elif temperature >= 24:
        tip = "短袖即可，早晚可备薄外套"
    elif temperature >= 18:
        tip = "长袖或薄外套"
    elif temperature >= 12:
        tip = "外套加长裤，早晚偏凉"
    elif temperature >= 5:
        tip = "厚外套或大衣，注意保暖"
    else:
        tip = "羽绒服，注意防寒"
    if rain_probability >= 60:
        tip += "；降水概率高，记得带伞"
    return tip


def query(city: Optional[str] = None, days: int = 3) -> ToolResult:
    """查询当前天气与未来几天预报。"""

    target_city = city or settings.weather_city
    params = {
        "latitude": settings.weather_latitude,
        "longitude": settings.weather_longitude,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
        "weather_code,wind_speed_10m,precipitation",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
        "precipitation_probability_max",
        "timezone": "Asia/Shanghai",
        "forecast_days": max(1, min(days, 7)),
    }

    try:
        payload = _get_json(_API_URL, params, settings.tool_timeout_seconds)
    except Exception as exc:  # noqa: BLE001 - 网络问题统一降级为可读错误
        return ToolResult(
            name="weather",
            ok=False,
            error=f"天气服务暂时不可用：{type(exc).__name__}",
            data={"city": target_city},
        )

    current = payload.get("current") or {}
    daily = payload.get("daily") or {}
    temperature = float(current.get("temperature_2m", 0.0))
    apparent = float(current.get("apparent_temperature", temperature))
    humidity = current.get("relative_humidity_2m")
    wind = current.get("wind_speed_10m")
    condition = _describe(current.get("weather_code"))

    dates = daily.get("time") or []
    highs = daily.get("temperature_2m_max") or []
    lows = daily.get("temperature_2m_min") or []
    rain = daily.get("precipitation_probability_max") or []
    codes = daily.get("weather_code") or []

    forecasts: List[Dict[str, Any]] = []
    for idx, date in enumerate(dates):
        forecasts.append(
            {
                "date": date,
                "weekday": _weekday_cn(date),
                "condition": _describe(codes[idx] if idx < len(codes) else None),
                "high": highs[idx] if idx < len(highs) else None,
                "low": lows[idx] if idx < len(lows) else None,
                "rain_probability": rain[idx] if idx < len(rain) else None,
            }
        )

    today_rain = float(rain[0]) if rain else 0.0
    dress = _dress_tip(temperature, today_rain)

    answer_parts = [
        f"{target_city}现在{condition}，气温 {temperature:.0f}℃"
        f"（体感 {apparent:.0f}℃）"
    ]
    if humidity is not None:
        answer_parts.append(f"湿度 {humidity}%")
    if wind is not None:
        answer_parts.append(f"风速 {wind} km/h")
    answer = "，".join(answer_parts) + "。"
    if forecasts:
        today = forecasts[0]
        if today.get("high") is not None and today.get("low") is not None:
            answer += (
                f"今天 {today['low']:.0f}~{today['high']:.0f}℃，"
                f"降水概率 {today_rain:.0f}%。"
            )
    answer += f"穿衣建议：{dress}。"

    return ToolResult(
        name="weather",
        ok=True,
        data={
            "city": target_city,
            "condition": condition,
            "temperature": round(temperature, 1),
            "apparent_temperature": round(apparent, 1),
            "humidity": humidity,
            "wind_speed": wind,
            "rain_probability": today_rain,
            "dress_tip": dress,
            "forecast": forecasts,
            "provider": "Open-Meteo",
            "fetched_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            "answer": answer,
        },
    )


def _weekday_cn(date_text: str) -> str:
    try:
        parsed = datetime.strptime(date_text, "%Y-%m-%d")
    except (TypeError, ValueError):
        return ""
    return "周" + "一二三四五六日"[parsed.weekday()]
