"""校园工具：当前时间、校历周次、常用电话。

这些问题的共同点是「不需要检索、也不需要模型」——答案要么来自系统时钟，
要么来自一张固定表。放进工具层可以保证稳定、零幻觉、毫秒级返回，
也避免把它们硬塞进制度库污染语料。
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from ..config import settings
from ..contracts import ToolResult


_TZ = ZoneInfo("Asia/Shanghai")
_WEEKDAYS = "一二三四五六日"

# 只收录学校官网 / 保卫处 / 校办公开的号码，不收录个人手机号。
_PUBLIC_PHONES: List[Dict[str, str]] = [
    {"name": "学校总机", "phone": "021-50215021", "scene": "总机可转各部门"},
    {"name": "学校总值班", "phone": "50214110", "scene": "24 小时应急"},
    {"name": "保卫处值班", "phone": "50214974", "scene": "治安、失物、门禁"},
    {"name": "校办（白天）", "phone": "50214090", "scene": "8:00–16:00 建议意见"},
    {"name": "学工部（夜间）", "phone": "50217729", "scene": "16:00–次日 8:00 学生事务"},
    {"name": "校长信箱", "phone": "sspu_xz@sspu.edu.cn", "scene": "邮件，全天受理"},
]


def _now() -> datetime:
    return datetime.now(_TZ)


def current_time() -> ToolResult:
    """当前日期时间 + 星期 + 教学周。"""

    now = _now()
    calendar_info = _calendar_payload(now.date())
    answer = (
        f"现在是 {now.strftime('%Y年%m月%d日')} 星期{_WEEKDAYS[now.weekday()]} "
        f"{now.strftime('%H:%M')}。"
    )
    if calendar_info.get("in_term"):
        answer += f"{calendar_info.get('term') or '本学期'}第 {calendar_info['week']} 周。"
    return ToolResult(
        name="campus_time",
        ok=True,
        data={
            "datetime": now.isoformat(timespec="seconds"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M"),
            "weekday": f"星期{_WEEKDAYS[now.weekday()]}",
            "timezone": "Asia/Shanghai",
            "calendar": calendar_info,
            "answer": answer,
        },
    )


def _parse_date(raw: Optional[str]) -> Optional[date]:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _load_terms() -> List[Dict[str, Any]]:
    """读取校历文件里的学期列表（已解析日期，按 start 升序）；文件缺失或损坏返回空。"""

    path = settings.academic_calendar_path
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError):
        return []

    terms: List[Dict[str, Any]] = []
    for item in raw.get("terms", []) if isinstance(raw, dict) else []:
        start = _parse_date(item.get("start"))
        if start is None:
            continue
        terms.append(
            {
                "name": str(item.get("name") or ""),
                "start": start,
                "weeks": int(item.get("weeks") or 20),
                "source": item.get("source"),
            }
        )
    terms.sort(key=lambda term: term["start"])
    return terms


def _calendar_payload(today: Optional[date] = None) -> Dict[str, Any]:
    """按学期独立计算教学周。

    教学周不是从某个固定日期一直往后数，而是春 / 夏 / 秋每个学期各自从第 1 周起算。
    优先用校历文件里覆盖今天的那个学期；没有覆盖则提示假期 / 未开学；
    校历文件不可用时才退回旧的单一起点配置（SEMESTER_START）。
    """

    today = today or _now().date()
    terms = _load_terms()

    if terms:
        current = None
        for term in terms:
            delta_days = (today - term["start"]).days
            if 0 <= delta_days < term["weeks"] * 7:
                current = (term, delta_days)
        if current is not None:
            term, delta_days = current
            week = delta_days // 7 + 1
            return {
                "in_term": True,
                "week": week,
                "term": term["name"],
                "semester_start": term["start"].isoformat(),
                "weeks": term["weeks"],
                "days_since_start": delta_days,
                "note": "具体以校历为准",
            }

        upcoming = next((term for term in terms if term["start"] > today), None)
        if upcoming is not None:
            return {
                "in_term": False,
                "week": None,
                "term": upcoming["name"],
                "semester_start": upcoming["start"].isoformat(),
                "days_until_start": (upcoming["start"] - today).days,
                "note": f"还没开学，距离 {upcoming['name']} 开学还有 {(upcoming['start'] - today).days} 天",
            }
        return {
            "in_term": False,
            "week": None,
            "term": None,
            "note": "当前不在任何已配置学期的教学周内（假期或校历未更新）",
        }

    # ---- 兼容：没有校历文件时按单一起点计算 ----
    start = _parse_date(settings.semester_start)
    if start is None:
        return {
            "in_term": False,
            "week": None,
            "semester_start": None,
            "note": "未配置校历，无法计算教学周",
        }

    delta_days = (today - start).days
    if delta_days < 0:
        return {
            "in_term": False,
            "week": None,
            "semester_start": start.isoformat(),
            "days_until_start": -delta_days,
            "note": f"距离开学还有 {-delta_days} 天",
        }

    week = delta_days // 7 + 1
    return {
        "in_term": week <= 20,
        "week": week,
        "semester_start": start.isoformat(),
        "days_since_start": delta_days,
        "note": "按 20 个教学周估算，具体以校历为准" if week > 20 else "",
    }


def calendar_info() -> ToolResult:
    """校历周次。"""

    payload = _calendar_payload()
    if payload.get("week"):
        term = f"{payload.get('term')}，" if payload.get("term") else ""
        answer = f"按{term}开学 {payload['semester_start']} 计算，现在是第 {payload['week']} 周。"
    elif payload.get("days_until_start"):
        answer = payload["note"]
    else:
        answer = payload.get("note") or "暂时无法计算教学周。"
    return ToolResult(name="campus_calendar", ok=True, data={**payload, "answer": answer})


def phones(keyword: Optional[str] = None) -> ToolResult:
    """常用电话；可按部门名过滤。"""

    items = _PUBLIC_PHONES
    if keyword:
        items = [
            item
            for item in _PUBLIC_PHONES
            if keyword in item["name"] or keyword in item["scene"]
        ] or _PUBLIC_PHONES

    lines = [f"{item['name']}：{item['phone']}（{item['scene']}）" for item in items]
    return ToolResult(
        name="campus_phones",
        ok=True,
        data={
            "items": items,
            "note": "以上为学校公开号码；个人手机号不在本系统收录范围",
            "answer": "常用联系方式：\n" + "\n".join(f"- {line}" for line in lines),
        },
    )
