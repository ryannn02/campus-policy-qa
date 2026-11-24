"""工具注册表。

工具是「不需要检索、不需要模型」的确定性能力。每个工具都返回
`contracts.ToolResult`，结构与文案分离（data 给前端，answer 给正文），
这样 Web 和 QQ 渠道可以各自渲染，不用解析自然语言。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ..contracts import ToolResult
from . import campus, weather


ToolFunc = Callable[..., ToolResult]

_REGISTRY: Dict[str, Dict[str, object]] = {
    "weather": {
        "label": "天气",
        "description": "当前天气与未来几天预报（Open-Meteo）",
        "func": weather.query,
    },
    "campus_time": {
        "label": "当前时间",
        "description": "现在几点、星期几、本学期第几周",
        "func": campus.current_time,
    },
    "campus_calendar": {
        "label": "校历",
        "description": "教学周次与开学时间",
        "func": campus.calendar_info,
    },
    "campus_phones": {
        "label": "常用电话",
        "description": "学校公开的部门联系方式",
        "func": campus.phones,
    },
}


def names() -> List[str]:
    return list(_REGISTRY)


def exists(name: str) -> bool:
    return name in _REGISTRY


def describe() -> List[Dict[str, object]]:
    """给前端 / 文档用的工具清单（不含函数对象）。"""

    return [
        {"name": name, "label": spec["label"], "description": spec["description"]}
        for name, spec in _REGISTRY.items()
    ]


def run(name: str, **kwargs: object) -> ToolResult:
    """执行工具；未知工具返回 ok=False，不抛异常。"""

    spec: Optional[Dict[str, object]] = _REGISTRY.get(name)
    if spec is None:
        return ToolResult(name=name, ok=False, error=f"未知工具：{name}")

    func: ToolFunc = spec["func"]  # type: ignore[assignment]
    try:
        return func(**kwargs)
    except TypeError:
        # 参数不匹配时退化为无参调用，避免调用方多传字段直接失败
        return func()
    except Exception as exc:  # noqa: BLE001 - 工具失败要降级，不能中断问答
        return ToolResult(name=name, ok=False, error=f"{type(exc).__name__}: {exc}")
