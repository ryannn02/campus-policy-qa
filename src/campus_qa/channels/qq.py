"""QQ / OneBot 渠道适配（校园墙）。

边界（重要）：

- **只处理「问题」与「回答」**：从消息里抽出文本 → 调统一问答 → 回一段文本；
- **不采集个人信息**：不落库 QQ 号、群号、昵称、头像，日志里也不打 `user_id`；
  只把 `group_id` / `message_type` 作为「往哪回」的路由信息用完即弃；
- 命中「查别人信息」的请求由 orchestrator 统一拒绝，本层不再重复判断；
- 同一个 `message_id` 做短时去重，避免 OneBot 重推导致重复回答。

OneBot v11 事件形如：

```
{"post_type": "message", "message_type": "group", "message_id": 123,
 "group_id": 456, "user_id": 789, "raw_message": "食堂几点开门",
 "message": [{"type": "text", "data": {"text": "食堂几点开门"}}]}
```

本模块只依赖标准库，方便以后单独部署成一个小进程。
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import settings
from ..orchestrator import answer


_MAX_DEDUPE = 500
_MAX_REPLY_CHARS = 1200

_seen_lock = threading.Lock()
_seen_ids: "OrderedDict[str, None]" = OrderedDict()


@dataclass(frozen=True)
class IncomingMessage:
    """渠道侧的入站消息：只保留回答所需的最小字段。"""

    text: str
    message_type: str
    group_id: Optional[str]
    message_id: str


def _remember(message_id: str) -> bool:
    """返回 True 表示这条消息第一次见到；False 表示重复推送。"""

    if not message_id:
        return True
    with _seen_lock:
        if message_id in _seen_ids:
            return False
        _seen_ids[message_id] = None
        while len(_seen_ids) > _MAX_DEDUPE:
            _seen_ids.popitem(last=False)
    return True


def _extract_text(event: Dict[str, Any]) -> str:
    """从 OneBot 消息段里抽纯文本，忽略图片、表情、@ 等。"""

    segments = event.get("message")
    if isinstance(segments, list):
        parts: List[str] = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            if segment.get("type") != "text":
                continue
            data = segment.get("data") or {}
            text = data.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        if parts:
            return " ".join(parts).strip()

    raw = event.get("raw_message")
    if isinstance(raw, str):
        # 去掉 CQ 码（[CQ:face,id=1] 之类），只留人话
        cleaned = []
        depth = 0
        for char in raw:
            if char == "[":
                depth += 1
                continue
            if char == "]" and depth:
                depth -= 1
                continue
            if depth == 0:
                cleaned.append(char)
        return "".join(cleaned).strip()
    return ""


def parse_event(event: Dict[str, Any]) -> Optional[IncomingMessage]:
    """把 OneBot 事件转成 `IncomingMessage`；非消息事件返回 None。"""

    if not isinstance(event, dict):
        return None
    if event.get("post_type") not in {None, "message", "message_sent"}:
        return None

    text = _extract_text(event)
    if not text:
        return None

    message_type = str(event.get("message_type") or "private")
    group_id = event.get("group_id")
    message_id = str(event.get("message_id") or "")

    return IncomingMessage(
        text=text,
        message_type=message_type,
        group_id=str(group_id) if group_id is not None else None,
        message_id=message_id,
    )


def build_reply(incoming: IncomingMessage, session_id: Optional[str] = None) -> Dict[str, Any]:
    """生成 OneBot 发送动作（`send_msg`），不包含任何用户身份字段。"""

    result = answer(incoming.text, session_id=session_id)
    text = result.answer or result.suggested_contact or "暂时没有查到相关信息。"
    if len(text) > _MAX_REPLY_CHARS:
        text = text[:_MAX_REPLY_CHARS] + "…（内容较长，建议在网页端查看完整引用）"

    params: Dict[str, Any] = {
        "message_type": incoming.message_type,
        "message": [{"type": "text", "data": {"text": text}}],
    }
    if incoming.message_type == "group" and incoming.group_id:
        params["group_id"] = incoming.group_id

    return {"action": "send_msg", "params": params}


def handle_event(
    event: Dict[str, Any],
    session_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """渠道入口：事件 → 回复动作。

    - 非消息 / 无文本 / 重复推送 → None（不回复）；
    - 群聊按 `group_id` 复用同一个匿名会话（只用于多轮上下文，不存成员信息）；
    - 私聊默认无状态，避免把单人对话沉淀成画像。
    """

    incoming = parse_event(event)
    if incoming is None:
        return None
    if not _remember(incoming.message_id):
        return None

    if session_id is None and incoming.message_type == "group" and incoming.group_id:
        session_id = f"qq_group_{incoming.group_id}"

    return build_reply(incoming, session_id=session_id)


def run_forever(webhook_url: str) -> None:  # pragma: no cover - 需要真实 OneBot 环境
    """接入占位：等校园墙的 OneBot 实现确定后再补。

    真实部署一般由 OneBot 实现（NapCat / LLOneBot 等）把事件 POST 到本服务，
    或由本服务反向 WS 连接。仓库里不预设某一种机器人框架，避免写死第三方。
    """

    raise NotImplementedError(
        "QQ 渠道需要先确定 OneBot 实现与接入方式（反向 WS 或 HTTP POST）。"
        f"当前配置：QQ_CHANNEL_ENABLED={settings.qq_channel_enabled}，webhook={webhook_url}"
    )
