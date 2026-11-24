"""多轮会话存储（SQLite）。

只存「问题 + 回答 + 来源」，不存任何个人信息：

- 不记录 IP、设备、QQ 号、学号；
- 渠道侧（QQ）只把 `session_id` 作为匿名会话键，不做用户画像；
- 会话可由用户随时重命名 / 删除，删除即物理删除消息行。

表结构刻意保持扁平（sessions / messages），因为这里要的是多轮上下文，
不是业务数据库；需要审计时直接读 SQLite 即可。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .config import settings


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    intent      TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    intent      TEXT,
    sources     TEXT,
    created_at  TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    intent: Optional[str] = None
    sources: List[Dict[str, object]] = field(default_factory=list)
    created_at: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "role": self.role,
            "content": self.content,
            "intent": self.intent,
            "sources": self.sources or [],
            "created_at": self.created_at,
        }

    def as_context(self) -> Dict[str, str]:
        """拼上下文时只要角色与正文，来源不进提示词。"""

        return {"role": self.role, "content": self.content}


class ConversationStore:
    """线程安全的 SQLite 会话存储，每次操作独立连接。"""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = Path(db_path or settings.session_db_path)
        self._lock = threading.Lock()
        self._initialized = False

    # ---- 连接与建表 ----

    @property
    def db_path(self) -> Path:
        return self._db_path

    def init_db(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
            self._initialized = True

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- 会话 ----

    def create_session(self, title: str = "新会话", intent: Optional[str] = None) -> str:
        self.init_db()
        session_id = new_session_id()
        timestamp = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, intent, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, title.strip() or "新会话", intent, timestamp, timestamp),
            )
        return session_id

    def list_sessions(self, limit: int = 100) -> List[Dict[str, object]]:
        self.init_db()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT s.id, s.title, s.intent, s.created_at, s.updated_at, "
                "       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count "
                "FROM sessions s ORDER BY s.updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_session(self, session_id: str) -> Optional[Dict[str, object]]:
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, title, intent, created_at, updated_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def rename_session(self, session_id: str, title: str) -> bool:
        self.init_db()
        clean = (title or "").strip()
        if not clean:
            return False
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (clean[:60], _now(), session_id),
            )
        return cursor.rowcount > 0

    def touch_session(self, session_id: str, intent: Optional[str] = None) -> None:
        self.init_db()
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET updated_at = ?, intent = COALESCE(?, intent) WHERE id = ?",
                (_now(), intent, session_id),
            )

    def delete_session(self, session_id: str) -> bool:
        self.init_db()
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0

    # ---- 消息 ----

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: Optional[str] = None,
        sources: Optional[List[Dict[str, object]]] = None,
    ) -> None:
        self.init_db()
        payload = json.dumps(sources or [], ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, intent, sources, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, role, content, intent, payload, _now()),
            )

    def get_messages(self, session_id: str, limit: int = 200) -> List[Message]:
        self.init_db()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, intent, sources, created_at FROM messages "
                "WHERE session_id = ? ORDER BY id ASC LIMIT ?",
                (session_id, limit),
            ).fetchall()

        messages: List[Message] = []
        for row in rows:
            try:
                sources = json.loads(row["sources"] or "[]")
            except json.JSONDecodeError:
                sources = []
            messages.append(
                Message(
                    role=row["role"],
                    content=row["content"],
                    intent=row["intent"],
                    sources=sources,
                    created_at=row["created_at"],
                )
            )
        return messages

    def get_context(self, session_id: str, turns: Optional[int] = None) -> List[Dict[str, str]]:
        """取最近 N 轮（1 轮 = 1 问 + 1 答）的上下文，供追问使用。"""

        limit = turns if turns is not None else settings.max_context_turns
        messages = self.get_messages(session_id)
        if not messages:
            return []
        keep = max(1, limit) * 2
        return [message.as_context() for message in messages[-keep:]]

    def last_intent(self, session_id: str) -> Optional[str]:
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT intent FROM messages WHERE session_id = ? AND role = 'user' "
                "ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return row["intent"] if row and row["intent"] else None

    def last_user_question(self, session_id: str) -> Optional[str]:
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content FROM messages WHERE session_id = ? AND role = 'user' "
                "ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return row["content"] if row else None

    def stats(self) -> Dict[str, object]:
        self.init_db()
        with self._connect() as conn:
            sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return {
            "sessions": sessions,
            "messages": messages,
            "db_path": str(self._db_path),
        }


# 进程级单例
store = ConversationStore()
