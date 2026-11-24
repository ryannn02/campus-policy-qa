"""会话存储测试（临时 SQLite，不联网）。"""

from __future__ import annotations

from pathlib import Path

from campus_qa.conversation import ConversationStore


def _store(tmp_path: Path) -> ConversationStore:
    return ConversationStore(db_path=tmp_path / "sessions.sqlite3")


def test_create_and_list_sessions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    sid = store.create_session("第一条会话")

    sessions = store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["id"] == sid
    assert sessions[0]["title"] == "第一条会话"
    assert sessions[0]["message_count"] == 0


def test_session_ids_are_unique(tmp_path: Path) -> None:
    store = _store(tmp_path)
    ids = {store.create_session(f"会话{i}") for i in range(5)}
    assert len(ids) == 5


def test_rename_session(tmp_path: Path) -> None:
    store = _store(tmp_path)
    sid = store.create_session("旧名字")

    assert store.rename_session(sid, "新名字") is True
    assert store.get_session(sid)["title"] == "新名字"
    assert store.rename_session(sid, "   ") is False
    assert store.rename_session("not-exist", "x") is False


def test_messages_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    sid = store.create_session("问答")
    store.add_message(sid, "user", "食堂几点开门？", intent="life")
    store.add_message(
        sid,
        "assistant",
        "早餐 6:30 开始",
        intent="life",
        sources=[{"kind": "life", "title": "食堂营业时间"}],
    )

    messages = store.get_messages(sid)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[1].sources[0]["title"] == "食堂营业时间"
    assert messages[0].as_context() == {"role": "user", "content": "食堂几点开门？"}


def test_context_keeps_last_n_turns(tmp_path: Path) -> None:
    store = _store(tmp_path)
    sid = store.create_session("多轮")
    for index in range(3):
        store.add_message(sid, "user", f"问题{index}", intent="life")
        store.add_message(sid, "assistant", f"回答{index}", intent="life")

    context = store.get_context(sid, turns=1)
    assert len(context) == 2
    assert context[0]["content"] == "问题2"


def test_last_intent_and_question(tmp_path: Path) -> None:
    store = _store(tmp_path)
    sid = store.create_session("意图")
    store.add_message(sid, "user", "快递在哪", intent="life")
    store.add_message(sid, "assistant", "生活区", intent="life")
    store.add_message(sid, "user", "缓考怎么申请", intent="policy")

    assert store.last_intent(sid) == "policy"
    assert store.last_user_question(sid) == "缓考怎么申请"


def test_delete_session_removes_messages(tmp_path: Path) -> None:
    store = _store(tmp_path)
    sid = store.create_session("待删除")
    store.add_message(sid, "user", "你好")

    assert store.delete_session(sid) is True
    assert store.get_session(sid) is None
    assert store.get_messages(sid) == []
    assert store.delete_session(sid) is False


def test_stats_counts(tmp_path: Path) -> None:
    store = _store(tmp_path)
    sid = store.create_session("统计")
    store.add_message(sid, "user", "你好")

    stats = store.stats()
    assert stats["sessions"] == 1
    assert stats["messages"] == 1
    assert Path(stats["db_path"]).name == "sessions.sqlite3"
