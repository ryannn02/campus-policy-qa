"""FastAPI 应用测试：契约兼容、会话 CRUD、流式、静态页。

全部依赖注入替身：不联网、不调 LLM、不加载向量模型。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from campus_qa import orchestrator
from campus_qa.app import main as app_main
from campus_qa.app.main import app
from campus_qa.contracts import AnswerResult, Intent, Source
from campus_qa.conversation import ConversationStore
from campus_qa.metrics import LatencyTracker


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    store = ConversationStore(db_path=tmp_path / "test.sqlite3")
    monkeypatch.setattr(app_main, "conversation_store", store)
    # 时延计数器是进程级单例，测试里换成新实例，避免跨用例污染
    monkeypatch.setattr(app_main, "tracker", LatencyTracker())

    def fake_answer(question: str, session_id: str | None = None) -> AnswerResult:
        return AnswerResult(
            question=question,
            answer="本科生最多可同时借阅 10 本图书。",
            confidence=0.92,
            top_retrieval_score=0.92,
            intent=Intent.POLICY,
            sources=[
                Source(
                    kind="policy",
                    title="library_policy",
                    snippet="本科生借阅数量上限为 10 本。",
                    score=0.92,
                    document_id="library_policy",
                )
            ],
        )

    monkeypatch.setattr(orchestrator, "answer", fake_answer)
    return TestClient(app)


def test_answer_contract_unchanged(client: TestClient) -> None:
    response = client.post("/api/answer", json={"question": "图书馆能借几本书？"})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "图书馆能借几本书？"
    assert body["refused"] is False
    assert body["answer"] == "本科生最多可同时借阅 10 本图书。"
    assert body["citations"] == []
    # 新增字段：有默认值，不破坏旧客户端
    assert body["intent"] == "policy"
    assert body["sources"][0]["document_id"] == "library_policy"
    assert body["session_id"] is None


def test_invalid_json_returns_400_not_422(client: TestClient) -> None:
    response = client.post(
        "/api/answer",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_json"


def test_non_object_json_returns_400(client: TestClient) -> None:
    response = client.post("/api/answer", json=[1, 2, 3])
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_json"


def test_non_string_question_returns_400(client: TestClient) -> None:
    response = client.post("/api/answer", json={"question": {"x": 1}})
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_question"


def test_invalid_session_id_returns_400(client: TestClient) -> None:
    response = client.post("/api/answer", json={"question": "你好", "session_id": 123})
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_session_id"


def test_session_crud(client: TestClient) -> None:
    created = client.post("/api/sessions", json={"title": "测试会话"}).json()
    session_id = created["session_id"]
    assert created["title"] == "测试会话"

    listed = client.get("/api/sessions").json()["sessions"]
    assert any(item["id"] == session_id for item in listed)

    renamed = client.patch(f"/api/sessions/{session_id}", json={"title": "改名后"})
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "改名后"

    messages = client.get(f"/api/sessions/{session_id}/messages")
    assert messages.status_code == 200
    assert messages.json()["messages"] == []

    deleted = client.delete(f"/api/sessions/{session_id}")
    assert deleted.status_code == 200
    assert client.get(f"/api/sessions/{session_id}/messages").status_code == 404


def test_rename_missing_session_returns_404(client: TestClient) -> None:
    assert client.patch("/api/sessions/nope", json={"title": "x"}).status_code == 404
    assert client.delete("/api/sessions/nope").status_code == 404


def test_stream_emits_meta_delta_done(client: TestClient) -> None:
    response = client.post("/api/answer/stream", json={"question": "图书馆能借几本书？"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    body = response.text
    assert "event: meta" in body
    assert "event: delta" in body
    assert "event: done" in body
    done_line = [line for line in body.splitlines() if line.startswith("data:")][-1]
    payload = json.loads(done_line[len("data:") :].strip())
    assert payload["intent"] == "policy"


def test_route_endpoint(client: TestClient) -> None:
    response = client.post("/api/route", json={"question": "现在几点？"})
    assert response.status_code == 200
    assert response.json()["intent"] == "tool"
    assert response.json()["tool"] == "campus_time"


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "campus-policy-qa"
    assert "life" in body


def test_stats_includes_life_and_conversation(client: TestClient) -> None:
    response = client.get("/api/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 0
    assert "life" in body
    assert "conversation" in body


def test_capabilities(client: TestClient) -> None:
    body = client.get("/api/capabilities").json()
    assert set(body["intents"]) == {"policy", "life", "tool", "chitchat"}
    assert any(tool["name"] == "weather" for tool in body["tools"])


def test_index_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "校园问答助手" in response.text
