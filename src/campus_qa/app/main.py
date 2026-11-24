"""FastAPI 应用：统一问答 + 多轮会话 + 流式输出。

契约兼容性：

- `POST /api/answer` 的请求体与响应仍是 `docs/web_api_contract.md` 的那一套，
  只是 `AnswerResult` 多了 `intent` / `sources` / `tool_result` 三个**可选**字段；
- 非法 JSON / question 非字符串仍然返回 400（不是 FastAPI 默认的 422），
  与旧 `scripts/serve.py` 的行为一致；
- `/health`、`/api/stats` 保留，字段只增不减。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .. import orchestrator
from ..config import settings
from ..contracts import AnswerResult
from ..conversation import store as conversation_store
from ..life import store as life_store
from ..metrics import tracker


_WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(
    title="校园问答助手",
    description="制度问答（RAG）+ 生活服务 + 实时工具 + 多轮会话",
    version="0.2.0",
)


app.mount("/assets", StaticFiles(directory=str(_WEB_DIR / "assets")), name="assets")


@app.on_event("startup")
async def warmup_models() -> None:
    """启动时预热检索索引与向量模型。

    向量模型是懒加载的，若不预热，服务启动后的第一个制度问题会卡住几十秒；
    预热把这部分开销挪到启动阶段，首个用户请求也能在秒级返回。
    """

    started = time.monotonic()

    def _warmup() -> None:
        from .. import retriever

        retriever._load_index()
        retriever._encode_question("校园问答预热")
        life_store.search("食堂")

    await run_in_threadpool(_warmup)
    print(f"检索索引与向量模型预热完成，耗时 {time.monotonic() - started:.1f}s")


# ---------------------------------------------------------------- 工具函数


async def _read_json(request: Request) -> tuple[Optional[Dict[str, Any]], Optional[JSONResponse]]:
    """按旧契约解析请求体：非法 JSON / 非对象返回 400。"""

    raw = await request.body()
    if not raw:
        return {}, None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, JSONResponse(
            status_code=400,
            content={"error": "invalid_json", "message": "请求体不是合法 JSON"},
        )
    if not isinstance(payload, dict):
        return None, JSONResponse(
            status_code=400,
            content={"error": "invalid_json", "message": "请求体必须是 JSON 对象"},
        )
    return payload, None


def _coerce_question(payload: Dict[str, Any]) -> tuple[Optional[str], Optional[JSONResponse]]:
    raw_question = payload.get("question", "")
    if raw_question is None:
        return "", None
    if not isinstance(raw_question, str):
        return None, JSONResponse(
            status_code=400,
            content={"error": "invalid_question", "message": "question 字段必须是字符串"},
        )
    return raw_question, None


# ---------------------------------------------------------------- 问答


@app.post("/api/answer")
async def api_answer(request: Request) -> Any:
    payload, error = await _read_json(request)
    if error is not None:
        return error

    question, error = _coerce_question(payload or {})
    if error is not None:
        return error

    session_id = (payload or {}).get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_session_id", "message": "session_id 必须是字符串"},
        )

    started = time.monotonic()
    try:
        result: AnswerResult = await run_in_threadpool(
            orchestrator.answer, question or "", session_id or None
        )
    except Exception:  # noqa: BLE001 - 边界兜底，不向客户端泄漏堆栈
        tracker.record(time.monotonic() - started)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": "服务暂时不可用，请稍后重试。"},
        )

    tracker.record(time.monotonic() - started)
    body = result.model_dump(mode="json")
    body["session_id"] = session_id
    body["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return JSONResponse(content=body)


@app.post("/api/answer/stream")
async def api_answer_stream(request: Request) -> Any:
    """SSE 流式输出。

    事件顺序：`meta`（意图/会话）→ 多个 `delta`（正文分段）→ `done`（完整结果）。
    """

    payload, error = await _read_json(request)
    if error is not None:
        return error

    question, error = _coerce_question(payload or {})
    if error is not None:
        return error

    session_id = (payload or {}).get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_session_id", "message": "session_id 必须是字符串"},
        )

    started = time.monotonic()
    result: AnswerResult = await run_in_threadpool(
        orchestrator.answer, question or "", session_id or None
    )
    tracker.record(time.monotonic() - started)

    async def event_stream():
        def _event(name: str, data: Dict[str, Any]) -> str:
            return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        yield _event(
            "meta",
            {
                "intent": result.intent.value if result.intent else None,
                "refused": result.refused,
                "session_id": session_id,
                "sources": [source.model_dump(mode="json") for source in result.sources],
            },
        )

        text = result.answer or result.suggested_contact or ""
        chunk_size = 18
        for index in range(0, len(text), chunk_size):
            yield _event("delta", {"text": text[index : index + chunk_size]})

        yield _event("done", result.model_dump(mode="json"))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/route")
async def api_route(request: Request) -> Any:
    """只做路由，不生成答案——调试与评测用。"""

    payload, error = await _read_json(request)
    if error is not None:
        return error
    question, error = _coerce_question(payload or {})
    if error is not None:
        return error

    session_id = (payload or {}).get("session_id")
    decision = await run_in_threadpool(
        orchestrator.route_question, question or "", session_id if isinstance(session_id, str) else None
    )
    return {
        "intent": decision.intent.value,
        "confidence": decision.confidence,
        "matched": decision.matched,
        "reason": decision.reason,
        "tool": decision.tool,
    }


# ---------------------------------------------------------------- 会话


@app.get("/api/sessions")
async def list_sessions() -> Dict[str, Any]:
    sessions = await run_in_threadpool(conversation_store.list_sessions)
    return {"sessions": sessions}


@app.post("/api/sessions")
async def create_session(request: Request) -> Any:
    payload, error = await _read_json(request)
    if error is not None:
        return error
    title = (payload or {}).get("title") or "新会话"
    if not isinstance(title, str):
        return JSONResponse(status_code=400, content={"error": "invalid_title"})
    session_id = await run_in_threadpool(conversation_store.create_session, title)
    return {"session_id": session_id, "title": title[:60]}


@app.patch("/api/sessions/{session_id}")
async def rename_session(session_id: str, request: Request) -> Any:
    payload, error = await _read_json(request)
    if error is not None:
        return error
    title = (payload or {}).get("title")
    if not isinstance(title, str) or not title.strip():
        return JSONResponse(status_code=400, content={"error": "invalid_title"})
    ok = await run_in_threadpool(conversation_store.rename_session, session_id, title)
    if not ok:
        return JSONResponse(status_code=404, content={"error": "session_not_found"})
    return {"ok": True, "session_id": session_id, "title": title[:60]}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> Any:
    ok = await run_in_threadpool(conversation_store.delete_session, session_id)
    if not ok:
        return JSONResponse(status_code=404, content={"error": "session_not_found"})
    return {"ok": True, "session_id": session_id}


@app.get("/api/sessions/{session_id}/messages")
async def session_messages(session_id: str) -> Any:
    session = await run_in_threadpool(conversation_store.get_session, session_id)
    if not session:
        return JSONResponse(status_code=404, content={"error": "session_not_found"})
    messages: List[Dict[str, Any]] = [
        message.as_dict()
        for message in await run_in_threadpool(conversation_store.get_messages, session_id)
    ]
    return {"session": session, "messages": messages}


# ---------------------------------------------------------------- 运维


@app.get("/health")
@app.get("/api/health")
async def health() -> Dict[str, Any]:
    ok, snapshot = orchestrator.health_snapshot()
    return {
        "status": "ok" if ok else "degraded",
        "service": "campus-policy-qa",
        "uptime_seconds": round(tracker.uptime_seconds(), 3),
        "index_files_present": (
            (settings.index_dir / "embeddings.npy").exists()
            and (settings.index_dir / "manifest.json").exists()
        ),
        "life_index_present": settings.life_index_path.exists(),
        **snapshot,
    }


@app.get("/api/stats")
async def stats() -> Dict[str, Any]:
    summary = tracker.summary()
    summary["life"] = life_store.stats()
    summary["conversation"] = conversation_store.stats()
    return summary


@app.get("/api/capabilities")
async def capabilities() -> Dict[str, Any]:
    return orchestrator.capabilities()


# ---------------------------------------------------------------- 静态页面


@app.get("/")
async def index() -> Any:
    target = _WEB_DIR / "index.html"
    if not target.exists():
        return JSONResponse(status_code=500, content={"error": "web_assets_missing"})
    return FileResponse(target)


@app.get("/favicon.ico")
async def favicon() -> Any:
    return JSONResponse(status_code=204, content=None)
