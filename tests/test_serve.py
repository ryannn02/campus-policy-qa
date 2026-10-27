"""Web/API 层（成员 D）测试：时延统计、请求解析与契约序列化。

不启动真实服务、不联网、不加载向量模型，只验证 `scripts.serve` 中
可独立测试的纯逻辑与契约序列化路径。
"""

from __future__ import annotations

from scripts import serve
from campus_qa.contracts import AnswerResult, Citation, RefuseReason


# --- 百分位数 / 时延统计 ---


def test_percentile_nearest_rank() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert serve._percentile(values, 0.50) == 2.0
    assert serve._percentile(values, 0.95) == 4.0
    assert serve._percentile([], 0.95) == 0.0


def test_latency_tracker_empty_summary() -> None:
    tracker = serve.LatencyTracker()
    summary = tracker.summary()

    assert summary["count"] == 0
    assert summary["p95_seconds"] is None
    assert summary["p95_within_base_target"] is None
    assert summary["p95_within_advanced_target"] is None


def test_latency_tracker_computes_percentiles() -> None:
    tracker = serve.LatencyTracker()
    for value in [1.0, 2.0, 3.0, 4.0, 5.0]:
        tracker.record(value)

    summary = tracker.summary()

    assert summary["count"] == 5
    assert summary["p50_seconds"] == 3.0
    assert summary["p95_seconds"] == 5.0
    assert summary["max_seconds"] == 5.0
    # 5s 落在基础档（≤8s）与进阶档（≤5s）目标内。
    assert summary["p95_within_base_target"] is True
    assert summary["p95_within_advanced_target"] is True


def test_latency_tracker_marks_p95_above_advanced_target() -> None:
    tracker = serve.LatencyTracker()
    for value in [1.0, 1.0, 1.0, 1.0, 9.0]:
        tracker.record(value)

    summary = tracker.summary()

    assert summary["p95_seconds"] == 9.0
    assert summary["p95_within_base_target"] is False
    assert summary["p95_within_advanced_target"] is False


# --- 请求体解析 ---


def test_parse_question_payload_valid() -> None:
    question, error = serve._parse_question_payload(
        '{"question": "图书馆能借几本书？"}'.encode("utf-8")
    )
    assert error is None
    assert question == "图书馆能借几本书？"


def test_parse_question_payload_missing_question() -> None:
    question, error = serve._parse_question_payload(b"{}")
    assert error is None
    assert question == ""


def test_parse_question_payload_null_question() -> None:
    question, error = serve._parse_question_payload(b'{"question": null}')
    assert error is None
    assert question == ""


def test_parse_question_payload_invalid_json() -> None:
    question, error = serve._parse_question_payload(b"not-json")
    assert question is None
    assert error is not None
    assert error["error"] == "invalid_json"


def test_parse_question_payload_non_object() -> None:
    question, error = serve._parse_question_payload(b"[1,2,3]")
    assert question is None
    assert error is not None
    assert error["error"] == "invalid_json"


def test_parse_question_payload_non_string_question() -> None:
    question, error = serve._parse_question_payload(b'{"question": {"x": 1}}')
    assert question is None
    assert error is not None
    assert error["error"] == "invalid_question"


# --- AnswerResult 契约序列化 ---


def test_build_answer_payload_matches_contract(monkeypatch) -> None:
    fake_result = AnswerResult(
        question="图书馆能借几本书？",
        answer="本科生最多可同时借阅 10 本图书。",
        citations=[
            Citation(
                document_id="library_policy",
                snippet="本科生借阅数量上限为 10 本。",
                score=0.92,
                page=3,
                section="借阅数量",
            )
        ],
        confidence=0.92,
        top_retrieval_score=0.92,
        refused=False,
    )
    monkeypatch.setattr(serve, "answer_question", lambda _q: fake_result)

    payload = serve.build_answer_payload("图书馆能借几本书？")

    assert payload["question"] == "图书馆能借几本书？"
    assert payload["refused"] is False
    assert payload["refuse_reason"] is None
    assert payload["answer"] == "本科生最多可同时借阅 10 本图书。"
    assert payload["citations"][0]["document_id"] == "library_policy"
    assert payload["citations"][0]["score"] == 0.92


def test_build_answer_payload_refused(monkeypatch) -> None:
    fake_result = AnswerResult(
        question="hello",
        refused=True,
        refuse_reason=RefuseReason.UNSUPPORTED_LANGUAGE,
        suggested_contact="请改用中文提问，或咨询辅导员。",
    )
    monkeypatch.setattr(serve, "answer_question", lambda _q: fake_result)

    payload = serve.build_answer_payload("hello")

    assert payload["refused"] is True
    assert payload["refuse_reason"] == "unsupported_language"
    assert payload["answer"] is None
    assert payload["suggested_contact"] == "请改用中文提问，或咨询辅导员。"


# --- 服务异常 5xx 兜底 ---


def test_handle_answer_request_ok(monkeypatch) -> None:
    fake_result = AnswerResult(
        question="图书馆能借几本书？",
        answer="本科生最多可同时借阅 10 本图书。",
        citations=[
            Citation(document_id="library_policy", snippet="第十条", score=0.9)
        ],
        confidence=0.9,
        top_retrieval_score=0.9,
        refused=False,
    )
    monkeypatch.setattr(serve, "answer_question", lambda _q: fake_result)

    status, body = serve.handle_answer_request("图书馆能借几本书？")

    assert status == 200
    assert body["refused"] is False
    assert body["answer"] == "本科生最多可同时借阅 10 本图书。"


def test_handle_answer_request_internal_error(monkeypatch) -> None:
    def _boom(_question: str) -> AnswerResult:
        raise RuntimeError("boom")

    monkeypatch.setattr(serve, "answer_question", _boom)

    status, body = serve.handle_answer_request("图书馆能借几本书？")

    assert status == 500
    assert body["error"] == "internal_error"
    assert body["message"]
