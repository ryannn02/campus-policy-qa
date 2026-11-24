"""兼容入口：把旧的 `python -m scripts.serve` 转发到 FastAPI 应用。

历史背景：M1–M4 的服务层是标准库 `ThreadingHTTPServer` 手写的，Web 页面
也内嵌在本文件里。M5 换成 FastAPI 之后，本文件保留为一层很薄的壳：

- `LatencyTracker` / `_percentile` 从 `campus_qa.metrics` 重新导出，
  旧的监控测试与外部脚本无需改动；
- `_parse_question_payload` / `build_answer_payload` / `handle_answer_request`
  保留原有语义（非法 JSON → 400、异常 → 500），只是不再负责起服务；
- `main()` 改为启动 uvicorn，页面与接口都来自 `campus_qa.app.main:app`。

约定：新代码请直接 `uvicorn campus_qa.app.main:app`，这个文件只为兼容而存在。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
from urllib.parse import urlparse  # noqa: F401 - 保留旧引用，供外部脚本导入

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from campus_qa.metrics import (  # noqa: E402 - 需先补 sys.path
    LATENCY_TARGET_ADVANCED_SECONDS,
    LATENCY_TARGET_BASE_SECONDS,
    LatencyTracker,
    _percentile,
    tracker as LATENCY_TRACKER,
)
from campus_qa.service import answer_question  # noqa: E402


def _parse_question_payload(raw_body: bytes) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """解析请求体，返回 ``(question, error)``。

    - 合法时返回 ``(question, None)``；
    - 非法 JSON 或非对象时返回 ``(None, error_dict)``，由调用方回 400。
    """

    if raw_body:
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, {"error": "invalid_json", "message": "请求体不是合法 JSON"}
        if not isinstance(payload, dict):
            return None, {"error": "invalid_json", "message": "请求体必须是 JSON 对象"}
    else:
        payload = {}

    raw_question = payload.get("question", "")
    if raw_question is None:
        question = ""
    elif isinstance(raw_question, str):
        question = raw_question
    else:
        return None, {"error": "invalid_question", "message": "question 字段必须是字符串"}

    return question, None


def build_answer_payload(question: str) -> Dict[str, Any]:
    """调用问答服务，返回 ``AnswerResult`` 的 JSON 字典。

    仍走 `service.answer_question`（而不是 orchestrator），因为这是制度链路的
    兼容入口；新功能请用 `campus_qa.orchestrator.answer`。
    """

    result = answer_question(question)
    return result.model_dump(mode="json")


def handle_answer_request(question: str) -> Tuple[int, Dict[str, Any]]:
    """执行问答并返回 ``(status, body)``；异常统一 500。"""

    try:
        return 200, build_answer_payload(question)
    except Exception:  # noqa: BLE001 - 边界兜底，不向客户端泄漏堆栈
        return 500, {"error": "internal_error", "message": "服务暂时不可用，请稍后重试。"}


def _lan_addresses() -> list[str]:
    """返回本机局域网 IPv4（排除回环与链路本地地址），用于打印可分享的访问地址。"""

    import socket

    addresses: list[str] = []
    try:
        host_name = socket.gethostname()
        for info in socket.getaddrinfo(host_name, None, socket.AF_INET):
            ip = info[4][0]
            if ip.startswith("127.") or ip.startswith("169.254."):
                continue
            if ip not in addresses:
                addresses.append(ip)
    except OSError:
        pass
    if not addresses:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(("10.255.255.255", 1))  # 不真正发包，仅用于选路
                ip = probe.getsockname()[0]
            if not ip.startswith("127."):
                addresses.append(ip)
        except OSError:
            pass
    return addresses


def main(argv: Optional[Sequence[str]] = None) -> None:
    """启动 FastAPI 服务（uvicorn）。

    默认监听 ``0.0.0.0``，同一局域网内的其他电脑 / 手机可以直接用本机 IP 访问；
    只想本机使用时传 ``--host 127.0.0.1``。
    """

    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="启动校园问答 Web 服务")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="监听地址，默认 0.0.0.0（局域网可访问）；只本机用 127.0.0.1",
    )
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    args = parser.parse_args(argv)

    host, port = args.host, args.port
    print(f"Serving campus QA on http://{host}:{port}")
    if host in ("0.0.0.0", "::"):
        print("  本机访问：      http://127.0.0.1:%d" % port)
        for ip in _lan_addresses():
            print("  局域网访问：    http://%s:%d   （其他电脑 / 手机连同一 Wi-Fi 后打开）" % (ip, port))
        print("  若其他电脑打不开：确认在同一网段，并放行 Windows 防火墙该端口的入站连接")
    print("  GET  /               Web 页面")
    print("  GET  /health         存活探针")
    print("  GET  /api/stats      问答时延统计（P50/P95/P99）")
    print("  POST /api/answer     问答接口（AnswerResult JSON）")
    print("  POST /api/answer/stream  SSE 流式输出")
    uvicorn.run("campus_qa.app.main:app", host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()
