"""最小 HTTP 服务与极简 Web 页面。

实现目标（成员 D：前端与接口负责人）：
- 提供 `POST /api/answer` JSON API，按 `docs/web_api_contract.md` 契约返回
  `AnswerResult` 的 JSON 表示；
- 提供一个简单的单页 Web 界面，围绕 `AnswerResult` 展示答案、引用与拒答提示；
- 监控 P95 响应时延：`GET /api/stats` 返回进程内累计的时延统计，
  对应 `docs/requirements.md` 的「基础档 ≤8s / 进阶档 ≤5s」目标；
- 提供 `GET /health` 存活探针，供进程守护与健康检查使用。

依赖约束：
- 只使用 Python 标准库和现有包（`campus_qa.service` 等），不额外引入 Web 框架，
  以便在当前依赖下快速跑通（遵循 development_requirements 的「不随意增加新依赖」）。
"""

from __future__ import annotations

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from campus_qa.config import settings
from campus_qa.service import answer_question


# 响应时延目标（秒），对应 docs/requirements.md 非功能需求。
LATENCY_TARGET_BASE_SECONDS = 8.0
LATENCY_TARGET_ADVANCED_SECONDS = 5.0

# 时延统计保留的最大样本数，超出后丢弃最旧样本。
_MAX_LATENCY_SAMPLES = 1000


PAGE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>校园制度问答助手</title>
    <style>
      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem 1rem;
        background: #f5f7fa;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;
        color: #1f2329;
      }
      .container { width: 100%; max-width: 680px; text-align: center; }
      h1 { font-size: 1.7rem; margin: 0 0 0.5rem; }
      .subtitle { color: #6b7280; font-size: 0.95rem; margin-bottom: 1.5rem; }
      .search-box { display: flex; gap: 0.5rem; justify-content: center; margin-bottom: 1rem; }
      input[type=text] {
        flex: 1;
        max-width: 460px;
        padding: 0.75rem 1rem;
        border: 1px solid #d1d5db;
        border-radius: 12px;
        font-size: 1rem;
      }
      button#ask {
        padding: 0.75rem 1.4rem;
        border: none;
        border-radius: 12px;
        background: #2563eb;
        color: #fff;
        font-size: 1rem;
        cursor: pointer;
      }
      button#ask:disabled { background: #9ca3af; cursor: not-allowed; }
      .section-label { color: #6b7280; font-size: 0.85rem; margin: 0.75rem 0 0.5rem; }
      .chips { display: flex; flex-wrap: wrap; gap: 0.5rem; justify-content: center; }
      .chip {
        padding: 0.4rem 0.9rem;
        border: 1px solid #d1d5db;
        border-radius: 999px;
        background: #fff;
        color: #374151;
        font-size: 0.85rem;
        cursor: pointer;
      }
      .chip:hover { border-color: #2563eb; color: #2563eb; }
      .status { color: #9ca3af; min-height: 1.2rem; margin: 0.5rem 0; }
      .answer {
        text-align: left;
        background: #fff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        line-height: 1.7;
        white-space: pre-wrap;
      }
      .citations { text-align: left; margin: 0.75rem 0; }
      .citations strong { display: block; margin-bottom: 0.4rem; }
      .citations ul { padding-left: 1.2rem; margin: 0.25rem 0 0; }
      .citations li { margin-bottom: 0.4rem; font-size: 0.9rem; color: #374151; }
      .citations a { color: #2563eb; text-decoration: none; margin-left: 0.3rem; }
      .meta { font-size: 0.85rem; color: #6b7280; margin: 0.5rem 0; text-align: left; }
      .followup { margin-top: 1.25rem; text-align: left; }
      .footer { margin-top: 1.5rem; font-size: 0.8rem; color: #9ca3af; }
      .answer:empty, .citations:empty, .followup:empty, .meta:empty, .status:empty { display: none; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>校园制度问答助手</h1>
      <div class="subtitle">缓考、休学、选课、奖助学金、借阅……校内制度问题都可以问我</div>

      <div class="search-box">
        <input id="question" type="text" placeholder="比如：缓考申请最晚什么时候交？" autocomplete="off" />
        <button id="ask">提问</button>
      </div>

      <div class="section-label">可以试试这样问</div>
      <div class="chips" id="suggestions"></div>

      <div id="status" class="status"></div>
      <div id="answer" class="answer"></div>
      <div id="citations" class="citations"></div>
      <div id="followup" class="followup"></div>
      <div id="meta" class="meta"></div>

      <div class="footer">本系统回答仅供参考，具体以学校正式制度文件为准。</div>
    </div>

    <script>
      const questionInput = document.getElementById('question');
      const askButton = document.getElementById('ask');
      const statusEl = document.getElementById('status');
      const answerEl = document.getElementById('answer');
      const citationsEl = document.getElementById('citations');
      const followupEl = document.getElementById('followup');
      const metaEl = document.getElementById('meta');

      const SUGGESTIONS = [
        '缓考申请最晚什么时候交？',
        '申请休学要满足哪些条件？',
        '图书馆能借几本书？',
        '宿舍可以用大功率电器吗？',
        '申请国家助学金需要哪些条件？',
      ];

      const FOLLOWUPS = [
        { keys: ['缓考', '补考', '考试', '考核'], qs: ['补考课程冲突怎么处理？', '缓考申请需要提交哪些材料？', '考试作弊会受到什么处分？'] },
        { keys: ['休学', '退学', '复学', '学籍'], qs: ['休学后宿舍和学费怎么处理？', '申请休学要走哪些审批流程？', '学生可以申请提前毕业吗？'] },
        { keys: ['转专业'], qs: ['转专业的申请时间和基本条件是什么？', '转专业后已修学分怎么认定？'] },
        { keys: ['选课', '学分', '退选', '退课'], qs: ['本科生每学期最多能修多少学分？', '退选课程需要满足什么条件？'] },
        { keys: ['奖学金', '助学金', '资助', '勤工助学'], qs: ['申请国家助学金需要哪些条件？', '奖学金怎么评选？'] },
        { keys: ['宿舍', '住宿', '宠物', '电器'], qs: ['宿舍可以使用大功率电器吗？', '学校允许在宿舍养宠物吗？'] },
        { keys: ['图书馆', '借书', '借阅', '图书'], qs: ['图书馆能借几本书？', '图书逾期未还会怎么处理？'] },
        { keys: ['军训'], qs: ['新生军训缺席的处理办法是什么？', '军训可以申请免训吗？'] },
        { keys: ['处分', '违纪', '作弊'], qs: ['学校对考试作弊的处分有哪些等级？', '学生违纪处分有哪些种类？'] },
        { keys: ['兼职', '打工'], qs: ['学校对学生校外兼职有什么规定？'] },
        { keys: ['毕业'], qs: ['学生可以申请提前毕业吗？', '补办学历学位证明需要什么流程？'] },
      ];

      const DEFAULT_FOLLOWUPS = [
        '缓考申请最晚什么时候交？',
        '本科生每学期最多能修多少学分？',
        '图书馆能借几本书？',
        '申请国家助学金需要哪些条件？',
      ];

      function refuseMessage(code) {
        switch (code) {
          case 'empty_question':
            return '请输入具体的校园制度问题。';
          case 'unsupported_language':
            return '目前仅支持中文提问，请改用中文。';
          case 'retrieval_empty':
            return '暂未找到相关制度条款，请确认问题表述。';
          case 'retrieval_low_score':
            return '检索匹配度较低，暂无法给出可靠回答。';
          case 'invalid_model_output':
            return '内部处理出现问题，请稍后重试。';
          case 'no_valid_citation':
            return '当前回答缺少有效引用，已拒绝作答。';
          case 'no_evidence_in_corpus':
            return '现有制度文件无法支撑该问题，请咨询辅导员或相关部门。';
          default:
            return '当前问题暂无法回答，请咨询相关老师。';
        }
      }

      function followUpSuggestions(q) {
        for (let i = 0; i < FOLLOWUPS.length; i++) {
          const group = FOLLOWUPS[i];
          const hit = group.keys.some(function (k) { return q.indexOf(k) !== -1; });
          if (hit) {
            return group.qs;
          }
        }
        return DEFAULT_FOLLOWUPS;
      }

      function renderChips(container, questions) {
        container.innerHTML = '';
        questions.forEach(function (q) {
          const chip = document.createElement('button');
          chip.type = 'button';
          chip.className = 'chip';
          chip.textContent = q;
          chip.addEventListener('click', function () { askQuestion(q); });
          container.appendChild(chip);
        });
      }

      function askQuestion(q) {
        questionInput.value = q;
        ask();
      }

      function renderFollowups(q) {
        followupEl.innerHTML = '';
        const label = document.createElement('div');
        label.className = 'section-label';
        label.textContent = '你可能还想问';
        followupEl.appendChild(label);
        const chips = document.createElement('div');
        chips.className = 'chips';
        followupEl.appendChild(chips);
        renderChips(chips, followUpSuggestions(q));
      }

      async function ask() {
        const q = (questionInput.value || '').trim();
        if (!q) {
          statusEl.textContent = '请先输入一个问题。';
          return;
        }

        askButton.disabled = true;
        statusEl.textContent = '正在查询制度条款，请稍候……';
        answerEl.textContent = '';
        citationsEl.innerHTML = '';
        followupEl.innerHTML = '';
        metaEl.textContent = '';

        try {
          const resp = await fetch('/api/answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: q })
          });

          if (!resp.ok) {
            let msg = 'HTTP ' + resp.status;
            try {
              const errData = await resp.json();
              if (errData && errData.message) msg = errData.message;
            } catch (e) { /* 忽略错误体解析失败 */ }
            throw new Error(msg);
          }

          const data = await resp.json();
          statusEl.textContent = '';

          if (data.refused) {
            const msg = refuseMessage(data.refuse_reason);
            answerEl.textContent = msg || (data.answer || '当前问题暂无法回答。');
          } else {
            answerEl.textContent = data.answer || '系统未返回答案。';
          }

          if (Array.isArray(data.citations) && data.citations.length > 0) {
            const list = document.createElement('ul');
            data.citations.forEach(function (c) {
              const li = document.createElement('li');
              const snippet = c.snippet || '';
              const docId = c.document_id || '[未命名文档]';
              const metaParts = [];
              if (typeof c.score === 'number') metaParts.push('score: ' + c.score.toFixed(2));
              if (typeof c.page === 'number') metaParts.push('page: ' + c.page);
              if (c.section) metaParts.push('section: ' + c.section);

              let text = '[' + docId + '] ' + snippet;
              if (metaParts.length) {
                text += ' (' + metaParts.join(', ') + ')';
              }

              li.textContent = text;

              if (c.url) {
                const link = document.createElement('a');
                link.href = c.url;
                link.target = '_blank';
                link.rel = 'noopener noreferrer';
                link.textContent = '查看原文';
                li.appendChild(link);
              }

              list.appendChild(li);
            });
            const heading = document.createElement('strong');
            heading.textContent = '引用条款';
            citationsEl.innerHTML = '';
            citationsEl.appendChild(heading);
            citationsEl.appendChild(list);
          } else {
            citationsEl.textContent = '';
          }

          const metaParts = [];
          if (typeof data.confidence === 'number') {
            metaParts.push('系统置信度：' + data.confidence.toFixed(2));
          }
          if (typeof data.top_retrieval_score === 'number') {
            metaParts.push('检索匹配度：' + data.top_retrieval_score.toFixed(2));
          }
          if (data.suggested_contact) {
            metaParts.push('建议：' + data.suggested_contact);
          }
          metaEl.textContent = metaParts.join('  |  ');

          renderFollowups(q);
        } catch (err) {
          console.error(err);
          statusEl.textContent = (err && err.message) || '服务暂时不可用，请稍后重试。';
        } finally {
          askButton.disabled = false;
        }
      }

      askButton.addEventListener('click', ask);
      questionInput.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter') {
          ask();
        }
      });

      renderChips(document.getElementById('suggestions'), SUGGESTIONS);
    </script>
  </body>
</html>
"""


def _percentile(sorted_values: List[float], p: float) -> float:
    """最近秩（nearest-rank）百分位数。

    输入要求已升序排列；空列表返回 0.0。
    """

    if not sorted_values:
        return 0.0
    rank = int(math.ceil(p * len(sorted_values)))
    index = max(0, min(len(sorted_values) - 1, rank - 1))
    return sorted_values[index]


class LatencyTracker:
    """线程安全的问答时延记录器，用于 P95 监控。

    - `record` 记录一次问答耗时（秒）；
    - `summary` 返回累计样本的统计：count、avg、p50/p95/p99、max，
      以及是否落在基础档 / 进阶档时延目标内。
    """

    def __init__(self, max_samples: int = _MAX_LATENCY_SAMPLES) -> None:
        self._lock = threading.Lock()
        self._latencies: List[float] = []
        self._max_samples = max_samples
        self._started_at = time.monotonic()

    def record(self, seconds: float) -> None:
        with self._lock:
            self._latencies.append(seconds)
            if len(self._latencies) > self._max_samples:
                self._latencies = self._latencies[-self._max_samples:]

    def uptime_seconds(self) -> float:
        return time.monotonic() - self._started_at

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            samples = sorted(self._latencies)
            count = len(samples)
            uptime = time.monotonic() - self._started_at

        if count == 0:
            return {
                "count": 0,
                "uptime_seconds": round(uptime, 3),
                "avg_seconds": None,
                "p50_seconds": None,
                "p95_seconds": None,
                "p99_seconds": None,
                "max_seconds": None,
                "p95_within_base_target": None,
                "p95_within_advanced_target": None,
            }

        avg = sum(samples) / count
        p95 = _percentile(samples, 0.95)
        return {
            "count": count,
            "uptime_seconds": round(uptime, 3),
            "avg_seconds": round(avg, 3),
            "p50_seconds": round(_percentile(samples, 0.50), 3),
            "p95_seconds": round(p95, 3),
            "p99_seconds": round(_percentile(samples, 0.99), 3),
            "max_seconds": round(samples[-1], 3),
            "p95_within_base_target": p95 <= LATENCY_TARGET_BASE_SECONDS,
            "p95_within_advanced_target": p95 <= LATENCY_TARGET_ADVANCED_SECONDS,
        }


# 进程级单例：所有请求共享同一个时延统计。
LATENCY_TRACKER = LatencyTracker()


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
        # 契约约定 question 必为字符串；其他类型直接 400，避免 str() 转出噪音。
        return None, {"error": "invalid_question", "message": "question 字段必须是字符串"}

    return question, None


def build_answer_payload(question: str) -> Dict[str, Any]:
    """调用问答服务，返回 ``AnswerResult`` 的 JSON 字典。

    抽成独立函数便于单元测试契约序列化，`do_POST` 与测试复用同一路径。
    """

    result = answer_question(question)
    # mode="json" 确保 RefuseReason 枚举等字段按字符串值序列化。
    return result.model_dump(mode="json")


def handle_answer_request(question: str) -> Tuple[int, Dict[str, Any]]:
    """执行问答并返回 ``(status, body)``。

    - 正常返回 ``(200, AnswerResult JSON)``；
    - 服务异常（如检索 / 生成链路意外抛出）返回 ``(500, error JSON)``，
      符合 `docs/web_api_contract.md` 的「服务异常返回 5xx 与错误信息」。
    """

    try:
        return 200, build_answer_payload(question)
    except Exception as exc:  # noqa: BLE001 - 边界兜底，不向客户端泄漏堆栈
        return 500, {"error": "internal_error", "message": "服务暂时不可用，请稍后重试。"}


class QAHandler(BaseHTTPRequestHandler):
    """简单的 HTTP 处理器。

    - GET `/` / `/index.html`：返回单页 HTML；
    - GET `/health`：存活探针；
    - GET `/api/stats`：问答时延统计（P50/P95/P99）；
    - POST `/api/answer`：接收 JSON 请求并返回 `AnswerResult` 的 JSON。
    """

    def do_GET(self) -> None:  # pragma: no cover - 手工测试为主
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            body = PAGE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": "campus-policy-qa",
                    "uptime_seconds": round(LATENCY_TRACKER.uptime_seconds(), 3),
                    "index_files_present": (
                        (settings.index_dir / "embeddings.npy").exists()
                        and (settings.index_dir / "manifest.json").exists()
                    ),
                },
            )
            return

        if parsed.path == "/api/stats":
            self._send_json(200, LATENCY_TRACKER.summary())
            return

        self.send_error(404, "Not Found")

    def do_POST(self) -> None:  # pragma: no cover - 手工测试为主
        parsed = urlparse(self.path)
        if parsed.path != "/api/answer":
            self.send_error(404, "Not Found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0

        raw_body = self.rfile.read(length) if length > 0 else b""

        question, error = _parse_question_payload(raw_body)
        if error is not None:
            self._send_json(400, error)
            return

        started = time.monotonic()
        status, body_dict = handle_answer_request(question)
        elapsed = time.monotonic() - started
        LATENCY_TRACKER.record(elapsed)

        self._send_json(status, body_dict)
        self.log_message(
            "answer status=%d elapsed=%.3fs question=%r",
            status,
            elapsed,
            question[:40],
        )

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        response = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover
        # 保留简单的控制台日志，便于本地调试（含时延与问题摘要）。
        super().log_message(format, *args)


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    """启动最小 HTTP 服务。"""

    server = ThreadingHTTPServer((host, port), QAHandler)
    print(f"Serving campus QA on http://{host}:{port}")
    print("  GET /health      存活探针")
    print("  GET /api/stats   问答时延统计（P50/P95/P99）")
    print("  POST /api/answer 问答接口（AnswerResult JSON）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        print("\nServer stopped by user")


if __name__ == "__main__":  # pragma: no cover
    main()
