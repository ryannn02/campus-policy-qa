"""时延统计（P50/P95/P99）。

独立成模块，避免 `scripts/serve.py`（兼容旧入口）与 FastAPI 应用互相
import 形成环。旧入口继续暴露 `LatencyTracker` / `_percentile` 同名对象，
`tests/test_serve.py` 的断言因此保持不变。
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, List


# 响应时延目标（秒），对应 docs/requirements.md 非功能需求。
LATENCY_TARGET_BASE_SECONDS = 8.0
LATENCY_TARGET_ADVANCED_SECONDS = 5.0

# 时延统计保留的最大样本数，超出后丢弃最旧样本。
MAX_LATENCY_SAMPLES = 1000


def _percentile(sorted_values: List[float], p: float) -> float:
    """最近秩（nearest-rank）百分位数；输入需已升序，空列表返回 0.0。"""

    if not sorted_values:
        return 0.0
    rank = int(math.ceil(p * len(sorted_values)))
    index = max(0, min(len(sorted_values) - 1, rank - 1))
    return sorted_values[index]


class LatencyTracker:
    """线程安全的问答时延记录器。"""

    def __init__(self, max_samples: int = MAX_LATENCY_SAMPLES) -> None:
        self._lock = threading.Lock()
        self._latencies: List[float] = []
        self._max_samples = max_samples
        self._started_at = time.monotonic()

    def record(self, seconds: float) -> None:
        with self._lock:
            self._latencies.append(seconds)
            if len(self._latencies) > self._max_samples:
                self._latencies = self._latencies[-self._max_samples :]

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
tracker = LatencyTracker()
