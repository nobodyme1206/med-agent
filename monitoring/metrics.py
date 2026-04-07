"""
运行时指标追踪：延迟、token消耗、工具调用次数、幻觉率等。
支持单次请求和聚合统计。
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

METRICS_DIR = Path(__file__).parent.parent / "data" / "metrics"


@dataclass
class RequestMetrics:
    """单次请求的指标"""
    request_id: str
    timestamp: float = field(default_factory=time.time)
    total_latency_ms: float = 0.0
    router_latency_ms: float = 0.0
    specialist_latency_ms: float = 0.0
    pharmacist_latency_ms: float = 0.0
    summary_latency_ms: float = 0.0
    total_tokens: int = 0
    tool_call_count: int = 0
    tool_success_count: int = 0
    department: str = ""
    escalated: bool = False
    loop_count: int = 0
    confidence: float = 0.0


class MetricsTracker:
    """
    指标追踪器。
    - 记录每次请求的详细指标
    - 聚合统计（P50/P99延迟、平均token、成功率等）
    - 持久化到 JSON 文件
    """

    def __init__(self, metrics_dir: Path = METRICS_DIR):
        self.metrics_dir = metrics_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self._history: List[RequestMetrics] = []
        self._current: Optional[RequestMetrics] = None

    def start_request(self, request_id: str):
        """开始追踪一次请求"""
        self._current = RequestMetrics(request_id=request_id)
        self._start_time = time.time()

    def record_stage_latency(self, stage: str, latency_ms: float):
        """记录某阶段的延迟"""
        if self._current is None:
            return
        attr = f"{stage}_latency_ms"
        if hasattr(self._current, attr):
            setattr(self._current, attr, latency_ms)

    def end_request(self, state: dict):
        """结束追踪，从最终 state 提取指标"""
        if self._current is None:
            return

        self._current.total_latency_ms = (time.time() - self._start_time) * 1000
        self._current.tool_call_count = len(state.get("tool_calls", []))
        self._current.tool_success_count = sum(
            1 for tc in state.get("tool_calls", []) if tc.get("success")
        )
        self._current.department = state.get("current_department", "")
        self._current.escalated = state.get("should_escalate", False)
        self._current.loop_count = state.get("loop_count", 0)
        self._current.confidence = state.get("confidence", 0.0)
        self._current.total_tokens = state.get("token_usage", 0)

        self._history.append(self._current)
        self._current = None

    def get_aggregate_stats(self, last_n: int = 0) -> Dict:
        """
        聚合统计。

        Args:
            last_n: 只统计最近N条，0=全部
        """
        records = self._history[-last_n:] if last_n > 0 else self._history
        if not records:
            return {"total_requests": 0}

        latencies = [r.total_latency_ms for r in records]
        latencies.sort()
        n = len(latencies)

        return {
            "total_requests": n,
            "avg_latency_ms": sum(latencies) / n,
            "p50_latency_ms": latencies[n // 2],
            "p99_latency_ms": latencies[int(n * 0.99)],
            "avg_tool_calls": sum(r.tool_call_count for r in records) / n,
            "tool_success_rate": (
                sum(r.tool_success_count for r in records) /
                max(sum(r.tool_call_count for r in records), 1)
            ),
            "escalation_rate": sum(1 for r in records if r.escalated) / n,
            "avg_confidence": sum(r.confidence for r in records) / n,
            "avg_loops": sum(r.loop_count for r in records) / n,
            "department_distribution": self._dept_dist(records),
        }

    def _dept_dist(self, records: List[RequestMetrics]) -> Dict[str, int]:
        dist = {}
        for r in records:
            dist[r.department] = dist.get(r.department, 0) + 1
        return dist

    def save(self, filename: str = "metrics_history.json"):
        """持久化指标历史"""
        path = self.metrics_dir / filename
        data = [asdict(r) for r in self._history]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"指标保存: {path} ({len(data)} 条)")

    def load(self, filename: str = "metrics_history.json"):
        """加载指标历史"""
        path = self.metrics_dir / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._history = [RequestMetrics(**d) for d in data]
            logger.info(f"指标加载: {len(self._history)} 条")


# 全局实例
tracker = MetricsTracker()
