"""
Agent 调用链 Tracing：记录每次问诊的完整执行轨迹。

功能：
  1. 记录每个 Agent 节点的输入/输出/延迟/token
  2. 记录工具调用链（含参数和结果摘要）
  3. 生成可视化友好的 trace 报告
  4. 支持导出为 JSON / 控制台格式

用法:
  from monitoring.tracing import Tracer, global_tracer
  tracer = global_tracer

  tracer.start_trace("req_001")
  with tracer.span("router") as span:
      span.set_input(state)
      result = route_patient(state)
      span.set_output(result)
  trace = tracer.end_trace()
"""

import time
import json
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)

TRACE_DIR = Path(__file__).parent.parent / "data" / "traces"


@dataclass
class Span:
    """单个执行步骤"""
    span_id: str
    name: str                          # 如 "router", "specialist", "tool:search_drug"
    span_type: str = "agent"           # agent / tool / llm / retrieval
    start_time: float = 0.0
    end_time: float = 0.0
    latency_ms: float = 0.0
    input_summary: str = ""
    output_summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["Span"] = field(default_factory=list)
    error: Optional[str] = None

    def set_input(self, data: Any, max_len: int = 200):
        """记录输入摘要"""
        if isinstance(data, dict):
            # 只记录关键字段
            keys_to_keep = ["messages", "current_department", "patient_info", "confidence"]
            summary_parts = []
            for k in keys_to_keep:
                if k in data:
                    v = data[k]
                    if isinstance(v, list):
                        summary_parts.append(f"{k}: [{len(v)} items]")
                    else:
                        summary_parts.append(f"{k}: {str(v)[:60]}")
            self.input_summary = ", ".join(summary_parts)[:max_len]
        else:
            self.input_summary = str(data)[:max_len]

    def set_output(self, data: Any, max_len: int = 200):
        """记录输出摘要"""
        if isinstance(data, dict):
            keys = list(data.keys())
            summary_parts = []
            for k in keys[:5]:
                v = data[k]
                if isinstance(v, list):
                    summary_parts.append(f"{k}: [{len(v)} items]")
                elif isinstance(v, str) and len(v) > 60:
                    summary_parts.append(f"{k}: {v[:60]}...")
                else:
                    summary_parts.append(f"{k}: {v}")
            self.output_summary = ", ".join(summary_parts)[:max_len]
        else:
            self.output_summary = str(data)[:max_len]

    def finish(self):
        self.end_time = time.time()
        self.latency_ms = (self.end_time - self.start_time) * 1000


@dataclass
class Trace:
    """一次完整问诊的执行轨迹"""
    trace_id: str
    request_id: str
    start_time: float = 0.0
    end_time: float = 0.0
    total_latency_ms: float = 0.0
    spans: List[Span] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finish(self):
        self.end_time = time.time()
        self.total_latency_ms = (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict:
        """转为可序列化字典"""
        def span_to_dict(span: Span) -> Dict:
            d = {
                "span_id": span.span_id,
                "name": span.name,
                "type": span.span_type,
                "latency_ms": round(span.latency_ms, 1),
                "input": span.input_summary,
                "output": span.output_summary,
            }
            if span.error:
                d["error"] = span.error
            if span.metadata:
                d["metadata"] = span.metadata
            if span.children:
                d["children"] = [span_to_dict(c) for c in span.children]
            return d

        return {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "span_count": len(self.spans),
            "spans": [span_to_dict(s) for s in self.spans],
            "metadata": self.metadata,
        }

    def to_console(self) -> str:
        """生成控制台友好的 trace 文本"""
        lines = [
            f"{'='*60}",
            f"Trace: {self.trace_id}  ({self.total_latency_ms:.0f}ms)",
            f"{'='*60}",
        ]
        for i, span in enumerate(self.spans):
            prefix = "├─" if i < len(self.spans) - 1 else "└─"
            status = "✓" if not span.error else "✗"
            lines.append(
                f"  {prefix} [{status}] {span.name} ({span.span_type}) "
                f"- {span.latency_ms:.0f}ms"
            )
            if span.input_summary:
                lines.append(f"  │    IN:  {span.input_summary[:80]}")
            if span.output_summary:
                lines.append(f"  │    OUT: {span.output_summary[:80]}")
            if span.error:
                lines.append(f"  │    ERR: {span.error}")
            for j, child in enumerate(span.children):
                child_prefix = "│  ├─" if j < len(span.children) - 1 else "│  └─"
                lines.append(
                    f"  {child_prefix} [{child.name}] {child.latency_ms:.0f}ms"
                )
        lines.append(f"{'='*60}")
        return "\n".join(lines)


class Tracer:
    """
    全局 Tracer，管理 trace 生命周期。
    """

    def __init__(self, trace_dir: Path = TRACE_DIR, enabled: bool = True):
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self._current_trace: Optional[Trace] = None
        self._current_span: Optional[Span] = None
        self._history: List[Trace] = []

    def start_trace(self, request_id: str) -> Optional[Trace]:
        """开始一次 trace"""
        if not self.enabled:
            return None
        trace = Trace(
            trace_id=str(uuid.uuid4())[:8],
            request_id=request_id,
            start_time=time.time(),
        )
        self._current_trace = trace
        return trace

    def end_trace(self) -> Optional[Trace]:
        """结束当前 trace"""
        if self._current_trace is None:
            return None
        self._current_trace.finish()
        self._history.append(self._current_trace)
        trace = self._current_trace
        self._current_trace = None
        self._current_span = None
        logger.debug(f"Trace 完成: {trace.trace_id} ({trace.total_latency_ms:.0f}ms)")
        return trace

    @contextmanager
    def span(self, name: str, span_type: str = "agent"):
        """
        上下文管理器：自动记录 span 的开始和结束。

        用法:
            with tracer.span("specialist", "agent") as s:
                s.set_input(state)
                result = specialist_analyze(state)
                s.set_output(result)
        """
        if not self.enabled or self._current_trace is None:
            yield _NoOpSpan()
            return

        s = Span(
            span_id=str(uuid.uuid4())[:8],
            name=name,
            span_type=span_type,
            start_time=time.time(),
        )

        parent_span = self._current_span
        self._current_span = s

        try:
            yield s
        except Exception as e:
            s.error = str(e)
            raise
        finally:
            s.finish()
            if parent_span is not None:
                parent_span.children.append(s)
            else:
                self._current_trace.spans.append(s)
            self._current_span = parent_span

    def get_current_trace(self) -> Optional[Trace]:
        return self._current_trace

    def get_history(self, last_n: int = 0) -> List[Trace]:
        if last_n > 0:
            return self._history[-last_n:]
        return self._history

    def save_trace(self, trace: Trace, filename: str = None):
        """保存单个 trace 到文件"""
        if filename is None:
            filename = f"trace_{trace.request_id}_{trace.trace_id}.json"
        path = self.trace_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trace.to_dict(), f, ensure_ascii=False, indent=2)

    def save_all(self, filename: str = "all_traces.json"):
        """保存所有 trace 历史"""
        path = self.trace_dir / filename
        data = [t.to_dict() for t in self._history]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Trace 历史保存: {path} ({len(data)} 条)")


class _NoOpSpan:
    """tracing 关闭时的空操作 span"""
    def set_input(self, *args, **kwargs): pass
    def set_output(self, *args, **kwargs): pass


# 全局实例
global_tracer = Tracer()
