"""
轨迹效率评测：Agent 执行步数、token 消耗、工具调用效率。
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def evaluate_trajectory_efficiency(results: List[Dict]) -> Dict:
    """
    评测 Agent 轨迹效率。

    Args:
        results: Agent 运行结果列表，每项含 tool_calls, loop_count, token_usage 等

    Returns:
        效率指标
    """
    total = len(results)
    if total == 0:
        return {"total": 0}

    loop_counts = [r.get("loop_count", 0) for r in results]
    token_usages = [r.get("token_usage", 0) for r in results]
    tool_counts = [len(r.get("tool_calls", [])) for r in results]
    tool_success = [
        sum(1 for tc in r.get("tool_calls", []) if tc.get("success", False))
        for r in results
    ]

    # 按是否升级分组
    escalated = [r for r in results if r.get("should_escalate")]
    normal = [r for r in results if not r.get("should_escalate")]

    return {
        "total_cases": total,
        "avg_loop_count": _mean(loop_counts),
        "max_loop_count": max(loop_counts) if loop_counts else 0,
        "avg_token_usage": _mean(token_usages),
        "total_token_usage": sum(token_usages),
        "avg_tool_calls": _mean(tool_counts),
        "tool_success_rate": (
            sum(tool_success) / max(sum(tool_counts), 1)
        ),
        "escalation_rate": len(escalated) / total,
        "normal_cases": len(normal),
        "escalated_cases": len(escalated),
        "efficiency_score": _compute_efficiency_score(
            loop_counts, tool_counts, token_usages
        ),
    }


def _mean(values: list) -> float:
    return sum(values) / max(len(values), 1)


def _compute_efficiency_score(loops, tools, tokens) -> float:
    """
    综合效率分数（0-1）。
    越少的步数和 token 完成任务 → 分数越高。
    """
    if not loops:
        return 0.0

    # 归一化：假设最优 1 轮、1-2 次工具、500 token
    avg_loop = _mean(loops)
    avg_tool = _mean(tools)
    avg_token = _mean(tokens)

    loop_score = max(1.0 - (avg_loop - 1) * 0.3, 0.0)
    tool_score = max(1.0 - abs(avg_tool - 1.5) * 0.2, 0.0)  # 1-2 次工具最优

    if avg_token > 0:
        # token 有追踪时，三项加权
        token_score = max(1.0 - avg_token / 8000, 0.0)
        return (loop_score * 0.3 + tool_score * 0.3 + token_score * 0.4)
    else:
        # token 未追踪时，只用 loop 和 tool 两项（避免虚高）
        return (loop_score * 0.5 + tool_score * 0.5)
