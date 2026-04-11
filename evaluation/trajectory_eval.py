"""
轨迹效率评测：Agent 执行步数、token 消耗、工具调用效率。
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def evaluate_trajectory_efficiency(results: List[Dict], references: Optional[List[Dict]] = None) -> Dict:
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
    tool_name_sequences = [[tc.get("tool_name", "") for tc in r.get("tool_calls", []) if tc.get("tool_name")] for r in results]
    tool_success = [
        sum(1 for tc in r.get("tool_calls", []) if tc.get("success", False))
        for r in results
    ]
    duplicate_rates = [
        (len(seq) - len(set(seq))) / max(len(seq), 1)
        for seq in tool_name_sequences
    ]
    plan_adherence = []
    stop_reasons = {}

    for result, seq in zip(results, tool_name_sequences):
        plan = result.get("tool_plan", []) or result.get("structured_output", {}).get("tool_plan", []) or []
        if plan:
            plan_set = set(plan)
            plan_adherence.append(sum(1 for name in seq if name in plan_set) / max(len(seq), 1))
        reason = result.get("stop_reason", "")
        if reason:
            stop_reasons[reason] = stop_reasons.get(reason, 0) + 1

    # Reflection 相关统计
    reflection_counts = [r.get("reflection_count", 0) for r in results]
    reflection_triggered = [r for r in results if r.get("reflection_count", 0) > 0]
    reflection_retried = [r for r in results if r.get("reflection_count", 0) > 1]

    escalated = [r for r in results if r.get("should_escalate")]
    normal = [r for r in results if not r.get("should_escalate")]

    first_tool_accuracy = None
    strict_sequence_accuracy = None
    if references:
        first_hits = []
        strict_hits = []
        for seq, ref in zip(tool_name_sequences, references):
            ref_seq = ref.get("preferred_tool_sequence") or ref.get("expected_tools") or ref.get("tools_used") or []
            pred_first = seq[0] if seq else ""
            ref_first = ref.get("expected_first_tool") or (ref_seq[0] if ref_seq else "")
            if ref_first or pred_first:
                first_hits.append(float(pred_first == ref_first))
            if ref_seq or seq:
                strict_hits.append(float(seq == ref_seq))
        if first_hits:
            first_tool_accuracy = sum(first_hits) / len(first_hits)
        if strict_hits:
            strict_sequence_accuracy = sum(strict_hits) / len(strict_hits)

    result = {
        "total_cases": total,
        "avg_loop_count": _mean(loop_counts),
        "median_loop_count": _percentile(loop_counts, 50),
        "max_loop_count": max(loop_counts) if loop_counts else 0,
        "avg_token_usage": _mean(token_usages),
        "median_token_usage": _percentile(token_usages, 50),
        "p90_token_usage": _percentile(token_usages, 90),
        "total_token_usage": sum(token_usages),
        "avg_tool_calls": _mean(tool_counts),
        "median_tool_calls": _percentile(tool_counts, 50),
        "p90_tool_calls": _percentile(tool_counts, 90),
        "tool_success_rate": (
            sum(tool_success) / max(sum(tool_counts), 1)
        ),
        "duplicate_tool_rate": _mean(duplicate_rates),
        "escalation_rate": len(escalated) / total,
        "normal_cases": len(normal),
        "escalated_cases": len(escalated),
        "stop_reason_distribution": stop_reasons,
        "efficiency_score": _compute_efficiency_score(
            loop_counts, tool_counts, token_usages, duplicate_rates
        ),
    }
    if plan_adherence:
        result["avg_plan_adherence"] = _mean(plan_adherence)
    if first_tool_accuracy is not None:
        result["first_tool_accuracy"] = first_tool_accuracy
    if strict_sequence_accuracy is not None:
        result["strict_sequence_accuracy"] = strict_sequence_accuracy

    # Reflection 指标
    result["reflection_triggered_rate"] = len(reflection_triggered) / total
    result["reflection_retry_rate"] = len(reflection_retried) / max(len(reflection_triggered), 1)
    result["avg_reflection_count"] = _mean(reflection_counts)

    return result


def _mean(values: list) -> float:
    return sum(values) / max(len(values), 1)


def _percentile(values: list, p: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((p / 100) * (len(ordered) - 1)))
    return ordered[idx]


def _compute_efficiency_score(loops, tools, tokens, duplicate_rates) -> float:
    if not loops:
        return 0.0

    avg_loop = _mean(loops)
    avg_tool = _mean(tools)
    avg_token = _mean(tokens)
    duplicate_penalty = _mean(duplicate_rates)

    loop_score = max(1.0 - (avg_loop - 1) * 0.3, 0.0)
    tool_score = max(1.0 - abs(avg_tool - 1.5) * 0.2, 0.0)
    duplicate_score = max(1.0 - duplicate_penalty * 2.0, 0.0)

    if avg_token > 0:
        token_score = max(1.0 - avg_token / 8000, 0.0)
        return (loop_score * 0.25 + tool_score * 0.25 + token_score * 0.35 + duplicate_score * 0.15)
    return (loop_score * 0.4 + tool_score * 0.35 + duplicate_score * 0.25)
