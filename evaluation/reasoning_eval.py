"""
推理链评测模块：评估 Agent 诊疗推理过程的质量。

四个维度：
  1. 推理完整性（Reasoning Completeness）：推理链是否覆盖从症状到结论的完整路径
  2. 证据锚定率（Evidence Grounding Rate）：结论中有多少比例有工具/知识库证据支撑
  3. 推理自洽性（Reasoning Consistency）：诊断方向是否与科室、症状、工具结果一致
  4. 工具归因分数（Tool Attribution Score）：工具调用对最终结论的贡献度
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def evaluate_reasoning(results: List[Dict], references: List[Dict] = None) -> Dict:
    """
    批量评测推理链质量。

    Args:
        results: Agent 输出列表，每个 dict 应包含：
            - reasoning_chain, differential_hypotheses, specialist_analysis,
            - tool_calls, final_response, current_department 等
        references: 可选的标准答案列表

    Returns:
        推理链评测汇总指标
    """
    if not results:
        return {"total": 0}

    total = len(results)
    completeness_scores = []
    grounding_scores = []
    consistency_scores = []
    attribution_scores = []
    details = []

    for i, result in enumerate(results):
        ref = references[i] if references and i < len(references) else None

        comp = _eval_completeness(result, ref)
        ground = _eval_evidence_grounding(result)
        consist = _eval_consistency(result)
        attrib = _eval_tool_attribution(result)

        completeness_scores.append(comp["score"])
        grounding_scores.append(ground["score"])
        consistency_scores.append(consist["score"])
        attribution_scores.append(attrib["score"])

        details.append({
            "index": i,
            "completeness": comp,
            "evidence_grounding": ground,
            "consistency": consist,
            "tool_attribution": attrib,
            "overall": _weighted_average([
                (comp["score"], 0.3),
                (ground["score"], 0.3),
                (consist["score"], 0.25),
                (attrib["score"], 0.15),
            ]),
        })

    return {
        "total": total,
        "avg_completeness": _mean(completeness_scores),
        "avg_evidence_grounding": _mean(grounding_scores),
        "avg_consistency": _mean(consistency_scores),
        "avg_tool_attribution": _mean(attribution_scores),
        "overall_reasoning_score": _mean([d["overall"] for d in details]),
        "details": details,
    }


# ─────────────────────────────────────────────
# 维度 1：推理完整性
# ─────────────────────────────────────────────

def _eval_completeness(result: Dict, ref: Dict = None) -> Dict:
    """
    评估推理链完整性：是否包含关键推理环节。

    检查项：
    - 有推理链文本（reasoning_chain）
    - 有鉴别诊断假设（differential_hypotheses）
    - Specialist 分析覆盖了假设
    - 最终回复中有结论
    """
    score = 0.0
    checks = {}
    max_points = 4.0

    # 检查 1：推理链是否存在
    reasoning_chain = result.get("reasoning_chain", "")
    has_chain = bool(reasoning_chain and len(reasoning_chain.strip()) > 10)
    checks["has_reasoning_chain"] = has_chain
    if has_chain:
        score += 1.0

    # 检查 2：鉴别诊断假设是否存在
    hypotheses = result.get("differential_hypotheses", [])
    has_hypotheses = bool(hypotheses and len(hypotheses) >= 1
                         and hypotheses[0] != "待专科分析后确定")
    checks["has_hypotheses"] = has_hypotheses
    checks["hypothesis_count"] = len(hypotheses)
    if has_hypotheses:
        score += 1.0

    # 检查 3：Specialist 分析是否覆盖假设
    analysis = result.get("specialist_analysis", "")
    if has_hypotheses and analysis:
        covered = sum(1 for h in hypotheses if h in analysis)
        coverage_rate = covered / max(len(hypotheses), 1)
        checks["hypothesis_coverage_rate"] = coverage_rate
        score += coverage_rate
    else:
        checks["hypothesis_coverage_rate"] = 0.0

    # 检查 4：最终回复是否有明确结论
    final = result.get("final_response", "")
    conclusion_markers = ["建议", "可能", "考虑", "排查", "就医", "诊断方向"]
    has_conclusion = any(m in final for m in conclusion_markers) if final else False
    checks["has_conclusion"] = has_conclusion
    if has_conclusion:
        score += 1.0

    return {
        "score": score / max_points,
        "checks": checks,
    }


# ─────────────────────────────────────────────
# 维度 2：证据锚定率
# ─────────────────────────────────────────────

def _eval_evidence_grounding(result: Dict) -> Dict:
    """
    评估证据锚定率：最终结论有多少被工具/知识证据支撑。

    策略：
    - 统计成功的工具调用数
    - 检查工具结果关键词是否出现在最终分析中
    - 检查 retrieved_knowledge 是否被引用
    """
    tool_calls = result.get("tool_calls", [])
    successful_tools = [tc for tc in tool_calls if tc.get("success") and tc.get("output")]
    analysis = result.get("specialist_analysis", "") + " " + result.get("final_response", "")

    if not successful_tools:
        # 没有工具调用，检查是否有 retrieved_knowledge
        knowledge = result.get("retrieved_knowledge", [])
        if knowledge:
            return {"score": 0.5, "detail": "有知识库检索但无工具调用", "grounded_tools": 0, "total_tools": 0}
        return {"score": 0.0, "detail": "无工具调用和知识检索", "grounded_tools": 0, "total_tools": 0}

    # 检查每个工具的输出是否在分析中被引用（粗略匹配）
    grounded = 0
    for tc in successful_tools:
        output_str = str(tc.get("output", ""))
        # 提取输出中的关键片段（取前100字的几个关键词）
        output_keywords = _extract_keywords_from_output(output_str)
        if output_keywords:
            matched = sum(1 for kw in output_keywords if kw in analysis)
            if matched >= max(1, len(output_keywords) // 3):
                grounded += 1

    total = len(successful_tools)
    rate = grounded / total if total > 0 else 0.0

    return {
        "score": rate,
        "detail": f"{grounded}/{total} 工具结果被引用",
        "grounded_tools": grounded,
        "total_tools": total,
    }


def _extract_keywords_from_output(output: str, max_keywords: int = 5) -> List[str]:
    """从工具输出中提取关键词片段"""
    if not output or len(output) < 5:
        return []
    # 简单策略：取长度 >= 2 的中文片段
    import re
    chunks = re.findall(r'[\u4e00-\u9fff]{2,8}', output[:200])
    # 过滤常见无意义词
    stopwords = {"的", "了", "是", "在", "有", "和", "与", "或", "及", "等", "为", "被", "将"}
    filtered = [c for c in chunks if c not in stopwords and len(c) >= 2]
    return filtered[:max_keywords]


# ─────────────────────────────────────────────
# 维度 3：推理自洽性
# ─────────────────────────────────────────────

def _eval_consistency(result: Dict) -> Dict:
    """
    评估推理自洽性：各环节之间是否逻辑一致。

    检查项：
    - 科室 vs 诊断方向
    - 推理链 vs 最终结论
    - 工具计划 vs 实际工具调用
    """
    score = 0.0
    checks = {}
    max_points = 3.0

    department = result.get("current_department", "")
    analysis = result.get("specialist_analysis", "")
    final_response = result.get("final_response", "")

    # 检查 1：科室是否在分析中被提及或匹配
    if department and analysis:
        dept_in_analysis = department in analysis or department in final_response
        checks["dept_consistent"] = dept_in_analysis
        if dept_in_analysis:
            score += 1.0
    else:
        checks["dept_consistent"] = None

    # 检查 2：计划工具 vs 实际调用
    planned_tools = set(result.get("tool_plan", []))
    actual_tools = {tc.get("tool_name", "") for tc in result.get("tool_calls", []) if tc.get("tool_name")}
    if planned_tools:
        overlap = len(planned_tools & actual_tools)
        plan_adherence = overlap / max(len(planned_tools), 1)
        checks["plan_adherence"] = plan_adherence
        score += plan_adherence
    else:
        checks["plan_adherence"] = None
        score += 0.5  # 没有计划不扣分

    # 检查 3：最终回复不为空且非兜底
    fallback_markers = ["暂时无法", "系统错误", "请稍后重试"]
    is_fallback = any(m in (final_response or "") for m in fallback_markers)
    checks["has_substantive_response"] = bool(final_response) and not is_fallback
    if checks["has_substantive_response"]:
        score += 1.0

    return {
        "score": score / max_points,
        "checks": checks,
    }


# ─────────────────────────────────────────────
# 维度 4：工具归因分数
# ─────────────────────────────────────────────

def _eval_tool_attribution(result: Dict) -> Dict:
    """
    评估工具对最终结论的贡献度。

    逻辑：
    - 有工具调用且成功 → 检查结论质量是否优于无工具情况
    - 工具结果被引用 → 工具贡献度高
    """
    tool_calls = result.get("tool_calls", [])
    successful = [tc for tc in tool_calls if tc.get("success")]
    total_calls = len(tool_calls)
    analysis = result.get("specialist_analysis", "")

    if total_calls == 0:
        return {
            "score": 0.5,  # 中性：未使用工具不惩罚
            "detail": "未使用工具",
            "successful_calls": 0,
            "total_calls": 0,
        }

    success_rate = len(successful) / total_calls

    # 检查是否有冗余调用（同一工具重复调用相同参数）
    call_signatures = [tc.get("call_signature", "") for tc in tool_calls]
    unique_sigs = set(call_signatures)
    redundancy = 1.0 - (len(unique_sigs) / max(len(call_signatures), 1))

    # 工具结果引用率
    grounding = _eval_evidence_grounding(result)
    grounding_rate = grounding["score"]

    # 综合分数：成功率 × 0.4 + 引用率 × 0.4 + (1 - 冗余率) × 0.2
    score = success_rate * 0.4 + grounding_rate * 0.4 + (1.0 - redundancy) * 0.2

    return {
        "score": score,
        "detail": f"成功率={success_rate:.2f}, 引用率={grounding_rate:.2f}, 冗余率={redundancy:.2f}",
        "successful_calls": len(successful),
        "total_calls": total_calls,
        "redundancy_rate": redundancy,
    }


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _mean(values: list) -> float:
    return sum(values) / max(len(values), 1)


def _weighted_average(pairs: List[tuple]) -> float:
    """加权平均：pairs = [(score, weight), ...]"""
    total_weight = sum(w for _, w in pairs)
    if total_weight == 0:
        return 0.0
    return sum(s * w for s, w in pairs) / total_weight
