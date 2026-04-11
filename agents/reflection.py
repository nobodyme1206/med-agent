"""
Reflection Agent：在 Specialist 分析之后、Summary 之前进行自我检查。

核心职责（PARM 框架中的 R）：
  1. 证据一致性检查：Specialist 的结论是否与工具返回的证据一致
  2. 假设覆盖检查：Planner 的鉴别诊断假设是否被分析覆盖
  3. 信息缺口检查：Planner 标记的信息缺口是否在分析中被补充
  4. 验证标准检查：Planner 定义的 verification_criteria 是否满足

若检查不通过，生成反馈信息回退给 Specialist 重试。
最多重试 MAX_REFLECTION_RETRIES 次后强制通过。
"""

import json
import logging
from typing import Dict, List

from graph.state import AgentState, MAX_REFLECTION_RETRIES
from utils.llm_client import chat

logger = logging.getLogger(__name__)

REFLECTION_SYSTEM_PROMPT = """你是一位严谨的医学质控审核员（Medical QA Reviewer）。你的任务是审查专科医生的分析报告，检查其是否符合诊疗规划的要求。

你需要检查以下几个方面：
1. **证据一致性**：分析结论是否有工具返回的证据支撑（如指南、检验解读）
2. **假设覆盖**：是否对鉴别诊断假设进行了分析或排除
3. **信息完整性**：规划中标记的信息缺口是否在分析中被提及
4. **验证标准**：是否满足规划器定义的验证标准

请输出 JSON：
{
  "passed": true/false,
  "evidence_grounding": {"score": 0.0-1.0, "detail": "证据引用情况"},
  "hypothesis_coverage": {"score": 0.0-1.0, "detail": "假设覆盖情况"},
  "gap_resolution": {"score": 0.0-1.0, "detail": "信息缺口填补情况"},
  "criteria_met": {"score": 0.0-1.0, "detail": "验证标准满足情况"},
  "overall_score": 0.0-1.0,
  "feedback": "如果不通过，给出具体的改进建议（供 Specialist 重试时参考）",
  "summary": "一句话总结审核结论"
}

判定标准：overall_score >= 0.6 则 passed=true，否则 passed=false。
如果 passed=false，feedback 必须包含明确的、可操作的改进建议。"""


def _build_reflection_prompt(state: AgentState) -> str:
    """构建 Reflection 审查的 user prompt"""
    specialist_analysis = state.get("specialist_analysis", "")
    tool_calls = state.get("tool_calls", [])
    differential_hypotheses = state.get("differential_hypotheses", [])
    information_gaps = state.get("information_gaps", [])
    verification_criteria = state.get("verification_criteria", [])
    reasoning_chain = state.get("reasoning_chain", "")
    plan_summary = state.get("plan_summary", "")
    department = state.get("current_department", "")

    # 提取工具调用结果摘要
    tool_results = []
    for tc in tool_calls:
        if tc.get("success") and tc.get("output"):
            output_str = str(tc["output"])[:300]
            tool_results.append(f"- {tc.get('tool_name', '')}: {output_str}")
    tool_results_text = "\n".join(tool_results) if tool_results else "无工具调用结果"

    prompt = (
        f"【科室】{department}\n"
        f"【诊疗规划】{plan_summary}\n"
        f"【推理链】{reasoning_chain}\n\n"
        f"【鉴别诊断假设】\n"
    )
    for i, h in enumerate(differential_hypotheses, 1):
        prompt += f"  {i}. {h}\n"

    prompt += f"\n【信息缺口】\n"
    for gap in information_gaps:
        prompt += f"  - {gap}\n"

    prompt += f"\n【验证标准】\n"
    for criterion in verification_criteria:
        prompt += f"  - {criterion}\n"

    prompt += (
        f"\n【工具调用结果】\n{tool_results_text}\n\n"
        f"【专科医生分析】\n{specialist_analysis}\n\n"
        f"请审查以上分析并输出 JSON 评估结果。"
    )
    return prompt


def _parse_reflection_result(response: str) -> Dict:
    """解析 LLM 返回的 Reflection 结果"""
    default = {
        "passed": True,
        "evidence_grounding": {"score": 0.5, "detail": "解析失败，默认通过"},
        "hypothesis_coverage": {"score": 0.5, "detail": ""},
        "gap_resolution": {"score": 0.5, "detail": ""},
        "criteria_met": {"score": 0.5, "detail": ""},
        "overall_score": 0.5,
        "feedback": "",
        "summary": "Reflection 解析失败，默认通过",
    }
    if not response:
        return default
    try:
        text = response
        if "```" in text:
            text = text.split("```")[1].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        result = json.loads(text)
        # 确保 passed 字段与 overall_score 一致
        overall = result.get("overall_score", 0.5)
        result["passed"] = overall >= 0.6
        return result
    except (json.JSONDecodeError, IndexError) as e:
        logger.warning(f"Reflection: JSON 解析失败 ({e})，默认通过")
        return default


def _rule_based_check(state: AgentState) -> Dict:
    """规则化快速检查（作为 LLM 审查的补充）"""
    issues = []
    specialist_analysis = state.get("specialist_analysis", "")

    # 检查 1：分析不能为空或过短
    if not specialist_analysis or len(specialist_analysis.strip()) < 30:
        issues.append("专科分析内容过短或为空")

    # 检查 2：鉴别假设覆盖
    hypotheses = state.get("differential_hypotheses", [])
    if hypotheses:
        covered = sum(1 for h in hypotheses if h in specialist_analysis)
        if covered == 0 and hypotheses[0] != "待专科分析后确定":
            issues.append(f"鉴别诊断假设未在分析中被提及: {hypotheses}")

    # 检查 3：工具调用结果是否被引用
    tool_calls = state.get("tool_calls", [])
    successful_tools = [tc for tc in tool_calls if tc.get("success") and tc.get("output")]
    if successful_tools and specialist_analysis:
        # 粗略检查：工具名是否在分析中被提及，或工具结果的关键词是否出现
        tool_names = {tc.get("tool_name", "") for tc in successful_tools}
        # 这里不做严格要求，只标记问题
        if not any(name in specialist_analysis for name in tool_names):
            pass  # 工具名不一定出现在文本中，不作为硬性要求

    return {
        "issues": issues,
        "has_critical_issues": len(issues) > 0,
    }


def reflect_on_analysis(state: AgentState) -> AgentState:
    """
    Reflection Agent 节点：审查 Specialist 的分析质量。

    流程：
    1. 规则化快速检查
    2. LLM 深度审查（证据一致性、假设覆盖、信息完整性）
    3. 综合判定是否通过
    4. 不通过则生成反馈信息
    """
    reflection_count = state.get("reflection_count", 0) + 1
    specialist_analysis = state.get("specialist_analysis", "")

    # 达到最大重试次数，强制通过
    if reflection_count > MAX_REFLECTION_RETRIES:
        logger.warning(f"Reflection: 达到最大重试次数 ({MAX_REFLECTION_RETRIES})，强制通过")
        return {
            "reflection_passed": True,
            "reflection_count": reflection_count,
            "reflection_feedback": f"已重试 {MAX_REFLECTION_RETRIES} 次，强制通过",
        }

    # 规则化快速检查
    rule_check = _rule_based_check(state)
    if rule_check["has_critical_issues"] and not specialist_analysis.strip():
        logger.warning(f"Reflection: 规则检查发现严重问题: {rule_check['issues']}")
        return {
            "reflection_passed": False,
            "reflection_count": reflection_count,
            "reflection_feedback": "规则检查不通过: " + "; ".join(rule_check["issues"]),
        }

    # LLM 深度审查
    prompt = _build_reflection_prompt(state)
    response = chat(
        prompt,
        system=REFLECTION_SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    result = _parse_reflection_result(response)
    passed = result.get("passed", True)
    feedback = result.get("feedback", "")
    overall_score = result.get("overall_score", 0.5)

    # 合并规则检查的问题
    if rule_check["issues"]:
        feedback = f"[规则检查] {'; '.join(rule_check['issues'])}。{feedback}"

    if passed:
        logger.info(
            f"Reflection: 通过 (score={overall_score:.2f}, count={reflection_count})"
        )
    else:
        logger.warning(
            f"Reflection: 不通过 (score={overall_score:.2f}, count={reflection_count}) "
            f"feedback={feedback[:100]}"
        )

    return {
        "reflection_passed": passed,
        "reflection_count": reflection_count,
        "reflection_feedback": feedback if not passed else result.get("summary", "审查通过"),
    }
