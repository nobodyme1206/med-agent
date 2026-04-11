"""
汇总 Agent：整合所有 Agent 的分析结果，生成最终回复。
包含安全检查和兜底逻辑。
"""

import logging
import json
from typing import Dict

from graph.state import AgentState, EMERGENCY_KEYWORDS, CONFIDENCE_THRESHOLD
from utils.llm_client import chat
from monitoring.fallback import fallback

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """你是一位资深的医学顾问，负责将多位专家的分析汇总为一份清晰、专业、有温度的回复。

你的职责：
1. 整合专科医生的诊断分析和药师的用药建议
2. 用通俗易懂的语言向患者解释
3. 给出明确的下一步建议（是否需要就医、做什么检查等）
4. 确保回复的安全性（不做超范围诊断、不替代线下就医）

回复格式要求：
- 简洁清晰，分段组织
- 先给结论，再解释原因
- 明确告知是否需要就医
- 用药建议要标注"仅供参考，遵医嘱"

安全红线（必须遵守）：
- 不能做确定性诊断（用"可能"、"建议排查"等措辞）
- 不能开具处方
- 涉及急危重症必须强调"立即就医"
- 不能给出自杀/自残相关建议"""


SAFETY_DISCLAIMER = "\n\n⚠️ *以上分析仅供参考，不构成医疗诊断。如症状持续或加重，请及时前往医院就诊。*"


def _base_structured_output(state: AgentState, final_response: str = "") -> Dict:
    tool_names = [tc.get("tool_name", "") for tc in state.get("tool_calls", []) if tc.get("tool_name")]
    return {
        "department": state.get("current_department", ""),
        "diagnosis_direction": "",
        "recommended_tests": [],
        "medication_advice": [],
        "need_followup": False,
        "followup_actions": [],
        "evidence_summary": [],
        "used_tools": tool_names,
        "tool_plan": state.get("tool_plan", []),
        "final_response": final_response,
    }


def _parse_structured_response(raw: str, state: AgentState) -> Dict:
    parsed = _base_structured_output(state)
    if not raw:
        return parsed

    def ensure_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    try:
        text = raw
        if "```" in text:
            text = text.split("```")[1].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        data = json.loads(text)
        parsed.update({
            "diagnosis_direction": data.get("diagnosis_direction", ""),
            "recommended_tests": ensure_list(data.get("recommended_tests", []) or []),
            "medication_advice": ensure_list(data.get("medication_advice", []) or []),
            "need_followup": bool(data.get("need_followup", False)),
            "followup_actions": ensure_list(data.get("followup_actions", []) or []),
            "evidence_summary": ensure_list(data.get("evidence_summary", []) or []),
            "final_response": data.get("final_response", ""),
        })
    except Exception as e:
        logger.warning(f"Summary: 结构化输出解析失败 ({e})")
    return parsed


def _check_safety(state: AgentState) -> tuple:
    """安全检查：返回 (should_escalate, reason)"""
    user_text = " ".join(
        m.get("content", "") for m in state.get("messages", [])
        if m.get("role") == "user"
    )

    # 急危重症检查（只检查患者输入，避免医生分析文本误触发）
    for kw in EMERGENCY_KEYWORDS:
        if kw in user_text:
            return True, f"检测到急危重症症状关键词: {kw}"

    # 置信度检查
    if state.get("confidence", 1.0) < CONFIDENCE_THRESHOLD:
        return True, "系统对当前分析的置信度较低，建议就医确认"

    # 专科分析为空
    if not state.get("specialist_analysis"):
        return True, "未能获得有效的专科分析结果"

    return False, ""


def summarize_response(state: AgentState) -> AgentState:
    """
    汇总 Agent 节点：生成最终回复。

    流程：
    1. 安全检查
    2. 整合多 Agent 分析结果
    3. LLM 生成最终回复
    4. 追加安全声明
    """
    department = state.get("current_department", "")
    specialist_analysis = state.get("specialist_analysis", "")
    drug_advice = state.get("drug_advice", "")
    patient_info = state.get("patient_info", "")
    plan_summary = state.get("plan_summary", "")
    expected_evidence = state.get("expected_evidence", [])

    # 三层兜底检查（monitoring/fallback.py）
    fallback_response = fallback.apply(state)

    # 补充：原有安全检查作为第四层
    should_escalate, escalate_reason = _check_safety(state)

    if fallback_response:
        logger.warning(f"Summary: 三层兜底触发")
        final_response = fallback_response
        if specialist_analysis:
            final_response += f"\n\n初步分析供参考：\n{specialist_analysis}"
        final_response += SAFETY_DISCLAIMER
        structured_output = _base_structured_output(state, final_response)
        return {
            "should_escalate": True,
            "escalate_reason": "三层兜底机制触发",
            "structured_output": structured_output,
            "final_response": final_response,
            "messages": [{"role": "assistant", "content": final_response}],
        }

    if should_escalate:
        logger.warning(f"Summary: 触发安全兜底 - {escalate_reason}")
        final_response = (
            f"根据您描述的情况，**建议您尽快前往医院{department}就诊**。\n\n"
            f"原因：{escalate_reason}\n\n"
        )
        if specialist_analysis:
            final_response += f"初步分析供参考：\n{specialist_analysis}\n"
        final_response += SAFETY_DISCLAIMER
        structured_output = _base_structured_output(state, final_response)
        structured_output["need_followup"] = True
        structured_output["followup_actions"] = ["尽快线下就医"]
        return {
            "should_escalate": True,
            "escalate_reason": escalate_reason,
            "structured_output": structured_output,
            "final_response": final_response,
            "messages": [{"role": "assistant", "content": final_response}],
        }

    reasoning_chain = state.get("reasoning_chain", "")
    differential_hypotheses = state.get("differential_hypotheses", [])
    reflection_feedback = state.get("reflection_feedback", "")
    reflection_count = state.get("reflection_count", 0)

    prompt = (
        f"【患者信息】{patient_info}\n"
        f"【就诊科室】{department}\n"
        f"【推理链】{reasoning_chain}\n"
        f"【鉴别诊断假设】{'、'.join(differential_hypotheses) if differential_hypotheses else '无'}\n"
        f"【执行计划】{plan_summary}\n"
        f"【期望证据】{'；'.join(expected_evidence) if expected_evidence else '无'}\n"
        f"【专科医生分析】\n{specialist_analysis}\n\n"
    )
    if drug_advice:
        prompt += f"【药师建议】\n{drug_advice}\n\n"

    if reflection_count > 0:
        prompt += f"【质控审核】经过 {reflection_count} 轮审查{'通过' if state.get('reflection_passed') else '（强制通过）'}。"
        if reflection_feedback:
            prompt += f" 审核意见：{reflection_feedback}\n\n"
        else:
            prompt += "\n\n"

    prompt += (
        "请输出 JSON："
        '{"diagnosis_direction":"","recommended_tests":[],"medication_advice":[],"need_followup":true,"followup_actions":[],"evidence_summary":[],"final_response":""}'
        "。其中 final_response 是面向患者的自然语言回复（需体现推理过程和证据来源），其他字段用于评测和训练。"
    )

    response = chat(
        prompt,
        system=SUMMARY_SYSTEM_PROMPT,
        temperature=0.3,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )

    if not response:
        final_response = specialist_analysis or "抱歉，系统暂时无法生成完整分析。建议您前往医院就诊。"
        final_response += SAFETY_DISCLAIMER
        structured_output = _base_structured_output(state, final_response)
    else:
        structured_output = _parse_structured_response(response, state)
        final_response = structured_output.get("final_response") or specialist_analysis or "抱歉，系统暂时无法生成完整分析。"
        final_response += SAFETY_DISCLAIMER
        structured_output["final_response"] = final_response

    logger.info(f"Summary: 最终回复生成完成 ({len(final_response)} 字)")

    return {
        "should_escalate": False,
        "structured_output": structured_output,
        "final_response": final_response,
        "messages": [{"role": "assistant", "content": final_response}],
    }
