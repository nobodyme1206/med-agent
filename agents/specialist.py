"""
专科 Agent：基于分诊结果进行问诊推理，调用 RAG 和检验解读工具辅助诊断。

使用统一工具调用引擎（utils/tool_agent.py），
优先 OpenAI Function Calling，fallback 文本解析。
"""

import logging
from typing import Dict

from graph.state import AgentState
from utils.tool_agent import run_tool_agent

logger = logging.getLogger(__name__)

# Specialist 可用的工具
SPECIALIST_TOOLS = ["search_guidelines", "interpret_lab_result"]

# FC 模式系统提示词（工具定义由 API 参数提供，不需要写在 prompt 里）
SPECIALIST_SYSTEM_PROMPT = """你是一位经验丰富的{department}医学顾问AI。请分析患者情况，必要时调用工具查询诊疗指南或解读检验结果，然后给出专业且安全的建议。"""


def specialist_analyze(state: AgentState) -> AgentState:
    """
    专科 Agent 节点：基于科室进行专业问诊推理。
    """
    department = state.get("current_department", "内科")
    patient_info = state.get("patient_info", "")
    messages = state.get("messages", [])
    retrieved_knowledge = list(state.get("retrieved_knowledge", []))
    tool_plan = state.get("tool_plan", [])
    expected_evidence = state.get("expected_evidence", [])
    plan_summary = state.get("plan_summary", "")
    differential_hypotheses = state.get("differential_hypotheses", [])
    information_gaps = state.get("information_gaps", [])
    reasoning_chain = state.get("reasoning_chain", "")
    reflection_feedback = state.get("reflection_feedback", "")
    memory_context = state.get("memory_context", "")

    # 构建对话上下文
    conversation = ""
    for msg in messages:
        role = "患者" if msg.get("role") == "user" else "医生"
        conversation += f"{role}：{msg['content']}\n"

    user_prompt = (
        f"【科室】{department}\n"
        f"【患者信息】{patient_info}\n\n"
        f"【推理链】{reasoning_chain}\n"
        f"【执行计划】{plan_summary}\n"
        f"【推荐工具】{', '.join(tool_plan) if tool_plan else '无'}\n"
        f"【期望证据】{'；'.join(expected_evidence) if expected_evidence else '无'}\n"
    )

    if differential_hypotheses:
        user_prompt += f"【鉴别诊断假设（需逐一分析或排除）】\n"
        for i, h in enumerate(differential_hypotheses, 1):
            user_prompt += f"  {i}. {h}\n"

    if information_gaps:
        user_prompt += f"【信息缺口（需重点关注）】\n"
        for gap in information_gaps:
            user_prompt += f"  - {gap}\n"

    if reflection_feedback:
        user_prompt += f"\n⚠️【质控反馈（上次分析未通过审查，请改进）】\n{reflection_feedback}\n"

    if memory_context:
        user_prompt += f"\n【相似历史病例参考（仅供辅助，不可直接复用结论）】\n{memory_context}\n"

    user_prompt += (
        f"\n【对话记录】\n{conversation}\n"
        f"请针对鉴别诊断假设逐一分析，引用工具返回的证据，给出回复。"
    )

    # 运行时配置（消融实验支持）
    from graph.workflow import _runtime_config
    tools_enabled = _runtime_config.get("use_tools", True)
    rag_enabled = _runtime_config.get("use_rag", True)
    max_tool_calls = _runtime_config.get("max_tool_calls")
    max_calls_per_tool = _runtime_config.get("max_calls_per_tool")

    system = SPECIALIST_SYSTEM_PROMPT.format(department=department)
    planned_tools = [name for name in tool_plan if name in SPECIALIST_TOOLS]
    if not planned_tools:
        planned_tools = list(SPECIALIST_TOOLS)

    # 调用统一引擎
    result = run_tool_agent(
        system_prompt=system,
        user_prompt=user_prompt,
        tool_names=planned_tools,
        tools_enabled=tools_enabled,
        rag_enabled=rag_enabled,
        temperature=0.3,
        agent_name="specialist",
        max_total_tool_calls=max_tool_calls,
        max_calls_per_tool=max_calls_per_tool,
    )

    analysis = result["response"]
    new_tool_calls = result["tool_calls"]
    retrieved_knowledge.extend(result.get("knowledge", []))

    if not analysis:
        analysis = "抱歉，系统暂时无法分析，请稍后重试。"

    logger.info(f"Specialist: 分析完成 (工具调用={len(new_tool_calls)})")

    return {
        "specialist_analysis": analysis,
        "tool_calls": new_tool_calls,
        "retrieved_knowledge": retrieved_knowledge,
        "stop_reason": result.get("stop_reason", ""),
        "messages": [{"role": "assistant", "content": f"[{department}医生分析] {analysis[:100]}..."}],
    }
