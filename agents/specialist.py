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

    # 构建对话上下文
    conversation = ""
    for msg in messages:
        role = "患者" if msg.get("role") == "user" else "医生"
        conversation += f"{role}：{msg['content']}\n"

    user_prompt = (
        f"【科室】{department}\n"
        f"【患者信息】{patient_info}\n\n"
        f"【对话记录】\n{conversation}\n"
        f"请进行分析并给出回复。"
    )

    # 运行时配置（消融实验支持）
    from graph.workflow import _runtime_config
    tools_enabled = _runtime_config.get("use_tools", True)
    rag_enabled = _runtime_config.get("use_rag", True)

    system = SPECIALIST_SYSTEM_PROMPT.format(department=department)

    # 调用统一引擎
    result = run_tool_agent(
        system_prompt=system,
        user_prompt=user_prompt,
        tool_names=SPECIALIST_TOOLS,
        tools_enabled=tools_enabled,
        rag_enabled=rag_enabled,
        temperature=0.3,
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
        "messages": [{"role": "assistant", "content": f"[{department}医生分析] {analysis[:100]}..."}],
    }
