"""
药师 Agent：基于专科 Agent 的诊断结果，提供用药建议和药物交互检查。

使用统一工具调用引擎（utils/tool_agent.py），
优先 OpenAI Function Calling，fallback 文本解析。
"""

import logging
from typing import Dict

from graph.state import AgentState
from utils.tool_agent import run_tool_agent

logger = logging.getLogger(__name__)

# Pharmacist 可用的工具
PHARMACIST_TOOLS = ["search_drug", "check_drug_interaction", "search_by_indication"]

PHARMACIST_SYSTEM_PROMPT = """你是一位临床药师，负责审核用药方案的安全性和合理性。

你的职责：
1. 根据诊断结果，建议可能的治疗药物
2. 检查患者当前用药是否存在药物交互
3. 提供用药注意事项和剂量建议
4. 对特殊人群（老人、儿童、孕妇、肝肾功能不全）给出调整建议

重要提醒：
- 所有用药建议都需注明"以下建议仅供参考，具体用药请遵医嘱"
- 对于处方药，必须强调需要医生开具处方"""


def pharmacist_review(state: AgentState) -> AgentState:
    """
    药师 Agent 节点：审核用药方案，提供药物建议。
    """
    patient_info = state.get("patient_info", "")
    specialist_analysis = state.get("specialist_analysis", "")
    department = state.get("current_department", "")
    tool_plan = state.get("tool_plan", [])
    expected_evidence = state.get("expected_evidence", [])
    plan_summary = state.get("plan_summary", "")

    # 从对话中提取上下文
    all_text = patient_info + " " + " ".join(
        m["content"] for m in state.get("messages", [])
    )

    user_prompt = (
        f"【患者信息】{patient_info}\n"
        f"【就诊科室】{department}\n"
        f"【专科医生分析】{specialist_analysis}\n"
        f"【执行计划】{plan_summary}\n"
        f"【推荐药学工具】{', '.join(t for t in tool_plan if t in PHARMACIST_TOOLS) or '无'}\n"
        f"【期望证据】{'；'.join(expected_evidence) if expected_evidence else '无'}\n"
        f"【对话上下文】{all_text[:500]}\n\n"
        f"请基于以上信息，提供用药评估和建议。"
        f"如果提到了具体药品，请先查询药品信息。"
        f"如果患者正在服用多种药物，请检查药物交互。"
    )

    # 运行时配置（消融实验支持）
    from graph.workflow import _runtime_config
    tools_enabled = _runtime_config.get("use_tools", True)
    max_tool_calls = _runtime_config.get("max_tool_calls")
    max_calls_per_tool = _runtime_config.get("max_calls_per_tool")
    planned_tools = [name for name in tool_plan if name in PHARMACIST_TOOLS]
    if not planned_tools:
        planned_tools = list(PHARMACIST_TOOLS)

    # 调用统一引擎
    result = run_tool_agent(
        system_prompt=PHARMACIST_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_names=planned_tools,
        tools_enabled=tools_enabled,
        temperature=0.2,
        agent_name="pharmacist",
        max_total_tool_calls=max_tool_calls,
        max_calls_per_tool=max_calls_per_tool,
    )

    drug_advice = result["response"]
    new_tool_calls = result["tool_calls"]

    if not drug_advice:
        drug_advice = "用药评估暂时不可用，请咨询线下药师。"

    logger.info(f"Pharmacist: 用药建议生成完成 (工具调用={len(new_tool_calls)})")

    return {
        "drug_advice": drug_advice,
        "tool_calls": new_tool_calls,
        "stop_reason": result.get("stop_reason", ""),
    }
