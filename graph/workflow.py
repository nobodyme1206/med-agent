"""
LangGraph 状态机编排：定义 Agent 之间的流转逻辑。

流程（PARM 架构）：
  START → Router → Planner → Specialist → Reflection ─┬→ Pharmacist → Summary → END
                       ↑                               │
                       └── Reflection 不通过时回退 ────┘

条件分支：
  - Router 判定急诊 → Summary（直接兜底）
  - Specialist 分析为空 → 重试 Specialist
  - Reflection 不通过 → 带反馈回退 Specialist（最多 MAX_REFLECTION_RETRIES 次）
  - Reflection 通过 + need_pharmacist → Pharmacist
  - Reflection 通过 → Summary
  - Summary 检测到风险 → 标记 should_escalate
"""

import logging
import uuid
from typing import Literal

try:
    from langgraph.graph import StateGraph, END
    _HAS_LANGGRAPH = True
except ImportError:
    StateGraph = None
    END = "__end__"
    _HAS_LANGGRAPH = False

from graph.state import AgentState, MAX_LOOP_COUNT, MAX_REFLECTION_RETRIES, MAX_TOOL_CALLS, MAX_CALLS_PER_TOOL
from agents.router import route_patient
from agents.planner import plan_consultation
from agents.specialist import specialist_analyze
from agents.reflection import reflect_on_analysis
from agents.pharmacist import pharmacist_review
from agents.summary import summarize_response
from monitoring.tracing import global_tracer
from memory.long_term import LongTermMemory

# 全局运行时配置（消融实验可覆盖）
_runtime_config = {
    "use_tools": True,
    "use_rag": True,
    "use_memory": False,
    "max_loops": MAX_LOOP_COUNT,
    "max_tool_calls": MAX_TOOL_CALLS,
    "max_calls_per_tool": MAX_CALLS_PER_TOOL,
}

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 条件路由函数
# ─────────────────────────────────────────────

def after_router(state: AgentState) -> Literal["planner", "summary"]:
    """Router 之后的路由决策"""
    if state.get("should_escalate", False):
        logger.info("Workflow: Router 触发升级 → 跳转 Summary")
        return "summary"
    return "planner"


def after_planner(state: AgentState) -> Literal["specialist", "summary"]:
    if state.get("should_escalate", False):
        logger.info("Workflow: Planner 触发升级 → 跳转 Summary")
        return "summary"
    return "specialist"


def after_specialist(state: AgentState) -> Literal["reflection", "specialist", "summary"]:
    """Specialist 之后的路由决策：进入 Reflection 或重试"""
    loop_count = state.get("loop_count", 0)
    max_loops = _runtime_config.get("max_loops", MAX_LOOP_COUNT)

    if loop_count >= max_loops:
        logger.info(f"Workflow: 达到最大循环次数 ({max_loops}) → 跳转 Summary")
        return "summary"

    analysis = state.get("specialist_analysis", "")
    if not analysis:
        logger.info("Workflow: 专科分析为空 → 重试 Specialist")
        return "specialist"

    return "reflection"


def after_reflection(state: AgentState) -> Literal["pharmacist", "specialist", "summary"]:
    """Reflection 之后的路由决策"""
    if state.get("reflection_passed", True):
        if state.get("need_pharmacist", False):
            logger.info("Workflow: Reflection 通过 → Pharmacist")
            return "pharmacist"
        logger.info("Workflow: Reflection 通过 → Summary")
        return "summary"

    # Reflection 不通过，回退 Specialist 重试
    reflection_count = state.get("reflection_count", 0)
    if reflection_count > MAX_REFLECTION_RETRIES:
        logger.warning("Workflow: Reflection 重试超限 → 强制进入 Summary")
        return "summary"

    logger.info(f"Workflow: Reflection 不通过 (count={reflection_count}) → 回退 Specialist")
    return "specialist"


# ─────────────────────────────────────────────
# 节点包装器（更新循环计数等元数据）
# ─────────────────────────────────────────────

def router_node(state: AgentState) -> AgentState:
    """Router 节点"""
    with global_tracer.span("router", "agent") as s:
        s.set_input(state)
        result = route_patient(state)
        s.set_output(result)
    return result


def planner_node(state: AgentState) -> AgentState:
    with global_tracer.span("planner", "agent") as s:
        s.set_input(state)
        result = plan_consultation(state)
        s.set_output(result)
    return result


def specialist_node(state: AgentState) -> AgentState:
    """Specialist 节点（带循环计数）"""
    loop_count = state.get("loop_count", 0)
    with global_tracer.span(f"specialist_loop{loop_count}", "agent") as s:
        s.set_input(state)
        result = specialist_analyze(state)
        result["loop_count"] = loop_count + 1
        s.set_output(result)
    return result


def pharmacist_node(state: AgentState) -> AgentState:
    """Pharmacist 节点"""
    with global_tracer.span("pharmacist", "agent") as s:
        s.set_input(state)
        result = pharmacist_review(state)
        s.set_output(result)
    return result


def reflection_node(state: AgentState) -> AgentState:
    """Reflection 节点"""
    with global_tracer.span("reflection", "agent") as s:
        s.set_input(state)
        result = reflect_on_analysis(state)
        s.set_output(result)
    return result


def summary_node(state: AgentState) -> AgentState:
    """Summary 节点"""
    with global_tracer.span("summary", "agent") as s:
        s.set_input(state)
        result = summarize_response(state)
        s.set_output(result)
    return result


def _merge_state(state: AgentState, patch: dict) -> AgentState:
    if not patch:
        return state
    merged = dict(state)
    for key, value in patch.items():
        if key in ("messages", "tool_calls") and isinstance(value, list):
            merged[key] = list(merged.get(key, [])) + value
        else:
            merged[key] = value
    return merged


class _SequentialWorkflow:
    def invoke(self, initial_state: AgentState) -> AgentState:
        state = dict(initial_state)
        state = _merge_state(state, router_node(state))
        route = after_router(state)

        if route == "summary":
            return _merge_state(state, summary_node(state))

        state = _merge_state(state, planner_node(state))
        route = after_planner(state)

        if route == "summary":
            return _merge_state(state, summary_node(state))

        while True:
            state = _merge_state(state, specialist_node(state))
            route = after_specialist(state)

            if route == "specialist":
                continue
            if route == "summary":
                return _merge_state(state, summary_node(state))

            # route == "reflection"
            state = _merge_state(state, reflection_node(state))
            route = after_reflection(state)

            if route == "specialist":
                continue
            if route == "summary":
                return _merge_state(state, summary_node(state))

            # route == "pharmacist"
            state = _merge_state(state, pharmacist_node(state))
            return _merge_state(state, summary_node(state))


# ─────────────────────────────────────────────
# 构建 LangGraph 工作流
# ─────────────────────────────────────────────

def build_workflow():
    """
    构建并返回编译后的 LangGraph 工作流。

    返回可直接调用 .invoke(state) 的 CompiledGraph。
    """
    if not _HAS_LANGGRAPH:
        logger.warning("LangGraph 不可用，使用顺序工作流")
        return _SequentialWorkflow()

    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("specialist", specialist_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("pharmacist", pharmacist_node)
    workflow.add_node("summary", summary_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        after_router,
        {
            "planner": "planner",
            "summary": "summary",
        },
    )

    workflow.add_conditional_edges(
        "planner",
        after_planner,
        {
            "specialist": "specialist",
            "summary": "summary",
        },
    )

    workflow.add_conditional_edges(
        "specialist",
        after_specialist,
        {
            "reflection": "reflection",
            "specialist": "specialist",
            "summary": "summary",
        },
    )

    workflow.add_conditional_edges(
        "reflection",
        after_reflection,
        {
            "pharmacist": "pharmacist",
            "specialist": "specialist",
            "summary": "summary",
        },
    )

    workflow.add_edge("pharmacist", "summary")
    workflow.add_edge("summary", END)

    compiled = workflow.compile()
    logger.info("LangGraph 工作流编译完成")

    return compiled


# ─────────────────────────────────────────────
# 便捷运行接口
# ─────────────────────────────────────────────

_compiled_workflow = None
_long_term_memory = LongTermMemory()


def get_workflow():
    """获取全局编译后的工作流（单例）"""
    global _compiled_workflow
    if _compiled_workflow is None:
        _compiled_workflow = build_workflow()
    return _compiled_workflow


def run_consultation(
    user_message: str,
    history: list = None,
    use_tools: bool = True,
    use_rag: bool = True,
    use_memory: bool = False,
    max_loops: int = None,
    max_tool_calls: int = None,
    max_calls_per_tool: int = None,
) -> dict:
    """
    运行一次完整的问诊流程。

    Args:
        user_message: 用户输入的症状描述
        history: 可选的历史对话消息列表

    Returns:
        完整的 AgentState，包含 final_response 和所有中间状态
    """
    # 更新运行时配置（消融实验使用）
    _runtime_config["use_tools"] = use_tools
    _runtime_config["use_rag"] = use_rag
    _runtime_config["use_memory"] = use_memory
    if max_loops is not None:
        _runtime_config["max_loops"] = max_loops
    else:
        _runtime_config["max_loops"] = MAX_LOOP_COUNT
    _runtime_config["max_tool_calls"] = max_tool_calls or MAX_TOOL_CALLS
    _runtime_config["max_calls_per_tool"] = max_calls_per_tool or MAX_CALLS_PER_TOOL

    workflow = get_workflow()

    messages = []

    memory_context = ""
    if use_memory:
        try:
            similar_cases = _long_term_memory.retrieve_similar_cases(
                chief_complaint=user_message, department="", top_k=3, min_score=0.4,
            )
            if similar_cases:
                context_parts = []
                for r in similar_cases:
                    meta = r.get("metadata", {})
                    summary = r.get("summary", "")
                    dept = meta.get("department", "")
                    diag = meta.get("diagnosis_direction", "")
                    part = summary
                    if dept and diag:
                        part = f"[{dept}] {diag}: {summary}"
                    elif summary:
                        part = summary
                    context_parts.append(part)

                memory_context = " | ".join(context_parts)
                messages.append({
                    "role": "system",
                    "content": f"[相似历史病例参考] {memory_context}",
                })
                logger.info(f"长期记忆: 检索到 {len(context_parts)} 条相似病例")
            else:
                # 回退到普通检索
                past_records = _long_term_memory.retrieve(user_message, top_k=2)
                if past_records:
                    context_parts = [r["summary"] for r in past_records if r.get("summary")]
                    if context_parts:
                        memory_context = " | ".join(context_parts)
                        messages.append({
                            "role": "system",
                            "content": f"[历史会话参考] {memory_context}",
                        })
                        logger.info(f"长期记忆: 检索到 {len(context_parts)} 条相关历史")
        except Exception as e:
            logger.debug(f"长期记忆检索跳过: {e}")

    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    initial_state: AgentState = {
        "messages": messages,
        "patient_info": "",
        "current_department": "",
        "router_reasoning": "",
        "problem_type": "",
        "tool_plan": [],
        "expected_evidence": [],
        "plan_summary": "",
        "need_pharmacist": False,
        "differential_hypotheses": [],
        "information_gaps": [],
        "verification_criteria": [],
        "reasoning_chain": "",
        "specialist_analysis": "",
        "drug_advice": "",
        "tool_calls": [],
        "retrieved_knowledge": [],
        "reflection_feedback": "",
        "reflection_passed": True,
        "reflection_count": 0,
        "structured_output": {
            "department": "",
            "diagnosis_direction": "",
            "recommended_tests": [],
            "medication_advice": [],
            "need_followup": False,
            "followup_actions": [],
            "evidence_summary": [],
            "used_tools": [],
            "tool_plan": [],
            "final_response": "",
        },
        "confidence": 0.0,
        "should_escalate": False,
        "escalate_reason": "",
        "loop_count": 0,
        "stop_reason": "",
        "final_response": "",
        "token_usage": 0,
        "memory_context": memory_context,
    }

    # Token 追踪
    from utils.llm_client import reset_token_usage, get_token_usage
    reset_token_usage()

    # Tracing
    request_id = str(uuid.uuid4())[:8]
    global_tracer.start_trace(request_id)
    result = workflow.invoke(initial_state)
    trace = global_tracer.end_trace()

    # 写回 token_usage
    result["token_usage"] = get_token_usage()
    if trace:
        logger.debug(trace.to_console())

    if use_memory:
        try:
            final_resp = result.get("final_response", "")
            if final_resp and len(final_resp) > 20:
                dept = result.get("current_department", "")
                structured = result.get("structured_output", {})
                # 存储结构化档案（P2 增强）
                _long_term_memory.store_structured_profile({
                    "department": dept,
                    "chief_complaint": user_message[:200],
                    "diagnosis_direction": structured.get("diagnosis_direction", ""),
                    "differential_hypotheses": result.get("differential_hypotheses", []),
                    "reasoning_chain": result.get("reasoning_chain", ""),
                    "medications": structured.get("medication_advice", []),
                    "tests": structured.get("recommended_tests", []),
                    "confidence": result.get("confidence", 0.0),
                    "request_id": request_id,
                })
        except Exception as e:
            logger.debug(f"长期记忆存储跳过: {e}")

    return result
