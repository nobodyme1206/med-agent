"""
LangGraph 状态机编排：定义 Agent 之间的流转逻辑。

流程：
  START → Router → Specialist → Pharmacist → Summary → END
                       ↑              │
                       └── 需要补充信息时循环（max 3 轮）

条件分支：
  - Router 判定急诊 → Summary（直接兜底）
  - Specialist 需要更多信息 → 循环回 Specialist
  - Summary 检测到风险 → 标记 should_escalate
"""

import logging
import uuid
from typing import Literal

from langgraph.graph import StateGraph, END

from graph.state import AgentState, MAX_LOOP_COUNT
from agents.router import route_patient
from agents.specialist import specialist_analyze
from agents.pharmacist import pharmacist_review
from agents.summary import summarize_response
from monitoring.tracing import global_tracer
from memory.long_term import LongTermMemory

# 全局运行时配置（消融实验可覆盖）
_runtime_config = {
    "use_tools": True,
    "use_rag": True,
    "max_loops": MAX_LOOP_COUNT,
}

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 条件路由函数
# ─────────────────────────────────────────────

def after_router(state: AgentState) -> Literal["specialist", "summary"]:
    """Router 之后的路由决策"""
    # 急诊 or 已标记升级 → 直接到 Summary 兜底
    if state.get("should_escalate", False):
        logger.info("Workflow: Router 触发升级 → 跳转 Summary")
        return "summary"
    return "specialist"


def after_specialist(state: AgentState) -> Literal["pharmacist", "specialist", "summary"]:
    """Specialist 之后的路由决策"""
    loop_count = state.get("loop_count", 0)
    max_loops = _runtime_config.get("max_loops", MAX_LOOP_COUNT)

    # 超过最大循环次数 → 直接到 Summary
    if loop_count >= max_loops:
        logger.info(f"Workflow: 达到最大循环次数 ({max_loops}) → 跳转 Summary")
        return "summary"

    # 如果专科分析为空或明确需要补充信息 → 循环
    analysis = state.get("specialist_analysis", "")
    if not analysis:
        logger.info("Workflow: 专科分析为空 → 重试 Specialist")
        return "specialist"

    # 正常流转到药师
    return "pharmacist"


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


def summary_node(state: AgentState) -> AgentState:
    """Summary 节点"""
    with global_tracer.span("summary", "agent") as s:
        s.set_input(state)
        result = summarize_response(state)
        s.set_output(result)
    return result


# ─────────────────────────────────────────────
# 构建 LangGraph 工作流
# ─────────────────────────────────────────────

def build_workflow() -> StateGraph:
    """
    构建并返回编译后的 LangGraph 工作流。

    返回可直接调用 .invoke(state) 的 CompiledGraph。
    """
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("router", router_node)
    workflow.add_node("specialist", specialist_node)
    workflow.add_node("pharmacist", pharmacist_node)
    workflow.add_node("summary", summary_node)

    # 设置入口
    workflow.set_entry_point("router")

    # 添加条件边
    workflow.add_conditional_edges(
        "router",
        after_router,
        {
            "specialist": "specialist",
            "summary": "summary",
        },
    )

    workflow.add_conditional_edges(
        "specialist",
        after_specialist,
        {
            "pharmacist": "pharmacist",
            "specialist": "specialist",
            "summary": "summary",
        },
    )

    # 固定边
    workflow.add_edge("pharmacist", "summary")
    workflow.add_edge("summary", END)

    # 编译
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
    max_loops: int = None,
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
    if max_loops is not None:
        _runtime_config["max_loops"] = max_loops
    else:
        _runtime_config["max_loops"] = MAX_LOOP_COUNT

    workflow = get_workflow()

    messages = []

    # 长期记忆：检索相关历史会话
    try:
        past_records = _long_term_memory.retrieve(user_message, top_k=2)
        if past_records:
            context_parts = [r["summary"] for r in past_records if r.get("summary")]
            if context_parts:
                messages.append({
                    "role": "system",
                    "content": f"[历史会话参考] {' | '.join(context_parts)}",
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
        "specialist_analysis": "",
        "drug_advice": "",
        "tool_calls": [],
        "retrieved_knowledge": [],
        "confidence": 0.0,
        "should_escalate": False,
        "escalate_reason": "",
        "loop_count": 0,
        "final_response": "",
        "token_usage": 0,
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

    # 长期记忆：存储本次会话摘要
    try:
        final_resp = result.get("final_response", "")
        if final_resp and len(final_resp) > 20:
            dept = result.get("current_department", "")
            summary = _long_term_memory.generate_session_summary(
                result.get("messages", []), department=dept
            )
            _long_term_memory.store_session(summary, metadata={
                "department": dept,
                "confidence": result.get("confidence", 0.0),
                "request_id": request_id,
            })
    except Exception as e:
        logger.debug(f"长期记忆存储跳过: {e}")

    return result
