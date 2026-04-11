"""
Agent 共享状态定义：LangGraph 状态机的核心数据结构。
所有 Agent 节点通过读写此状态进行通信。
"""

from typing import TypedDict, List, Dict, Optional, Any

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

import operator


class ToolCallRecord(TypedDict):
    """单次工具调用记录"""
    tool_name: str
    input_args: Dict[str, Any]
    output: Any
    success: bool
    agent_name: str
    round_id: int
    call_signature: str
    skipped_reason: str


class StructuredOutput(TypedDict):
    department: str
    diagnosis_direction: str
    recommended_tests: List[str]
    medication_advice: List[str]
    need_followup: bool
    followup_actions: List[str]
    evidence_summary: List[str]
    used_tools: List[str]
    tool_plan: List[str]
    final_response: str


class AgentState(TypedDict):
    """
    多 Agent 共享状态。LangGraph 每个节点可读写此状态。

    字段说明：
    - messages: 完整对话历史（用户 + Agent 回复）
    - patient_info: 从对话中提取的患者信息摘要
    - current_department: Router 分配的科室
    - specialist_analysis: 专科 Agent 的分析结果
    - drug_advice: 药师 Agent 的用药建议
    - tool_calls: 本轮所有工具调用记录
    - retrieved_knowledge: RAG 检索到的知识块
    - confidence: 当前回答的置信度（0-1）
    - should_escalate: 是否需要人工接管
    - escalate_reason: 接管原因
    - loop_count: 当前循环次数（防止无限循环）
    - final_response: 最终输出给用户的回复
    - token_usage: 累计 token 消耗
    """
    messages: Annotated[List[Dict[str, str]], operator.add]
    patient_info: str
    current_department: str
    router_reasoning: str
    problem_type: str
    tool_plan: List[str]
    expected_evidence: List[str]
    plan_summary: str
    need_pharmacist: bool
    differential_hypotheses: List[str]
    information_gaps: List[str]
    verification_criteria: List[str]
    reasoning_chain: str
    specialist_analysis: str
    drug_advice: str
    tool_calls: Annotated[List[ToolCallRecord], operator.add]
    retrieved_knowledge: List[Dict]
    reflection_feedback: str
    reflection_passed: bool
    reflection_count: int
    structured_output: StructuredOutput
    confidence: float
    should_escalate: bool
    escalate_reason: str
    loop_count: int
    stop_reason: str
    final_response: str
    token_usage: int
    memory_context: str


# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────

DEPARTMENTS = [
    "内科", "外科", "妇产科", "儿科", "急诊科",
    "神经内科", "心血管内科", "呼吸内科", "消化内科",
    "内分泌科", "骨科", "泌尿外科", "皮肤科", "眼科", "耳鼻喉科",
    "精神科", "风湿免疫科",
]

EMERGENCY_KEYWORDS = [
    "胸痛", "呼吸困难", "大出血", "意识丧失", "昏迷",
    "心脏骤停", "窒息", "严重过敏", "中毒", "高热惊厥",
    "剧烈头痛伴呕吐", "急性腹痛", "外伤大出血",
]

MAX_LOOP_COUNT = 3
MAX_REFLECTION_RETRIES = 2
CONFIDENCE_THRESHOLD = 0.6
TOKEN_BUDGET = 8000
MAX_TOOL_CALLS = 6
MAX_CALLS_PER_TOOL = 2
