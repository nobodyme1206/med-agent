"""
分诊 Agent：解析患者症状，判断应分配的科室并路由。
使用 LLM few-shot prompt 实现意图识别 + 科室分类。
"""

import json
import logging
from typing import Dict

from graph.state import AgentState, DEPARTMENTS, EMERGENCY_KEYWORDS
from utils.llm_client import chat

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """你是一位经验丰富的医院分诊台护士。你的职责是根据患者的症状描述，判断应该挂哪个科室。

规则：
1. 仔细分析患者的主诉和症状
2. 如果症状涉及急危重症（胸痛、呼吸困难、大出血、意识丧失等），优先分配到"急诊科"
3. 根据症状选择最匹配的科室
4. 同时提取患者的关键信息（年龄、性别、病史、主诉）

可选科室：{departments}

请严格按照以下JSON格式输出：
{{
  "department": "科室名称",
  "confidence": 0.0-1.0,
  "patient_info": "患者关键信息摘要",
  "reasoning": "分诊理由（一句话）"
}}"""

ROUTER_EXAMPLES = """示例1：
患者：我最近总是头晕，血压偏高，吃了硝苯地平但没效果。
输出：{{"department": "心血管内科", "confidence": 0.85, "patient_info": "高血压患者，服用硝苯地平控制不佳，主诉头晕", "reasoning": "血压控制不佳伴头晕，属于心血管内科范畴"}}

示例2：
患者：我孩子3岁，发烧39度两天了，还有点咳嗽。
输出：{{"department": "儿科", "confidence": 0.9, "patient_info": "3岁儿童，发热39°C持续2天伴咳嗽", "reasoning": "儿童发热伴呼吸道症状，应由儿科处理"}}

示例3：
患者：突然胸口剧烈疼痛，出了一身冷汗，感觉喘不上气。
输出：{{"department": "急诊科", "confidence": 0.95, "patient_info": "突发剧烈胸痛伴大汗、呼吸困难", "reasoning": "急性胸痛伴大汗和呼吸困难，需急诊排除心梗"}}"""


def check_emergency(text: str) -> bool:
    """检查是否包含急危重症关键词"""
    return any(kw in text for kw in EMERGENCY_KEYWORDS)


def route_patient(state: AgentState) -> AgentState:
    """
    分诊 Agent 节点：分析患者输入，分配科室。

    输入：state.messages 中最后一条用户消息
    输出：更新 state.current_department, state.patient_info, state.confidence
    """
    # 获取最后一条用户消息
    user_messages = [m for m in state["messages"] if m.get("role") == "user"]
    if not user_messages:
        logger.warning("Router: 无用户消息")
        return state

    last_user_msg = user_messages[-1]["content"]

    # 急诊快速通道
    if check_emergency(last_user_msg):
        logger.info("Router: 检测到急危重症关键词，直接分配急诊科")
        return {
            "current_department": "急诊科",
            "patient_info": f"急危重症：{last_user_msg[:100]}",
            "router_reasoning": "检测到急危重症关键词，直接进入急诊路径",
            "confidence": 0.95,
            "should_escalate": True,
            "escalate_reason": "检测到急危重症症状，建议立即就医",
        }

    # LLM 分诊（使用 JSON mode 保证输出格式）
    system = ROUTER_SYSTEM_PROMPT.format(departments="、".join(DEPARTMENTS))
    prompt = f"{ROUTER_EXAMPLES}\n\n现在请分诊以下患者：\n患者：{last_user_msg}\n输出："

    response = chat(
        prompt,
        system=system,
        temperature=0.1,
        max_tokens=256,
        response_format={"type": "json_object"},
    )

    if not response:
        logger.error("Router: LLM 调用失败，默认分配内科")
        return {
            "current_department": "内科",
            "patient_info": last_user_msg[:200],
            "router_reasoning": "路由模型失败，使用默认内科回退",
            "confidence": 0.3,
        }

    # 解析 JSON 输出
    try:
        # JSON mode 保证输出是合法 JSON，但仍做容错
        json_str = response
        if "```" in json_str:
            json_str = json_str.split("```")[1].strip()
            if json_str.startswith("json"):
                json_str = json_str[4:].strip()
        result = json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        logger.warning(f"Router: JSON 解析失败，原始输出: {response[:200]}")
        department = "内科"
        for dept in DEPARTMENTS:
            if dept in response:
                department = dept
                break
        return {
            "current_department": department,
            "patient_info": last_user_msg[:200],
            "router_reasoning": "路由输出解析失败，采用字符串匹配结果",
            "confidence": 0.5,
        }

    department = result.get("department", "内科")
    if department not in DEPARTMENTS:
        # 模糊匹配
        for dept in DEPARTMENTS:
            if dept in department or department in dept:
                department = dept
                break
        else:
            department = "内科"

    logger.info(f"Router: 分配科室={department}, 置信度={result.get('confidence', 0.5)}")

    return {
        "current_department": department,
        "patient_info": result.get("patient_info", last_user_msg[:200]),
        "router_reasoning": result.get("reasoning", ""),
        "confidence": result.get("confidence", 0.5),
    }
