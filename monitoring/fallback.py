"""
兜底策略：置信度/急危重症/工具连续失败 → 降级回复 + 人工接管标记。
"""

import logging
from typing import Dict, Optional

from graph.state import (
    AgentState,
    EMERGENCY_KEYWORDS,
    CONFIDENCE_THRESHOLD,
    TOKEN_BUDGET,
)

logger = logging.getLogger(__name__)


class FallbackHandler:
    """
    三层兜底机制：
    1. 急危重症拦截：检测到关键词 → 直接输出急救指引
    2. 置信度兜底：confidence < 阈值 → 建议就医
    3. 工具连续失败：连续 2 次失败 → 降级纯 LLM 回答
    """

    EMERGENCY_RESPONSE = (
        "⚠️ **紧急提醒**\n\n"
        "根据您描述的症状，可能涉及紧急情况。请您：\n"
        "1. **立即拨打 120 急救电话**\n"
        "2. 如有胸痛，保持坐位或半卧位，避免剧烈活动\n"
        "3. 如有出血，用干净织物按压伤口\n"
        "4. 如有意识丧失，将患者侧卧位防止窒息\n\n"
        "请不要等待在线咨询结果，**立即就医**。"
    )

    LOW_CONFIDENCE_RESPONSE = (
        "根据目前的信息，我的分析置信度不够高。"
        "**建议您前往医院{department}做进一步检查和确诊**。\n\n"
        "您可以将以上初步分析提供给接诊医生参考。"
    )

    TOOL_FAILURE_RESPONSE = (
        "抱歉，我在查询相关信息时遇到了技术问题。"
        "以下是基于已有知识的初步分析，仅供参考：\n\n{analysis}"
    )

    def check_emergency(self, text: str) -> Optional[str]:
        """检查是否包含急危重症关键词"""
        for kw in EMERGENCY_KEYWORDS:
            if kw in text:
                logger.warning(f"Fallback: 检测到急危重症关键词 '{kw}'")
                return self.EMERGENCY_RESPONSE
        return None

    def check_confidence(self, state: AgentState) -> Optional[str]:
        """检查置信度是否过低"""
        confidence = state.get("confidence", 1.0)
        if confidence < CONFIDENCE_THRESHOLD:
            dept = state.get("current_department", "相关科室")
            logger.warning(f"Fallback: 置信度过低 ({confidence:.2f} < {CONFIDENCE_THRESHOLD})")
            return self.LOW_CONFIDENCE_RESPONSE.format(department=dept)
        return None

    def check_tool_failures(self, state: AgentState) -> Optional[str]:
        """检查是否有连续工具调用失败"""
        tool_calls = state.get("tool_calls", [])
        if len(tool_calls) >= 2:
            last_two = tool_calls[-2:]
            if all(not tc.get("success") for tc in last_two):
                analysis = state.get("specialist_analysis", "暂无分析结果")
                logger.warning("Fallback: 连续2次工具调用失败，降级回复")
                return self.TOOL_FAILURE_RESPONSE.format(analysis=analysis)
        return None

    def check_token_budget(self, state: AgentState) -> bool:
        """检查是否超出 token budget"""
        usage = state.get("token_usage", 0)
        if usage > TOKEN_BUDGET:
            logger.warning(f"Fallback: Token 超预算 ({usage} > {TOKEN_BUDGET})")
            return True
        return False

    def apply(self, state: AgentState) -> Optional[str]:
        """
        按优先级检查所有兜底条件，返回兜底回复或 None。

        优先级：急危重症 > 工具失败 > 置信度低
        """
        # 1. 急危重症（只检查患者输入，避免医生分析文本误触发）
        user_text = " ".join(
            m.get("content", "") for m in state.get("messages", [])
            if m.get("role") == "user"
        )
        emergency = self.check_emergency(user_text)
        if emergency:
            return emergency

        # 2. 工具连续失败
        tool_fail = self.check_tool_failures(state)
        if tool_fail:
            return tool_fail

        # 3. 置信度低
        low_conf = self.check_confidence(state)
        if low_conf:
            return low_conf

        return None


# 全局实例
fallback = FallbackHandler()
