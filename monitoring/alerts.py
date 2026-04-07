"""
告警规则：监控关键指标，超阈值时触发告警。
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """告警规则定义"""
    name: str
    metric: str
    threshold: float
    condition: str    # "gt" (大于) 或 "lt" (小于)
    severity: str     # "warning" 或 "critical"
    message: str


# 默认告警规则
DEFAULT_RULES = [
    AlertRule(
        name="high_latency",
        metric="p99_latency_ms",
        threshold=30000,
        condition="gt",
        severity="warning",
        message="P99延迟超过30秒，请检查API响应速度",
    ),
    AlertRule(
        name="high_error_rate",
        metric="tool_success_rate",
        threshold=0.8,
        condition="lt",
        severity="critical",
        message="工具调用成功率低于80%，请检查工具服务",
    ),
    AlertRule(
        name="high_escalation",
        metric="escalation_rate",
        threshold=0.3,
        condition="gt",
        severity="warning",
        message="升级率超过30%，模型置信度普遍偏低",
    ),
    AlertRule(
        name="low_confidence",
        metric="avg_confidence",
        threshold=0.5,
        condition="lt",
        severity="warning",
        message="平均置信度低于0.5，模型表现可能下降",
    ),
]


class AlertManager:
    """告警管理器"""

    def __init__(self, rules: List[AlertRule] = None):
        self.rules = rules or DEFAULT_RULES
        self.alert_history: List[Dict] = []

    def check(self, stats: Dict) -> List[Dict]:
        """
        检查统计指标是否触发告警。

        Args:
            stats: MetricsTracker.get_aggregate_stats() 的输出

        Returns:
            触发的告警列表
        """
        triggered = []
        for rule in self.rules:
            value = stats.get(rule.metric)
            if value is None:
                continue

            fired = False
            if rule.condition == "gt" and value > rule.threshold:
                fired = True
            elif rule.condition == "lt" and value < rule.threshold:
                fired = True

            if fired:
                alert = {
                    "rule": rule.name,
                    "severity": rule.severity,
                    "message": rule.message,
                    "metric": rule.metric,
                    "value": value,
                    "threshold": rule.threshold,
                }
                triggered.append(alert)
                self.alert_history.append(alert)
                log_fn = logger.critical if rule.severity == "critical" else logger.warning
                log_fn(f"🚨 [{rule.severity.upper()}] {rule.message} ({rule.metric}={value:.2f})")

        return triggered


# 全局实例
alert_manager = AlertManager()
