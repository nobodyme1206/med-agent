"""
工具注册中心：统一管理所有 Agent 可调用的工具。
- JSON Schema 标准化定义（兼容 OpenAI Function Calling + MCP 协议）
- 工具调用日志记录
- 统一调用入口
"""

import json
import time
import logging
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """工具定义（MCP 兼容）"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema format
    handler: Callable           # 实际执行函数
    category: str = "general"   # 工具分类：drug / rag / lab / general


@dataclass
class ToolCallRecord:
    """工具调用记录"""
    tool_name: str
    input_args: Dict[str, Any]
    output: Any
    latency_ms: float
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class ToolRegistry:
    """
    工具注册中心。
    - register(): 注册工具
    - call(): 统一调用入口（带日志和异常处理）
    - get_openai_tools(): 导出 OpenAI Function Calling 格式
    - get_mcp_tools(): 导出 MCP 协议格式
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._call_history: List[ToolCallRecord] = []

    def register(self, tool_def: ToolDefinition):
        """注册一个工具"""
        self._tools[tool_def.name] = tool_def
        logger.info(f"工具注册: {tool_def.name} ({tool_def.category})")

    def call(self, tool_name: str, **kwargs) -> Any:
        """统一调用入口，带日志和异常处理"""
        if tool_name not in self._tools:
            raise ValueError(f"未注册的工具: {tool_name}")

        tool = self._tools[tool_name]
        start = time.time()
        try:
            result = tool.handler(**kwargs)
            latency = (time.time() - start) * 1000
            record = ToolCallRecord(
                tool_name=tool_name,
                input_args=kwargs,
                output=result,
                latency_ms=latency,
                success=True,
            )
            self._call_history.append(record)
            logger.info(f"工具调用成功: {tool_name} ({latency:.1f}ms)")
            return result
        except Exception as e:
            latency = (time.time() - start) * 1000
            record = ToolCallRecord(
                tool_name=tool_name,
                input_args=kwargs,
                output=None,
                latency_ms=latency,
                success=False,
                error=str(e),
            )
            self._call_history.append(record)
            logger.error(f"工具调用失败: {tool_name} - {e}")
            return {"error": str(e)}

    def get_openai_tools(self, filter_names: List[str] = None, filter_category: str = None) -> List[Dict]:
        """导出 OpenAI Function Calling 格式的工具定义列表。

        Args:
            filter_names: 只返回指定名称的工具
            filter_category: 只返回指定分类的工具
        """
        tools = []
        for name, tool_def in self._tools.items():
            if filter_names and name not in filter_names:
                continue
            if filter_category and tool_def.category != filter_category:
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "parameters": tool_def.parameters,
                },
            })
        return tools

    def get_mcp_tools(self) -> List[Dict]:
        """导出 MCP（Model Context Protocol）兼容格式"""
        tools = []
        for name, tool_def in self._tools.items():
            tools.append({
                "name": tool_def.name,
                "description": tool_def.description,
                "inputSchema": tool_def.parameters,
            })
        return tools

    def get_call_history(self) -> List[ToolCallRecord]:
        """获取工具调用历史"""
        return self._call_history

    def get_call_stats(self) -> Dict:
        """获取工具调用统计"""
        total = len(self._call_history)
        if total == 0:
            return {"total_calls": 0}
        success = sum(1 for r in self._call_history if r.success)
        avg_latency = sum(r.latency_ms for r in self._call_history) / total
        by_tool = {}
        for r in self._call_history:
            if r.tool_name not in by_tool:
                by_tool[r.tool_name] = {"calls": 0, "success": 0}
            by_tool[r.tool_name]["calls"] += 1
            if r.success:
                by_tool[r.tool_name]["success"] += 1
        return {
            "total_calls": total,
            "success_rate": success / total,
            "avg_latency_ms": avg_latency,
            "by_tool": by_tool,
        }

    def reset_history(self):
        """重置调用历史"""
        self._call_history = []

    def list_tools(self) -> List[str]:
        """列出所有已注册工具名"""
        return list(self._tools.keys())


# 全局注册中心实例
registry = ToolRegistry()
