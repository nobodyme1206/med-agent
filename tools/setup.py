"""
工具初始化：将所有工具注册到全局 ToolRegistry。
在 Agent 启动时调用 setup_tools() 即可。
"""

import logging
from tools.registry import registry, ToolDefinition

logger = logging.getLogger(__name__)


def setup_tools():
    """注册所有工具到全局 registry"""
    from tools.drug_lookup import TOOL_DEFINITIONS as drug_tools
    from tools.lab_interpreter import TOOL_DEFINITIONS as lab_tools
    from tools.guideline_rag import TOOL_DEFINITIONS as rag_tools

    all_tools = drug_tools + lab_tools + rag_tools

    for tool_cfg in all_tools:
        tool_def = ToolDefinition(
            name=tool_cfg["name"],
            description=tool_cfg["description"],
            parameters=tool_cfg["parameters"],
            handler=tool_cfg["handler"],
            category=tool_cfg.get("category", "general"),
        )
        registry.register(tool_def)

    logger.info(f"工具注册完成: {registry.list_tools()}")
    return registry
