"""
检验值解读工具：输入检验项目名+数值，输出正常/偏高/偏低+可能原因。
数据源：data/lab_ranges/lab_ranges.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "lab_ranges"
RANGES_PATH = DATA_DIR / "lab_ranges.json"

_lab_ranges: Optional[Dict] = None


def _load_ranges() -> Dict:
    """懒加载检验值参考范围表"""
    global _lab_ranges
    if _lab_ranges is None:
        if not RANGES_PATH.exists():
            logger.warning(f"检验范围表不存在: {RANGES_PATH}，返回空字典")
            _lab_ranges = {}
        else:
            with open(RANGES_PATH, "r", encoding="utf-8") as f:
                _lab_ranges = json.load(f)
            logger.info(f"检验范围表加载完成: {len(_lab_ranges)} 个项目")
    return _lab_ranges


def interpret_lab_result(test_name: str, value: float, unit: str = "") -> Dict:
    """
    解读单个检验结果。

    Args:
        test_name: 检验项目名称（如"血红蛋白"、"ALT"、"空腹血糖"）
        value: 检验数值
        unit: 单位（可选，用于辅助匹配）

    Returns:
        解读结果，包含正常范围、状态（正常/偏高/偏低）、可能原因。
    """
    ranges = _load_ranges()
    test_name_lower = test_name.lower()

    # 模糊匹配检验项目
    matched_key = None
    for key in ranges:
        key_lower = key.lower()
        aliases = [a.lower() for a in ranges[key].get("aliases", [])]
        if test_name_lower == key_lower or test_name_lower in aliases:
            matched_key = key
            break
        if test_name_lower in key_lower or key_lower in test_name_lower:
            matched_key = key
            break

    if not matched_key:
        return {
            "found": False,
            "test_name": test_name,
            "message": f"未找到检验项目: {test_name}",
        }

    ref = ranges[matched_key]
    low = ref.get("low", float("-inf"))
    high = ref.get("high", float("inf"))
    ref_unit = ref.get("unit", "")

    # LLM 可能传入字符串，强制转 float
    try:
        value = float(value)
    except (ValueError, TypeError):
        return {
            "found": True,
            "test_name": matched_key,
            "value": value,
            "message": f"无法解析数值: {value}",
        }

    if value < low:
        status = "偏低"
        causes = ref.get("low_causes", [])
    elif value > high:
        status = "偏高"
        causes = ref.get("high_causes", [])
    else:
        status = "正常"
        causes = []

    return {
        "found": True,
        "test_name": matched_key,
        "value": value,
        "unit": ref_unit or unit,
        "reference_range": f"{low}-{high} {ref_unit}",
        "status": status,
        "possible_causes": causes,
        "clinical_significance": ref.get("significance", ""),
    }


def interpret_lab_panel(results: list) -> list:
    """
    批量解读检验报告。

    Args:
        results: 检验结果列表，每项为 {"test_name": str, "value": float, "unit": str}

    Returns:
        解读结果列表。
    """
    interpretations = []
    for item in results:
        interp = interpret_lab_result(
            test_name=item.get("test_name", ""),
            value=item.get("value", 0),
            unit=item.get("unit", ""),
        )
        interpretations.append(interp)
    return interpretations


# ─────────────────────────────────────────────
# 工具注册定义
# ─────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "interpret_lab_result",
        "description": "解读单个检验结果，输入检验项目名和数值，返回是否正常、参考范围及可能原因。",
        "parameters": {
            "type": "object",
            "properties": {
                "test_name": {
                    "type": "string",
                    "description": "检验项目名称，如'血红蛋白'、'ALT'、'空腹血糖'、'白细胞计数'",
                },
                "value": {
                    "type": "number",
                    "description": "检验数值",
                },
                "unit": {
                    "type": "string",
                    "description": "单位（可选）",
                },
            },
            "required": ["test_name", "value"],
        },
        "handler": interpret_lab_result,
        "category": "lab",
    },
]
