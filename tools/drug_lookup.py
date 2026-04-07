"""
药品知识库查询工具：支持按药品名 / 适应症 / 药物交互查询。
数据源：data/drug_kb/drug_database.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "drug_kb"
DB_PATH = DATA_DIR / "drug_database.json"

_drug_db: Optional[List[Dict]] = None


def _load_db() -> List[Dict]:
    """懒加载药品数据库"""
    global _drug_db
    if _drug_db is None:
        if not DB_PATH.exists():
            logger.warning(f"药品数据库不存在: {DB_PATH}，返回空列表")
            _drug_db = []
        else:
            with open(DB_PATH, "r", encoding="utf-8") as f:
                _drug_db = json.load(f)
            logger.info(f"药品数据库加载完成: {len(_drug_db)} 条记录")
    return _drug_db


def search_drug(drug_name: str) -> Dict:
    """
    按药品名查询药品信息。

    Args:
        drug_name: 药品名称（支持通用名和商品名模糊匹配）

    Returns:
        药品详细信息，包含适应症、禁忌症、不良反应、用法用量等。
        未找到时返回空结果。
    """
    db = _load_db()
    results = []
    drug_name_lower = drug_name.lower()
    for drug in db:
        names = [drug.get("generic_name", ""), drug.get("trade_name", "")]
        aliases = drug.get("aliases", [])
        all_names = [n.lower() for n in names + aliases if n]
        if any(drug_name_lower in n or n in drug_name_lower for n in all_names):
            results.append(drug)

    if not results:
        return {"found": False, "message": f"未找到药品: {drug_name}"}

    # 返回最匹配的一条
    drug = results[0]
    return {
        "found": True,
        "generic_name": drug.get("generic_name", ""),
        "trade_name": drug.get("trade_name", ""),
        "category": drug.get("category", ""),
        "indications": drug.get("indications", []),
        "contraindications": drug.get("contraindications", []),
        "adverse_reactions": drug.get("adverse_reactions", []),
        "dosage": drug.get("dosage", ""),
        "interactions": drug.get("interactions", []),
    }


def check_drug_interaction(drug_a: str, drug_b: str) -> Dict:
    """
    检查两种药物之间是否存在交互作用。

    Args:
        drug_a: 第一种药品名称
        drug_b: 第二种药品名称

    Returns:
        交互信息，包含严重程度和建议。
    """
    db = _load_db()

    # 查找两种药物
    info_a = search_drug(drug_a)
    info_b = search_drug(drug_b)

    if not info_a.get("found") or not info_b.get("found"):
        missing = []
        if not info_a.get("found"):
            missing.append(drug_a)
        if not info_b.get("found"):
            missing.append(drug_b)
        return {"has_interaction": False, "message": f"未找到药品: {', '.join(missing)}"}

    # 检查交互
    interactions_a = info_a.get("interactions", [])
    drug_b_lower = drug_b.lower()
    for interaction in interactions_a:
        target = interaction.get("drug", "").lower()
        if drug_b_lower in target or target in drug_b_lower:
            return {
                "has_interaction": True,
                "drug_a": drug_a,
                "drug_b": drug_b,
                "severity": interaction.get("severity", "unknown"),
                "description": interaction.get("description", ""),
                "recommendation": interaction.get("recommendation", ""),
            }

    return {
        "has_interaction": False,
        "drug_a": drug_a,
        "drug_b": drug_b,
        "message": "未发现已知药物交互作用",
    }


def search_by_indication(indication: str) -> List[Dict]:
    """
    按适应症查询可用药品列表。

    Args:
        indication: 适应症/疾病名称

    Returns:
        匹配的药品列表（简要信息）。
    """
    db = _load_db()
    results = []
    indication_lower = indication.lower()

    for drug in db:
        indications = drug.get("indications", [])
        for ind in indications:
            if indication_lower in ind.lower():
                results.append({
                    "generic_name": drug.get("generic_name", ""),
                    "category": drug.get("category", ""),
                    "indication_match": ind,
                    "dosage": drug.get("dosage", ""),
                })
                break

    return results if results else [{"message": f"未找到适应症为 '{indication}' 的药品"}]


# ─────────────────────────────────────────────
# 工具注册定义
# ─────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "search_drug",
        "description": "按药品名查询药品详细信息，包含适应症、禁忌症、不良反应、用法用量和药物交互。",
        "parameters": {
            "type": "object",
            "properties": {
                "drug_name": {
                    "type": "string",
                    "description": "药品名称（通用名或商品名）",
                },
            },
            "required": ["drug_name"],
        },
        "handler": search_drug,
        "category": "drug",
    },
    {
        "name": "check_drug_interaction",
        "description": "检查两种药物之间是否存在交互作用，返回交互严重程度和用药建议。",
        "parameters": {
            "type": "object",
            "properties": {
                "drug_a": {"type": "string", "description": "第一种药品名称"},
                "drug_b": {"type": "string", "description": "第二种药品名称"},
            },
            "required": ["drug_a", "drug_b"],
        },
        "handler": check_drug_interaction,
        "category": "drug",
    },
    {
        "name": "search_by_indication",
        "description": "按适应症/疾病名称查询可用药品列表。",
        "parameters": {
            "type": "object",
            "properties": {
                "indication": {"type": "string", "description": "适应症或疾病名称"},
            },
            "required": ["indication"],
        },
        "handler": search_by_indication,
        "category": "drug",
    },
]
