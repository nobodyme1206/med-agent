"""
药品知识库构建/扩展脚本。
可从 JSON 源文件补充更多药品数据。

用法:
  python scripts/build_drug_kb.py --source data/drug_kb/extra_drugs.json
"""

import json
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "drug_kb" / "drug_database.json"


def load_db() -> list:
    if DB_PATH.exists():
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_db(data: list):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"药品数据库保存: {DB_PATH} ({len(data)} 条)")


def merge_drugs(existing: list, new_drugs: list) -> list:
    """合并药品，按 generic_name 去重"""
    seen = {d["generic_name"] for d in existing}
    added = 0
    for drug in new_drugs:
        name = drug.get("generic_name", "")
        if name and name not in seen:
            existing.append(drug)
            seen.add(name)
            added += 1
    logger.info(f"新增 {added} 种药品")
    return existing


def validate_drug(drug: dict) -> bool:
    """校验药品数据结构"""
    required = ["generic_name", "category", "indications"]
    return all(drug.get(k) for k in required)


def print_stats(data: list):
    """打印统计"""
    categories = {}
    for d in data:
        cat = d.get("category", "未分类")
        categories[cat] = categories.get(cat, 0) + 1
    logger.info(f"总计 {len(data)} 种药品，分类统计:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="", help="额外药品数据 JSON 路径")
    parser.add_argument("--stats", action="store_true", help="仅打印统计")
    args = parser.parse_args()

    db = load_db()

    if args.stats:
        print_stats(db)
        return

    if args.source:
        with open(args.source, "r", encoding="utf-8") as f:
            new_drugs = json.load(f)
        new_drugs = [d for d in new_drugs if validate_drug(d)]
        db = merge_drugs(db, new_drugs)
        save_db(db)

    print_stats(db)


if __name__ == "__main__":
    main()
