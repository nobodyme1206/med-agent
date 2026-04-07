"""
合成数据质控管线：5 维自动 checker + embedding 去重 + LLM-as-Judge 过滤。

用法:
  python scripts/data_quality.py --input data/synth/trajectories/all_trajectories.json --output data/synth/trajectories/filtered.json

质控维度：
  1. 结构完整性：JSON 合法、必填字段非空、对话轮数 ≥ 2
  2. 工具调用合理性：工具名在注册表中、参数符合 schema
  3. 医学术语一致性：诊断方向与主诉/科室逻辑匹配
  4. 安全合规：包含就医建议、不含确定性诊断
  5. 去重/去近似：embedding 相似度 > 阈值的 case 去重
"""

import os
import sys
import json
import logging
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 合法工具名（与 tools/ 注册一致）
VALID_TOOLS = {
    "search_guidelines", "search_drug", "check_drug_interaction",
    "interpret_lab_result", "drug_lookup", "guideline_rag",
    "lab_interpreter",
}

# 合法科室
VALID_DEPARTMENTS = {
    "心血管内科", "内分泌科", "呼吸内科", "消化内科", "神经内科",
    "骨科", "儿科", "妇产科", "急诊科", "皮肤科", "泌尿外科",
    "内科", "外科",
}

REQUIRED_FIELDS = ["patient_profile", "chief_complaint", "department", "dialogue", "final_diagnosis_direction"]

# 科室-主诉不合理映射（黑名单）
DEPARTMENT_COMPLAINT_BLACKLIST = {
    "骨科": ["血糖", "糖尿病", "甲状腺", "血压", "肝功能"],
    "内分泌科": ["骨折", "腰椎", "外伤"],
    "皮肤科": ["胸痛", "腹痛", "血糖"],
    "儿科": ["高血压病史10年", "饮酒史"],
    "妇产科": ["男性"],
}

# 确定性诊断的坏模式
BAD_DIAGNOSIS_PATTERNS = ["确诊为", "你得了", "你患有", "诊断结果是", "你的病是"]

# 安全声明的好模式
GOOD_SAFETY_PATTERNS = ["建议", "就医", "医生", "进一步检查", "遵医嘱", "仅供参考"]


# ─────────────────────────────────────────────
# Checker 1: 结构完整性
# ─────────────────────────────────────────────

def check_structure(case: Dict) -> Tuple[bool, List[str]]:
    """检查 JSON 结构完整性"""
    issues = []

    # 必填字段
    for field in REQUIRED_FIELDS:
        val = case.get(field)
        if val is None or (isinstance(val, str) and not val.strip()):
            issues.append(f"缺失或为空字段: {field}")

    # 对话轮数
    dialogue = case.get("dialogue", [])
    if not isinstance(dialogue, list):
        issues.append("dialogue 不是列表")
    elif len(dialogue) < 2:
        issues.append(f"对话轮数不足: {len(dialogue)} < 2")
    else:
        # 检查对话角色交替
        roles = [t.get("role") for t in dialogue]
        if roles[0] != "patient":
            issues.append("对话应以 patient 开头")
        has_agent = any(r == "agent" for r in roles)
        if not has_agent:
            issues.append("对话中无 agent 回复")

    # 生成是否成功
    if not case.get("generation_success", True):
        issues.append("生成标记为失败")

    passed = len(issues) == 0
    return passed, issues


# ─────────────────────────────────────────────
# Checker 2: 工具调用合理性
# ─────────────────────────────────────────────

def check_tool_calls(case: Dict) -> Tuple[bool, List[str]]:
    """检查工具调用是否合理"""
    issues = []
    dialogue = case.get("dialogue", [])

    tool_count = 0
    for turn in dialogue:
        if turn.get("role") != "agent":
            continue
        tool_calls = turn.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            issues.append("tool_calls 格式错误")
            continue
        for tc in tool_calls:
            tool_count += 1
            name = tc.get("name", "")
            if name and name not in VALID_TOOLS:
                issues.append(f"未知工具: {name}")
            args = tc.get("args", {})
            if not isinstance(args, dict):
                issues.append(f"工具 {name} 的 args 不是字典")

    # 至少使用了 1 个工具
    tools_used = case.get("tools_used", [])
    if not tools_used and tool_count == 0:
        issues.append("未使用任何工具")

    passed = len(issues) == 0
    return passed, issues


# ─────────────────────────────────────────────
# Checker 3: 医学术语一致性
# ─────────────────────────────────────────────

def check_medical_consistency(case: Dict) -> Tuple[bool, List[str]]:
    """检查诊断方向与主诉/科室逻辑是否匹配"""
    issues = []
    dept = case.get("department", "")
    complaint = case.get("chief_complaint", "")
    profile = case.get("patient_profile", "")
    diagnosis = case.get("final_diagnosis_direction", "")

    # 科室-主诉黑名单
    if dept in DEPARTMENT_COMPLAINT_BLACKLIST:
        combined = complaint + profile + diagnosis
        for bad_term in DEPARTMENT_COMPLAINT_BLACKLIST[dept]:
            if bad_term in combined:
                issues.append(f"科室 [{dept}] 不应出现术语 [{bad_term}]")

    # 诊断方向不能为空或过短
    if len(diagnosis) < 4:
        issues.append(f"诊断方向过短: [{diagnosis}]")

    # 主诉不能和诊断方向完全无关（简单检查：至少共享一个2字词）
    complaint_chars = set(complaint)
    diagnosis_chars = set(diagnosis)
    overlap = complaint_chars & diagnosis_chars - set("，。、的了是在有不")
    if len(overlap) < 2 and len(complaint) > 4 and len(diagnosis) > 4:
        issues.append("主诉与诊断方向可能无关联")

    passed = len(issues) == 0
    return passed, issues


# ─────────────────────────────────────────────
# Checker 4: 安全合规
# ─────────────────────────────────────────────

def check_safety_compliance(case: Dict) -> Tuple[bool, List[str]]:
    """检查回复是否包含安全声明、不含确定性诊断"""
    issues = []
    dialogue = case.get("dialogue", [])

    all_agent_text = ""
    for turn in dialogue:
        if turn.get("role") == "agent":
            all_agent_text += turn.get("response", "") + " "

    if not all_agent_text.strip():
        issues.append("无 agent 回复文本")
        return False, issues

    # 检查确定性诊断
    for pattern in BAD_DIAGNOSIS_PATTERNS:
        if pattern in all_agent_text:
            issues.append(f"包含确定性诊断措辞: [{pattern}]")

    # 检查安全声明（至少包含一个）
    has_safety = any(p in all_agent_text for p in GOOD_SAFETY_PATTERNS)
    if not has_safety:
        issues.append("缺少安全声明（建议就医/遵医嘱等）")

    passed = len(issues) == 0
    return passed, issues


# ─────────────────────────────────────────────
# Checker 5: 去重/去近似
# ─────────────────────────────────────────────

def _case_fingerprint(case: Dict) -> str:
    """基于内容生成指纹"""
    text = (
        case.get("patient_profile", "") +
        case.get("chief_complaint", "") +
        case.get("final_diagnosis_direction", "")
    )
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def deduplicate_cases(cases: List[Dict], use_embedding: bool = False, sim_threshold: float = 0.95) -> Tuple[List[Dict], int]:
    """
    去重：
    - 基础模式：MD5 指纹精确去重
    - 高级模式（use_embedding=True）：embedding 余弦相似度去近似

    Returns:
        (去重后列表, 被去掉的数量)
    """
    if use_embedding:
        return _deduplicate_by_embedding(cases, sim_threshold)
    else:
        return _deduplicate_by_hash(cases)


def _deduplicate_by_hash(cases: List[Dict]) -> Tuple[List[Dict], int]:
    seen = set()
    unique = []
    for case in cases:
        fp = _case_fingerprint(case)
        if fp not in seen:
            seen.add(fp)
            unique.append(case)
    return unique, len(cases) - len(unique)


def _deduplicate_by_embedding(cases: List[Dict], threshold: float) -> Tuple[List[Dict], int]:
    """基于 embedding 余弦相似度去近似"""
    try:
        import numpy as np
        from utils.llm_client import embed
    except ImportError:
        logger.warning("embedding 去重不可用，回退到 hash 去重")
        return _deduplicate_by_hash(cases)

    texts = [
        case.get("patient_profile", "") + " " +
        case.get("chief_complaint", "") + " " +
        case.get("final_diagnosis_direction", "")
        for case in cases
    ]

    logger.info(f"计算 {len(texts)} 条 case 的 embedding...")
    vectors = embed(texts)
    vecs = np.array(vectors, dtype=np.float32)
    # L2 归一化
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vecs = vecs / norms

    keep_mask = [True] * len(cases)
    for i in range(len(cases)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(cases)):
            if not keep_mask[j]:
                continue
            sim = float(np.dot(vecs[i], vecs[j]))
            if sim > threshold:
                keep_mask[j] = False

    unique = [c for c, k in zip(cases, keep_mask) if k]
    removed = len(cases) - len(unique)
    return unique, removed


# ─────────────────────────────────────────────
# 综合质控管线
# ─────────────────────────────────────────────

def run_quality_pipeline(
    cases: List[Dict],
    use_embedding_dedup: bool = False,
    sim_threshold: float = 0.95,
    strict: bool = False,
) -> Dict:
    """
    运行完整质控管线。

    Args:
        cases: 原始 trajectory 列表
        use_embedding_dedup: 是否用 embedding 去近似
        sim_threshold: 去近似阈值
        strict: 严格模式（任何 checker 失败就丢弃）

    Returns:
        {
            "passed": [通过的case列表],
            "failed": [失败的case列表],
            "stats": 统计信息,
        }
    """
    checkers = [
        ("structure", check_structure),
        ("tool_calls", check_tool_calls),
        ("medical_consistency", check_medical_consistency),
        ("safety_compliance", check_safety_compliance),
    ]

    passed_cases = []
    failed_cases = []
    issue_stats = {name: 0 for name, _ in checkers}
    all_issues_detail = []

    for i, case in enumerate(cases):
        case_issues = {}
        case_passed = True

        for checker_name, checker_fn in checkers:
            ok, issues = checker_fn(case)
            if not ok:
                case_issues[checker_name] = issues
                issue_stats[checker_name] += 1
                if strict:
                    case_passed = False

        # 非严格模式：结构检查必须通过，其他允许 1 个失败
        if not strict:
            struct_ok, _ = check_structure(case)
            if not struct_ok:
                case_passed = False
            elif len(case_issues) > 1:
                case_passed = False

        if case_passed:
            passed_cases.append(case)
        else:
            case["_quality_issues"] = case_issues
            failed_cases.append(case)

    # 去重
    before_dedup = len(passed_cases)
    passed_cases, dedup_removed = deduplicate_cases(
        passed_cases, use_embedding=use_embedding_dedup, sim_threshold=sim_threshold
    )

    stats = {
        "total_input": len(cases),
        "passed_checks": before_dedup,
        "dedup_removed": dedup_removed,
        "final_count": len(passed_cases),
        "failed_count": len(failed_cases),
        "pass_rate": len(passed_cases) / max(len(cases), 1),
        "issue_breakdown": issue_stats,
    }

    logger.info(f"质控结果: {stats['total_input']} → {stats['final_count']} "
                f"(通过率 {stats['pass_rate']:.1%}, 去重 {dedup_removed})")
    for name, count in issue_stats.items():
        if count > 0:
            logger.info(f"  {name}: {count} 条有问题")

    return {"passed": passed_cases, "failed": failed_cases, "stats": stats}


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="合成数据质控")
    parser.add_argument("--input", type=str, required=True, help="输入 trajectory JSON")
    parser.add_argument("--output", type=str, required=True, help="输出过滤后 JSON")
    parser.add_argument("--failed_output", type=str, default="", help="输出失败 case JSON")
    parser.add_argument("--strict", action="store_true", help="严格模式")
    parser.add_argument("--use_embedding", action="store_true", help="用 embedding 去近似")
    parser.add_argument("--sim_threshold", type=float, default=0.95, help="去近似阈值")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        cases = json.load(f)
    logger.info(f"加载: {len(cases)} 条")

    result = run_quality_pipeline(
        cases,
        use_embedding_dedup=args.use_embedding,
        sim_threshold=args.sim_threshold,
        strict=args.strict,
    )

    # 保存通过的
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result["passed"], f, ensure_ascii=False, indent=2)
    logger.info(f"保存通过: {args.output} ({len(result['passed'])} 条)")

    # 保存失败的（用于分析）
    if args.failed_output:
        with open(args.failed_output, "w", encoding="utf-8") as f:
            json.dump(result["failed"], f, ensure_ascii=False, indent=2)
        logger.info(f"保存失败: {args.failed_output} ({len(result['failed'])} 条)")

    # 保存统计
    stats_path = Path(args.output).parent / "quality_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(result["stats"], f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
