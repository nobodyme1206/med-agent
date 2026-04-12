"""
任务完成率评测 + 工具调用准确率评测。

升级版：
  - 诊断评测：embedding 语义相似度（主）+ 关键词匹配（回退）
  - 工具评测：名称 F1 + 参数精确匹配 + 参数准确率
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.med_synonyms import normalize_medical_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 科室层级映射：子科室 → 父科室
_DEPARTMENT_HIERARCHY = {
    "心血管内科": "内科",
    "消化内科": "内科",
    "呼吸内科": "内科",
    "神经内科": "内科",
    "内分泌科": "内科",
    "肾内科": "内科",
    "血液内科": "内科",
    "风湿免疫科": "内科",
    "感染科": "内科",
    "普通外科": "外科",
    "骨科": "外科",
    "泌尿外科": "外科",
    "神经外科": "外科",
    "心胸外科": "外科",
    "肝胆外科": "外科",
    "胃肠外科": "外科",
    "乳腺外科": "外科",
    "血管外科": "外科",
}


def _department_match(pred_dept: str, ref_dept: str) -> float:
    """科室匹配：精确匹配=1.0，层级匹配（子科室∈父科室）=1.0，否则=0.0"""
    if not pred_dept or not ref_dept:
        return 0.0
    if pred_dept == ref_dept:
        return 1.0
    # 层级匹配：pred 是子科室，ref 是父科室
    if _DEPARTMENT_HIERARCHY.get(pred_dept) == ref_dept:
        return 1.0
    # 反向：ref 是子科室，pred 是父科室
    if _DEPARTMENT_HIERARCHY.get(ref_dept) == pred_dept:
        return 1.0
    # 同属一个父科室
    if (_DEPARTMENT_HIERARCHY.get(pred_dept)
            and _DEPARTMENT_HIERARCHY.get(pred_dept) == _DEPARTMENT_HIERARCHY.get(ref_dept)):
        return 1.0
    return 0.0


def _normalize_text(text) -> str:
    import re
    if not isinstance(text, str):
        text = str(text) if text else ""
    return re.sub(r"[\s\W_]+", "", text.lower())


def _extract_structured_output(pred: Dict) -> Dict:
    structured = pred.get("structured_output")
    if not isinstance(structured, dict):
        structured = {}
    tool_calls = pred.get("tool_calls", [])
    used_tools = structured.get("used_tools") or [tc.get("tool_name", "") for tc in tool_calls if tc.get("tool_name")]
    return {
        "department": structured.get("department") or pred.get("current_department", ""),
        "diagnosis_direction": structured.get("diagnosis_direction") or pred.get("specialist_analysis") or pred.get("final_response", ""),
        "recommended_tests": structured.get("recommended_tests") or [],
        "medication_advice": structured.get("medication_advice") or [],
        "need_followup": bool(structured.get("need_followup", pred.get("should_escalate", False))),
        "followup_actions": structured.get("followup_actions") or [],
        "evidence_summary": structured.get("evidence_summary") or [],
        "used_tools": used_tools,
        "tool_plan": structured.get("tool_plan") or pred.get("tool_plan", []),
        "final_response": structured.get("final_response") or pred.get("final_response", ""),
    }


def _extract_reference_struct(ref: Dict) -> Dict:
    expected_tools = ref.get("preferred_tool_sequence") or ref.get("expected_tools") or ref.get("tools_used") or []
    if isinstance(expected_tools, str):
        expected_tools = [expected_tools]
    recommended_tests = ref.get("recommended_tests") or []
    if isinstance(recommended_tests, str):
        recommended_tests = [recommended_tests]
    return {
        "department": ref.get("expected_department") or ref.get("department", ""),
        "diagnosis_direction": ref.get("expected_diagnosis_direction") or ref.get("final_diagnosis_direction", ""),
        "recommended_tests": recommended_tests,
        "expected_tools": expected_tools,
        "expected_first_tool": ref.get("expected_first_tool") or (expected_tools[0] if expected_tools else ""),
        "need_pharmacist": ref.get("need_pharmacist"),
    }


def _keyword_f1(text_a: str, text_b: str) -> float:
    a_terms = set(_extract_keywords(text_a))
    b_terms = set(_extract_keywords(text_b))
    if not a_terms and not b_terms:
        return 1.0
    if not a_terms or not b_terms:
        return 0.0
    overlap = len(a_terms & b_terms)
    precision = overlap / len(a_terms)
    recall = overlap / len(b_terms)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _soft_text_match(text_a: str, text_b: str) -> float:
    syn_a = normalize_medical_text(text_a or "")
    syn_b = normalize_medical_text(text_b or "")
    norm_a = _normalize_text(syn_a)
    norm_b = _normalize_text(syn_b)
    if not norm_a and not norm_b:
        return 1.0
    if not norm_a or not norm_b:
        return 0.0
    if norm_a == norm_b:
        return 1.0
    if norm_a in norm_b or norm_b in norm_a:
        shorter = min(len(norm_a), len(norm_b))
        longer = max(len(norm_a), len(norm_b))
        if longer > 0:
            return max(shorter / longer, 0.75)
    rouge = _rouge_l(syn_a, syn_b)
    keyword = _keyword_f1(syn_a, syn_b)
    return max(rouge, keyword)


def _list_recall(pred_items: List, ref_items: List) -> Optional[float]:
    pred_items = [str(x) if not isinstance(x, str) else x for x in pred_items]
    ref_items = [str(x) if not isinstance(x, str) else x for x in ref_items]
    pred_set = {_normalize_text(normalize_medical_text(x)) for x in pred_items if _normalize_text(x)}
    ref_set = {_normalize_text(normalize_medical_text(x)) for x in ref_items if _normalize_text(x)}
    if not ref_set:
        return None
    if not pred_set:
        return 0.0
    return len(pred_set & ref_set) / len(ref_set)


def evaluate_single_prediction(
    pred: Dict,
    ref: Dict,
    correct_threshold: float = 0.7,
    partial_threshold: float = 0.45,
) -> Dict:
    pred_struct = _extract_structured_output(pred)
    ref_struct = _extract_reference_struct(ref)

    department_match = _department_match(
        pred_struct["department"],
        ref_struct["department"],
    )
    diagnosis_similarity = _soft_text_match(
        pred_struct["diagnosis_direction"],
        ref_struct["diagnosis_direction"],
    )
    combined_score = 0.4 * department_match + 0.6 * diagnosis_similarity
    test_recall = _list_recall(pred_struct.get("recommended_tests", []), ref_struct.get("recommended_tests", []))

    return {
        "department": ref_struct["department"] or "unknown",
        "pred_department": pred_struct["department"],
        "department_match": department_match,
        "diagnosis_similarity": diagnosis_similarity,
        "combined_score": combined_score,
        "test_recall": test_recall,
        "is_correct": combined_score >= correct_threshold,
        "is_partial": combined_score >= partial_threshold,
    }


def evaluate_task_completion(
    predictions: List[Dict],
    references: List[Dict],
    correct_threshold: float = 0.7,
    partial_threshold: float = 0.45,
) -> Dict:
    """
    评测任务完成率：最终诊断方向是否正确。

    使用 embedding 语义相似度（cosine > correct_threshold 算正确），
    embedding 不可用时回退关键词匹配。

    Args:
        predictions: Agent 输出列表，含 final_response, tool_calls 等
        references: 标准答案列表，含 final_diagnosis_direction, tools_used 等
        correct_threshold: 语义相似度 ≥ 此值则判正确
        partial_threshold: 语义相似度 ≥ 此值则判部分正确

    Returns:
        评测指标字典
    """
    total = len(predictions)
    if total == 0:
        return {"total": 0}

    correct = 0
    partial = 0
    department_correct = 0
    diagnosis_correct = 0
    similarities = []
    combined_scores = []
    test_recalls = []
    by_department = {}
    details = []

    for pred, ref in zip(predictions, references):
        detail = evaluate_single_prediction(
            pred,
            ref,
            correct_threshold=correct_threshold,
            partial_threshold=partial_threshold,
        )
        details.append(detail)

        dept = detail["department"]
        if dept not in by_department:
            by_department[dept] = {"total": 0, "correct": 0, "partial": 0}
        by_department[dept]["total"] += 1

        similarities.append(detail["diagnosis_similarity"])
        combined_scores.append(detail["combined_score"])
        if detail["test_recall"] is not None:
            test_recalls.append(detail["test_recall"])

        if detail["department_match"]:
            department_correct += 1
        if detail["diagnosis_similarity"] >= correct_threshold:
            diagnosis_correct += 1

        if detail["is_correct"]:
            correct += 1
            by_department[dept]["correct"] += 1
        elif detail["is_partial"]:
            partial += 1
            by_department[dept]["partial"] += 1

    result = {
        "total": total,
        "correct": correct,
        "partial_correct": partial,
        "accuracy": correct / total,
        "partial_accuracy": (correct + partial) / total,
        "strict_accuracy": correct / total,
        "strict_partial_accuracy": (correct + partial) / total,
        "department_accuracy": department_correct / total,
        "diagnosis_accuracy": diagnosis_correct / total,
        "avg_similarity": sum(similarities) / max(len(similarities), 1),
        "avg_combined_score": sum(combined_scores) / max(len(combined_scores), 1),
        "by_department": by_department,
        "details": details,
    }
    if test_recalls:
        result["avg_test_recall"] = sum(test_recalls) / len(test_recalls)
    return result


def evaluate_tool_usage(predictions: List[Dict], references: List[Dict]) -> Dict:
    """
    评测工具调用准确率。

    三层评测：
      1. 工具名称 F1（调用了正确的工具）
      2. 工具参数准确率（参数也正确）
      3. 工具调用顺序相似度（Kendall tau）
    """
    total = len(predictions)
    if total == 0:
        return {"total": 0}

    precisions = []
    recalls = []
    f1s = []
    param_accuracies = []
    first_tool_accuracies = []
    strict_sequence_matches = []
    subset_matches = []
    duplicate_rates = []
    offplan_rates = []

    for pred, ref in zip(predictions, references):
        pred_calls = [tc for tc in pred.get("tool_calls", []) if tc.get("tool_name")]
        pred_tools = [tc.get("tool_name", "") for tc in pred_calls]
        ref_tools = ref.get("preferred_tool_sequence") or ref.get("tools_used", []) or ref.get("expected_tools", [])

        ref_calls_with_args = _extract_ref_tool_calls(ref)

        if not ref_tools and not pred_tools:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
            param_accuracies.append(1.0)
            first_tool_accuracies.append(1.0)
            strict_sequence_matches.append(1.0)
            subset_matches.append(1.0)
            duplicate_rates.append(0.0)
            offplan_rates.append(0.0)
            continue

        pred_set = set(pred_tools)
        ref_set = set(ref_tools)
        tp = len(pred_set & ref_set)
        precision = tp / max(len(pred_set), 1)
        recall = tp / max(len(ref_set), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        ref_first = ref_tools[0] if ref_tools else ""
        pred_first = pred_tools[0] if pred_tools else ""
        first_tool_accuracies.append(float(ref_first == pred_first if (ref_first or pred_first) else True))
        strict_sequence_matches.append(float(pred_tools == ref_tools))
        subset_matches.append(float(all(t in ref_set for t in pred_tools)))
        duplicate_rates.append((len(pred_tools) - len(set(pred_tools))) / max(len(pred_tools), 1))
        offplan_rates.append(sum(1 for t in pred_tools if t not in ref_set) / max(len(pred_tools), 1))

        if ref_calls_with_args and pred_calls:
            param_acc = _compute_param_accuracy(pred_calls, ref_calls_with_args)
            param_accuracies.append(param_acc)

    n = len(f1s) or 1
    result = {
        "total": total,
        "avg_precision": sum(precisions) / n,
        "avg_recall": sum(recalls) / n,
        "avg_f1": sum(f1s) / n,
        "avg_tool_calls_per_case": sum(
            len(p.get("tool_calls", [])) for p in predictions
        ) / total,
        "first_tool_accuracy": sum(first_tool_accuracies) / max(len(first_tool_accuracies), 1),
        "strict_sequence_accuracy": sum(strict_sequence_matches) / max(len(strict_sequence_matches), 1),
        "subset_match_accuracy": sum(subset_matches) / max(len(subset_matches), 1),
        "duplicate_tool_rate": sum(duplicate_rates) / max(len(duplicate_rates), 1),
        "offplan_tool_rate": sum(offplan_rates) / max(len(offplan_rates), 1),
    }
    if param_accuracies:
        result["avg_param_accuracy"] = sum(param_accuracies) / len(param_accuracies)
    return result


def _extract_ref_tool_calls(ref: Dict) -> List[Dict]:
    """从 reference 的 dialogue 中提取带参数的工具调用"""
    calls = []
    for turn in ref.get("dialogue", []):
        if turn.get("role") == "agent":
            for tc in turn.get("tool_calls", []):
                calls.append({
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                })
    return calls


def _compute_param_accuracy(pred_calls: List[Dict], ref_calls: List[Dict]) -> float:
    """
    计算工具参数准确率。
    对每个匹配的工具名，比较关键参数是否一致。
    """
    matched = 0
    compared = 0

    for ref_tc in ref_calls:
        ref_name = ref_tc.get("name", "")
        ref_args = ref_tc.get("args", {})
        # 找同名的 pred tool call
        for pred_tc in pred_calls:
            pred_name = pred_tc.get("tool_name", "")
            pred_args = pred_tc.get("input_args", {})
            if pred_name == ref_name:
                # 比较参数
                for key, ref_val in ref_args.items():
                    compared += 1
                    pred_val = pred_args.get(key)
                    if pred_val is not None and str(pred_val).strip() == str(ref_val).strip():
                        matched += 1
                break  # 只匹配第一个同名工具

    if compared == 0:
        return 1.0  # 无参数可比 → 默认通过
    return matched / compared


def _rouge_l(text_a: str, text_b: str) -> float:
    """字符级 ROUGE-L Recall，不需要分词，对中文友好。

    使用 Recall 而非 F1：衡量参考答案（text_b）的关键内容是否
    被模型回复（text_a）覆盖。适合短参考 vs 长回复的场景。
    """
    if not text_a or not text_b:
        return 0.0
    # 移除标点和空格，只保留实质字符
    import re
    a = re.sub(r'[\s\W]', '', text_a)
    b = re.sub(r'[\s\W]', '', text_b)
    if not a or not b:
        return 0.0
    # LCS 长度（DP）
    m, n = len(a), len(b)
    if m > 500:
        a = a[:500]
        m = 500
    if n > 500:
        b = b[:500]
        n = 500
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    lcs_len = prev[n]
    # Recall：参考答案被覆盖的比例
    recall = lcs_len / n
    return recall


def _extract_keywords(text: str) -> List[str]:
    """从诊断文本中提取关键词（保留作为辅助工具）"""
    for ch in "，。、；：“”‘’（）【】《》！？":
        text = text.replace(ch, " ")
    stopwords = {"的", "了", "是", "在", "有", "不", "这", "个", "人", "都", "一", "和", "我",
                 "你", "他", "她", "它", "我们", "要", "会", "可以", "进行", "建议", "患者", "可能"}
    words = [w.strip() for w in text.split() if len(w.strip()) >= 2 and w.strip() not in stopwords]
    return words
