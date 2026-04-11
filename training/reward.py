"""
GRPO 奖励函数：多维度组合奖励。
- 任务完成奖励（诊断是否正确）
- 工具调用准确性（是否调了正确的工具、传了正确的参数）
- 安全合规（是否遵守安全红线）
- 格式正确性（是否遵循 ReAct 格式）
"""

import re
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def _extract_tool_calls(completion: str) -> List[Dict[str, Any]]:
    tool_calls = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', completion, re.DOTALL)
    parsed = []
    for tc_str in tool_calls:
        try:
            tc = json.loads(tc_str)
            args = tc.get("args", tc.get("params", tc.get("parameters", {})))
            parsed.append({"name": tc.get("name", ""), "args": args if isinstance(args, dict) else {}})
        except json.JSONDecodeError:
            continue
    return parsed


def _extract_structured_output(completion: str) -> Dict[str, Any]:
    match = re.search(r'<structured_output>(.*?)</structured_output>', completion, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return {}


def _normalize_text(text: str) -> str:
    return re.sub(r'[\s\W_]+', '', (text or '').lower())


def _text_match_score(pred: str, ref: str) -> float:
    pred_norm = _normalize_text(pred)
    ref_norm = _normalize_text(ref)
    if not pred_norm and not ref_norm:
        return 1.0
    if not pred_norm or not ref_norm:
        return 0.0
    if pred_norm == ref_norm:
        return 1.0
    if pred_norm in ref_norm or ref_norm in pred_norm:
        shorter = min(len(pred_norm), len(ref_norm))
        longer = max(len(pred_norm), len(ref_norm))
        return max(shorter / max(longer, 1), 0.75)
    ref_terms = set(re.findall(r'[\u4e00-\u9fff]{2,4}', ref))
    if ref_terms:
        overlap = sum(1 for term in ref_terms if term in pred)
        return overlap / len(ref_terms)
    return 0.0


# ─────────────────────────────────────────────
# 子奖励函数
# ─────────────────────────────────────────────

def task_completion_reward(completion: str, ground_truth: str = "", **kwargs) -> float:
    """
    任务完成奖励：检查最终回复是否包含关键诊断信息。

    v2：使用中文 n-gram 匹配替代空格分词，增加渐进式评分：
    - 回复长度和实质性内容 → 基础分
    - <response> 标签内有实质内容 → 加分
    - 与 ground_truth 的中文关键词匹配 → 核心分
    """
    if not completion.strip():
        return 0.0

    score = 0.0

    # 基础分：回复长度（鼓励生成实质性内容）
    if len(completion) > 50:
        score += 0.05
    if len(completion) > 150:
        score += 0.05

    # <response> 标签内有实质内容
    response_match = re.search(r'<response>(.*?)</response>', completion, re.DOTALL)
    if response_match and len(response_match.group(1).strip()) > 30:
        score += 0.15

    if not ground_truth:
        return min(score + 0.3, 1.0)

    # 中文 n-gram 关键词匹配（2-4 字词提取）
    gt_terms = set(re.findall(r'[\u4e00-\u9fff]{2,4}', ground_truth))
    if not gt_terms:
        return min(score + 0.3, 1.0)

    comp_text = completion.lower()
    matched = sum(1 for term in gt_terms if term in comp_text)
    match_ratio = matched / len(gt_terms)
    score += 0.75 * match_ratio

    return min(score, 1.0)


def structured_output_reward(completion: str, structured_output_target: Dict = None, ground_truth: str = "", **kwargs) -> float:
    structured = _extract_structured_output(completion)
    if not structured:
        return 0.0

    score = 0.2
    diagnosis_target = ""
    if isinstance(structured_output_target, dict):
        diagnosis_target = structured_output_target.get("diagnosis_direction", "")
        dept_target = structured_output_target.get("department", "")
        if dept_target and structured.get("department") == dept_target:
            score += 0.2
    if not diagnosis_target:
        diagnosis_target = ground_truth

    if diagnosis_target:
        score += 0.4 * _text_match_score(structured.get("diagnosis_direction", ""), diagnosis_target)

    if isinstance(structured.get("recommended_tests", []), list):
        score += 0.1
    if isinstance(structured.get("evidence_summary", []), list):
        score += 0.1

    final_response = structured.get("final_response", "")
    if isinstance(final_response, str) and len(final_response.strip()) > 20:
        score += 0.1

    return min(score, 1.0)


def plan_adherence_reward(
    completion: str,
    expected_tools: List[str] = None,
    expected_first_tool: str = "",
    preferred_tool_sequence: List[str] = None,
    tool_plan: List[str] = None,
    **kwargs,
) -> float:
    calls = _extract_tool_calls(completion)
    called_names = [tc.get("name", "") for tc in calls if tc.get("name")]
    reference = preferred_tool_sequence or tool_plan or expected_tools or []
    if not reference:
        return 0.8 if not called_names else 0.5
    if not called_names:
        return 0.0

    ref_set = set(reference)
    hit_rate = sum(1 for name in called_names if name in ref_set) / len(called_names)
    coverage = sum(1 for name in reference if name in called_names) / len(reference)
    score = 0.4 * hit_rate + 0.4 * coverage

    if expected_first_tool:
        score += 0.2 if called_names[0] == expected_first_tool else 0.0
    elif reference:
        score += 0.2 if called_names[0] == reference[0] else 0.0

    return min(score, 1.0)


def duplicate_control_reward(completion: str, **kwargs) -> float:
    calls = _extract_tool_calls(completion)
    if not calls:
        return 1.0
    names = [tc.get("name", "") for tc in calls if tc.get("name")]
    if not names:
        return 1.0
    duplicate_rate = (len(names) - len(set(names))) / len(names)
    return max(1.0 - duplicate_rate * 2.0, 0.0)


def tool_accuracy_reward(completion: str, expected_tools: List[str] = None, **kwargs) -> float:
    """
    工具调用准确性奖励。

    v2：分层评分，重点奖励格式正确性而非精确工具名称匹配：
    - 有 <tool_call> 标签 → 基础分
    - JSON 格式正确（name + args/params）→ 加分
    - 参数非空且合理 → 加分
    - 精确匹配 expected_tools → 额外加分
    """
    tool_calls = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', completion, re.DOTALL)

    if not tool_calls:
        if expected_tools:
            return 0.1  # 该调工具但没调
        return 0.5  # 不需要工具，没调也行

    score = 0.0
    valid_calls = 0
    called_names = set()

    for tc_str in tool_calls:
        try:
            tc = json.loads(tc_str)
            has_name = "name" in tc and isinstance(tc["name"], str) and len(tc["name"]) > 0
            has_args = "args" in tc or "params" in tc or "parameters" in tc
            if has_name and has_args:
                valid_calls += 1
                called_names.add(tc["name"])
                # 参数非空加分
                args = tc.get("args", tc.get("params", tc.get("parameters", {})))
                if isinstance(args, dict) and len(args) > 0:
                    score += 0.1
            elif has_name:
                valid_calls += 0.5
                called_names.add(tc["name"])
        except json.JSONDecodeError:
            pass

    # 格式正确率
    format_ratio = valid_calls / max(len(tool_calls), 1)
    score += 0.5 * format_ratio

    # 精确匹配 expected_tools（额外奖励，不是必须）
    if expected_tools:
        expected_set = set(expected_tools)
        tp = len(expected_set & called_names)
        if tp > 0:
            score += 0.3 * (tp / len(expected_set))

    return min(score, 1.0)


def safety_reward(completion: str, **kwargs) -> float:
    """
    安全合规奖励。

    检查项：
    - 不做确定性诊断（用"可能"、"建议"等措辞）
    - 不开具处方
    - 包含就医建议
    - 不包含有害内容
    """
    score = 1.0

    # 确定性诊断惩罚
    definitive_patterns = ["你得了", "你患有", "确诊为", "诊断为"]
    for pat in definitive_patterns:
        if pat in completion:
            score -= 0.3
            break

    # 开处方惩罚
    prescription_patterns = ["处方：", "给你开", "请服用以下药物"]
    for pat in prescription_patterns:
        if pat in completion:
            score -= 0.3
            break

    # 包含安全提示加分
    safe_patterns = ["建议就医", "遵医嘱", "仅供参考", "进一步检查"]
    if any(pat in completion for pat in safe_patterns):
        score += 0.1

    return max(min(score, 1.0), 0.0)


def format_reward(completion: str, **kwargs) -> float:
    """
    格式正确性奖励：检查是否遵循 ReAct 格式。

    评分标准：
    - 包含 <think>...</think> → +0.3
    - 包含 <response>...</response> → +0.4
    - 工具调用格式正确 → +0.3
    """
    score = 0.0

    if "<think>" in completion and "</think>" in completion:
        score += 0.3

    if "<response>" in completion and "</response>" in completion:
        score += 0.4

    tool_calls = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', completion, re.DOTALL)
    if tool_calls:
        valid = sum(1 for tc in tool_calls if _is_valid_tool_call(tc))
        score += 0.3 * (valid / max(len(tool_calls), 1))
    elif "<response>" in completion:
        # 没有工具调用但有 response，也给分
        score += 0.3

    return min(score, 1.0)


def _is_valid_tool_call(tc_str: str) -> bool:
    """检查工具调用 JSON 是否有效（接受 args/params/parameters）"""
    try:
        tc = json.loads(tc_str) if isinstance(tc_str, str) else tc_str
        return "name" in tc and ("args" in tc or "params" in tc or "parameters" in tc)
    except (json.JSONDecodeError, TypeError):
        return False


# ─────────────────────────────────────────────
# PARM 架构对齐奖励（P2 新增）
# ─────────────────────────────────────────────

def reasoning_chain_reward(completion: str, **kwargs) -> float:
    """
    推理链质量奖励：评估模型是否生成了完整的推理过程。

    评分维度：
    - 有 <think> 标签且内容 > 50 字 → 基础分
    - 包含鉴别诊断关键词（"考虑"、"排除"、"鉴别"）→ 加分
    - 包含证据引用（工具结果引用）→ 加分
    - 推理步骤 ≥ 2（用序号/分号分隔）→ 加分
    - 有明确结论指向 → 加分
    """
    score = 0.0

    # 提取 <think> 内容
    think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""

    if not think_text:
        return 0.0

    # 基础分：推理内容长度
    if len(think_text) > 30:
        score += 0.15
    if len(think_text) > 80:
        score += 0.10

    # 鉴别诊断关键词
    differential_keywords = ["考虑", "排除", "鉴别", "可能性", "假设", "不排除", "需要排查"]
    diff_hits = sum(1 for kw in differential_keywords if kw in think_text)
    score += min(diff_hits * 0.08, 0.20)

    # 证据引用（引用了工具结果或检查结果）
    evidence_keywords = ["根据", "检查显示", "结果提示", "指南建议", "文献", "证据"]
    evidence_hits = sum(1 for kw in evidence_keywords if kw in think_text)
    score += min(evidence_hits * 0.08, 0.20)

    # 推理步骤数（序号、分号、换行分隔）
    step_markers = re.findall(r'[1-9][.、)）]|[；;\n]', think_text)
    if len(step_markers) >= 2:
        score += 0.15
    elif len(step_markers) >= 1:
        score += 0.08

    # 明确结论指向
    conclusion_keywords = ["综合", "因此", "初步判断", "总结", "诊断方向", "最终"]
    if any(kw in think_text for kw in conclusion_keywords):
        score += 0.15

    return min(score, 1.0)


def reflection_quality_reward(completion: str, **kwargs) -> float:
    """
    Reflection 质量奖励：评估模型是否对自身输出进行了审查和修正。

    评分维度：
    - 有自我检查/验证的迹象 → 基础分
    - 包含修正措辞 → 加分
    - 工具结果和结论一致性 → 加分
    - 对不确定性有表述 → 加分
    """
    score = 0.0
    full_text = completion

    # 提取 <think> 和 <response> 内容
    think_match = re.search(r'<think>(.*?)</think>', full_text, re.DOTALL)
    response_match = re.search(r'<response>(.*?)</response>', full_text, re.DOTALL)
    think_text = think_match.group(1) if think_match else ""
    response_text = response_match.group(1) if response_match else ""
    combined = think_text + " " + response_text

    if not combined.strip():
        return 0.0

    # 自我检查迹象
    check_keywords = ["验证", "核实", "检查一下", "再看看", "回顾", "确认", "审查"]
    check_hits = sum(1 for kw in check_keywords if kw in combined)
    score += min(check_hits * 0.10, 0.25)

    # 修正措辞（说明模型有自我纠错能力）
    correction_keywords = ["修正", "补充", "更正", "需要注意", "之前遗漏", "进一步", "另外"]
    correction_hits = sum(1 for kw in correction_keywords if kw in combined)
    score += min(correction_hits * 0.10, 0.25)

    # 不确定性表达（校准意识）
    uncertainty_keywords = ["可能", "不确定", "有待", "需进一步", "尚不能确定", "概率"]
    unc_hits = sum(1 for kw in uncertainty_keywords if kw in combined)
    score += min(unc_hits * 0.08, 0.25)

    # 工具结果引用（证据有据可查）
    tool_results = re.findall(r'<observation>(.*?)</observation>', full_text, re.DOTALL)
    if tool_results:
        # 检查结论中是否引用了工具关键片段
        for obs in tool_results:
            obs_keywords = re.findall(r'[\u4e00-\u9fff]{2,4}', obs[:100])
            if obs_keywords:
                referenced = sum(1 for kw in obs_keywords[:5] if kw in response_text)
                if referenced >= 1:
                    score += 0.10
                    break

    # 安全声明存在性（reflection 的核心价值）
    safety_keywords = ["建议就医", "遵医嘱", "仅供参考", "专业医生"]
    if any(kw in response_text for kw in safety_keywords):
        score += 0.15

    return min(score, 1.0)


# ─────────────────────────────────────────────
# 组合奖励函数（GRPO 使用）
# ─────────────────────────────────────────────

def med_agent_reward(
    completions: List[str],
    ground_truth: List[str] = None,
    expected_tools: List[List[str]] = None,
    expected_first_tool: List[str] = None,
    preferred_tool_sequence: List[List[str]] = None,
    tool_plan: List[List[str]] = None,
    structured_output_target: List[Dict] = None,
    **kwargs,
) -> List[float]:
    """
    MedAgent 组合奖励函数，供 GRPOTrainer 使用。

    权重（v2，重新平衡使各维度均可获得有效分数）：
    - 格式正确: 0.30（最可控，产生最大方差）
    - 任务完成: 0.30（中文 n-gram 匹配，渐进评分）
    - 工具调用: 0.20（奖励格式，不强求精确名称）
    - 安全合规: 0.20（保持不变）

    Args:
        completions: 模型生成的回复列表
        ground_truth: 标准诊断（可选）
        expected_tools: 标准工具调用序列（可选）

    Returns:
        每个 completion 的奖励分数列表
    """
    rewards = []
    for i, completion in enumerate(completions):
        gt = ground_truth[i] if ground_truth and i < len(ground_truth) else ""
        et = expected_tools[i] if expected_tools and i < len(expected_tools) else None
        eft = expected_first_tool[i] if expected_first_tool and i < len(expected_first_tool) else ""
        pref = preferred_tool_sequence[i] if preferred_tool_sequence and i < len(preferred_tool_sequence) else None
        plan = tool_plan[i] if tool_plan and i < len(tool_plan) else None
        struct_target = structured_output_target[i] if structured_output_target and i < len(structured_output_target) else None

        r_task = task_completion_reward(completion, ground_truth=gt)
        r_tool = tool_accuracy_reward(completion, expected_tools=et)
        r_safe = safety_reward(completion)
        r_fmt = format_reward(completion)
        r_struct = structured_output_reward(completion, structured_output_target=struct_target, ground_truth=gt)
        r_plan = plan_adherence_reward(
            completion,
            expected_tools=et,
            expected_first_tool=eft,
            preferred_tool_sequence=pref,
            tool_plan=plan,
        )
        r_dup = duplicate_control_reward(completion)
        r_reasoning = reasoning_chain_reward(completion)
        r_reflection = reflection_quality_reward(completion)

        total = (
            0.18 * r_task
            + 0.10 * r_tool
            + 0.10 * r_safe
            + 0.15 * r_fmt
            + 0.12 * r_struct
            + 0.10 * r_plan
            + 0.05 * r_dup
            + 0.12 * r_reasoning
            + 0.08 * r_reflection
        )
        rewards.append(total)

    return rewards
