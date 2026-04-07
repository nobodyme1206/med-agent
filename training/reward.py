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
# 组合奖励函数（GRPO 使用）
# ─────────────────────────────────────────────

def med_agent_reward(
    completions: List[str],
    ground_truth: List[str] = None,
    expected_tools: List[List[str]] = None,
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

        r_task = task_completion_reward(completion, ground_truth=gt)
        r_tool = tool_accuracy_reward(completion, expected_tools=et)
        r_safe = safety_reward(completion)
        r_fmt = format_reward(completion)

        total = 0.30 * r_task + 0.20 * r_tool + 0.20 * r_safe + 0.30 * r_fmt
        rewards.append(total)

    return rewards
