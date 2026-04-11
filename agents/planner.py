import json
import logging
from typing import Dict, List

from graph.state import AgentState
from utils.llm_client import chat

logger = logging.getLogger(__name__)

_ALLOWED_TOOLS = [
    "search_guidelines",
    "interpret_lab_result",
    "search_drug",
    "search_by_indication",
    "check_drug_interaction",
]

_PLANNER_SYSTEM_PROMPT = """你是一位资深医学规划器（Medical Reasoning Planner）。你的任务是根据患者描述和分诊结果，进行医学推理并生成诊疗规划。

你需要完成以下推理步骤：
1. **鉴别诊断假设**：根据症状生成 2-4 个可能的诊断方向（按可能性排序）
2. **信息缺口分析**：当前信息不足以确认诊断的部分（需要什么检查/问诊）
3. **推理链**：从症状到假设的推理过程（一段简洁的文字）
4. **工具规划**：选择必要的工具来验证假设或填补信息缺口
5. **验证标准**：Specialist 的分析需要满足哪些条件才算合格

可用工具：search_guidelines、interpret_lab_result、search_drug、search_by_indication、check_drug_interaction。

请只输出 JSON，格式如下：
{
  "problem_type": "symptom_only | lab_related | drug_related | multi_factor",
  "differential_hypotheses": ["最可能的诊断1", "需排查的诊断2"],
  "information_gaps": ["缺少XX检查结果", "需确认XX病史"],
  "reasoning_chain": "患者表现为XX症状→结合XX因素→首先考虑XX，需排查XX",
  "tool_plan": ["tool_a", "tool_b"],
  "verification_criteria": ["分析需覆盖至少一个鉴别假设", "需引用指南依据"],
  "need_pharmacist": true,
  "expected_evidence": ["指南依据", "检验解读"],
  "plan_summary": "一句话概括计划"
}

要求：
1. differential_hypotheses 控制在 2-4 个，按可能性排序。
2. information_gaps 列出当前无法确认的关键信息。
3. reasoning_chain 要体现从症状到假设的逻辑推导过程。
4. tool_plan 只保留必要工具，控制在 1-4 个。
5. verification_criteria 是给下游 Reflection Agent 的检查清单。
6. 若问题涉及药物/副作用/相互作用，设置 need_pharmacist=true。"""


def _dedupe(seq: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in seq:
        if item and item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _last_user_message(state: AgentState) -> str:
    user_messages = [m.get("content", "") for m in state.get("messages", []) if m.get("role") == "user"]
    return user_messages[-1] if user_messages else ""


def _heuristic_plan(user_text: str, patient_info: str) -> Dict:
    text = f"{user_text}\n{patient_info}"
    tool_plan: List[str] = []
    expected_evidence: List[str] = []
    problem_type = "symptom_only"
    need_pharmacist = False
    differential_hypotheses: List[str] = []
    information_gaps: List[str] = []

    drug_keywords = [
        "药", "服用", "吃了", "用药", "剂量", "副作用", "不良反应", "相互作用",
        "阿司匹林", "布洛芬", "二甲双胍", "硝苯地平", "抗生素", "降压药",
    ]
    lab_keywords = [
        "化验", "检验", "血糖", "血压", "mmol", "HbA1c", "血常规", "白带", "尿酸",
        "黑便", "胸片", "胃镜", "结果", "指标",
    ]

    if any(kw in text for kw in drug_keywords):
        problem_type = "drug_related"
        need_pharmacist = True
        tool_plan.extend(["search_drug", "search_by_indication"])
        expected_evidence.extend(["药物适应症", "用药注意事项"])
        information_gaps.append("具体用药方案和剂量")

    if any(kw in text for kw in lab_keywords):
        if problem_type == "drug_related":
            problem_type = "multi_factor"
        else:
            problem_type = "lab_related"
        tool_plan.append("interpret_lab_result")
        expected_evidence.append("检验结果解读")
        information_gaps.append("完整检验报告数据")

    if any(kw in text for kw in ["同时", "一起", "联用", "合用", "交互", "相互作用"]):
        need_pharmacist = True
        if problem_type == "symptom_only":
            problem_type = "drug_related"
        tool_plan.append("check_drug_interaction")
        expected_evidence.append("药物相互作用风险")

    tool_plan.insert(0, "search_guidelines")
    expected_evidence.insert(0, "指南依据")

    tool_plan = [t for t in _dedupe(tool_plan) if t in _ALLOWED_TOOLS][:4]
    expected_evidence = _dedupe(expected_evidence)[:4]

    if not tool_plan:
        tool_plan = ["search_guidelines"]
    if not expected_evidence:
        expected_evidence = ["指南依据"]
    if not information_gaps:
        information_gaps = ["详细病史和体检信息"]
    if not differential_hypotheses:
        differential_hypotheses = ["待专科分析后确定"]

    plan_summary = "先检索指南，再结合症状形成专科分析。"
    reasoning_chain = "根据患者描述进行初步判断，需结合指南和工具进一步分析。"
    if problem_type == "lab_related":
        plan_summary = "先检索指南并解读关键检验指标，再输出诊断方向。"
        reasoning_chain = "患者提供了检验相关信息→需解读检验指标→结合指南确定诊断方向。"
    elif problem_type == "drug_related":
        plan_summary = "先做专科判断，再补充药物检索和适应症匹配。"
        reasoning_chain = "患者涉及用药问题→需检索药物信息→评估用药合理性。"
    elif problem_type == "multi_factor":
        plan_summary = "先检索指南和检验结果，再补充药学分析，最后汇总。"
        reasoning_chain = "患者同时涉及检验和用药→需多维度分析→检验+药物+指南综合判断。"

    verification_criteria = [
        "分析需覆盖至少一个鉴别诊断假设",
        "需引用指南或工具返回的证据",
        "建议需具备可操作性",
    ]

    return {
        "problem_type": problem_type,
        "differential_hypotheses": differential_hypotheses,
        "information_gaps": information_gaps,
        "reasoning_chain": reasoning_chain,
        "tool_plan": tool_plan,
        "verification_criteria": verification_criteria,
        "need_pharmacist": need_pharmacist,
        "expected_evidence": expected_evidence,
        "plan_summary": plan_summary,
    }


def _ensure_list(value, fallback=None) -> List[str]:
    if value is None:
        return fallback or []
    if isinstance(value, str):
        return [value]
    return list(value)


def _merge_plan(model_plan: Dict, heuristic_plan: Dict) -> Dict:
    tool_plan = model_plan.get("tool_plan") or heuristic_plan["tool_plan"]
    if isinstance(tool_plan, str):
        tool_plan = [tool_plan]
    tool_plan = [t for t in _dedupe(tool_plan or []) if t in _ALLOWED_TOOLS]
    if not tool_plan:
        tool_plan = heuristic_plan["tool_plan"]

    expected_evidence = model_plan.get("expected_evidence") or heuristic_plan["expected_evidence"]
    if isinstance(expected_evidence, str):
        expected_evidence = [expected_evidence]
    expected_evidence = _dedupe(expected_evidence or heuristic_plan["expected_evidence"])

    differential_hypotheses = _ensure_list(
        model_plan.get("differential_hypotheses"),
        heuristic_plan["differential_hypotheses"],
    )
    information_gaps = _ensure_list(
        model_plan.get("information_gaps"),
        heuristic_plan["information_gaps"],
    )
    verification_criteria = _ensure_list(
        model_plan.get("verification_criteria"),
        heuristic_plan["verification_criteria"],
    )

    return {
        "problem_type": model_plan.get("problem_type") or heuristic_plan["problem_type"],
        "differential_hypotheses": differential_hypotheses[:4],
        "information_gaps": information_gaps,
        "reasoning_chain": model_plan.get("reasoning_chain") or heuristic_plan["reasoning_chain"],
        "tool_plan": tool_plan,
        "verification_criteria": verification_criteria,
        "need_pharmacist": bool(model_plan.get("need_pharmacist", heuristic_plan["need_pharmacist"])),
        "expected_evidence": expected_evidence,
        "plan_summary": model_plan.get("plan_summary") or heuristic_plan["plan_summary"],
    }


def plan_consultation(state: AgentState) -> AgentState:
    user_text = _last_user_message(state)
    patient_info = state.get("patient_info", "")
    department = state.get("current_department", "")
    router_reasoning = state.get("router_reasoning", "")

    memory_context = state.get("memory_context", "")

    heuristic_plan = _heuristic_plan(user_text, patient_info)
    prompt = (
        f"【患者输入】{user_text}\n"
        f"【患者摘要】{patient_info}\n"
        f"【分诊科室】{department}\n"
        f"【分诊理由】{router_reasoning}\n"
    )
    if memory_context:
        prompt += f"【相似历史病例参考】{memory_context}\n"
    prompt += "\n请生成最小必要工具计划。"

    merged_plan = heuristic_plan
    response = chat(
        prompt,
        system=_PLANNER_SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=256,
        response_format={"type": "json_object"},
    )
    if response:
        try:
            raw = response
            if "```" in raw:
                raw = raw.split("```")[1].strip()
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            merged_plan = _merge_plan(json.loads(raw), heuristic_plan)
        except Exception as e:
            logger.warning(f"Planner: JSON 解析失败，使用启发式计划 ({e})")

    logger.info(
        "Planner: type=%s, tools=%s, pharmacist=%s, hypotheses=%s",
        merged_plan["problem_type"],
        merged_plan["tool_plan"],
        merged_plan["need_pharmacist"],
        merged_plan.get("differential_hypotheses", []),
    )

    return {
        "problem_type": merged_plan["problem_type"],
        "differential_hypotheses": merged_plan.get("differential_hypotheses", []),
        "information_gaps": merged_plan.get("information_gaps", []),
        "reasoning_chain": merged_plan.get("reasoning_chain", ""),
        "tool_plan": merged_plan["tool_plan"],
        "verification_criteria": merged_plan.get("verification_criteria", []),
        "expected_evidence": merged_plan["expected_evidence"],
        "need_pharmacist": merged_plan["need_pharmacist"],
        "plan_summary": merged_plan["plan_summary"],
    }
