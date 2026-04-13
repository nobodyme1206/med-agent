"""
将合成 trajectory 转换为 SFT 训练数据（ShareGPT 格式，兼容 LLaMA-Factory）。
同时生成 GRPO prompt 数据。

用法:
  python scripts/convert_traj_to_sft.py \
    --input data/synth/trajectories/all_trajectories.json \
    --sft_output data/synth/sft_data/agent_sft.json \
    --grpo_output data/synth/sft_data/grpo_prompts.json
"""

import json
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是一位经验丰富的医学顾问AI。在回答问题时，你会先思考分析，"
    "必要时调用工具查询信息，然后给出专业且安全的建议。"
    "请使用 <think>...</think> 标签展示思考过程，"
    "使用 <tool_call>...</tool_call> 调用工具，"
    "可选地使用 <structured_output>...</structured_output> 输出结构化结果，"
    "使用 <response>...</response> 给出最终回复。"
)


def _dedupe(seq):
    seen = set()
    ordered = []
    for item in seq:
        if item and item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _first_patient_turn(dialogue: list) -> str:
    for turn in dialogue:
        if turn.get("role") == "patient":
            return turn.get("content", "")
    return ""


def _tool_sequence_from_dialogue(dialogue: list) -> list:
    names = []
    for turn in dialogue:
        if turn.get("role") == "agent":
            for tc in turn.get("tool_calls", []):
                if tc.get("name"):
                    names.append(tc.get("name"))
    return _dedupe(names)


def _ensure_list(value):
    if not value:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _infer_structured_output(traj: dict) -> dict:
    dialogue = traj.get("dialogue", [])
    structured = traj.get("structured_output") or {}
    tool_sequence = (
        traj.get("preferred_tool_sequence")
        or traj.get("tool_plan")
        or traj.get("tools_used")
        or structured.get("tool_plan")
        or _tool_sequence_from_dialogue(dialogue)
    )
    diagnosis = structured.get("diagnosis_direction") or traj.get("final_diagnosis_direction", "")
    department = structured.get("department") or traj.get("department", "")
    final_response = structured.get("final_response") or traj.get("final_response", "")
    return {
        "department": department,
        "diagnosis_direction": diagnosis,
        "recommended_tests": _ensure_list(structured.get("recommended_tests") or traj.get("recommended_tests")),
        "medication_advice": _ensure_list(structured.get("medication_advice") or traj.get("medication_advice")),
        "need_followup": bool(structured.get("need_followup", False)),
        "followup_actions": _ensure_list(structured.get("followup_actions") or traj.get("followup_actions")),
        "evidence_summary": _ensure_list(structured.get("evidence_summary") or traj.get("expected_evidence")),
        "used_tools": tool_sequence,
        "tool_plan": tool_sequence,
        "final_response": final_response,
        # PARM 新字段
        "differential_hypotheses": _ensure_list(
            structured.get("differential_hypotheses") or traj.get("differential_hypotheses")
        ),
        "reasoning_chain": structured.get("reasoning_chain") or traj.get("reasoning_chain", ""),
        "verification_criteria": _ensure_list(
            structured.get("verification_criteria") or traj.get("verification_criteria")
        ),
        "reflection_feedback": structured.get("reflection_feedback") or traj.get("reflection_feedback", ""),
    }


def _planner_target(traj: dict) -> dict:
    structured = _infer_structured_output(traj)
    tool_plan = structured.get("tool_plan", [])
    drug_tools = {"search_drug", "search_by_indication", "check_drug_interaction"}
    if any(t in tool_plan for t in drug_tools) and any(t == "interpret_lab_result" for t in tool_plan):
        problem_type = "multi_factor"
    elif any(t in tool_plan for t in drug_tools):
        problem_type = "drug_related"
    elif "interpret_lab_result" in tool_plan:
        problem_type = "lab_related"
    else:
        problem_type = "symptom_only"
    return {
        "problem_type": problem_type,
        "tool_plan": tool_plan,
        "need_pharmacist": any(t in tool_plan for t in drug_tools),
        "expected_evidence": structured.get("evidence_summary", []),
        "plan_summary": traj.get("plan_summary", "先检索指南，再结合证据形成最终结论。"),
        # PARM 新字段
        "differential_hypotheses": structured.get("differential_hypotheses", []),
        "information_gaps": _ensure_list(traj.get("information_gaps")),
        "reasoning_chain": structured.get("reasoning_chain", ""),
        "verification_criteria": structured.get("verification_criteria", []),
    }


def _stage_sample(system_prompt: str, user_input: str, assistant_output: str, metadata: dict) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": user_input},
            {"from": "gpt", "value": assistant_output},
        ],
        "metadata": metadata,
    }


def convert_trajectory_to_sft(traj: dict) -> list:
    """
    将一条 trajectory 转换为 ShareGPT 格式的多轮对话。

    返回格式（LLaMA-Factory ShareGPT）:
    {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
    """
    if not traj.get("generation_success", False):
        return []

    dialogue = traj.get("dialogue", [])
    if not dialogue:
        return []

    conversations = []
    # 添加 system
    conversations.append({"from": "system", "value": SYSTEM_PROMPT})
    structured_output = _infer_structured_output(traj)
    last_agent_idx = max((i for i, turn in enumerate(dialogue) if turn.get("role") == "agent"), default=-1)

    for idx, turn in enumerate(dialogue):
        role = turn.get("role", "")

        if role == "patient":
            conversations.append({
                "from": "human",
                "value": turn.get("content", ""),
            })
        elif role == "agent":
            # 构建 Agent 回复（包含思考链和工具调用）
            agent_text = ""

            thought = turn.get("thought", "")
            if thought:
                agent_text += f"<think>{thought}</think>\n"

            tool_calls = turn.get("tool_calls", [])
            for tc in tool_calls:
                tc_json = json.dumps(
                    {"name": tc.get("name", ""), "args": tc.get("args", {})},
                    ensure_ascii=False,
                )
                agent_text += f"<tool_call>{tc_json}</tool_call>\n"
                result_summary = tc.get("result_summary", "")
                if result_summary:
                    agent_text += f"<observation>{result_summary}</observation>\n"

            response = turn.get("response", "")
            if idx == last_agent_idx and any(structured_output.values()):
                # PARM: 嵌入推理链和鉴别诊断
                reasoning = structured_output.get("reasoning_chain", "")
                if reasoning:
                    agent_text += f"<reasoning_chain>{reasoning}</reasoning_chain>\n"
                diff_hyps = structured_output.get("differential_hypotheses", [])
                if diff_hyps and diff_hyps != ["待专科分析后确定"]:
                    agent_text += f"<differential>{json.dumps(diff_hyps, ensure_ascii=False)}</differential>\n"
                reflection = structured_output.get("reflection_feedback", "")
                if reflection:
                    agent_text += f"<reflection>{reflection}</reflection>\n"
                agent_text += f"<structured_output>{json.dumps(structured_output, ensure_ascii=False)}</structured_output>\n"
            if response:
                agent_text += f"<response>{response}</response>"

            if agent_text.strip():
                conversations.append({
                    "from": "gpt",
                    "value": agent_text.strip(),
                })

    # 至少需要一轮 human-gpt 对话
    human_turns = [c for c in conversations if c["from"] == "human"]
    gpt_turns = [c for c in conversations if c["from"] == "gpt"]
    if not human_turns or not gpt_turns:
        return []

    return [{"conversations": conversations}]


def convert_trajectory_to_grpo(traj: dict) -> list:
    """
    将一条 trajectory 转换为 GRPO prompt 数据。
    只保留第一轮 patient 输入作为 prompt，加上标准答案。
    """
    if not traj.get("generation_success", False):
        return []

    dialogue = traj.get("dialogue", [])
    if not dialogue:
        return []

    # 第一个 patient turn 作为 prompt
    first_patient = None
    for turn in dialogue:
        if turn.get("role") == "patient":
            first_patient = turn.get("content", "")
            break

    if not first_patient:
        return []

    structured_output = _infer_structured_output(traj)
    planner_target = _planner_target(traj)
    tools_used = planner_target.get("tool_plan", [])
    diagnosis = structured_output.get("diagnosis_direction", "")

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": first_patient},
    ]

    return [{
        "prompt": prompt,
        "ground_truth": diagnosis,
        "expected_tools": tools_used,
        "expected_first_tool": tools_used[0] if tools_used else "",
        "preferred_tool_sequence": tools_used,
        "tool_plan": planner_target.get("tool_plan", []),
        "need_pharmacist": planner_target.get("need_pharmacist", False),
        "department": structured_output.get("department", ""),
        "structured_output_target": structured_output,
    }]


def convert_trajectory_to_router_sft(traj: dict) -> list:
    if not traj.get("generation_success", False):
        return []
    dialogue = traj.get("dialogue", [])
    user_input = _first_patient_turn(dialogue)
    if not user_input:
        return []
    output = {
        "department": traj.get("department", ""),
        "patient_info": traj.get("patient_info", user_input[:120]),
        "reasoning": traj.get("router_reasoning", "根据主诉和症状分配科室"),
        "confidence": traj.get("confidence", 0.85),
    }
    return [_stage_sample(
        "你是一位分诊路由器，请输出 JSON。",
        user_input,
        json.dumps(output, ensure_ascii=False),
        {"stage": "router", "department": output["department"]},
    )]


def convert_trajectory_to_planner_sft(traj: dict) -> list:
    if not traj.get("generation_success", False):
        return []
    dialogue = traj.get("dialogue", [])
    user_input = _first_patient_turn(dialogue)
    if not user_input:
        return []
    planner_target = _planner_target(traj)
    assistant_output = json.dumps(planner_target, ensure_ascii=False)
    return [_stage_sample(
        "你是一位流程规划器，请输出最小必要的工具计划 JSON。",
        user_input,
        assistant_output,
        {"stage": "planner", "department": traj.get("department", "")},
    )]


def convert_trajectory_to_summary_sft(traj: dict) -> list:
    if not traj.get("generation_success", False):
        return []
    dialogue = traj.get("dialogue", [])
    user_input = _first_patient_turn(dialogue)
    if not user_input:
        return []
    structured_output = _infer_structured_output(traj)
    final_response = structured_output.get("final_response") or traj.get("final_response", "")
    assistant_output = f"<structured_output>{json.dumps(structured_output, ensure_ascii=False)}</structured_output>\n<response>{final_response}</response>"
    return [_stage_sample(
        "你是一位汇总 Agent，请输出 structured_output 与最终回复。",
        user_input,
        assistant_output,
        {"stage": "summary", "department": structured_output.get("department", "")},
    )]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--sft_output", type=str, default="data/synth/sft_data/agent_sft.json")
    parser.add_argument("--grpo_output", type=str, default="data/synth/sft_data/grpo_prompts.json")
    parser.add_argument("--router_output", type=str, default="")
    parser.add_argument("--planner_output", type=str, default="")
    parser.add_argument("--summary_output", type=str, default="")
    parser.add_argument("--min_turns", type=int, default=2, help="最少对话轮数")
    args = parser.parse_args()

    # 加载 trajectories
    with open(args.input, "r", encoding="utf-8") as f:
        trajectories = json.load(f)
    logger.info(f"加载 {len(trajectories)} 条 trajectory")

    # 转换 SFT 数据
    sft_data = []
    grpo_data = []
    router_data = []
    planner_data = []
    summary_data = []
    skipped = 0

    for traj in trajectories:
        sft_items = convert_trajectory_to_sft(traj)
        grpo_items = convert_trajectory_to_grpo(traj)
        router_items = convert_trajectory_to_router_sft(traj)
        planner_items = convert_trajectory_to_planner_sft(traj)
        summary_items = convert_trajectory_to_summary_sft(traj)

        for item in sft_items:
            convs = item["conversations"]
            human_count = sum(1 for c in convs if c["from"] == "human")
            if human_count >= args.min_turns:
                sft_data.append(item)
            else:
                skipped += 1

        grpo_data.extend(grpo_items)
        router_data.extend(router_items)
        planner_data.extend(planner_items)
        summary_data.extend(summary_items)

    # 保存 SFT 数据
    sft_path = Path(args.sft_output)
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    logger.info(f"SFT 数据: {len(sft_data)} 条 (跳过 {skipped} 条)")

    # 保存 GRPO prompt 数据
    grpo_path = Path(args.grpo_output)
    grpo_path.parent.mkdir(parents=True, exist_ok=True)
    with open(grpo_path, "w", encoding="utf-8") as f:
        json.dump(grpo_data, f, ensure_ascii=False, indent=2)
    logger.info(f"GRPO 数据: {len(grpo_data)} 条")

    router_path = Path(args.router_output) if args.router_output else sft_path.parent / "router_sft.json"
    planner_path = Path(args.planner_output) if args.planner_output else sft_path.parent / "planner_sft.json"
    summary_path = Path(args.summary_output) if args.summary_output else sft_path.parent / "summary_sft.json"
    for path, data in ((router_path, router_data), (planner_path, planner_data), (summary_path, summary_data)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Router 数据: {len(router_data)} 条")
    logger.info(f"Planner 数据: {len(planner_data)} 条")
    logger.info(f"Summary 数据: {len(summary_data)} 条")

    # 统计
    dept_stats = {}
    for item in grpo_data:
        dept = item.get("department", "unknown")
        dept_stats[dept] = dept_stats.get(dept, 0) + 1
    logger.info(f"GRPO 科室分布: {json.dumps(dept_stats, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
