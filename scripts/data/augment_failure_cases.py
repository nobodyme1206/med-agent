import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.task_eval import evaluate_single_prediction

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

ROUTER_PROMPT = "你是一位分诊路由器，请输出 JSON。"
PLANNER_PROMPT = "你是一位流程规划器，请输出最小必要的工具计划 JSON。"
SUMMARY_PROMPT = "你是一位汇总 Agent，请输出 structured_output 与最终回复。"


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _extract_patient_input(case: dict) -> str:
    for turn in case.get("dialogue", []):
        if turn.get("role") == "patient":
            return turn.get("content", "")
    return case.get("chief_complaint", "") or case.get("patient_input", "")


def _expected_tools(ref: dict) -> list:
    tools = ref.get("preferred_tool_sequence") or ref.get("expected_tools") or ref.get("tools_used") or []
    if isinstance(tools, str):
        return [tools]
    return [tool for tool in tools if tool]


def _extract_reference_tool_calls(ref: dict) -> list:
    calls = []
    for turn in ref.get("dialogue", []):
        for tc in turn.get("tool_calls", []) or []:
            if not tc.get("name"):
                continue
            calls.append({
                "name": tc.get("name", ""),
                "args": tc.get("args", {}) if isinstance(tc.get("args", {}), dict) else {},
                "result_summary": tc.get("result_summary", ""),
            })
    return calls


def _render_final_response(structured: dict) -> str:
    parts = []
    diagnosis = structured.get("diagnosis_direction", "")
    tests = _ensure_list(structured.get("recommended_tests"))
    meds = _ensure_list(structured.get("medication_advice"))
    followups = _ensure_list(structured.get("followup_actions"))
    if diagnosis:
        parts.append(f"初步考虑{diagnosis}。")
    if tests:
        parts.append(f"建议完善{'、'.join(tests)}。")
    if meds:
        parts.append(f"用药建议：{'；'.join(str(x) for x in meds if x)}。")
    if followups:
        parts.append(f"后续建议：{'；'.join(str(x) for x in followups if x)}。")
    if not parts:
        parts.append("建议结合面诊和进一步检查综合判断。")
    return "".join(parts)


def _structured_target(ref: dict) -> dict:
    tool_plan = _expected_tools(ref)
    structured = {
        "department": ref.get("expected_department") or ref.get("department", ""),
        "diagnosis_direction": ref.get("expected_diagnosis_direction") or ref.get("final_diagnosis_direction", ""),
        "recommended_tests": _ensure_list(ref.get("recommended_tests")),
        "medication_advice": _ensure_list(ref.get("medication_advice")),
        "need_followup": bool(ref.get("need_followup", False)),
        "followup_actions": _ensure_list(ref.get("followup_actions")),
        "evidence_summary": _ensure_list(ref.get("expected_evidence")),
        "used_tools": tool_plan,
        "tool_plan": tool_plan,
        "final_response": ref.get("expected_final_response", "") or ref.get("final_response", ""),
        # PARM 新字段
        "differential_hypotheses": _ensure_list(ref.get("differential_hypotheses")),
        "reasoning_chain": ref.get("reasoning_chain", ""),
        "verification_criteria": _ensure_list(ref.get("verification_criteria")),
        "reflection_feedback": ref.get("reflection_feedback", ""),
    }
    if not structured["final_response"]:
        structured["final_response"] = _render_final_response(structured)
    return structured


def _planner_target(ref: dict, structured: dict) -> dict:
    tool_plan = structured.get("tool_plan", [])
    drug_tools = {"search_drug", "search_by_indication", "check_drug_interaction"}
    if any(tool in drug_tools for tool in tool_plan) and "interpret_lab_result" in tool_plan:
        problem_type = "multi_factor"
    elif any(tool in drug_tools for tool in tool_plan):
        problem_type = "drug_related"
    elif "interpret_lab_result" in tool_plan:
        problem_type = "lab_related"
    else:
        problem_type = "symptom_only"
    return {
        "problem_type": problem_type,
        "tool_plan": tool_plan,
        "need_pharmacist": bool(ref.get("need_pharmacist", False)),
        "expected_evidence": structured.get("evidence_summary", []),
        "plan_summary": ref.get("plan_summary", "先完成关键工具检索，再汇总结论。"),
    }


def _failure_tags(pred: dict, ref: dict, detail: dict) -> list:
    expected_tools = _expected_tools(ref)
    predicted_tools = [tc.get("tool_name", "") for tc in pred.get("tool_calls", []) if tc.get("tool_name")]
    expected_first_tool = ref.get("expected_first_tool") or (expected_tools[0] if expected_tools else "")
    predicted_first_tool = predicted_tools[0] if predicted_tools else ""
    tags = []
    if not detail.get("department_match"):
        tags.append("department_mismatch")
    if detail.get("diagnosis_similarity", 0.0) < 0.7:
        tags.append("diagnosis_mismatch")
    test_recall = detail.get("test_recall")
    if test_recall is not None and test_recall < 1.0:
        tags.append("missing_tests")
    if expected_first_tool and predicted_first_tool != expected_first_tool:
        tags.append("wrong_first_tool")
    if any(tool not in set(expected_tools) for tool in predicted_tools):
        tags.append("offplan_tool")
    if len(predicted_tools) > len(set(predicted_tools)):
        tags.append("duplicate_tool")
    # PARM 相关 tag
    if not pred.get("reasoning_chain") and not pred.get("differential_hypotheses"):
        tags.append("missing_reasoning")
    if pred.get("reflection_count", 0) > 1:
        tags.append("reflection_retry")
    return tags


def _normalize_failure_case(item: dict) -> dict:
    ref = item.get("reference", {}) if isinstance(item.get("reference"), dict) else {}
    pred = item.get("prediction", {}) if isinstance(item.get("prediction"), dict) else {}
    detail = {}
    if ref and pred:
        detail = evaluate_single_prediction(pred, ref)
    elif "combined_score" in item:
        detail = {
            "combined_score": item.get("combined_score", 0.0),
            "diagnosis_similarity": item.get("diagnosis_similarity", 0.0),
            "department_match": item.get("department_match", 0.0),
            "test_recall": item.get("test_recall"),
            "pred_department": item.get("predicted_department", ""),
            "is_correct": False,
            "is_partial": item.get("combined_score", 0.0) >= 0.45,
        }
    structured = _structured_target(ref)
    planner = _planner_target(ref, structured)
    failure_tags = item.get("failure_tags") or _failure_tags(pred, ref, detail)
    return {
        "case_id": item.get("case_id", 0),
        "patient_input": item.get("patient_input", "") or _extract_patient_input(ref),
        "pred": pred,
        "ref": ref,
        "detail": detail,
        "structured_output_target": structured,
        "planner_target": planner,
        "reference_tool_calls": _extract_reference_tool_calls(ref),
        "failure_tags": failure_tags,
    }


def _mine_failures(predictions: list, references: list, skip_partial: bool) -> list:
    failures = []
    for idx, (pred, ref) in enumerate(zip(predictions, references)):
        detail = evaluate_single_prediction(pred, ref)
        if detail.get("is_correct"):
            continue
        if skip_partial and detail.get("is_partial"):
            continue
        failures.append(_normalize_failure_case({
            "case_id": idx,
            "patient_input": _extract_patient_input(ref),
            "prediction": pred,
            "reference": ref,
        }))
    return failures


def _sharegpt_sample(system_prompt: str, user_text: str, assistant_text: str, metadata: dict = None) -> dict:
    sample = {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": user_text},
            {"from": "gpt", "value": assistant_text},
        ]
    }
    if metadata is not None:
        sample["metadata"] = metadata
    return sample


def _planner_input(case: dict) -> str:
    structured = case["structured_output_target"]
    lines = [case.get("patient_input", "")]
    if structured.get("department"):
        lines.append(f"[路由科室] {structured['department']}")
    if case.get("failure_tags"):
        lines.append(f"[重点修正] {', '.join(case['failure_tags'])}")
    return "\n".join(line for line in lines if line)


def _summary_input(case: dict) -> str:
    structured = case["structured_output_target"]
    planner = case["planner_target"]
    lines = [case.get("patient_input", "")]
    if structured.get("department"):
        lines.append(f"[科室] {structured['department']}")
    if planner.get("tool_plan"):
        lines.append(f"[工具计划] {' -> '.join(planner['tool_plan'])}")
    if structured.get("evidence_summary"):
        lines.append(f"[关键证据] {'；'.join(str(x) for x in structured['evidence_summary'])}")
    return "\n".join(line for line in lines if line)


def _agent_completion(case: dict) -> str:
    structured = case["structured_output_target"]
    planner = case["planner_target"]
    tool_calls = case["reference_tool_calls"]
    think_parts = []
    if structured.get("diagnosis_direction"):
        think_parts.append(f"诊断方向：{structured['diagnosis_direction']}")
    if planner.get("tool_plan"):
        think_parts.append(f"工具计划：{' -> '.join(planner['tool_plan'])}")
    if case.get("failure_tags"):
        think_parts.append(f"重点修正：{', '.join(case['failure_tags'])}")
    segments = []
    if think_parts:
        segments.append(f"<think>{'；'.join(think_parts)}</think>")
    for tc in tool_calls[:3]:
        payload = {"name": tc.get("name", ""), "args": tc.get("args", {})}
        segments.append(f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>")
        if tc.get("result_summary"):
            segments.append(f"<observation>{tc['result_summary']}</observation>")
    # PARM: 嵌入推理链和鉴别诊断
    reasoning = structured.get("reasoning_chain", "")
    if reasoning:
        segments.append(f"<reasoning_chain>{reasoning}</reasoning_chain>")
    diff_hyps = structured.get("differential_hypotheses", [])
    if diff_hyps:
        segments.append(f"<differential>{json.dumps(diff_hyps, ensure_ascii=False)}</differential>")
    reflection = structured.get("reflection_feedback", "")
    if reflection:
        segments.append(f"<reflection>{reflection}</reflection>")
    segments.append(f"<structured_output>{json.dumps(structured, ensure_ascii=False)}</structured_output>")
    segments.append(f"<response>{structured.get('final_response', '')}</response>")
    return "\n".join(segment for segment in segments if segment)


def _router_output(case: dict) -> str:
    ref = case["ref"]
    payload = {
        "department": case["structured_output_target"].get("department", ""),
        "patient_info": ref.get("patient_info", "") or case.get("patient_input", "")[:120],
        "reasoning": ref.get("router_reasoning", "根据主诉和症状进行科室分诊"),
        "confidence": ref.get("confidence", 0.85),
    }
    return json.dumps(payload, ensure_ascii=False)


def _planner_output(case: dict) -> str:
    return json.dumps(case["planner_target"], ensure_ascii=False)


def _summary_output(case: dict) -> str:
    structured = case["structured_output_target"]
    return (
        f"<structured_output>{json.dumps(structured, ensure_ascii=False)}</structured_output>\n"
        f"<response>{structured.get('final_response', '')}</response>"
    )


def build_outputs(failures: list) -> dict:
    manifest = []
    agent_sft = []
    grpo = []
    router = []
    planner = []
    summary = []
    tag_counts = {}
    for case in failures:
        structured = case["structured_output_target"]
        planner_target = case["planner_target"]
        user_text = case.get("patient_input", "")
        if not user_text:
            continue
        for tag in case.get("failure_tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        manifest.append({
            "case_id": case.get("case_id", 0),
            "patient_input": user_text,
            "failure_tags": case.get("failure_tags", []),
            "combined_score": case.get("detail", {}).get("combined_score", 0.0),
            "structured_output_target": structured,
            "planner_target": planner_target,
        })
        agent_sft.append(_sharegpt_sample(SYSTEM_PROMPT, user_text, _agent_completion(case)))
        grpo.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            "ground_truth": structured.get("diagnosis_direction", ""),
            "expected_tools": planner_target.get("tool_plan", []),
            "expected_first_tool": (planner_target.get("tool_plan") or [""])[0] if planner_target.get("tool_plan") else "",
            "preferred_tool_sequence": planner_target.get("tool_plan", []),
            "tool_plan": planner_target.get("tool_plan", []),
            "need_pharmacist": planner_target.get("need_pharmacist", False),
            "department": structured.get("department", ""),
            "structured_output_target": structured,
        })
        router.append(_sharegpt_sample(
            ROUTER_PROMPT,
            user_text,
            _router_output(case),
            {"stage": "router", "department": structured.get("department", "")},
        ))
        planner.append(_sharegpt_sample(
            PLANNER_PROMPT,
            _planner_input(case),
            _planner_output(case),
            {"stage": "planner", "department": structured.get("department", ""), "failure_tags": case.get("failure_tags", [])},
        ))
        summary.append(_sharegpt_sample(
            SUMMARY_PROMPT,
            _summary_input(case),
            _summary_output(case),
            {"stage": "summary", "department": structured.get("department", ""), "failure_tags": case.get("failure_tags", [])},
        ))
    return {
        "manifest": manifest,
        "agent_sft": agent_sft,
        "grpo": grpo,
        "router": router,
        "planner": planner,
        "summary": summary,
        "tag_counts": tag_counts,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure_cases", type=str, default="")
    parser.add_argument("--predictions", type=str, default="")
    parser.add_argument("--eval_data", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--skip_partial", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    if args.failure_cases:
        with open(args.failure_cases, "r", encoding="utf-8") as f:
            raw_failures = json.load(f)
        failures = [_normalize_failure_case(item) for item in raw_failures]
    elif args.predictions and args.eval_data:
        with open(args.predictions, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        with open(args.eval_data, "r", encoding="utf-8") as f:
            references = json.load(f)
        failures = _mine_failures(predictions, references, skip_partial=args.skip_partial)
    else:
        raise ValueError("需要提供 --failure_cases 或同时提供 --predictions 与 --eval_data")

    outputs = build_outputs(failures)
    files = {
        "failure_manifest.json": outputs["manifest"],
        "hard_case_sft.json": outputs["agent_sft"],
        "hard_case_grpo.json": outputs["grpo"],
        "hard_case_router_sft.json": outputs["router"],
        "hard_case_planner_sft.json": outputs["planner"],
        "hard_case_summary_sft.json": outputs["summary"],
        "failure_stats.json": {
            "total_failures": len(outputs["manifest"]),
            "tag_counts": outputs["tag_counts"],
        },
    }
    for name, payload in files.items():
        with open(output_dir / name, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"failure augmentation 完成: {len(outputs['manifest'])} 条 hard cases")
    logger.info(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
