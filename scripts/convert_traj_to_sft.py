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
    "使用 <response>...</response> 给出最终回复。"
)


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

    for turn in dialogue:
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

    # 标准工具列表
    tools_used = traj.get("tools_used", [])
    diagnosis = traj.get("final_diagnosis_direction", "")

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": first_patient},
    ]

    return [{
        "prompt": prompt,
        "ground_truth": diagnosis,
        "expected_tools": tools_used,
        "department": traj.get("department", ""),
    }]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--sft_output", type=str, default="data/synth/sft_data/agent_sft.json")
    parser.add_argument("--grpo_output", type=str, default="data/synth/sft_data/grpo_prompts.json")
    parser.add_argument("--min_turns", type=int, default=2, help="最少对话轮数")
    args = parser.parse_args()

    # 加载 trajectories
    with open(args.input, "r", encoding="utf-8") as f:
        trajectories = json.load(f)
    logger.info(f"加载 {len(trajectories)} 条 trajectory")

    # 转换 SFT 数据
    sft_data = []
    grpo_data = []
    skipped = 0

    for traj in trajectories:
        sft_items = convert_trajectory_to_sft(traj)
        grpo_items = convert_trajectory_to_grpo(traj)

        for item in sft_items:
            convs = item["conversations"]
            human_count = sum(1 for c in convs if c["from"] == "human")
            if human_count >= args.min_turns:
                sft_data.append(item)
            else:
                skipped += 1

        grpo_data.extend(grpo_items)

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

    # 统计
    dept_stats = {}
    for item in grpo_data:
        dept = item.get("department", "unknown")
        dept_stats[dept] = dept_stats.get(dept, 0) + 1
    logger.info(f"GRPO 科室分布: {json.dumps(dept_stats, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
