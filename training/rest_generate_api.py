"""
ReST 数据生成（API 模式）— 通过 OpenAI 兼容 API 生成 + 本地 reward 打分。

优势：
  - fp16 推理（API 端合并 adapter），比 4-bit 本地推理快 ~30%
  - 并发请求，重叠生成与打分
  - 参数可调：num_generations=4 + max_tokens=256 → 比原始配置快 3-4x

用法：
  python training/rest_generate_api.py \
    --api_base http://localhost:8000/v1 \
    --data_path data/synth/sft_data/grpo_prompts.json \
    --output_path data/synth/sft_data/rest_sft_r1.json \
    --num_generations 4 --max_tokens 256 --reward_threshold 0.4
"""

import os
import sys
import json
import argparse
import logging
import time
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.reward import (
    task_completion_reward,
    tool_accuracy_reward,
    safety_reward,
    format_reward,
    structured_output_reward,
    plan_adherence_reward,
    duplicate_control_reward,
    reasoning_chain_reward,
    reflection_quality_reward,
)


def compute_reward(completion, ground_truth="", expected_tools=None,
                   expected_first_tool="", preferred_tool_sequence=None,
                   tool_plan=None, structured_output_target=None):
    r_task = task_completion_reward(completion, ground_truth=ground_truth)
    r_tool = tool_accuracy_reward(completion, expected_tools=expected_tools)
    r_safe = safety_reward(completion)
    r_fmt = format_reward(completion)
    r_struct = structured_output_reward(completion, structured_output_target=structured_output_target, ground_truth=ground_truth)
    r_plan = plan_adherence_reward(
        completion, expected_tools=expected_tools,
        expected_first_tool=expected_first_tool,
        preferred_tool_sequence=preferred_tool_sequence,
        tool_plan=tool_plan,
    )
    r_dup = duplicate_control_reward(completion)
    r_reasoning = reasoning_chain_reward(completion)
    r_reflection = reflection_quality_reward(completion)
    total = (0.18 * r_task + 0.10 * r_tool + 0.10 * r_safe + 0.15 * r_fmt
             + 0.12 * r_struct + 0.10 * r_plan + 0.05 * r_dup
             + 0.12 * r_reasoning + 0.08 * r_reflection)
    return {"total": total}


def to_sft_format(prompt_messages, completion):
    conversations = []
    for msg in prompt_messages:
        role = msg["role"]
        if role == "system":
            conversations.append({"from": "system", "value": msg["content"]})
        elif role == "user":
            conversations.append({"from": "human", "value": msg["content"]})
        elif role == "assistant":
            conversations.append({"from": "gpt", "value": msg["content"]})
    conversations.append({"from": "gpt", "value": completion})
    return {"conversations": conversations}


def call_api(api_base, messages, temperature, max_tokens, max_retries=3):
    """调用 OpenAI 兼容 API 生成单个 completion（含重试 + 退避）"""
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{api_base}/chat/completions",
                json={
                    "model": "default",
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": 0.9,
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                logger.warning(f"API call failed after {max_retries} retries: {e}")
                return None


def generate_for_prompt(api_base, prompt_messages, num_gen, temperature, max_tokens):
    """串行生成多个 completion（LlamaFactory API 不支持并发）"""
    completions = []
    for _ in range(num_gen):
        result = call_api(api_base, prompt_messages, temperature, max_tokens)
        if result:
            completions.append(result)
    return completions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_base", default="http://localhost:8000/v1")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--reward_threshold", type=float, default=0.4)
    parser.add_argument("--top_k_per_prompt", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--rest_round", type=int, default=1)
    parser.add_argument("--prev_rest_data", type=str, default="")
    args = parser.parse_args()

    with open(args.data_path) as f:
        prompts_data = json.load(f)
    logger.info(f"Loaded {len(prompts_data)} prompts (API mode, gen={args.num_generations}, max_tokens={args.max_tokens})")

    sft_samples = []
    stats = {"total_generated": 0, "total_kept": 0, "rewards": []}
    t_start = time.time()

    for idx, item in enumerate(prompts_data):
        prompt = item["prompt"]
        gt = item.get("ground_truth", "")
        et = item.get("expected_tools", [])
        eft = item.get("expected_first_tool", "")
        pref = item.get("preferred_tool_sequence", [])
        plan = item.get("tool_plan", [])
        struct_target = item.get("structured_output_target", None)

        # 并发生成
        completions_text = generate_for_prompt(
            args.api_base, prompt, args.num_generations, args.temperature, args.max_tokens
        )

        # 打分 + 筛选
        scored = []
        for comp in completions_text:
            reward = compute_reward(
                comp, ground_truth=gt, expected_tools=et,
                expected_first_tool=eft, preferred_tool_sequence=pref,
                tool_plan=plan, structured_output_target=struct_target,
            )
            scored.append((comp, reward))
            stats["total_generated"] += 1

        scored.sort(key=lambda x: x[1]["total"], reverse=True)
        kept = 0
        for comp, reward in scored:
            if kept >= args.top_k_per_prompt:
                break
            if reward["total"] >= args.reward_threshold:
                sft_samples.append(to_sft_format(prompt, comp))
                stats["rewards"].append(reward["total"])
                kept += 1
                stats["total_kept"] += 1

        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - t_start
            speed = (idx + 1) / elapsed * 60
            eta = (len(prompts_data) - idx - 1) / speed if speed > 0 else 0
            best_r = scored[0][1]["total"] if scored else 0
            logger.info(
                f"[{idx+1}/{len(prompts_data)}] kept={kept}, "
                f"best={best_r:.3f}, "
                f"speed={speed:.1f} prompts/min, ETA={eta:.0f}min"
            )

    # 迭代合并
    if args.rest_round > 1 and args.prev_rest_data and os.path.exists(args.prev_rest_data):
        with open(args.prev_rest_data, "r", encoding="utf-8") as f:
            prev_data = json.load(f)
        logger.info(f"ReST Round {args.rest_round}: merge prev {len(prev_data)} + new {len(sft_samples)}")
        sft_samples = prev_data + sft_samples

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(sft_samples, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    avg_reward = sum(stats["rewards"]) / max(len(stats["rewards"]), 1)
    logger.info(f"\n{'='*60}")
    logger.info(f"ReST Generation Done! (API mode, Round {args.rest_round})")
    logger.info(f"  Prompts: {len(prompts_data)}, Generated: {stats['total_generated']}")
    logger.info(f"  Kept: {stats['total_kept']} ({stats['total_kept']/max(stats['total_generated'],1)*100:.1f}%)")
    logger.info(f"  Total samples: {len(sft_samples)}, Avg reward: {avg_reward:.4f}")
    logger.info(f"  Time: {elapsed/60:.1f} min")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
