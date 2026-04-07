"""
ReST (Reinforced Self-Training) 数据生成脚本。

流程：
  1. 加载 SFT 模型
  2. 对每个 GRPO prompt 生成 K 个 completion
  3. 用 reward 函数打分
  4. 保留 reward > 阈值的高质量 completion
  5. 转换为 LLaMA-Factory SFT 格式

用法：
  python training/rest_generate.py \
    --model_path /root/autodl-tmp/output/qwen2.5-7b-med-agent-sft \
    --data_path /root/autodl-tmp/data/grpo_prompts.json \
    --output_path /root/autodl-tmp/data/rest_sft.json \
    --num_generations 8 \
    --reward_threshold 0.4
"""

import os
import sys
import json
import argparse
import logging
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 确保能 import training.reward
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.reward import (
    task_completion_reward,
    tool_accuracy_reward,
    safety_reward,
    format_reward,
)


def compute_reward(completion: str, ground_truth: str = "", expected_tools=None) -> dict:
    """计算单个 completion 的多维度奖励"""
    r_task = task_completion_reward(completion, ground_truth=ground_truth)
    r_tool = tool_accuracy_reward(completion, expected_tools=expected_tools)
    r_safe = safety_reward(completion)
    r_fmt = format_reward(completion)
    total = 0.30 * r_task + 0.20 * r_tool + 0.20 * r_safe + 0.30 * r_fmt
    return {
        "total": total,
        "task": r_task,
        "tool": r_tool,
        "safe": r_safe,
        "format": r_fmt,
    }


def to_sft_format(prompt_messages, completion):
    """将 prompt + completion 转为 LLaMA-Factory sharegpt 格式"""
    conversations = []
    for msg in prompt_messages:
        role = msg["role"]
        if role == "system":
            conversations.append({"from": "system", "value": msg["content"]})
        elif role == "user":
            conversations.append({"from": "human", "value": msg["content"]})
        elif role == "assistant":
            conversations.append({"from": "gpt", "value": msg["content"]})

    # 添加模型生成的 completion 作为 assistant 回复
    conversations.append({"from": "gpt", "value": completion})
    return {"conversations": conversations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_model_path", default=None,
                        help="Base model path for QLoRA. If not set, assumes model_path is full model or auto-detect.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--reward_threshold", type=float, default=0.4,
                        help="Minimum reward to keep a completion")
    parser.add_argument("--top_k_per_prompt", type=int, default=2,
                        help="Max completions to keep per prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of prompts to process at once")
    args = parser.parse_args()

    # ─── 加载模型 ───
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 检测是否是 LoRA adapter（有 adapter_config.json）
    adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
    is_lora = os.path.exists(adapter_config_path)

    if is_lora:
        # 读取 base_model_name_or_path
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)
        base_path = args.base_model_path or adapter_cfg.get("base_model_name_or_path", "")
        logger.info(f"Detected LoRA adapter. Loading base model from {base_path}")

        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            ),
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, args.model_path)
        logger.info("LoRA adapter loaded and merged")
    else:
        logger.info(f"Loading full model from {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            ),
            device_map="auto",
        )

    model.eval()

    # ─── 加载数据 ───
    with open(args.data_path) as f:
        prompts_data = json.load(f)
    logger.info(f"Loaded {len(prompts_data)} prompts")

    # ─── 批量生成 + 打分 + 筛选 ───
    sft_samples = []
    stats = {"total_generated": 0, "total_kept": 0, "rewards": []}

    for idx, item in enumerate(prompts_data):
        prompt = item["prompt"]
        gt = item.get("ground_truth", "")
        et = item.get("expected_tools", [])

        # 构建输入
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        prompt_len = input_ids.shape[1]

        # 批量生成 K 个 completion（一次 generate 调用）
        completions = []
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.9,
                num_return_sequences=args.num_generations,
                pad_token_id=tokenizer.pad_token_id,
            )
        for seq in outputs:
            comp = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            reward = compute_reward(comp, ground_truth=gt, expected_tools=et)
            completions.append((comp, reward))
            stats["total_generated"] += 1

        # 按 reward 排序，筛选
        completions.sort(key=lambda x: x[1]["total"], reverse=True)
        kept = 0
        for comp, reward in completions:
            if kept >= args.top_k_per_prompt:
                break
            if reward["total"] >= args.reward_threshold:
                sft_sample = to_sft_format(prompt, comp)
                sft_samples.append(sft_sample)
                stats["rewards"].append(reward["total"])
                kept += 1
                stats["total_kept"] += 1

        if (idx + 1) % 10 == 0 or idx == 0:
            avg_r = sum(stats["rewards"][-kept:]) / max(kept, 1) if kept > 0 else 0
            logger.info(
                f"[{idx+1}/{len(prompts_data)}] "
                f"generated={args.num_generations}, kept={kept}, "
                f"best_reward={completions[0][1]['total']:.3f}, "
                f"avg_kept_reward={avg_r:.3f}"
            )

    # ─── 保存结果 ───
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(sft_samples, f, ensure_ascii=False, indent=2)

    avg_reward = sum(stats["rewards"]) / max(len(stats["rewards"]), 1)
    logger.info(f"\n{'='*60}")
    logger.info(f"ReST Generation 完成!")
    logger.info(f"  总 prompt 数: {len(prompts_data)}")
    logger.info(f"  总生成数: {stats['total_generated']}")
    logger.info(f"  保留样本数: {stats['total_kept']} ({stats['total_kept']/max(stats['total_generated'],1)*100:.1f}%)")
    logger.info(f"  平均 reward: {avg_reward:.4f}")
    logger.info(f"  输出文件: {args.output_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
