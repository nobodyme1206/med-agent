"""
GRPO 强化学习训练脚本。
使用 HuggingFace TRL 的 GRPOTrainer + QLoRA 4bit 适配单卡 RTX 4090。

用法（AutoDL）：
  accelerate launch training/grpo_train.py \
    --model_path /root/autodl-tmp/output/qwen2.5-7b-med-agent-sft \
    --data_path data/synth/sft_data/grpo_prompts.json \
    --output_dir /root/autodl-tmp/output/qwen2.5-7b-med-agent-grpo
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training for MedAgent")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Agentic SFT 模型路径（LoRA adapter 或 merged）")
    parser.add_argument("--data_path", type=str, required=True,
                        help="GRPO prompt 数据路径（JSON）")
    parser.add_argument("--output_dir", type=str, default="output/grpo",
                        help="输出目录")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="每个 prompt 采样数（组大小），最低 4")
    parser.add_argument("--max_completion_length", type=int, default=512,
                        help="最大生成长度")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL 散度惩罚系数")
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--no_qlora", action="store_true", default=False,
                        help="禁用 QLoRA 4bit 量化（默认启用）")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="从 checkpoint 恢复训练（路径或 'auto' 自动查找最新）")
    return parser.parse_args()


def load_dataset(data_path: str):
    """加载 GRPO prompt 数据集"""
    from datasets import Dataset

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 期望格式: [{"prompt": [{"role": "user", "content": "..."}], "ground_truth": "...", "expected_tools": [...]}]
    # GRPOTrainer 需要 conversational format 的 prompt
    prompts = []
    metadata = {}
    for i, item in enumerate(data):
        if isinstance(item["prompt"], str):
            prompts.append([{"role": "user", "content": item["prompt"]}])
        else:
            prompts.append(item["prompt"])

        if "ground_truth" in item:
            if "ground_truth" not in metadata:
                metadata["ground_truth"] = []
            metadata["ground_truth"].append(item["ground_truth"])

        if "expected_tools" in item:
            if "expected_tools" not in metadata:
                metadata["expected_tools"] = []
            metadata["expected_tools"].append(item["expected_tools"])

    ds_dict = {"prompt": prompts}
    ds_dict.update(metadata)
    dataset = Dataset.from_dict(ds_dict)

    logger.info(f"数据集加载: {len(dataset)} 条 prompt")
    return dataset


def main():
    args = parse_args()

    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOTrainer, GRPOConfig
    from peft import LoraConfig

    # ─── 模型量化配置（QLoRA 4bit）───
    quantization_config = None
    if not args.no_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        logger.info("QLoRA 4bit 量化已启用")

    # ─── LoRA 配置 ───
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    # ─── GRPO 训练配置 ───
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        beta=args.beta,
        logging_steps=1,
        save_steps=5,
        save_total_limit=3,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        max_prompt_length=512,
        gradient_checkpointing=True,
        warmup_ratio=0.1,
    )

    # ─── 加载数据 ───
    dataset = load_dataset(args.data_path)

    # ─── 奖励函数 ───
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.reward import med_agent_reward

    def reward_func(completions, **kwargs):
        """GRPO 奖励函数包装器（兼容 TRL 各版本）"""
        # 提取文本内容（TRL 可能传 list[str] 或 list[list[dict]]）
        texts = []
        for c in completions:
            if isinstance(c, list):
                # conversational format: [{"role": "assistant", "content": "..."}]
                texts.append(c[-1]["content"] if c else "")
            elif isinstance(c, dict):
                texts.append(c.get("content", str(c)))
            else:
                texts.append(str(c))

        # TRL GRPOTrainer 将 dataset 中的额外列作为 kwargs 传入
        gt = kwargs.get("ground_truth", None)
        et = kwargs.get("expected_tools", None)
        # gt/et 可能是单值（当前 batch 切片），需要包装为 list
        if gt is not None and not isinstance(gt, list):
            gt = [gt] * len(texts)
        if et is not None and not isinstance(et, list):
            et = [et] * len(texts)
        return med_agent_reward(texts, ground_truth=gt, expected_tools=et)

    # ─── 加载模型 ───
    logger.info(f"初始化 GRPOTrainer: model={args.model_path}")

    from transformers import AutoModelForCausalLM
    model_load_kwargs = {"device_map": "auto"}
    if quantization_config:
        model_load_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, **model_load_kwargs
    )

    # ─── 初始化 Trainer ───
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_func,
        peft_config=peft_config,
    )

    # ─── 修复 KV cache：patch TRL 的 unwrap_model_for_generation ───
    # TRL 0.14.0 的 unwrap_model_for_generation 不处理 gradient_checkpointing/use_cache，
    # 导致 generation 时 KV cache 被禁用，速度慢 2-3x。
    # 正确做法：在 generation 上下文中关闭 gradient_checkpointing 并启用 use_cache。
    from contextlib import contextmanager
    import trl.trainer.grpo_trainer as _grpo_module
    _orig_unwrap = _grpo_module.unwrap_model_for_generation

    @contextmanager
    def _patched_unwrap(model, accelerator, **kw):
        with _orig_unwrap(model, accelerator, **kw) as unwrapped:
            unwrapped.gradient_checkpointing_disable()
            unwrapped.config.use_cache = True
            logger.debug("generation: gradient_checkpointing OFF, use_cache ON")
            try:
                yield unwrapped
            finally:
                unwrapped.config.use_cache = False
                unwrapped.gradient_checkpointing_enable()
                logger.debug("generation done: gradient_checkpointing ON, use_cache OFF")

    _grpo_module.unwrap_model_for_generation = _patched_unwrap
    logger.info("已 patch unwrap_model_for_generation，generation 时启用 KV cache")

    # ─── 训练 ───
    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt == "auto":
        import glob
        ckpts = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
                       key=lambda x: int(x.split("-")[-1]))
        resume_ckpt = ckpts[-1] if ckpts else None
        logger.info(f"自动查找 checkpoint: {resume_ckpt}")
    logger.info(f"开始 GRPO 训练... resume={resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ─── 保存 ───
    trainer.save_model(args.output_dir)
    logger.info(f"模型保存至: {args.output_dir}")

    # 保存训练配置
    config = vars(args)
    with open(os.path.join(args.output_dir, "grpo_train_config.json"), "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
