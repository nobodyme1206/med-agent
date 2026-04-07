#!/bin/bash
# GRPO 训练自动重试脚本
# 如果 OOM 或其他错误导致 crash，自动从最近 checkpoint 恢复
# 用法: bash run_grpo_with_retry.sh [max_retries]

MAX_RETRIES=${1:-10}
ATTEMPT=0
EXIT_CODE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=/root/autodl-tmp/venv_grpo/bin:$PATH

COMMON_ARGS=(
    /root/autodl-tmp/med-agent/training/grpo_train.py
    --model_path /root/autodl-tmp/output/qwen2.5-7b-med-agent-sft
    --data_path /root/autodl-tmp/data/grpo_prompts.json
    --output_dir /root/autodl-tmp/output/qwen2.5-7b-med-agent-grpo
    --num_generations 4
    --max_completion_length 512
    --num_train_epochs 2
)

while [ $ATTEMPT -lt $MAX_RETRIES ] && [ $EXIT_CODE -ne 0 ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "========== 第 ${ATTEMPT}/${MAX_RETRIES} 次尝试 $(date) =========="

    if [ $ATTEMPT -eq 1 ]; then
        echo "首次运行，不加载 checkpoint"
        python "${COMMON_ARGS[@]}"
    else
        echo "从最近 checkpoint 恢复..."
        python "${COMMON_ARGS[@]}" --resume_from_checkpoint auto
    fi

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "训练异常退出 (exit=$EXIT_CODE)，等待 10 秒后重试..."
        sleep 10
    fi
done

if [ $EXIT_CODE -eq 0 ]; then
    echo "========== GRPO 训练成功完成！=========="
else
    echo "========== 达到最大重试次数 ${MAX_RETRIES}，训练未完成 =========="
fi
