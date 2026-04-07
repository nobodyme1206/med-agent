#!/bin/bash
# MedAgent GRPO 训练脚本
# 用法: bash scripts/train_grpo.sh [round]
# 前提: 已完成 SFT 训练

set -e

ROUND=${1:-1}
WORK_DIR=/root/autodl-tmp
SFT_MODEL=$WORK_DIR/output/qwen2.5-7b-med-agent-sft
GRPO_OUTPUT=$WORK_DIR/output/qwen2.5-7b-med-agent-grpo-r${ROUND}
DATA_PATH=/root/med-agent/data/synth/sft_data/grpo_prompts.json

echo "=========================================="
echo "  MedAgent GRPO 训练 (Round $ROUND)"
echo "=========================================="

# 确定输入模型：R1 用 SFT 产出，R2+ 用上一轮 GRPO 产出
if [ "$ROUND" -eq 1 ]; then
    MODEL_PATH=$SFT_MODEL
else
    PREV_ROUND=$((ROUND - 1))
    MODEL_PATH=$WORK_DIR/output/qwen2.5-7b-med-agent-grpo-r${PREV_ROUND}
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 输入模型不存在: $MODEL_PATH"
    echo "请先完成上一阶段训练"
    exit 1
fi

echo "  输入模型: $MODEL_PATH"
echo "  输出目录: $GRPO_OUTPUT"
echo "  数据: $DATA_PATH"
echo "  方法: GRPO + QLoRA 4bit"
echo "  采样数: 4 / prompt"
echo "  KL beta: 0.1"
echo ""

# 设置环境变量
export PYTHONPATH=/root/med-agent:$PYTHONPATH

# 启动训练
accelerate launch /root/med-agent/training/grpo_train.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $GRPO_OUTPUT \
    --num_generations 4 \
    --beta 0.1 \
    --learning_rate 5e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_completion_length 1024

echo ""
echo "=========================================="
echo "  GRPO Round $ROUND 训练完成！"
echo "  模型路径: $GRPO_OUTPUT"
echo "=========================================="
echo ""
echo "建议下一步:"
echo "  1. 跑 50 条评测: python evaluation/run_eval.py --eval_data data/eval/eval_cases.json --output results/grpo_r${ROUND}/ --run_agent --run_judge"
echo "  2. 对比 SFT vs GRPO-R${ROUND} 指标"
echo "  3. 分析 Bad Case，决定是否需要 Round $((ROUND+1))"
