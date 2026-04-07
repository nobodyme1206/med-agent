#!/bin/bash
# MedAgent Agentic SFT 训练脚本
# 用法: bash scripts/train_sft.sh
# 前提: 已运行 bash scripts/prepare_autodl.sh

set -e

echo "=========================================="
echo "  MedAgent Agentic SFT 训练"
echo "=========================================="

WORK_DIR=/root/autodl-tmp
LLAMA_FACTORY_DIR=$WORK_DIR/LLaMA-Factory
CONFIG_PATH=/root/med-agent/training/configs/sft_config.yaml

# 检查环境
if [ ! -d "$LLAMA_FACTORY_DIR" ]; then
    echo "错误: LLaMA-Factory 未安装，请先运行 bash scripts/prepare_autodl.sh"
    exit 1
fi

# 检查数据集注册
python -c "
import json
with open('$LLAMA_FACTORY_DIR/data/dataset_info.json') as f:
    info = json.load(f)
assert 'med_agent_sft' in info, 'med_agent_sft 未注册!'
print('数据集注册: OK')
"

# 训练
echo "开始 SFT 训练..."
echo "  模型: Qwen2.5-7B-Instruct"
echo "  方法: LoRA (rank=16, alpha=32)"
echo "  数据: 499 条 Agent trajectory"
echo "  Epochs: 2"
echo "  输出: $WORK_DIR/output/qwen2.5-7b-med-agent-sft/"
echo ""

cd $LLAMA_FACTORY_DIR
llamafactory-cli train $CONFIG_PATH

echo ""
echo "=========================================="
echo "  SFT 训练完成！"
echo "  模型路径: $WORK_DIR/output/qwen2.5-7b-med-agent-sft/"
echo "  下一步: bash scripts/train_grpo.sh"
echo "=========================================="

# SFT 中间检查（可选）
echo ""
echo "建议: 运行 SFT 中间检查确认模型学会 ReAct 格式"
echo "  python -c \"
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_path = '$WORK_DIR/output/qwen2.5-7b-med-agent-sft'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
prompt = '患者: 我最近血压偏高，头晕，吃了硝苯地平没效果'
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
output = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
print(tokenizer.decode(output[0], skip_special_tokens=True))
\""
