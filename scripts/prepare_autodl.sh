#!/bin/bash
# MedAgent AutoDL 一键环境配置脚本
# 用法: bash scripts/prepare_autodl.sh
# 前提: AutoDL RTX 4090 实例，已选择 PyTorch 2.x + CUDA 12.x 镜像

set -e

echo "=========================================="
echo "  MedAgent AutoDL 环境配置"
echo "=========================================="

# ─── 1. 基础目录 ───
WORK_DIR=/root/autodl-tmp
MODEL_DIR=$WORK_DIR/models
DATA_DIR=$WORK_DIR/data
OUTPUT_DIR=$WORK_DIR/output
LLAMA_FACTORY_DIR=$WORK_DIR/LLaMA-Factory

mkdir -p $MODEL_DIR $DATA_DIR $OUTPUT_DIR

# ─── 2. 下载基座模型（如果不存在）───
MODEL_PATH=$MODEL_DIR/Qwen/Qwen2___5-7B-Instruct
if [ ! -d "$MODEL_PATH" ]; then
    echo "[1/5] 下载 Qwen2.5-7B-Instruct..."
    pip install modelscope -q
    python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='$MODEL_DIR')
"
    echo "模型下载完成: $MODEL_PATH"
else
    echo "[1/5] 模型已存在，跳过下载"
fi

# ─── 3. 安装 LLaMA-Factory ───
if [ ! -d "$LLAMA_FACTORY_DIR" ]; then
    echo "[2/5] 安装 LLaMA-Factory..."
    cd $WORK_DIR
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory
    pip install -e ".[torch,metrics]" -q
else
    echo "[2/5] LLaMA-Factory 已存在，跳过"
fi

# ─── 4. 安装 GRPO 训练依赖 ───
echo "[3/5] 安装训练依赖..."
cd /root/med-agent
pip install -r requirements_autodl.txt -q

# ─── 5. 注册 SFT 数据集到 LLaMA-Factory ───
echo "[4/5] 注册 SFT 数据集..."
# 复制数据到 LLaMA-Factory 数据目录
cp data/synth/sft_data/agent_sft.json $LLAMA_FACTORY_DIR/data/agent_sft.json

# 合并 dataset_info.json
python -c "
import json
info_path = '$LLAMA_FACTORY_DIR/data/dataset_info.json'
with open(info_path, 'r') as f:
    info = json.load(f)
# 注册 med_agent_sft 数据集
info['med_agent_sft'] = {
    'file_name': 'agent_sft.json',
    'formatting': 'sharegpt',
    'columns': {'messages': 'messages'},
    'tags': {
        'role_tag': 'role',
        'content_tag': 'content',
        'user_tag': 'user',
        'assistant_tag': 'assistant',
        'system_tag': 'system'
    }
}
with open(info_path, 'w') as f:
    json.dump(info, f, ensure_ascii=False, indent=2)
print(f'已注册 med_agent_sft 到 {info_path}')
"

# ─── 6. 验证 ───
echo "[5/5] 环境验证..."
python -c "
import torch, transformers, peft, trl, bitsandbytes
print(f'PyTorch:       {torch.__version__}')
print(f'CUDA:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'Transformers:  {transformers.__version__}')
print(f'PEFT:          {peft.__version__}')
print(f'TRL:           {trl.__version__}')
print(f'BitsAndBytes:  {bitsandbytes.__version__}')
print(f'GPU Memory:    {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

echo ""
echo "=========================================="
echo "  环境配置完成！"
echo "  SFT 训练: bash scripts/train_sft.sh"
echo "  GRPO 训练: bash scripts/train_grpo.sh"
echo "=========================================="
