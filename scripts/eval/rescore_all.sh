#!/bin/bash
# ─────────────────────────────────────────────
# 重算评测报告：利用已有的 predictions.json + judge_checkpoint.jsonl
# 不需要重跑 Agent 和 Judge，只重新计算指标
# 用法: bash scripts/rescore_all.sh
# ─────────────────────────────────────────────

set -e
PROJECT="/root/autodl-tmp/med-agent"
EVAL_PY="/root/autodl-tmp/venv_grpo/bin/python"
EVAL_DATA="$PROJECT/data/eval/eval_cases.json"

for model in base sft rest; do
    RESULT_DIR="$PROJECT/results/$model"
    PRED="$RESULT_DIR/predictions.json"
    
    if [ ! -f "$PRED" ]; then
        echo "[$model] predictions.json 不存在，跳过"
        continue
    fi
    
    echo ""
    echo "============================================"
    echo "  重算 $model 评测报告"
    echo "============================================"
    
    # 备份旧报告
    cp "$RESULT_DIR/evaluation_report.json" "$RESULT_DIR/evaluation_report_old.json" 2>/dev/null || true
    
    cd $PROJECT
    $EVAL_PY -m evaluation.run_eval \
        --eval_data $EVAL_DATA \
        --output "$RESULT_DIR" \
        --predictions "$PRED" \
        --run_judge \
        2>&1 | tail -20
    
    echo "[$model] 重算完成"
done

echo ""
echo "============================================"
echo "  全部重算完成"
echo "============================================"
