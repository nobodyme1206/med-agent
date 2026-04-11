#!/bin/bash
# ─────────────────────────────────────────────
# 一键评测脚本：base → SFT → ReST 三版模型顺序评测
# 在服务器上直接运行：bash scripts/run_full_eval.sh
# ─────────────────────────────────────────────

set -e

# ─── 配置 ───
WORK_DIR="${WORK_DIR:-/root/autodl-tmp}"
BASE_MODEL="${BASE_MODEL:-$WORK_DIR/models/Qwen/Qwen2___5-7B-Instruct}"
SFT_ADAPTER="${SFT_ADAPTER:-$WORK_DIR/output/qwen2.5-7b-med-agent-sft}"
REST_ADAPTER="${REST_ADAPTER:-$WORK_DIR/output/qwen2.5-7b-med-agent-rest}"
LLAMA_CLI="${LLAMA_CLI:-$WORK_DIR/venv_api/bin/llamafactory-cli}"
EVAL_PY="${EVAL_PY:-$WORK_DIR/venv_grpo/bin/python}"
PROJECT="${PROJECT:-$WORK_DIR/med-agent}"
LOG_DIR="${LOG_DIR:-$WORK_DIR/logs}"
mkdir -p "$LOG_DIR"
EVAL_DATA="$PROJECT/data/eval/eval_cases.json"
MAX_CASES=50
API_PORT=8000
API_URL="http://localhost:$API_PORT/v1"
AUGMENT_FAILURES="${AUGMENT_FAILURES:-1}"

# ─── 函数 ───
wait_for_api() {
    echo "[$(date '+%H:%M:%S')] 等待 API 服务就绪..."
    for i in $(seq 1 60); do
        if curl -s "$API_URL/models" > /dev/null 2>&1; then
            echo "[$(date '+%H:%M:%S')] API 已就绪 (${i}s)"
            return 0
        fi
        sleep 2
    done
    echo "[$(date '+%H:%M:%S')] ERROR: API 启动超时"
    return 1
}

kill_api() {
    fuser -k $API_PORT/tcp 2>/dev/null || true
    sleep 2
}

start_api() {
    local model_name=$1
    local adapter_path=$2

    kill_api
    echo ""
    echo "============================================"
    echo "[$(date '+%H:%M:%S')] 启动 API: $model_name"
    echo "============================================"

    if [ -z "$adapter_path" ]; then
        # Base 模型（无 adapter）
        nohup $LLAMA_CLI api \
            --model_name_or_path $BASE_MODEL \
            --template qwen \
            > $LOG_DIR/api_${model_name}.log 2>&1 &
    else
        # 带 adapter
        nohup $LLAMA_CLI api \
            --model_name_or_path $BASE_MODEL \
            --adapter_name_or_path $adapter_path \
            --template qwen \
            > $LOG_DIR/api_${model_name}.log 2>&1 &
    fi

    wait_for_api
}

run_eval() {
    local model_name=$1
    local output_dir="$PROJECT/results/$model_name"
    rm -rf "$output_dir"
    mkdir -p "$output_dir"

    echo "[$(date '+%H:%M:%S')] 开始评测: $model_name → $output_dir"

    cd $PROJECT
    $EVAL_PY -m evaluation.run_eval \
        --eval_data $EVAL_DATA \
        --output "$output_dir" \
        --run_agent \
        --run_safety \
        --run_judge \
        --max_cases $MAX_CASES \
        2>&1 | tee "$output_dir/eval.log"

    if [ "$AUGMENT_FAILURES" = "1" ] && [ -f "$output_dir/failure_cases.json" ]; then
        $EVAL_PY "$PROJECT/scripts/augment_failure_cases.py" \
            --failure_cases "$output_dir/failure_cases.json" \
            --output_dir "$output_dir/failure_augmented" \
            2>&1 | tee "$output_dir/failure_augment.log"
    fi

    echo "[$(date '+%H:%M:%S')] $model_name 评测完成"
    echo ""
    cat "$output_dir/evaluation_report.json" | python3 -m json.tool 2>/dev/null || true
    echo ""
}

# ─── 主流程 ───
echo "============================================"
echo "  MedAgent 三版模型评测"
echo "  开始时间: $(date)"
echo "============================================"
START_TIME=$(date +%s)

  # 1. Base 模型
  start_api "base" ""
  run_eval "base"
  
  # 2. SFT 模型
 if [ -d "$SFT_ADAPTER" ]; then
     start_api "sft" "$SFT_ADAPTER"
     run_eval "sft"
 else
     echo "[$(date '+%H:%M:%S')] 跳过 sft：adapter 不存在 ($SFT_ADAPTER)"
 fi
  
  # 3. ReST 模型
 if [ -d "$REST_ADAPTER" ]; then
     start_api "rest" "$REST_ADAPTER"
     run_eval "rest"
 else
     echo "[$(date '+%H:%M:%S')] 跳过 rest：adapter 不存在 ($REST_ADAPTER)"
 fi

# 清理
kill_api

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "============================================"
echo "  全部评测完成！总耗时: ${ELAPSED} 分钟"
echo "  结果目录: $PROJECT/results/"
echo "============================================"
echo ""

# 汇总对比
echo "=== 三版模型对比 ==="
for m in base sft rest; do
    REPORT="$PROJECT/results/$m/evaluation_report.json"
    if [ -f "$REPORT" ]; then
        echo "--- $m ---"
        python3 -c "
import json
with open('$REPORT') as f:
    r = json.load(f)
tc = r.get('task_completion', {})
tu = r.get('tool_usage', {})
te = r.get('trajectory_efficiency', {})
re = r.get('reasoning_quality', {})
sf = r.get('safety', {})
lj = r.get('llm_judge', {})
fa = r.get('failure_analysis', {})
print(f'  任务完成率: {tc.get(\"accuracy\", 0):.2%}')
print(f'  结构化分数: {tc.get(\"avg_combined_score\", 0):.3f}')
print(f'  科室准确率: {tc.get(\"department_accuracy\", 0):.2%}')
print(f'  工具F1:     {tu.get(\"avg_f1\", 0):.3f}')
print(f'  FirstTool:  {tu.get(\"first_tool_accuracy\", 0):.2%}')
print(f'  效率分:     {te.get(\"efficiency_score\", 0):.3f}')
print(f'  推理完整:   {re.get(\"avg_completeness\", 0):.3f}')
print(f'  证据锚定:   {re.get(\"avg_evidence_grounding\", 0):.3f}')
print(f'  工具归因:   {re.get(\"avg_tool_attribution\", 0):.3f}')
print(f'  安全拒绝率: {sf.get(\"overall_refusal_rate\", 0):.2%}')
print(f'  Judge总分:  {lj.get(\"avg_overall\", 0):.2f}/5.0')
print(f'  失败样本数: {fa.get(\"total_failures\", 0)}')
" 2>/dev/null || echo "  (报告解析失败)"
    fi
done
