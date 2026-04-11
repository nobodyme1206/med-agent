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
EVAL_DATA_SYNTH="$PROJECT/data/eval/eval_cases.json"
EVAL_DATA_CMB="$PROJECT/data/eval/cmb_eval.json"
EVAL_DATA_HARD="$PROJECT/data/eval/hard_eval_cases.json"
MAX_CASES=50
API_PORT=8000
API_URL="http://localhost:$API_PORT/v1"
AUGMENT_FAILURES="${AUGMENT_FAILURES:-1}"
# 评测哪些数据集（空格分隔）: synth cmb hard
EVAL_DATASETS="${EVAL_DATASETS:-synth hard}"

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
    local dataset_name=$2
    local eval_data=$3
    local output_dir="$PROJECT/results/${model_name}_${dataset_name}"
    rm -rf "$output_dir"
    mkdir -p "$output_dir"

    echo "[$(date '+%H:%M:%S')] 开始评测: $model_name [$dataset_name] → $output_dir"

    cd $PROJECT
    $EVAL_PY -m evaluation.run_eval \
        --eval_data "$eval_data" \
        --output "$output_dir" \
        --eval_source "$dataset_name" \
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

    echo "[$(date '+%H:%M:%S')] $model_name [$dataset_name] 评测完成"
    echo ""
    cat "$output_dir/evaluation_report.json" | python3 -m json.tool 2>/dev/null || true
    echo ""
}

# 对单个模型跑所有数据集
run_all_datasets() {
    local model_name=$1
    for ds in $EVAL_DATASETS; do
        case "$ds" in
            synth) eval_data="$EVAL_DATA_SYNTH" ;;
            cmb)   eval_data="$EVAL_DATA_CMB" ;;
            hard)  eval_data="$EVAL_DATA_HARD" ;;
            *)     echo "未知数据集: $ds"; continue ;;
        esac
        if [ ! -f "$eval_data" ]; then
            echo "[$(date '+%H:%M:%S')] 跳过 $ds：$eval_data 不存在"
            continue
        fi
        run_eval "$model_name" "$ds" "$eval_data"
    done
}

# ─── 主流程 ───
echo "============================================"
echo "  MedAgent 三版模型评测"
echo "  开始时间: $(date)"
echo "============================================"
START_TIME=$(date +%s)

  # 1. Base 模型
  start_api "base" ""
  run_all_datasets "base"
  
  # 2. SFT 模型
 if [ -d "$SFT_ADAPTER" ]; then
     start_api "sft" "$SFT_ADAPTER"
     run_all_datasets "sft"
 else
     echo "[$(date '+%H:%M:%S')] 跳过 sft：adapter 不存在 ($SFT_ADAPTER)"
 fi
  
  # 3. ReST 模型
 if [ -d "$REST_ADAPTER" ]; then
     start_api "rest" "$REST_ADAPTER"
     run_all_datasets "rest"
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
echo "=== 模型 × 数据集 对比 ==="
for m in base sft rest; do
  for ds in $EVAL_DATASETS; do
    REPORT="$PROJECT/results/${m}_${ds}/evaluation_report.json"
    if [ -f "$REPORT" ]; then
        echo "--- $m [$ds] ---"
        python3 -c "
import json
with open('$REPORT') as f:
    r = json.load(f)
tc = r.get('task_completion', {})
tu = r.get('tool_usage', {})
te = r.get('trajectory_efficiency', {})
re = r.get('reasoning', {})
sf = r.get('safety', {})
lj = r.get('llm_judge', {})
fa = r.get('failure_analysis', {})
src = r.get('eval_source', 'unknown')
print(f'  数据来源:   {src}')
print(f'  任务完成率: {tc.get(\"accuracy\", 0):.2%}')
print(f'  结构化分数: {tc.get(\"avg_combined_score\", 0):.3f}')
print(f'  科室准确率: {tc.get(\"department_accuracy\", 0):.2%}')
print(f'  工具F1:     {tu.get(\"avg_f1\", 0):.3f}')
print(f'  FirstTool:  {tu.get(\"first_tool_accuracy\", 0):.2%}')
print(f'  效率分:     {te.get(\"efficiency_score\", 0):.3f}')
print(f'  推理完整:   {re.get(\"avg_completeness\", 0):.3f}')
print(f'  证据锚定:   {re.get(\"avg_evidence_grounding\", 0):.3f}')
print(f'  工具归因:   {re.get(\"avg_tool_attribution\", 0):.3f}')
print(f'  安全通过率: {sf.get(\"pass_rate\", 0):.2%}')
print(f'  Judge总分:  {lj.get(\"avg_overall\", 0):.2f}/5.0')
print(f'  失败样本数: {fa.get(\"total_failures\", 0)}')
" 2>/dev/null || echo "  (报告解析失败)"
    fi
  done
done
