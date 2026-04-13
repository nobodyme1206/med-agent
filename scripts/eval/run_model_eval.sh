#!/bin/bash
# 一键跑指定模型的三层评测（synth + hard + CMB）
# 用法:
#   MODEL_TAG=sft bash scripts/run_model_eval.sh
#   MODEL_TAG=rest bash scripts/run_model_eval.sh
#   MODEL_TAG=base bash scripts/run_model_eval.sh
set -euo pipefail

MODEL_TAG="${MODEL_TAG:-sft}"
WORK_DIR=/root/autodl-tmp
PROJECT="$WORK_DIR/med-agent"
EVAL_PY="$WORK_DIR/venv_eval/bin/python"
LOG_DIR="$WORK_DIR/logs"
export PYTHONPATH="$PROJECT"

JUDGE_MODEL="${JUDGE_MODEL:-DeepSeek-V3.1}"

mkdir -p "$LOG_DIR"

echo "============================================="
echo "  MedAgent ${MODEL_TAG} 模型三层评测"
echo "  Judge: $JUDGE_MODEL"
echo "  时间: $(date)"
echo "============================================="

# ─── 1. Synth (50条) ───
echo ""
echo ">>> [1/3] Synth 评测 (50条)..."
mkdir -p "$PROJECT/results/${MODEL_TAG}_synth"
"$EVAL_PY" -m evaluation.run_eval \
  --eval_data "$PROJECT/data/eval/eval_cases.json" \
  --output "$PROJECT/results/${MODEL_TAG}_synth" \
  --eval_source synth \
  --run_agent --run_judge --run_safety \
  --judge_model "$JUDGE_MODEL" \
  --safety_sample 2 \
  --max_cases 50 \
  2>&1 | tee "$LOG_DIR/${MODEL_TAG}_synth.log"
echo ">>> [1/3] Synth 完成 ✓"

# ─── 2. Hard (20条) ───
echo ""
echo ">>> [2/3] Hard-case 评测 (20条)..."
mkdir -p "$PROJECT/results/${MODEL_TAG}_hard"
"$EVAL_PY" -m evaluation.run_eval \
  --eval_data "$PROJECT/data/eval/hard_eval_cases.json" \
  --output "$PROJECT/results/${MODEL_TAG}_hard" \
  --eval_source hard_case \
  --run_agent --run_judge --run_safety \
  --judge_model "$JUDGE_MODEL" \
  --safety_sample 2 \
  2>&1 | tee "$LOG_DIR/${MODEL_TAG}_hard.log"
echo ">>> [2/3] Hard-case 完成 ✓"

# ─── 3. CMB (30条) ───
echo ""
echo ">>> [3/3] CMB 真实数据评测 (30条)..."
mkdir -p "$PROJECT/results/${MODEL_TAG}_cmb"
"$EVAL_PY" -m evaluation.run_eval \
  --eval_data "$PROJECT/data/eval/cmb_eval.json" \
  --output "$PROJECT/results/${MODEL_TAG}_cmb" \
  --eval_source CMB \
  --run_agent --run_judge --run_safety \
  --judge_model "$JUDGE_MODEL" \
  --safety_sample 2 \
  --max_cases 30 \
  2>&1 | tee "$LOG_DIR/${MODEL_TAG}_cmb.log"
echo ">>> [3/3] CMB 完成 ✓"

# ─── 汇总 ───
echo ""
echo "============================================="
echo "  ${MODEL_TAG} 三层评测全部完成！"
echo "  结果目录:"
echo "    results/${MODEL_TAG}_synth/"
echo "    results/${MODEL_TAG}_hard/"
echo "    results/${MODEL_TAG}_cmb/"
echo "  时间: $(date)"
echo "============================================="

for ds in synth hard cmb; do
  REPORT="$PROJECT/results/${MODEL_TAG}_${ds}/evaluation_report.json"
  if [ -f "$REPORT" ]; then
    echo ""
    echo "--- ${MODEL_TAG}_${ds} ---"
    "$EVAL_PY" -c "
import json
r = json.load(open('$REPORT'))
lj = r.get('llm_judge', {})
tc = r.get('task_completion', {})
tu = r.get('tool_usage', {})
re_ = r.get('reasoning', {})
sa = r.get('safety', {})
print(f'  Judge 综合分:   {lj.get(\"avg_overall\", 0):.2f}/5.0')
print(f'  诊断准确率:     {lj.get(\"diagnostic_accuracy\", 0):.1%}')
print(f'  科室准确率:     {tc.get(\"department_accuracy\", 0):.1%}')
print(f'  工具 F1:        {tu.get(\"avg_f1\", 0):.3f}')
print(f'  安全通过率:     {sa.get(\"pass_rate\", 0):.2%}')
print(f'  推理综合分:     {re_.get(\"overall_reasoning_score\", 0):.3f}')
"
  fi
done
