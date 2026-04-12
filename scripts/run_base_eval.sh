#!/bin/bash
# 一键跑 Base 模型三层评测（synth + hard + CMB）
set -euo pipefail

WORK_DIR=/root/autodl-tmp
PROJECT="$WORK_DIR/med-agent"
EVAL_PY="$WORK_DIR/venv_eval/bin/python"
LOG_DIR="$WORK_DIR/logs"
export PYTHONPATH="$PROJECT"

JUDGE_MODEL="${JUDGE_MODEL:-DeepSeek-V3.1}"

mkdir -p "$LOG_DIR"

echo "============================================="
echo "  MedAgent Base 模型三层评测"
echo "  Judge: $JUDGE_MODEL"
echo "  时间: $(date)"
echo "============================================="

# ─── 1. Synth (50条) ───
echo ""
echo ">>> [1/3] Synth 评测 (50条)..."
mkdir -p "$PROJECT/results/base_synth"
"$EVAL_PY" -m evaluation.run_eval \
  --eval_data "$PROJECT/data/eval/eval_cases.json" \
  --output "$PROJECT/results/base_synth" \
  --eval_source synth \
  --run_agent --run_judge --run_safety \
  --judge_model "$JUDGE_MODEL" \
  --safety_sample 2 \
  --max_cases 50 \
  2>&1 | tee "$LOG_DIR/base_synth.log"
echo ">>> [1/3] Synth 完成 ✓"

# ─── 2. Hard (20条) ───
echo ""
echo ">>> [2/3] Hard-case 评测 (20条)..."
mkdir -p "$PROJECT/results/base_hard"
"$EVAL_PY" -m evaluation.run_eval \
  --eval_data "$PROJECT/data/eval/hard_eval_cases.json" \
  --output "$PROJECT/results/base_hard" \
  --eval_source hard_case \
  --run_agent --run_judge --run_safety \
  --judge_model "$JUDGE_MODEL" \
  --safety_sample 2 \
  2>&1 | tee "$LOG_DIR/base_hard.log"
echo ">>> [2/3] Hard-case 完成 ✓"

# ─── 3. CMB (90条) ───
echo ""
echo ">>> [3/3] CMB 真实数据评测 (90条)..."
mkdir -p "$PROJECT/results/base_cmb"
"$EVAL_PY" -m evaluation.run_eval \
  --eval_data "$PROJECT/data/eval/cmb_eval.json" \
  --output "$PROJECT/results/base_cmb" \
  --eval_source CMB \
  --run_agent --run_judge --run_safety \
  --judge_model "$JUDGE_MODEL" \
  --safety_sample 2 \
  --max_cases 30 \
  2>&1 | tee "$LOG_DIR/base_cmb.log"
echo ">>> [3/3] CMB 完成 ✓"

# ─── 汇总 ───
echo ""
echo "============================================="
echo "  三层评测全部完成！"
echo "  结果目录:"
echo "    results/base_synth/"
echo "    results/base_hard/"
echo "    results/base_cmb/"
echo "  时间: $(date)"
echo "============================================="

# 打印各数据集关键指标
for ds in synth hard cmb; do
  REPORT="$PROJECT/results/base_${ds}/evaluation_report.json"
  if [ -f "$REPORT" ]; then
    echo ""
    echo "--- base_${ds} ---"
    "$EVAL_PY" -c "
import json
r = json.load(open('$REPORT'))
tc = r.get('task_completion', {})
j = r.get('judge', {})
eff = r.get('trajectory_efficiency', {})
print(f'  科室准确率:     {tc.get(\"department_accuracy\", 0):.1%}')
print(f'  诊断准确率:     {tc.get(\"diagnosis_accuracy\", 0):.1%}')
print(f'  工具F1:         {tc.get(\"tool_f1\", tc.get(\"avg_combined_score\", 0)):.3f}')
print(f'  效率分:         {eff.get(\"avg_efficiency\", 0):.3f}')
print(f'  Judge总分:      {j.get(\"avg_total\", 0):.2f}/5.0')
print(f'  Judge准确率:    {j.get(\"accuracy\", 0):.1%}')
"
  fi
done
