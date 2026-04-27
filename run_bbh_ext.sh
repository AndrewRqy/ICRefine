#!/usr/bin/env bash
# run_bbh_ext.sh — Pipeline run for 6 new BBH tasks with pk_regression_guard.
#
# Tasks: formal_fallacies, logical_deduction_three, web_of_lies,
#        date_understanding, navigate, snarks
#
# Parameters:
#   Model:       gpt-4.1-mini
#   Concurrency: 32  (override with CONCURRENCY=N ./run_bbh_ext.sh)
#   CS iters:    5
#   Rule iters:  3 (Phase 1 skipped — no rule-set, but bootstrap init runs)
#   Bootstrap N: 20
#   PK guard:    enabled, tolerance 0.03
#   Output:      runs/bbh_ext/
#
# Usage:
#   chmod +x run_bbh_ext.sh
#   ./run_bbh_ext.sh

set -euo pipefail

MODEL="openai/gpt-4.1-mini"
CONCURRENCY="${CONCURRENCY:-32}"
MAX_CS_ITERS=5
MAX_RULE_ITERS=3
BOOTSTRAP_N=20

DATA_DIR="datasets/bbh"
CS_ICL_DIR="../cheat-sheet-icl/data/cheat_prompt"
OUT_BASE="runs/bbh_ext"

run_task() {
    local pipeline_task="$1"
    local eval_task="$2"
    local train_jsonl="$3"
    local out_dir="${OUT_BASE}/${eval_task}"

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  TASK: ${pipeline_task}  (eval key: ${eval_task})"
    echo "  Start: $(date)"
    echo "════════════════════════════════════════════════════════"

    mkdir -p "${out_dir}"

    python3 -m ICR_hybrid.pipeline \
        --task "${pipeline_task}" \
        --dataset "${train_jsonl}" \
        --no-oracle \
        --model-score "${MODEL}" \
        --model-rule-patch "${MODEL}" \
        --model-casestudy "${MODEL}" \
        --rule-concurrency "${CONCURRENCY}" \
        --cs-concurrency "${CONCURRENCY}" \
        --auto-rule-init \
        --bootstrap-n "${BOOTSTRAP_N}" \
        --rule-acc-goal 1.01 \
        --max-rule-iters "${MAX_RULE_ITERS}" \
        --max-cs-iters "${MAX_CS_ITERS}" \
        --pk-regression-guard \
        --pk-regression-tolerance 0.03 \
        --output-dir "${out_dir}" \
        2>&1 | tee "${out_dir}/run.log"

    echo "  Pipeline done: $(date)"

    echo "  Running comparison eval for ${eval_task} ..."
    python3 eval_bbh_comparison.py \
        --run-dir "${OUT_BASE}" \
        --cs-icl-dir "${CS_ICL_DIR}" \
        --model "${MODEL}" \
        --concurrency "${CONCURRENCY}" \
        --no-csicl \
        --tasks "${eval_task}" \
        2>&1 | tee "${out_dir}/comparison.log"

    echo "  Done: $(date)"
}

cd "$(dirname "$0")"

mkdir -p "${OUT_BASE}"
exec > >(tee -a "${OUT_BASE}/run_main.log") 2>&1

echo "BBH Extended Tasks Run — $(date)"
echo "Model: ${MODEL}  Concurrency: ${CONCURRENCY}"
echo "Output: ${OUT_BASE}"

run_task "formal_fallacies"        "formal_fallacies"        "${DATA_DIR}/formal_fallacies_train.jsonl"
run_task "logical_deduction_three" "logical_deduction_three" "${DATA_DIR}/logical_deduction_three_objects_train.jsonl"
run_task "web_of_lies"             "web_of_lies"             "${DATA_DIR}/web_of_lies_train.jsonl"
run_task "date_understanding"      "date_understanding"      "${DATA_DIR}/date_understanding_train.jsonl"
run_task "navigate"                "navigate"                "${DATA_DIR}/navigate_train.jsonl"
run_task "snarks"                  "snarks"                  "${DATA_DIR}/snarks_train.jsonl"

echo ""
echo "════════════════════════════════════════════════════════"
echo "ALL PIPELINE TASKS DONE — $(date)"
echo "════════════════════════════════════════════════════════"

echo ""
echo "════════════════════════════════════════════════════════"
echo "FINAL ACCURACY SUMMARY (test split)"
echo "════════════════════════════════════════════════════════"

python3 - <<'EOF'
import json, pathlib, sys
p = pathlib.Path("runs/bbh_ext/comparison_results.json")
if not p.exists():
    print("No results found."); sys.exit(0)
results = json.loads(p.read_text())
print("\n" + "="*70)
print("FINAL ACCURACY SUMMARY (test split)")
print(f"{'Task':<30} {'N_test':>6}  {'CS-ICL':>8}  {'Ours':>8}  {'Delta':>8}")
print("-"*70)
for task, row in results.items():
    cs  = f"{row['cs_icl_acc']:.1%}" if row.get("cs_icl_acc") is not None else "N/A"
    our = f"{row['ours_acc']:.1%}"   if row.get("ours_acc")   is not None else "N/A"
    delta = (f"{row['ours_acc'] - row['cs_icl_acc']:+.1%}"
             if row.get("cs_icl_acc") and row.get("ours_acc") else "N/A")
    print(f"{task:<30} {row['n_test']:>6}  {cs:>8}  {our:>8}  {delta:>8}")
print("="*70)
EOF

echo ""
echo "════════════════════════════════════════════════════════"
echo "COMPLETE — $(date)"
echo "════════════════════════════════════════════════════════"
