#!/usr/bin/env bash
# run_bbh_concrete.sh — Overnight run with concrete-example (CS-ICL-style) bootstrap.
#
# Differences from run_bbh_rerun.sh:
#   - causal_judgement uses bootstrap_cheatsheet_fn (named scenarios) instead of
#     abstract rule bootstrap; Phase 1 rule-patch is skipped for that task.
#   - Phase 2 uses concrete_cs_gen_fn per failure cluster instead of structured CaseStudy.
#   - Results land in runs/bbh_concrete/ so the previous run is preserved.
#
# Usage:
#   chmod +x run_bbh_concrete.sh
#   nohup ./run_bbh_concrete.sh > runs/bbh_concrete/main.log 2>&1 &

set -euo pipefail

MODEL="openai/gpt-4.1-2025-04-14"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_CS_ITERS=5
MAX_RULE_ITERS=3
BOOTSTRAP_N=20

DATA_DIR="datasets/bbh"
CS_ICL_DIR="../cheat-sheet-icl/data/cheat_prompt"
OUT_BASE="runs/bbh_concrete"

run_task() {
    local pipeline_task="$1"
    local eval_task="$2"
    local train_jsonl="$3"
    local out_dir="${OUT_BASE}/${eval_task}"

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  TASK: ${pipeline_task}  (eval key: ${eval_task})"
    echo "  Train: ${train_jsonl}"
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
        --output-dir "${out_dir}" \
        2>&1 | tee "${out_dir}/run.log"

    echo "  Pipeline done: $(date)"

    echo "  Running comparison eval for ${eval_task} ..."
    python3 eval_bbh_comparison.py \
        --run-dir "${OUT_BASE}" \
        --cs-icl-dir "${CS_ICL_DIR}" \
        --model "${MODEL}" \
        --concurrency "${CONCURRENCY}" \
        --tasks "${eval_task}" \
        2>&1 | tee "${out_dir}/comparison.log"

    echo "  Done: $(date)"
}

cd "$(dirname "$0")"

mkdir -p "${OUT_BASE}"
exec > >(tee -a "${OUT_BASE}/run_main.log") 2>&1

echo "BBH Concrete-Example Run — $(date)"
echo "Model: ${MODEL}  Concurrency: ${CONCURRENCY}"
echo "Output: ${OUT_BASE}"

run_task "bbh_boolean"          "boolean_expressions"   "${DATA_DIR}/boolean_expressions_train.jsonl"
run_task "causal_judgement"     "causal_judgement"      "${DATA_DIR}/causal_judgement_train.jsonl"
run_task "sports_understanding" "sports_understanding"  "${DATA_DIR}/sports_understanding_train.jsonl"
run_task "disambiguation_qa"    "disambiguation_qa"     "${DATA_DIR}/disambiguation_qa_train.jsonl"
run_task "geometric_shapes"     "geometric_shapes"      "${DATA_DIR}/geometric_shapes_train.jsonl"

echo ""
echo "════════════════════════════════════════════════════════"
echo "ALL TASKS DONE — final summary — $(date)"
echo "════════════════════════════════════════════════════════"

python3 - <<'EOF'
import json, pathlib, sys
p = pathlib.Path("runs/bbh_concrete/comparison_results.json")
if not p.exists():
    print("No results found."); sys.exit(0)
results = json.loads(p.read_text())
print("\n" + "="*70)
print("FINAL ACCURACY SUMMARY (test split)")
print(f"{'Task':<25} {'N_test':>6}  {'CS-ICL':>8}  {'Ours':>8}  {'Delta':>8}")
print("-"*70)
for task, row in results.items():
    cs  = f"{row['cs_icl_acc']:.1%}" if row.get("cs_icl_acc") is not None else "N/A"
    our = f"{row['ours_acc']:.1%}"   if row.get("ours_acc")   is not None else "N/A"
    if row.get("cs_icl_acc") and row.get("ours_acc"):
        delta = f"{row['ours_acc'] - row['cs_icl_acc']:+.1%}"
    else:
        delta = "N/A"
    print(f"{task:<25} {row['n_test']:>6}  {cs:>8}  {our:>8}  {delta:>8}")
print("="*70)
EOF

echo ""
echo "════════════════════════════════════════════════════════"
echo "COMPLETE — $(date)"
echo "════════════════════════════════════════════════════════"
