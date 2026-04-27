#!/usr/bin/env bash
# run_bbh_bootstrap.sh — CS-ICL bootstrap + ICR refinement on 6 BBH tasks.
#
# Approach:
#   1. Generate a CS-ICL style cheat sheet from the first 75 train items
#      (50 for snarks) → saved as Cheatsheet JSON in runs/bbh_bootstrap/{task}/
#   2. Run ICR Phase 2 on ALL 150 train items with the bootstrap cheat sheet
#      as the initial roadmap (--init-cheatsheet).
#   3. Eval against CS-ICL baseline (same eval_bbh_comparison.py as before).
#
# The hypothesis: seeding ICR with a CS-ICL derived roadmap (general principles)
# lets Phase 2 case studies address residual failures rather than rebuilding
# general knowledge from scratch.
#
# Models:  gpt-4.1-mini for gen/patch/casestudy and scoring.
# Output:  runs/bbh_bootstrap/
#
# Usage:
#   chmod +x run_bbh_bootstrap.sh
#   ./run_bbh_bootstrap.sh

set -euo pipefail

MODEL="openai/gpt-4.1-mini"
CONCURRENCY="${CONCURRENCY:-32}"
MAX_CS_ITERS=5
MAX_RULE_ITERS=3
BOOTSTRAP_N=20

DATA_DIR="datasets/bbh"
CS_ICL_DIR="../cheat-sheet-icl/data/cheat_prompt"
OUT_BASE="runs/bbh_bootstrap"
BOOTSTRAP_DIR="${OUT_BASE}/bootstrap_cheatsheets"

cd "$(dirname "$0")"
mkdir -p "${OUT_BASE}"
exec > >(tee -a "${OUT_BASE}/run_main.log") 2>&1

echo "BBH Bootstrap+ICR Run — $(date)"
echo "Model: ${MODEL}  Concurrency: ${CONCURRENCY}"
echo "Output: ${OUT_BASE}"
echo ""

# ── Step 1: Generate bootstrap cheat sheets ──────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  Step 1: Generate CS-ICL bootstrap cheat sheets"
echo "  (75 items per task, 50 for snarks)"
echo "════════════════════════════════════════════════════════"

python3 gen_icr_bootstrap.py \
    --model "${MODEL}" \
    --max-tokens 4000 \
    --out-base "${BOOTSTRAP_DIR}" \
    2>&1 | tee "${OUT_BASE}/bootstrap_gen.log"

echo "  Bootstrap cheat sheets generated: $(date)"
echo ""

# ── Step 2+3: Run each task ──────────────────────────────────────────────────

run_task() {
    local pipeline_task="$1"
    local eval_task="$2"
    local train_jsonl="$3"
    local bootstrap_json="${BOOTSTRAP_DIR}/${pipeline_task}/bootstrap_cs.json"
    local out_dir="${OUT_BASE}/${eval_task}"

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  TASK: ${pipeline_task}  (eval key: ${eval_task})"
    echo "  Bootstrap: ${bootstrap_json}"
    echo "  Start: $(date)"
    echo "════════════════════════════════════════════════════════"

    mkdir -p "${out_dir}"

    python3 -m ICR_hybrid.pipeline \
        --task "${pipeline_task}" \
        --dataset "${train_jsonl}" \
        --no-oracle \
        --init-cheatsheet "${bootstrap_json}" \
        --model-score "${MODEL}" \
        --model-rule-patch "${MODEL}" \
        --model-casestudy "${MODEL}" \
        --rule-concurrency "${CONCURRENCY}" \
        --cs-concurrency "${CONCURRENCY}" \
        --auto-rule-init \
        --bootstrap-n "${BOOTSTRAP_N}" \
        --rule-acc-goal 0.95 \
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
        --tasks "${eval_task}" \
        2>&1 | tee "${out_dir}/comparison.log"

    echo "  Done: $(date)"
}

run_task "formal_fallacies"        "formal_fallacies"        "${DATA_DIR}/formal_fallacies_train.jsonl"
run_task "logical_deduction_three" "logical_deduction_three" "${DATA_DIR}/logical_deduction_three_objects_train.jsonl"
run_task "web_of_lies"             "web_of_lies"             "${DATA_DIR}/web_of_lies_train.jsonl"
run_task "date_understanding"      "date_understanding"      "${DATA_DIR}/date_understanding_train.jsonl"
run_task "navigate"                "navigate"                "${DATA_DIR}/navigate_train.jsonl"
run_task "snarks"                  "snarks"                  "${DATA_DIR}/snarks_train.jsonl"

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "ALL PIPELINE TASKS DONE — $(date)"
echo "════════════════════════════════════════════════════════"

python3 - <<'EOF'
import json, pathlib, sys

def load(p):
    p = pathlib.Path(p)
    return json.loads(p.read_text()) if p.exists() else {}

bootstrap = load("runs/bbh_bootstrap/comparison_results.json")
baseline  = load("runs/bbh_ext/comparison_results.json")

tasks = ["formal_fallacies", "logical_deduction_three", "web_of_lies",
         "date_understanding", "navigate", "snarks"]

print("\n" + "="*85)
print("FINAL ACCURACY SUMMARY (test split)")
print(f"{'Task':<30} {'N_test':>6}  {'CS-ICL':>8}  {'ICR-base':>9}  {'Bootstrap+ICR':>13}  {'Δ-cs':>6}  {'Δ-base':>7}")
print("-"*85)
for task in tasks:
    row_b  = bootstrap.get(task, {})
    row_bl = baseline.get(task, {})
    cs   = f"{row_b['cs_icl_acc']:.1%}"  if row_b.get("cs_icl_acc")  is not None else "N/A"
    ours = f"{row_b['ours_acc']:.1%}"    if row_b.get("ours_acc")     is not None else "N/A"
    base = f"{row_bl['ours_acc']:.1%}"   if row_bl.get("ours_acc")    is not None else "N/A"
    ntest = row_b.get("n_test", "?")
    d_cs   = (f"{row_b['ours_acc'] - row_b['cs_icl_acc']:+.1%}"
              if row_b.get("cs_icl_acc") and row_b.get("ours_acc") else "N/A")
    d_base = (f"{row_b['ours_acc'] - row_bl['ours_acc']:+.1%}"
              if row_bl.get("ours_acc") and row_b.get("ours_acc") else "N/A")
    print(f"{task:<30} {str(ntest):>6}  {cs:>8}  {base:>9}  {ours:>13}  {d_cs:>6}  {d_base:>7}")
print("="*85)
EOF

echo ""
echo "════════════════════════════════════════════════════════"
echo "COMPLETE — $(date)"
echo "════════════════════════════════════════════════════════"
