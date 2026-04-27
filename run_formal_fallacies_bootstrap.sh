#!/usr/bin/env bash
# run_formal_fallacies_bootstrap.sh — Bootstrap+ICR on formal_fallacies.
#
# Skips any step whose output already exists so re-runs are safe.
#
# Steps:
#   1. Generate CS-ICL bootstrap cheat sheet from first 75 train items
#   2. Run ICR pipeline with bootstrap seed + Phase 1 gap-filling + Phase 2 CS
#   3. Eval against CS-ICL baseline
#
# Baseline to beat: CS-ICL 60%  ICR-only 65%
#
# Usage:
#   chmod +x run_formal_fallacies_bootstrap.sh
#   ./run_formal_fallacies_bootstrap.sh

set -euo pipefail

MODEL="openai/gpt-4.1-mini"
CONCURRENCY="${CONCURRENCY:-32}"
TASK="formal_fallacies"
TRAIN_JSONL="datasets/bbh/formal_fallacies_train.jsonl"
CS_ICL_DIR="../cheat-sheet-icl/data/cheat_prompt"
BOOTSTRAP_DIR="runs/bbh_bootstrap/bootstrap_cheatsheets"
BOOTSTRAP_JSON="${BOOTSTRAP_DIR}/${TASK}/bootstrap_cs.json"
OUT_DIR="runs/bbh_bootstrap/${TASK}"
RUN_BASE="runs/bbh_bootstrap"

cd "$(dirname "$0")"
mkdir -p "${OUT_DIR}" "${RUN_BASE}"
exec > >(tee -a "${RUN_BASE}/${TASK}_run_main.log") 2>&1

echo "Bootstrap+ICR: ${TASK} — $(date)"
echo "Model: ${MODEL}  Concurrency: ${CONCURRENCY}"
echo ""

# ── Step 1: Bootstrap cheat sheet ────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  Step 1: CS-ICL bootstrap cheat sheet"
echo "════════════════════════════════════════════════════════"

if [ -f "${BOOTSTRAP_JSON}" ]; then
    echo "  Already exists — skipping. (${BOOTSTRAP_JSON})"
else
    python3 gen_icr_bootstrap.py \
        --tasks "${TASK}" \
        --model "${MODEL}" \
        --out-base "${BOOTSTRAP_DIR}" \
        2>&1 | tee "${RUN_BASE}/${TASK}_bootstrap_gen.log"
    echo "  Done: $(date)"
fi

echo ""

# ── Step 2: ICR pipeline ──────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  Step 2: ICR pipeline (Phase 1 gap-fill + Phase 2 CS)"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════"

FINAL_CS="${OUT_DIR}/cheatsheet_final.json"

if [ -f "${FINAL_CS}" ]; then
    echo "  Already exists — skipping. (${FINAL_CS})"
else
    python3 -m ICR_hybrid.pipeline \
        --task "${TASK}" \
        --dataset "${TRAIN_JSONL}" \
        --no-oracle \
        --init-cheatsheet "${BOOTSTRAP_JSON}" \
        --model-score "${MODEL}" \
        --model-rule-patch "${MODEL}" \
        --model-casestudy "${MODEL}" \
        --rule-concurrency "${CONCURRENCY}" \
        --cs-concurrency "${CONCURRENCY}" \
        --auto-rule-init \
        --bootstrap-n 20 \
        --rule-acc-goal 0.95 \
        --max-rule-iters 3 \
        --max-cs-iters 5 \
        --pk-regression-guard \
        --pk-regression-tolerance 0.03 \
        --output-dir "${OUT_DIR}" \
        2>&1 | tee "${RUN_BASE}/${TASK}_icr_run.log"
    echo "  Pipeline done: $(date)"
fi

echo ""

# ── Step 3: Comparison eval ───────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  Step 3: Comparison eval vs CS-ICL baseline"
echo "════════════════════════════════════════════════════════"

RESULTS_JSON="${RUN_BASE}/comparison_results.json"
TASK_IN_RESULTS=$(python3 -c "
import json, pathlib, sys
p = pathlib.Path('${RESULTS_JSON}')
if p.exists():
    d = json.loads(p.read_text())
    sys.exit(0 if '${TASK}' in d else 1)
sys.exit(1)
" 2>/dev/null && echo "yes" || echo "no")

if [ "${TASK_IN_RESULTS}" = "yes" ]; then
    echo "  Already in results — skipping eval."
else
    python3 eval_bbh_comparison.py \
        --run-dir "${RUN_BASE}" \
        --cs-icl-dir "${CS_ICL_DIR}" \
        --model "${MODEL}" \
        --concurrency "${CONCURRENCY}" \
        --tasks "${TASK}" \
        2>&1 | tee "${RUN_BASE}/${TASK}_comparison.log"
fi

echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
python3 - <<'EOF'
import json, pathlib, sys

p = pathlib.Path("runs/bbh_bootstrap/comparison_results.json")
if not p.exists():
    print("No results yet."); sys.exit(0)

results = json.loads(p.read_text())
row = results.get("formal_fallacies", {})
cs   = f"{row['cs_icl_acc']:.1%}" if row.get("cs_icl_acc") is not None else "N/A"
ours = f"{row['ours_acc']:.1%}"   if row.get("ours_acc")   is not None else "N/A"
delta = (f"{row['ours_acc'] - row['cs_icl_acc']:+.1%}"
         if row.get("cs_icl_acc") and row.get("ours_acc") else "N/A")

print("=" * 50)
print(f"  formal_fallacies  (test={row.get('n_test','?')})")
print(f"  CS-ICL baseline : {cs}")
print(f"  Bootstrap + ICR : {ours}")
print(f"  Delta           : {delta}")
print("=" * 50)
EOF

echo ""
echo "COMPLETE — $(date)"
