#!/usr/bin/env bash
# run_magma_comparison.sh — CS-ICL vs ICR comparison on MAGMA hard3 subset.
#
# Dataset: hard3, 100 items with correct GPT-5.4 reasoning (50T/50F)
#          train=100, test=42
#
# Models:
#   Cheat-sheet generation : gpt-4.1  (both CS-ICL and ICR gen/patch)
#   Scoring / eval         : gpt-oss-120b  (via OpenRouter — strong enough for MAGMA)
#
# Usage:
#   chmod +x run_magma_comparison.sh
#   ./run_magma_comparison.sh

set -euo pipefail

GEN_MODEL="openai/gpt-4.1"
SCORE_MODEL="openai/gpt-oss-120b"
CONCURRENCY="${CONCURRENCY:-16}"
OUT_DIR="runs/magma_comparison_gpt41"

cd "$(dirname "$0")"
mkdir -p "${OUT_DIR}"
exec > >(tee -a "${OUT_DIR}/run_main.log") 2>&1

echo "MAGMA Comparison Run — $(date)"
echo "Gen model:   ${GEN_MODEL}"
echo "Score model: ${SCORE_MODEL}  Concurrency: ${CONCURRENCY}"
echo "Dataset:     hard3 subset, GPT-5.4 correct only (train=100 / test=42)"
echo ""

# ── Step 1: Generate CS-ICL cheat sheet ──────────────────────────────────────
echo "════════════════════════════════════════════"
echo "  Step 1: Generate CS-ICL cheat sheet"
echo "════════════════════════════════════════════"

python3 gen_csicl_magma.py \
    --model "${GEN_MODEL}" \
    --max-tokens 4000 \
    2>&1 | tee "${OUT_DIR}/csicl_gen.log"

# ── Step 2: Run ICR pipeline ──────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "  Step 2: ICR pipeline"
echo "  Start: $(date)"
echo "════════════════════════════════════════════"

python3 -m ICR_hybrid.pipeline \
    --task magma \
    --dataset datasets/magma_train.jsonl \
    --oracle-csv datasets/hard3_gpt54_oracle.csv \
    --model-score "${SCORE_MODEL}" \
    --model-rule-patch "${GEN_MODEL}" \
    --model-casestudy "${GEN_MODEL}" \
    --rule-concurrency "${CONCURRENCY}" \
    --cs-concurrency "${CONCURRENCY}" \
    --auto-rule-init \
    --bootstrap-n 20 \
    --rule-acc-goal 1.01 \
    --max-rule-iters 3 \
    --max-cs-iters 5 \
    --pk-regression-guard \
    --pk-regression-tolerance 0.03 \
    --output-dir "${OUT_DIR}/magma" \
    2>&1 | tee "${OUT_DIR}/icr_run.log"

echo "  Pipeline done: $(date)"

# ── Step 3: Comparison eval ───────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "  Step 3: Comparison eval"
echo "════════════════════════════════════════════"

python3 eval_bbh_comparison.py \
    --run-dir "${OUT_DIR}" \
    --cs-icl-dir "../cheat-sheet-icl/data/cheat_prompt" \
    --model "${SCORE_MODEL}" \
    --concurrency "${CONCURRENCY}" \
    --tasks magma \
    2>&1 | tee "${OUT_DIR}/comparison.log"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
python3 - <<'EOF'
import json, pathlib, sys
p = pathlib.Path("runs/magma_comparison_gpt41/comparison_results.json")
if not p.exists():
    print("No results found."); sys.exit(0)
results = json.loads(p.read_text())
row = results.get("magma", {})
cs  = f"{row['cs_icl_acc']:.1%}" if row.get("cs_icl_acc") is not None else "N/A"
our = f"{row['ours_acc']:.1%}"   if row.get("ours_acc")   is not None else "N/A"
delta = (f"{row['ours_acc'] - row['cs_icl_acc']:+.1%}"
         if row.get("cs_icl_acc") and row.get("ours_acc") else "N/A")
print("=" * 50)
print(f"  MAGMA hard3 subset  (train=100 / test=42)")
print(f"  CS-ICL (oracle reasoning): {cs}")
print(f"  Ours (ICR):                {our}")
print(f"  Delta:                     {delta}")
print("=" * 50)
EOF

echo ""
echo "COMPLETE — $(date)"
