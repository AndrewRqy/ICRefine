#!/usr/bin/env bash
# magma_large.sh — Magma pipeline on the combined hard2+hard3+normal dataset.
#
# Training set : datasets/magma_large_train.jsonl  (954 items, 477T/477F)
# Test set     : datasets/magma_large_test.jsonl   (636 items, 318T/318F)
#                evaluated separately via eval_magma_large.py
#
# Key differences from the original 100-item magma run:
#   - 9.5× more training items (954 vs 100)
#   - bootstrap-n-items scaled to 200 (vs default 75) for richer Phase 0
#   - gpt-oss-120b as scorer (strong reasoning → correct patch/CS acceptance)
#   - gpt-4.1-mini for generation (patches + case studies, unchanged)
#   - Oracle: combined hard2+hard3 oracle CSV (327 correct traces) used for
#     Phase 0 bootstrap + Phase 2 CS generation; Phase 1 oracle disabled
#     (--no-phase1-oracle) since Phase 1 does 0 patches on magma anyway
#   - cot-first (RF mode)
#
# Usage:
#   bash scripts/pipeline/magma_large.sh
#   bash scripts/pipeline/magma_large.sh --dry-run

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

if [ -f "$ROOT/.env" ]; then set -a; source "$ROOT/.env"; set +a; fi

MODEL_SCORE="openai/gpt-oss-120b"
MODEL_GEN="openai/gpt-4.1-mini"
DATASET="datasets/magma_large_train.jsonl"
OUT_DIR="runs/magma_large"
CONCUR=20
LOG="$OUT_DIR/pipeline.log"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=true; fi

echo "=== Magma large pipeline ==="
echo "  score model       : $MODEL_SCORE"
echo "  gen model         : $MODEL_GEN"
echo "  dataset           : $DATASET  (954 items)"
echo "  bootstrap-n-items : 200  (scaled from default 75)"
echo "  oracle            : datasets/magma_combined_oracle.csv  (327 traces, P2 only)"
echo "  output            : $OUT_DIR"
echo "  dry_run           : $DRY_RUN"
echo ""

cmd=(
    python3 -m ICR_hybrid.pipeline
    --task                magma
    --dataset             "$DATASET"
    --oracle-csv          "datasets/magma_combined_oracle.csv"
    --no-phase1-oracle
    --model-score         "$MODEL_SCORE"
    --model-rule-patch    "$MODEL_GEN"
    --model-casestudy     "$MODEL_GEN"
    --rule-concurrency    "$CONCUR"
    --cs-concurrency      "$CONCUR"
    --max-rule-iters      4
    --max-cs-iters        5
    --cot-first
    --bootstrap-n-items   200
    --output-dir          "$OUT_DIR"
)

if $DRY_RUN; then
    echo "[dry-run] ${cmd[*]}"
else
    mkdir -p "$OUT_DIR"
    echo "Launching pipeline → $LOG"
    "${cmd[@]}" 2>&1 | tee "$LOG"
    echo ""
    echo "=== Pipeline complete ==="
    echo "Cheatsheet: $OUT_DIR/cheatsheet_final.txt"
    echo ""
    echo "Next: run test eval:"
    echo "  python3 scripts/eval/eval_magma_large.py"
fi
