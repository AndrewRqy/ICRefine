#!/usr/bin/env bash
# compare_modes.sh — Train and compare ICR modes on the same dataset.
#
# Usage:
#   bash compare_modes.sh           # full comparison (100 training items)
#   bash compare_modes.sh smoke     # quick smoke test (15 items, no similarity gate)
#
# Run from the ICRefine/ directory:
#   cd "path/to/ICRefine"
#   bash compare_modes.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Config — edit these before running
# ---------------------------------------------------------------------------
DATASET="path/to/dataset.jsonl"
BASE_CHEATSHEET="path/to/prior_knowledge.txt"
MODEL_SCORE="openai/gpt-oss-120b"
MODEL_CS="openai/gpt-4o"

TRAIN_LIMIT=100          # items used for training in each mode
SMOKE_LIMIT=15           # items used in smoke test
BIN_THRESHOLD=3
BATCH_SIZE=5

MODE="${1:-full}"        # "smoke" | "full"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUT_REASONING="runs/compare_reasoning"
OUT_SELECT_CS="runs/compare_select_cs"
OUT_SELECT_DT="runs/compare_select_dt"
OUT_SMOKE_CS="runs/smoke_cs"
OUT_SMOKE_DT="runs/smoke_dt"

CS_REASONING="runs/compare_reasoning/cheatsheet_final.txt"
CS_SELECT_CS="runs/compare_select_cs/cheatsheet_final.txt"
CS_SELECT_DT="runs/compare_select_dt/cheatsheet_final.txt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
HR="────────────────────────────────────────────────────────────"

log() { echo -e "\n$HR\n$1\n$HR"; }

# ---------------------------------------------------------------------------
# SMOKE TEST — verifies all gates and DT revision fire without errors
# ---------------------------------------------------------------------------
if [[ "$MODE" == "smoke" ]]; then
    log "SMOKE: ICR_select CS-only ($SMOKE_LIMIT items, 2 candidates, no similarity gate)"
    python -m ICR_select.pipeline \
        --dataset "$DATASET" \
        --prior-knowledge "$BASE_CHEATSHEET" \
        --model-score "$MODEL_SCORE" \
        --model-casestudy "$MODEL_CS" \
        --bin-threshold "$BIN_THRESHOLD" \
        --batch-size "$BATCH_SIZE" \
        --limit "$SMOKE_LIMIT" \
        --n-candidates 2 \
        --no-similarity-gate \
        --dt-rounds 1 \
        --output-dir "$OUT_SMOKE_CS"

    log "SMOKE: ICR_select + DT revision ($SMOKE_LIMIT items, 2 rounds, min-failures=3)"
    python -m ICR_select.pipeline \
        --dataset "$DATASET" \
        --prior-knowledge "$BASE_CHEATSHEET" \
        --model-score "$MODEL_SCORE" \
        --model-casestudy "$MODEL_CS" \
        --bin-threshold "$BIN_THRESHOLD" \
        --batch-size "$BATCH_SIZE" \
        --limit "$SMOKE_LIMIT" \
        --n-candidates 2 \
        --no-similarity-gate \
        --dt-rounds 2 \
        --min-failures-for-dt 3 \
        --output-dir "$OUT_SMOKE_DT"

    log "SMOKE complete. Check logs above for [gate:fix_rate], [gate:regression], [dt_revise] messages."
    exit 0
fi

# ---------------------------------------------------------------------------
# FULL — train all 3 modes
# ---------------------------------------------------------------------------

# Step 1: Train A — ICR_reasoning
log "STEP 1/3: Train A — ICR_reasoning (${TRAIN_LIMIT} items)"
python -m ICR_reasoning.pipeline \
    --dataset "$DATASET" \
    --init-txt "$BASE_CHEATSHEET" \
    --model-score "$MODEL_SCORE" \
    --model-casestudy "$MODEL_CS" \
    --bin-threshold "$BIN_THRESHOLD" \
    --batch-size "$BATCH_SIZE" \
    --val-split 0.0 \
    --limit "$TRAIN_LIMIT" \
    --no-analysis \
    --output-dir "$OUT_REASONING" \
    --cheatsheet-out "$CS_REASONING"

# Step 2: Train B — ICR_select CS-only
log "STEP 2/3: Train B — ICR_select CS-only (${TRAIN_LIMIT} items)"
python -m ICR_select.pipeline \
    --dataset "$DATASET" \
    --prior-knowledge "$BASE_CHEATSHEET" \
    --model-score "$MODEL_SCORE" \
    --model-casestudy "$MODEL_CS" \
    --bin-threshold "$BIN_THRESHOLD" \
    --batch-size "$BATCH_SIZE" \
    --limit "$TRAIN_LIMIT" \
    --n-candidates 3 \
    --fix-rate-threshold 0.5 \
    --regress-threshold 0.1 \
    --dt-rounds 1 \
    --output-dir "$OUT_SELECT_CS" \
    --cheatsheet-out "$CS_SELECT_CS"

# Step 3: Train C — ICR_select + DT revision
log "STEP 3/3: Train C — ICR_select + DT revision (${TRAIN_LIMIT} items, 3 rounds)"
python -m ICR_select.pipeline \
    --dataset "$DATASET" \
    --prior-knowledge "$BASE_CHEATSHEET" \
    --model-score "$MODEL_SCORE" \
    --model-casestudy "$MODEL_CS" \
    --bin-threshold "$BIN_THRESHOLD" \
    --batch-size "$BATCH_SIZE" \
    --limit "$TRAIN_LIMIT" \
    --n-candidates 3 \
    --fix-rate-threshold 0.5 \
    --regress-threshold 0.1 \
    --dt-rounds 3 \
    --plateau-threshold 0.02 \
    --output-dir "$OUT_SELECT_DT" \
    --cheatsheet-out "$CS_SELECT_DT"

log "All done. Cheatsheets saved to:"
echo "  A (ICR_reasoning):         $CS_REASONING"
echo "  B (ICR_select CS-only):    $CS_SELECT_CS"
echo "  C (ICR_select + DT):       $CS_SELECT_DT"
