#!/usr/bin/env bash
# compare_modes.sh — Smoke-test and full comparison of ICR modes on normal dataset.
#
# Usage:
#   bash compare_modes.sh           # full comparison (100 training items)
#   bash compare_modes.sh smoke     # quick smoke test (15 items, no similarity gate)
#   bash compare_modes.sh eval      # eval only (skips training, reads existing cheatsheets)
#
# Run from the ICRefine/ directory:
#   cd "path/to/CHAI Project/ICRefine"
#   bash compare_modes.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Config — edit these if needed
# ---------------------------------------------------------------------------
SAIR_DIR="../SAIR_evaluation_pipeline"
DATASET="$SAIR_DIR/datasets/normal.jsonl"
BASE_CHEATSHEET="$SAIR_DIR/prompts/NeuriCo_cheatsheet.txt"
MODEL_SCORE="openai/gpt-oss-120b"
MODEL_CS="openai/gpt-4o"

TRAIN_LIMIT=100          # items used for training in each mode
EVAL_K=150               # held-out eval items
EVAL_SEED=99             # held-out seed — never used in training

SMOKE_LIMIT=100          # items used in smoke test
BIN_THRESHOLD=3
BATCH_SIZE=5

MODE="${1:-full}"        # "smoke" | "eval" | "full"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUT_REASONING="runs/compare_reasoning"
OUT_SELECT_CS="runs/compare_select_cs"
OUT_SELECT_DT="runs/compare_select_dt"
OUT_SMOKE_CS="runs/smoke_cs"
OUT_SMOKE_DT="runs/smoke_dt"

CS_REASONING="$SAIR_DIR/prompts/compare_reasoning.txt"
CS_SELECT_CS="$SAIR_DIR/prompts/compare_select_cs.txt"
CS_SELECT_DT="$SAIR_DIR/prompts/compare_select_dt.txt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
HR="────────────────────────────────────────────────────────────"

log() { echo -e "\n$HR\n$1\n$HR"; }

run_eval() {
    local label="$1"
    local cheatsheet="$2"
    log "EVAL: $label"
    python "$SAIR_DIR/run_evaluation.py" \
        --models "$MODEL_SCORE" \
        --datasets normal \
        --sample-k "$EVAL_K" \
        --sample-seed "$EVAL_SEED" \
        --cheatsheet "$cheatsheet" \
        --reasoning-effort low
}

# ---------------------------------------------------------------------------
# SMOKE TEST — verifies all gates and DT revision fire without errors
# ---------------------------------------------------------------------------
if [[ "$MODE" == "smoke" ]]; then
    log "SMOKE: ICR_select CS-only ($SMOKE_LIMIT items, 2 candidates, no similarity gate)"
    python -m ICR_select.pipeline \
        --dataset "$DATASET" \
        --init-txt "$BASE_CHEATSHEET" \
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
        --init-txt "$BASE_CHEATSHEET" \
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
# EVAL ONLY — run evaluation on already-generated cheatsheets
# ---------------------------------------------------------------------------
if [[ "$MODE" == "eval" ]]; then
    log "BASELINE: NeuriCo (no refinement)"
    run_eval "baseline (NeuriCo)" "$BASE_CHEATSHEET"

    for label_cs in \
        "A: ICR_reasoning|$CS_REASONING" \
        "B: ICR_select CS-only|$CS_SELECT_CS" \
        "C: ICR_select + DT revision|$CS_SELECT_DT"
    do
        label="${label_cs%%|*}"
        cs="${label_cs##*|}"
        if [[ -f "$cs" ]]; then
            run_eval "$label" "$cs"
        else
            echo "  [skip] $label — cheatsheet not found: $cs"
        fi
    done

    log "Eval complete. Compare accuracies above."
    exit 0
fi

# ---------------------------------------------------------------------------
# FULL — baseline eval → train all 3 modes → eval all 3 cheatsheets
# ---------------------------------------------------------------------------

# Step 1: Baseline
log "STEP 1/5: Baseline eval (NeuriCo, no refinement)"
run_eval "baseline (NeuriCo)" "$BASE_CHEATSHEET"

# Step 2: Train A — ICR_reasoning
log "STEP 2/5: Train A — ICR_reasoning (${TRAIN_LIMIT} items)"
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

# Step 3: Train B — ICR_select CS-only
log "STEP 3/5: Train B — ICR_select CS-only (${TRAIN_LIMIT} items)"
python -m ICR_select.pipeline \
    --dataset "$DATASET" \
    --init-txt "$BASE_CHEATSHEET" \
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

# Step 4: Train C — ICR_select + DT revision
log "STEP 4/5: Train C — ICR_select + DT revision (${TRAIN_LIMIT} items, 3 rounds)"
python -m ICR_select.pipeline \
    --dataset "$DATASET" \
    --init-txt "$BASE_CHEATSHEET" \
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

# Step 5: Eval all three on locked held-out set
log "STEP 5/5: Eval all cheatsheets on held-out set (seed=$EVAL_SEED, k=$EVAL_K)"
run_eval "A: ICR_reasoning"         "$CS_REASONING"
run_eval "B: ICR_select CS-only"    "$CS_SELECT_CS"
run_eval "C: ICR_select + DT revision" "$CS_SELECT_DT"

log "All done. Summary of held-out accuracies is in the eval output above."
