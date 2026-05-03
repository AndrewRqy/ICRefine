#!/usr/bin/env bash
# ea_no_oracle_ff.sh — EA + no-oracle pipeline for formal_fallacies, 3 seeds.
#
# formal_fallacies was omitted from the original ea_phase1.sh run (CJ/GS/SN/DQ
# only). This script fills that gap so the combined EA+no-oracle head-to-head
# table can cover all 5 non-ceiling tasks uniformly.
#
# Saves into the same directory layout as the existing EA variance seeds:
#   seed 1 → runs/bbh_ea_phase1/formal_fallacies/
#   seed 2 → runs/variance/ea_seed2/formal_fallacies/
#   seed 3 → runs/variance/ea_seed3/formal_fallacies/
#
# All 3 seeds launch in parallel.
#
# Usage:
#   bash scripts/pipeline/ea_no_oracle_ff.sh
#   bash scripts/pipeline/ea_no_oracle_ff.sh --dry-run
#
# Estimated wall time: ~1-2h (3 parallel EA runs on a single task).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
if [ -f "$ROOT/.env" ]; then set -a; source "$ROOT/.env"; set +a; fi

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=true; fi

MODEL="openai/gpt-4.1-mini"
CONCUR=25
TASK="formal_fallacies"
DATASET="datasets/bbh/${TASK}_train.jsonl"
LOG_DIR="runs/variance/logs"

mkdir -p "$LOG_DIR"

echo "=== EA + no-oracle: formal_fallacies × 3 seeds ==="
echo "  model   : $MODEL"
echo "  dry_run : $DRY_RUN"
echo ""

PIDS=()

run_seed() {
    local seed="$1"
    local out_dir="$2"
    local log="$LOG_DIR/ea_seed${seed}_${TASK}.log"

    local cmd=(
        python3 -m ICR_hybrid.pipeline
        --task         "$TASK"
        --dataset      "$DATASET"
        --no-oracle
        --model-score      "$MODEL"
        --model-rule-patch "$MODEL"
        --model-casestudy  "$MODEL"
        --rule-concurrency "$CONCUR"
        --cs-concurrency   "$CONCUR"
        --max-rule-iters 4
        --max-cs-iters   5
        --cot-first
        --use-ea
        --ea-population-size     3
        --ea-n-survivors         2
        --ea-pk-size-budget      12000
        --ea-val-fraction        0.20
        --ea-failure-sample-frac 0.60
        --output-dir "$out_dir"
    )

    if $DRY_RUN; then
        echo "[dry-run] seed${seed}: ${cmd[*]}"
    else
        mkdir -p "$out_dir"
        echo "  starting seed ${seed} → $log"
        "${cmd[@]}" >> "$log" 2>&1 &
        PIDS+=($!)
    fi
}

run_seed 1 "runs/bbh_ea_phase1/$TASK"
run_seed 2 "runs/variance/ea_seed2/$TASK"
run_seed 3 "runs/variance/ea_seed3/$TASK"

if ! $DRY_RUN; then
    echo ""
    echo "Waiting for ${#PIDS[@]} runs..."
    FAILED=0
    for pid in "${PIDS[@]}"; do
        wait "$pid" || { echo "  FAILED  pid=$pid"; FAILED=$((FAILED+1)); }
    done
    [[ $FAILED -gt 0 ]] && { echo "$FAILED run(s) failed"; exit 1; }
    echo "  all done."
fi

echo ""
echo "=== Done ==="
echo "Next: bash scripts/eval/eval_ea_combined.sh"
