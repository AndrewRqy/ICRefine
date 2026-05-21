#!/usr/bin/env bash
# run_v7_bbh5_remaining.sh — Faithful v7 rebuild on GS, FF, SN × 3 seeds.
#
# Uses icr_holistic_v7 (label-only rewriter) with mini CS-ICL init.
# Matches historical v7 params used for CJ/DQ.
# Runs 3 jobs at a time.
#
# Usage:
#   bash scripts/pipeline/run_v7_bbh5_remaining.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
if [ -f "$ROOT/.env" ]; then set -a; source "$ROOT/.env"; set +a; fi

MODEL="openai/gpt-4.1-mini"
SEEDS=(1000 2000 3000)

mkdir -p logs

TASK_DEFS=(
    "geometric_shapes|gs|datasets/bbh/geometric_shapes_train_labeled.jsonl"
    "formal_fallacies|ff|datasets/bbh/formal_fallacies_train_labeled.jsonl"
    "snarks|sn|datasets/bbh/snarks_train_labeled.jsonl"
)

# Build job list
JOBS=()
for TASK_DEF in "${TASK_DEFS[@]}"; do
    IFS='|' read -r TASK ABBR DATASET <<< "$TASK_DEF"
    for SEED in "${SEEDS[@]}"; do
        JOBS+=("${TASK}|${ABBR}|${DATASET}|${SEED}")
    done
done

echo "=== ${#JOBS[@]} jobs queued (3 at a time) ==="
echo ""

i=0; total=${#JOBS[@]}; batch=0
while [ $i -lt $total ]; do
    batch=$((batch+1))
    batch_pids=(); batch_labels=()
    for ((b=0; b<3 && i<total; b++, i++)); do
        IFS='|' read -r TASK ABBR DATASET SEED <<< "${JOBS[$i]}"
        OUTDIR="ICR_paper_ready/holistic_${ABBR}_v7_${SEED}"
        LOG="logs/v7_${ABBR}_${SEED}_$(date +%Y%m%d_%H%M%S).log"
        echo "--- Launching v7 ${ABBR}_${SEED}  →  ${LOG}"
        python3 -m icr_holistic_v7.pipeline \
            --task "$TASK" \
            --dataset "$DATASET" \
            --model-score "$MODEL" \
            --model-gen   "$MODEL" \
            --output-dir  "$OUTDIR" \
            --max-iters 8 \
            --fix-rate 0.10 \
            --regression-pool 100 \
            --bin-threshold 2 \
            --concurrency 30 \
            --bin-retry 2 \
            --rollback \
            --min-pool-for-net-gate 4 \
            --early-stop-patience 3 \
            --beam-size 2 \
            --cs-icl-tokens "$SEED" \
            > "$LOG" 2>&1 &
        batch_pids+=($!); batch_labels+=("${ABBR}_${SEED}")
    done
    echo "--- Batch ${batch}: waiting for ${batch_labels[*]}..."
    for pid in "${batch_pids[@]}"; do wait "$pid"; done
    echo "[$(date '+%H:%M:%S')] Batch ${batch} done: ${batch_labels[*]}"
    echo ""
done

echo "=== All faithful v7 BBH-remaining runs complete ==="
echo "  GS/FF/SN seeds 1000/2000/3000 → ICR_paper_ready/holistic_{gs,ff,sn}_v7_{seed}"
