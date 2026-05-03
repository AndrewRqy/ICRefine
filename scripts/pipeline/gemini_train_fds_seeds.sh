#!/usr/bin/env bash
# gemini_train_fds_seeds.sh — Seeds 2 & 3 for the Gemini-trained pipeline
# on FF, DQ, and Snarks (the three tasks not covered by gemini_train_cj_gs_seeds.sh).
#
# Pre-requisites:
#   - gemini_train_cj_gs_seeds.sh must have completed (CJ+GS seeds 2&3 done)
#   - runs/gemini_train_v3/{task}/ must exist for all 5 tasks (seed 1)
#
# After this script completes, all 5 tasks × 3 seeds are covered.
# The eval step re-runs all 3 seeds × 5 tasks (overwriting the CJ+GS-only
# eval files written by gemini_train_cj_gs_seeds.sh) and re-aggregates.
#
# Output:
#   Pipeline: runs/gemini_train_v3_seed{2,3}/{formal_fallacies,disambiguation_qa,snarks}/
#   Eval:     runs/gemini_train_v3_eval_results/seed{1,2,3}_rf.json  (5-task, overwrites)
#   3-seed:   runs/gemini_train_v3_3seed_mean.json
#
# Usage:
#   bash scripts/pipeline/gemini_train_fds_seeds.sh
#   bash scripts/pipeline/gemini_train_fds_seeds.sh --dry-run
#
# Estimated wall time: ~4-6h pipeline (6 parallel runs) + ~15min eval.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
if [ -f "$ROOT/.env" ]; then set -a; source "$ROOT/.env"; set +a; fi

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=true; fi

MODEL="google/gemini-2.0-flash-001"
CONCUR=25
TASKS="formal_fallacies disambiguation_qa snarks"
EVAL_OUT="runs/gemini_train_v3_eval_results"
MEAN_OUT="runs/gemini_train_v3_3seed_mean.json"
LOG_DIR="runs/gemini_train_v3_seed2/logs"

mkdir -p "$LOG_DIR" "$EVAL_OUT"

echo "=== Gemini-trained seeds 2 & 3: FF + DQ + Snarks ==="
echo "  model   : $MODEL"
echo "  dry_run : $DRY_RUN"
echo ""

PIDS=()
LABELS=()

run_pipeline() {
    local seed="$1"
    local task="$2"
    local out_dir="runs/gemini_train_v3_seed${seed}/$task"
    local log="runs/gemini_train_v3_seed${seed}/logs/${task}.log"

    mkdir -p "$(dirname "$log")"

    local cmd=(
        python3 -m ICR_hybrid.pipeline
        --task         "$task"
        --dataset      "datasets/bbh/${task}_train.jsonl"
        --no-oracle
        --model-score      "$MODEL"
        --model-rule-patch "$MODEL"
        --model-casestudy  "$MODEL"
        --rule-concurrency "$CONCUR"
        --cs-concurrency   "$CONCUR"
        --max-rule-iters 4
        --max-cs-iters   5
        --cot-first
        --output-dir "$out_dir"
    )

    if $DRY_RUN; then
        echo "[dry-run] seed${seed} $task: ${cmd[*]}"
    else
        mkdir -p "$out_dir"
        echo "  starting seed${seed} $task → $log"
        "${cmd[@]}" >> "$log" 2>&1 &
        PIDS+=($!)
    fi
    LABELS+=("seed${seed}_${task}")
}

for seed in 2 3; do
    for task in formal_fallacies disambiguation_qa snarks; do
        run_pipeline "$seed" "$task"
    done
done

if ! $DRY_RUN; then
    echo ""
    echo "Waiting for ${#PIDS[@]} pipeline runs..."
    FAILED=0
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo "  OK      ${LABELS[$i]}"
        else
            echo "  FAILED  ${LABELS[$i]}"
            FAILED=$((FAILED+1))
        fi
    done
    [[ $FAILED -gt 0 ]] && { echo "$FAILED run(s) failed"; exit 1; }
fi

echo ""
echo "=== Pipeline complete — evaluating all 3 seeds × 5 tasks ==="
echo ""

EVAL_PIDS=()
EVAL_LABELS=()
EVAL_LOGS=()

launch_eval() {
    local seed="$1"
    local out="$2"
    local log="$EVAL_OUT/seed${seed}_eval.log"
    shift 2

    EVAL_LABELS+=("seed${seed}")
    EVAL_LOGS+=("$log")

    local cmd=(
        python3 scripts/eval/eval_cs_ablation.py
        --tasks causal_judgement geometric_shapes formal_fallacies disambiguation_qa snarks
        --reasoning-first
        --no-csicl
        --concurrency "$CONCUR"
        --run-dir-overrides "$@"
        --out "$out"
    )

    if $DRY_RUN; then
        echo "[dry-run] eval seed${seed}: ${cmd[*]}"
        EVAL_PIDS+=("skip")
    else
        echo "  evaluating seed${seed} → $log"
        "${cmd[@]}" >> "$log" 2>&1 &
        EVAL_PIDS+=($!)
    fi
}

# seed 1: all 5 tasks from original gemini_train_v3 run
launch_eval 1 "$EVAL_OUT/seed1_rf.json" \
    "causal_judgement:runs/gemini_train_v3/causal_judgement" \
    "geometric_shapes:runs/gemini_train_v3/geometric_shapes" \
    "formal_fallacies:runs/gemini_train_v3/formal_fallacies" \
    "disambiguation_qa:runs/gemini_train_v3/disambiguation_qa" \
    "snarks:runs/gemini_train_v3/snarks"

# seeds 2 & 3: CJ+GS from cj_gs script, FF+DQ+Snarks from this script
for seed in 2 3; do
    launch_eval "$seed" "$EVAL_OUT/seed${seed}_rf.json" \
        "causal_judgement:runs/gemini_train_v3_seed${seed}/causal_judgement" \
        "geometric_shapes:runs/gemini_train_v3_seed${seed}/geometric_shapes" \
        "formal_fallacies:runs/gemini_train_v3_seed${seed}/formal_fallacies" \
        "disambiguation_qa:runs/gemini_train_v3_seed${seed}/disambiguation_qa" \
        "snarks:runs/gemini_train_v3_seed${seed}/snarks"
done

if ! $DRY_RUN; then
    echo ""
    echo "Waiting for ${#EVAL_PIDS[@]} eval jobs..."
    FAILED=0
    for i in "${!EVAL_PIDS[@]}"; do
        if wait "${EVAL_PIDS[$i]}"; then
            echo "  OK      ${EVAL_LABELS[$i]}"
        else
            echo "  FAILED  ${EVAL_LABELS[$i]}  (see ${EVAL_LOGS[$i]})"
            FAILED=$((FAILED+1))
        fi
    done
    [[ $FAILED -gt 0 ]] && { echo "$FAILED eval(s) failed"; exit 1; }
fi

echo ""
echo "=== Aggregating 5-task 3-seed mean ==="
cmd=(
    python3 scripts/eval/aggregate_variance.py
    --condition gemini_train
    --seed1 "$EVAL_OUT/seed1_rf.json"
    --seed2 "$EVAL_OUT/seed2_rf.json"
    --seed3 "$EVAL_OUT/seed3_rf.json"
    --out   "$MEAN_OUT"
)
if $DRY_RUN; then
    echo "[dry-run] ${cmd[*]}"
else
    "${cmd[@]}"
fi

echo ""
echo "=== Done ==="
echo "  5-task 3-seed mean (Gemini-trained): $MEAN_OUT"
