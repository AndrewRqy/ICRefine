#!/usr/bin/env bash
# variance_runs.sh — Seed variance runs for key claims.
#
# Fires 16 pipeline re-runs across 3 groups to estimate pipeline variance
# on non-ceiling tasks for the claims that need statistical support.
#
# Group A — v3 baseline (2 seeds each for GS, CJ, DQ, FF, Snarks):
#   5 non-ceiling tasks × 2 seeds = 10 runs.
#   Validates Phase 2 penalties, oracle contamination, non-train transfer.
#
# Group B — EA Phase 1 (2 seeds each for GS, DQ):
#   Validates EA pk_only +11pp on GS and Gemini collapse on DQ.
#
# Group C — E3 no-oracle (2 seeds for CJ):
#   Validates +9.2pp oracle contamination recovery.
#
# All 16 runs launch in parallel. Output dirs:
#   runs/variance/v3_seed{2,3}/{task}/
#   runs/variance/ea_seed{2,3}/{task}/
#   runs/variance/e3_seed{2,3}/causal_judgement/
#
# After completion, eval with:
#   bash scripts/eval/run_variance_eval.sh
#
# Estimated wall time: ~3-4h (all parallel).
#
# Usage:
#   bash scripts/pipeline/variance_runs.sh
#   bash scripts/pipeline/variance_runs.sh --dry-run

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

MODEL="openai/gpt-4.1-mini"
CONCUR=25
OUT_BASE="runs/variance"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

LOG_DIR="$OUT_BASE/logs"

echo "=== Variance runs (16 parallel pipeline runs) ==="
echo "  model   : $MODEL"
echo "  out     : $OUT_BASE"
echo "  dry_run : $DRY_RUN"
echo ""

run_pipeline() {
    local task="$1"
    local out_dir="$2"
    local log_file="$LOG_DIR/$(basename $(dirname $out_dir))_${task}.log"
    shift 2
    local extra_args=("$@")

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
        --output-dir   "$out_dir"
        "${extra_args[@]}"
    )

    if $DRY_RUN; then
        echo "[dry-run] ${cmd[*]}"
    else
        mkdir -p "$LOG_DIR"
        echo "  launching $task → $out_dir"
        "${cmd[@]}" >> "$log_file" 2>&1 &
    fi
}

# ── Group A: v3 baseline seeds ────────────────────────────────────────────────
# 5 tasks × 2 seeds = 10 runs
# Tasks: GS, CJ, DQ (primary claims) + FF, Snarks (borderline but included
# for a fuller variance picture across all non-ceiling tasks)
echo "--- Group A: v3 baseline (GS, CJ, DQ, FF, Snarks — seeds 2 & 3) ---"
for seed in 2 3; do
    for task in geometric_shapes causal_judgement disambiguation_qa \
                formal_fallacies snarks; do
        run_pipeline "$task" "$OUT_BASE/v3_seed${seed}/$task"
    done
done

# ── Group B: EA Phase 1 seeds ─────────────────────────────────────────────────
# 2 tasks × 2 seeds = 4 runs
echo ""
echo "--- Group B: EA Phase 1 (GS, DQ — seeds 2 & 3) ---"
for seed in 2 3; do
    for task in geometric_shapes disambiguation_qa; do
        run_pipeline "$task" "$OUT_BASE/ea_seed${seed}/$task" \
            --use-ea \
            --ea-population-size     3 \
            --ea-n-survivors         2 \
            --ea-pk-size-budget      12000 \
            --ea-val-fraction        0.20 \
            --ea-failure-sample-frac 0.60
    done
done

# ── Group C: E3 no-oracle seeds ───────────────────────────────────────────────
# 1 task × 2 seeds = 2 runs
echo ""
echo "--- Group C: E3 no-oracle (CJ — seeds 2 & 3) ---"
for seed in 2 3; do
    run_pipeline "causal_judgement" "$OUT_BASE/e3_seed${seed}/causal_judgement"
done

# ── Wait ──────────────────────────────────────────────────────────────────────
if ! $DRY_RUN; then
    echo ""
    echo "All 16 runs launched — waiting ..."
    wait
    echo "All done."
fi

echo ""
echo "=== Variance runs complete ==="
echo "Results in: $OUT_BASE/"
echo ""
echo "Next: eval all seeds:"
echo "  bash scripts/eval/run_variance_eval.sh"
