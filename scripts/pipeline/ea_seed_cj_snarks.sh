#!/usr/bin/env bash
# ea_seed_cj_snarks.sh — Seeds 2 & 3 for EA Phase 1 on CJ and Snarks.
#
# Seed 1 already exists in runs/bbh_ea_phase1/ (causal_judgement, snarks).
# This script runs seeds 2 & 3 to complete the 3-seed mean for those tasks,
# matching the EA variance evaluation already done for GS/DQ.
#
# Pipeline: ICR_hybrid with --use-ea, same hyperparams as original ea_phase1.sh.
# Eval: appended to existing ea_seed{2,3}_rf.json files (new tasks only), then
#       re-aggregates all variance means via aggregate_variance.py --all.
#
# Usage:
#   bash scripts/pipeline/ea_seed_cj_snarks.sh
#   bash scripts/pipeline/ea_seed_cj_snarks.sh --dry-run
#
# Estimated wall time: ~3-4h (4 parallel EA runs).

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
OUT_BASE="runs/variance"
EVAL_OUT="runs/variance/eval_results"
LOG_DIR="$OUT_BASE/logs"

mkdir -p "$LOG_DIR" "$EVAL_OUT"

echo "=== EA seed variance: CJ + Snarks (seeds 2 & 3) ==="
echo "  model   : $MODEL"
echo "  dry_run : $DRY_RUN"
echo ""

PIDS=()
LABELS=()

# ── Launch 4 pipeline runs in parallel ────────────────────────────────────────
for seed in 2 3; do
    for task in causal_judgement snarks; do
        label="ea_seed${seed}_${task}"
        log="$LOG_DIR/${label}.log"
        out_dir="$OUT_BASE/ea_seed${seed}/$task"

        cmd=(
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
            --use-ea
            --ea-population-size     3
            --ea-n-survivors         2
            --ea-pk-size-budget      12000
            --ea-val-fraction        0.20
            --ea-failure-sample-frac 0.60
            --output-dir "$out_dir"
        )

        if $DRY_RUN; then
            echo "[dry-run] $label: ${cmd[*]}"
            PIDS+=("skip")
        else
            mkdir -p "$out_dir"
            echo "  starting $label → $log"
            "${cmd[@]}" >> "$log" 2>&1 &
            PIDS+=($!)
        fi
        LABELS+=("$label")
    done
done

# ── Wait ──────────────────────────────────────────────────────────────────────
if ! $DRY_RUN; then
    echo ""
    echo "Waiting for ${#PIDS[@]} pipeline runs..."
    FAILED=()
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo "  OK      ${LABELS[$i]}"
        else
            echo "  FAILED  ${LABELS[$i]}  (see $LOG_DIR/${LABELS[$i]}.log)"
            FAILED+=("${LABELS[$i]}")
        fi
    done
    if [[ ${#FAILED[@]} -gt 0 ]]; then
        echo ""
        echo "${#FAILED[@]} pipeline run(s) failed: ${FAILED[*]}"
        exit 1
    fi
fi

echo ""
echo "=== Pipeline complete — running evals ==="
echo ""

# ── Eval all 3 seeds in parallel ─────────────────────────────────────────────
EVAL_PIDS=()
EVAL_LABELS=()
EVAL_LOGS=()

launch_eval() {
    local label="$1"; local log="$2"; shift 2
    EVAL_LABELS+=("$label")
    EVAL_LOGS+=("$log")
    if $DRY_RUN; then
        echo "[dry-run] eval $label: $*"
        EVAL_PIDS+=("skip")
    else
        echo "  evaluating $label → $log"
        "$@" >> "$log" 2>&1 &
        EVAL_PIDS+=($!)
    fi
}

launch_eval "ea_seed1_cj_sn" "$LOG_DIR/ea_seed1_cj_sn_eval.log" \
    python3 scripts/eval/eval_cs_ablation.py \
    --tasks causal_judgement snarks \
    --reasoning-first --no-csicl --concurrency "$CONCUR" \
    --run-dir-overrides \
        "causal_judgement:runs/bbh_ea_phase1/causal_judgement" \
        "snarks:runs/bbh_ea_phase1/snarks" \
    --out "$EVAL_OUT/ea_seed1_cj_sn_rf.json"

for seed in 2 3; do
    launch_eval "ea_seed${seed}_cj_sn" "$LOG_DIR/ea_seed${seed}_cj_sn_eval.log" \
        python3 scripts/eval/eval_cs_ablation.py \
        --tasks causal_judgement snarks \
        --reasoning-first --no-csicl --concurrency "$CONCUR" \
        --run-dir-overrides \
            "causal_judgement:$OUT_BASE/ea_seed${seed}/causal_judgement" \
            "snarks:$OUT_BASE/ea_seed${seed}/snarks" \
        --out "$EVAL_OUT/ea_seed${seed}_cj_sn_rf.json"
done

if ! $DRY_RUN; then
    echo ""
    echo "Waiting for ${#EVAL_PIDS[@]} eval jobs..."
    FAILED=()
    for i in "${!EVAL_PIDS[@]}"; do
        if wait "${EVAL_PIDS[$i]}"; then
            echo "  OK      ${EVAL_LABELS[$i]}"
        else
            echo "  FAILED  ${EVAL_LABELS[$i]}  (see ${EVAL_LOGS[$i]})"
            FAILED+=("${EVAL_LABELS[$i]}")
        fi
    done
    if [[ ${#FAILED[@]} -gt 0 ]]; then
        echo "${#FAILED[@]} eval(s) failed: ${FAILED[*]}"
        exit 1
    fi
fi

# ── Re-aggregate all variance means ───────────────────────────────────────────
echo ""
if $DRY_RUN; then
    echo "[dry-run] python3 scripts/eval/aggregate_variance.py --all"
else
    echo "  re-aggregating 3-seed means..."
    python3 scripts/eval/aggregate_variance.py --all
    echo "  OK  aggregate_variance"
fi

echo ""
echo "=== Done ==="
echo "Outputs:"
echo "  $EVAL_OUT/ea_seed1_cj_sn_rf.json"
echo "  $EVAL_OUT/ea_seed2_cj_sn_rf.json"
echo "  $EVAL_OUT/ea_seed3_cj_sn_rf.json"
echo "  runs/variance/ea_3seed_mean.json  (updated)"
