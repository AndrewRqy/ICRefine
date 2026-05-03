#!/usr/bin/env bash
# run_variance_eval.sh — Evaluate all variance seed cheatsheets for 3-seed averaging.
#
# Variance seeds produced by runs in runs/variance/:
#   v3_seed2, v3_seed3  : standard v3 pipeline, tasks CJ/GS/DQ/FF/SN
#   e3_seed2, e3_seed3  : v3 pipeline (--no-oracle, same oracle mode as v3), CJ only
#   ea_seed2, ea_seed3  : EA Phase 1 pipeline, tasks GS/DQ
#
# All seeds use --reasoning-first RF scoring for consistency with paper evals.
# CS-ICL scoring is skipped (--no-csicl) since CS-ICL cheatsheet is static.
#
# Output: runs/variance/eval_results/<seed>_rf.json (one file per seed)
#
# Usage:
#   bash scripts/eval/run_variance_eval.sh
#   bash scripts/eval/run_variance_eval.sh --dry-run

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
if [ -f "$ROOT/.env" ]; then set -a; source "$ROOT/.env"; set +a; fi

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=true; fi

OUT_DIR="runs/variance/eval_results"
CONCUR=25
EVAL="scripts/eval/eval_cs_ablation.py"

mkdir -p "$OUT_DIR"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

PIDS=()
LABELS=()
LOGS=()

run_eval() {
    local label="$1"; shift
    local log="$LOG_DIR/${label}.log"
    LABELS+=("$label")
    LOGS+=("$log")

    if $DRY_RUN; then
        echo "[dry-run] $label: python3 $EVAL $*"
        PIDS+=("skip")
        return
    fi

    echo "  starting $label → $log"
    python3 "$EVAL" "$@" > "$log" 2>&1 &
    PIDS+=($!)
}

echo "=== Variance seed evaluation ==="
echo "  output dir : $OUT_DIR"
echo "  concurrency: $CONCUR"
echo "  dry_run    : $DRY_RUN"
echo ""

# ── v3_seed2: CJ, GS, DQ, FF, SN ─────────────────────────────────────────────
run_eval "v3_seed2" \
    --tasks causal_judgement geometric_shapes disambiguation_qa formal_fallacies snarks \
    --reasoning-first \
    --no-csicl \
    --concurrency "$CONCUR" \
    --run-dir-overrides \
        "causal_judgement:runs/variance/v3_seed2/causal_judgement" \
        "geometric_shapes:runs/variance/v3_seed2/geometric_shapes" \
        "disambiguation_qa:runs/variance/v3_seed2/disambiguation_qa" \
        "formal_fallacies:runs/variance/v3_seed2/formal_fallacies" \
        "snarks:runs/variance/v3_seed2/snarks" \
    --out "$OUT_DIR/v3_seed2_rf.json"

# ── v3_seed3: CJ, GS, DQ, FF, SN ─────────────────────────────────────────────
run_eval "v3_seed3" \
    --tasks causal_judgement geometric_shapes disambiguation_qa formal_fallacies snarks \
    --reasoning-first \
    --no-csicl \
    --concurrency "$CONCUR" \
    --run-dir-overrides \
        "causal_judgement:runs/variance/v3_seed3/causal_judgement" \
        "geometric_shapes:runs/variance/v3_seed3/geometric_shapes" \
        "disambiguation_qa:runs/variance/v3_seed3/disambiguation_qa" \
        "formal_fallacies:runs/variance/v3_seed3/formal_fallacies" \
        "snarks:runs/variance/v3_seed3/snarks" \
    --out "$OUT_DIR/v3_seed3_rf.json"

# ── e3_seed2: CJ only ─────────────────────────────────────────────────────────
run_eval "e3_seed2" \
    --tasks causal_judgement \
    --reasoning-first \
    --no-csicl \
    --concurrency "$CONCUR" \
    --run-dir-overrides \
        "causal_judgement:runs/variance/e3_seed2/causal_judgement" \
    --out "$OUT_DIR/e3_seed2_rf.json"

# ── e3_seed3: CJ only ─────────────────────────────────────────────────────────
run_eval "e3_seed3" \
    --tasks causal_judgement \
    --reasoning-first \
    --no-csicl \
    --concurrency "$CONCUR" \
    --run-dir-overrides \
        "causal_judgement:runs/variance/e3_seed3/causal_judgement" \
    --out "$OUT_DIR/e3_seed3_rf.json"

# ── ea_seed2: GS, DQ ──────────────────────────────────────────────────────────
run_eval "ea_seed2" \
    --tasks geometric_shapes disambiguation_qa \
    --reasoning-first \
    --no-csicl \
    --concurrency "$CONCUR" \
    --run-dir-overrides \
        "geometric_shapes:runs/variance/ea_seed2/geometric_shapes" \
        "disambiguation_qa:runs/variance/ea_seed2/disambiguation_qa" \
    --out "$OUT_DIR/ea_seed2_rf.json"

# ── ea_seed3: GS, DQ ──────────────────────────────────────────────────────────
run_eval "ea_seed3" \
    --tasks geometric_shapes disambiguation_qa \
    --reasoning-first \
    --no-csicl \
    --concurrency "$CONCUR" \
    --run-dir-overrides \
        "geometric_shapes:runs/variance/ea_seed3/geometric_shapes" \
        "disambiguation_qa:runs/variance/ea_seed3/disambiguation_qa" \
    --out "$OUT_DIR/ea_seed3_rf.json"

if $DRY_RUN; then
    echo ""
    echo "[dry-run] 6 eval jobs described above."
    exit 0
fi

# ── Wait and report ───────────────────────────────────────────────────────────
echo ""
echo "Waiting for ${#PIDS[@]} jobs..."
FAILED=()
for i in "${!PIDS[@]}"; do
    label="${LABELS[$i]}"
    pid="${PIDS[$i]}"
    log="${LOGS[$i]}"
    if wait "$pid"; then
        echo "  OK      $label"
    else
        echo "  FAILED  $label  (see $log)"
        FAILED+=("$label")
    fi
done

echo ""
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All variance evals complete."
    echo "Next: python3 scripts/eval/aggregate_variance.py"
else
    echo "${#FAILED[@]} job(s) failed: ${FAILED[*]}"
    exit 1
fi
