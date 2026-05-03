#!/usr/bin/env bash
# ea_phase1.sh — EA Phase 1 extension runs.
#
# Replaces sequential PK patching (Phase 1) with the evolutionary algorithm:
# population of candidate PK rewrites → tournament selection → survivors
# crossed into the next generation.  Phase 2 (partition CS generation) is
# identical to the standard pipeline.
#
# Target tasks — those where Phase 2 CS hurt or fail in v3:
#   causal_judgement   — Phase 2 CS actively harmful (oracle-contaminated)
#   geometric_shapes   — Phase 2 CS hurt -6pp vs Phase-1-only
#   snarks             — v3 full -4.2pp below CS-ICL despite 2 CS added
#   disambiguation_qa  — 0 CS generated in v3 (threshold too tight)
#
# Comparison baseline: runs/bbh_v3/{task}/ (same model, same oracle defaults)
# EA Phase 1 defaults used here:
#   population-size  = 3   (3 candidate PK rewrites per generation)
#   n-survivors      = 2   (top 2 survive to next gen / crossover)
#   pk-size-budget   = 12000  (max chars for EA PK output — matches ablation P1 cap)
#   val-fraction     = 0.20   (20% of train held out as EA validation set)
#   failure-sample-frac = 0.60
#
# Scoring uses reasoning-first (RF) mode (--cot-first), matching v3 evaluation.
#
# Usage:
#   bash scripts/pipeline/ea_phase1.sh
#   bash scripts/pipeline/ea_phase1.sh --dry-run
#
# Estimated wall time: ~4-6h (EA Phase 1 is more expensive than single-path patching).

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL="openai/gpt-4.1-mini"
CONCUR=25
TASKS="causal_judgement geometric_shapes snarks disambiguation_qa"
OUT_BASE="runs/bbh_ea_phase1"

# EA Phase 1 hyperparams (defaults — explicit for reproducibility)
EA_POP=3
EA_SURVIVORS=2
EA_PK_BUDGET=12000
EA_VAL_FRAC=0.20
EA_FAIL_FRAC=0.60

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# ── Main ──────────────────────────────────────────────────────────────────────
echo "=== EA Phase 1 Extension Runs ==="
echo "  model   : $MODEL"
echo "  tasks   : $TASKS"
echo "  out     : $OUT_BASE"
echo "  ea_pop  : $EA_POP  survivors=$EA_SURVIVORS  pk_budget=$EA_PK_BUDGET"
echo "  dry_run : $DRY_RUN"
echo ""

LOG_DIR="$OUT_BASE/logs"

for task in $TASKS; do
    dataset="datasets/bbh/${task}_train.jsonl"
    task_dir="$OUT_BASE/$task"
    log_file="$LOG_DIR/${task}.log"

    cmd=(
        python3 -m ICR_hybrid.pipeline
        --task "$task"
        --dataset "$dataset"
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
        --ea-population-size    "$EA_POP"
        --ea-n-survivors        "$EA_SURVIVORS"
        --ea-pk-size-budget     "$EA_PK_BUDGET"
        --ea-val-fraction       "$EA_VAL_FRAC"
        --ea-failure-sample-frac "$EA_FAIL_FRAC"
        --output-dir "$task_dir"
    )

    if $DRY_RUN; then
        echo "[dry-run] ${cmd[*]} >> $log_file 2>&1 &"
    else
        mkdir -p "$LOG_DIR"
        echo "  starting $task → $log_file"
        "${cmd[@]}" >> "$log_file" 2>&1 &
    fi
done

if ! $DRY_RUN; then
    echo ""
    echo "  waiting for all EA runs to finish..."
    wait
    echo "  all done."
fi

echo ""
echo "=== EA Phase 1 runs complete ==="
echo "Results in: $OUT_BASE/"
echo ""
echo "Next: eval and compare against v3 baseline:"
echo "  python3 scripts/eval/eval_cs_ablation.py \\"
echo "    --tasks causal_judgement geometric_shapes snarks disambiguation_qa \\"
echo "    --reasoning-first \\"
echo "    --run-dir-overrides <task>:$OUT_BASE/<task> \\"
echo "    --out runs/bbh_ea_phase1_rf.json"
