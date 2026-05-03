#!/usr/bin/env bash
# ablation_size.sh — Cheatsheet size ablation study.
#
# Sweeps two independent dimensions for each task:
#   Phase 1 PK char limit  : 3000 | 6000 | 12000 | unlimited (default)
#   Phase 2 CS count limit : 1 | 3 | unlimited (default)
#
# Tasks: geometric_shapes, formal_fallacies, snarks, disambiguation_qa
# These were chosen because Phase 1 and/or Phase 2 produce meaningful
# content for v3 (non-ceiling tasks with variable CS output).
#
# v3 baseline prior_knowledge sizes (PK field only, not CS):
#   GS=9K  FF=6K  Snarks=7K  DQ=16K  avg≈9.4K chars
# v3 baseline CS counts: GS=2   FF=4   Snarks=2   DQ=0
#   → Phase 1 limits (3K/6K/12K) bracket the baseline PK sizes.
#     The char limit is now enforced at Phase 0 (bootstrap generation)
#     with up to 3 retries before hard-truncation, so the PK starts
#     within budget before Phase 1 patching runs.
#   → Phase 2 limits (1/3) select from a best-of-N pool; the pipeline
#     auto-loosens the fix-rate gate (0.30→0.20, floor 0.15) so the pool
#     can exceed the v3 baseline counts before trimming to N.
#
# Scoring uses reasoning-first (RF) mode throughout training (--cot-first),
# matching the evaluation regime.
#
# All conditions are launched in parallel at once to minimise wall time.
# Peak API concurrency: 6 conditions × 4 tasks × 25 requests ≈ 600 max.
#
# Output layout:
#   runs/ablation_size2/
#     p1_{N}chars/{task}/     Phase 1 PK cap at N chars, Phase 2 uncapped
#     p1_unlimited/{task}/    Baseline (no cap), Phase 2 uncapped
#     p2_{N}cs/{task}/        Phase 2 best-of-N CS, Phase 1 uncapped
#     p2_unlimited → aliased to p1_unlimited (skipped)
#
# Usage:
#   bash scripts/pipeline/ablation_size.sh
#   bash scripts/pipeline/ablation_size.sh --dry-run    # print commands only
#
# Estimated wall time: ~2-3h (all conditions parallel).

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL="openai/gpt-4.1-mini"
CONCUR=25
TASKS="geometric_shapes formal_fallacies snarks disambiguation_qa"
OUT_BASE="runs/ablation_size2"

# Phase 1 PK char limits (chars). "0" = unlimited.
P1_LIMITS=(3000 6000 12000 0)

# Phase 2 CS count limits. "0" = unlimited (aliased to p1_unlimited).
P2_LIMITS=(1 3 0)

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
dataset_file() { echo "datasets/bbh/${1}_train.jsonl"; }

label_p1() { [[ $1 -eq 0 ]] && echo "unlimited" || echo "${1}chars"; }
label_p2() { [[ $1 -eq 0 ]] && echo "unlimited" || echo "${1}cs"; }

launch() {
    local run_dir="$1"; shift
    local extra_args=("$@")
    local log_dir="$run_dir/logs"

    for task in $TASKS; do
        local task_dir="$run_dir/$task"
        local log_file="$log_dir/${task}.log"
        local cmd=(
            python3 -m ICR_hybrid.pipeline
            --task "$task"
            --dataset "$(dataset_file "$task")"
            --no-oracle
            --model-score      "$MODEL"
            --model-rule-patch "$MODEL"
            --model-casestudy  "$MODEL"
            --rule-concurrency "$CONCUR"
            --cs-concurrency   "$CONCUR"
            --max-rule-iters 4
            --max-cs-iters   5
            --cot-first
            --output-dir "$task_dir"
            "${extra_args[@]}"
        )
        if $DRY_RUN; then
            echo "[dry-run] ${cmd[*]} >> $log_file 2>&1 &"
        else
            mkdir -p "$log_dir"
            echo "  $task → $log_file"
            "${cmd[@]}" >> "$log_file" 2>&1 &
        fi
    done
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo "=== Cheatsheet Size Ablation (all conditions parallel) ==="
echo "  model   : $MODEL"
echo "  tasks   : $TASKS"
echo "  P1 caps : ${P1_LIMITS[*]}"
echo "  P2 caps : ${P2_LIMITS[*]}"
echo "  out     : $OUT_BASE"
echo "  dry_run : $DRY_RUN"
echo ""

# ── Sweep A: Phase 1 PK char limit (Phase 2 uncapped) ────────────────────────
echo "=== Sweep A: Phase 1 PK char limit ==="
for limit in "${P1_LIMITS[@]}"; do
    lbl="$(label_p1 "$limit")"
    echo "  launching p1_${lbl} ..."
    if [[ $limit -eq 0 ]]; then
        launch "$OUT_BASE/p1_unlimited"
    else
        launch "$OUT_BASE/p1_${lbl}" --max-pk-chars "$limit"
    fi
done

# ── Sweep B: Phase 2 CS count limit (Phase 1 uncapped) ───────────────────────
echo ""
echo "=== Sweep B: Phase 2 CS count limit (Phase 1 uncapped) ==="
for limit in "${P2_LIMITS[@]}"; do
    lbl="$(label_p2 "$limit")"
    if [[ $limit -eq 0 ]]; then
        echo "  p2_unlimited → aliased to p1_unlimited (skip re-run)"
        continue
    fi
    echo "  launching p2_${lbl} ..."
    launch "$OUT_BASE/p2_${lbl}" --max-case-studies "$limit"
done

# ── Wait for all 24 background jobs ──────────────────────────────────────────
if ! $DRY_RUN; then
    echo ""
    echo "All conditions launched — waiting for all $(jobs -p | wc -l | tr -d ' ') jobs ..."
    wait
    echo "All done."
fi

echo ""
echo "=== All ablation runs complete ==="
echo "Results in: $OUT_BASE/"
echo ""
echo "Next: eval all conditions:"
echo "  python3 scripts/eval/eval_cs_ablation.py \\"
echo "    --tasks geometric_shapes formal_fallacies snarks disambiguation_qa \\"
echo "    --reasoning-first \\"
echo "    --run-dir-overrides <task>:$OUT_BASE/<condition>/<task> \\"
echo "    --out runs/ablation_size2_rf.json"
