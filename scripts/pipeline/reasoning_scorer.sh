#!/usr/bin/env bash
# reasoning_scorer.sh — Scoring-model ablation study.
#
# Tests whether replacing gpt-4.1-mini as the scoring model with a stronger
# reasoning model (openai/gpt-oss-120b) improves cheatsheet quality, while
# keeping the generation models (rule-patch, casestudy) fixed at gpt-4.1-mini.
#
# Hypothesis: gpt-4.1-mini accepts PK patches and case studies that overfit
# training failures without generalising, because it cannot reliably judge
# which revisions reflect genuine improvement vs. pattern-matching to the
# scorer's own outputs.  A stronger reasoning model as scorer should:
#   (a) apply more Phase 1 patches on GS (where mini scorer accepts 0 patches
#       because the bootstrap immediately satisfies its own acceptance gate),
#   (b) reject the near-duplicate FF CS (CS1≈CS4) that a weaker scorer lets through,
#   (c) potentially unlock DQ CS that mini scorer rejects via overly tight
#       regression gating.
#
# Baseline: runs/bbh_v3/{task}/   (all mini, reuse — no re-run needed)
#
# Tasks:
#   geometric_shapes   — scorer applies 0 patches in v3 (bootstrap convergence)
#   formal_fallacies   — scorer accepts 3 near-duplicate illicit-conversion CS
#   disambiguation_qa  — scorer generates 0 CS in v3 (all fail regression gate)
#
# Scoring model  : openai/gpt-oss-120b  (117B MoE, $0.039/$0.18 per M tokens)
# Gen model      : openai/gpt-4.1-mini  (unchanged)
# Eval model     : openai/gpt-4.1-mini  (cot-first, matching paper regime)
#
# All three tasks launch in parallel.
# Estimated wall time: ~2-3h (gpt-oss-120b scoring is fast; generation is mini).
#
# Usage:
#   bash scripts/pipeline/reasoning_scorer.sh
#   bash scripts/pipeline/reasoning_scorer.sh --dry-run

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_SCORE="openai/gpt-oss-120b"   # reasoning scorer
MODEL_GEN="openai/gpt-4.1-mini"    # generation (patch + casestudy) — unchanged

TASKS="geometric_shapes formal_fallacies disambiguation_qa"
OUT_BASE="runs/bbh_reasoning_scorer"
CONCUR=20

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# ── Main ──────────────────────────────────────────────────────────────────────
echo "=== Scoring-model ablation: reasoning scorer ==="
echo "  score model : $MODEL_SCORE"
echo "  gen model   : $MODEL_GEN"
echo "  tasks       : $TASKS"
echo "  out         : $OUT_BASE"
echo "  dry_run     : $DRY_RUN"
echo ""

LOG_DIR="$OUT_BASE/logs"

for task in $TASKS; do
    dataset="datasets/bbh/${task}_train.jsonl"
    task_dir="$OUT_BASE/$task"
    log_file="$LOG_DIR/${task}.log"

    cmd=(
        python3 -m ICR_hybrid.pipeline
        --task         "$task"
        --dataset      "$dataset"
        --no-oracle
        --model-score      "$MODEL_SCORE"
        --model-rule-patch "$MODEL_GEN"
        --model-casestudy  "$MODEL_GEN"
        --rule-concurrency "$CONCUR"
        --cs-concurrency   "$CONCUR"
        --max-rule-iters 4
        --max-cs-iters   5
        --cot-first
        --output-dir   "$task_dir"
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
    echo "  waiting for all 3 runs to finish ..."
    wait
    echo "  all done."
fi

echo ""
echo "=== Reasoning scorer runs complete ==="
echo "Results in: $OUT_BASE/"
echo ""
echo "Next: eval vs v3 baseline (all mini):"
echo "  python3 scripts/eval/eval_cs_ablation.py \\"
echo "    --tasks geometric_shapes formal_fallacies disambiguation_qa \\"
echo "    --reasoning-first \\"
echo "    --run-dir-overrides \\"
echo "      geometric_shapes:$OUT_BASE/geometric_shapes \\"
echo "      formal_fallacies:$OUT_BASE/formal_fallacies \\"
echo "      disambiguation_qa:$OUT_BASE/disambiguation_qa \\"
echo "    --out runs/bbh_reasoning_scorer_rf.json"
