#!/usr/bin/env bash
# eval_ea_combined.sh — Evaluate and aggregate the full EA+no-oracle combined
# pipeline across all 5 non-ceiling tasks × 3 seeds.
#
# Pre-requisites (all pipeline runs must be complete):
#   bash scripts/pipeline/ea_no_oracle_ff.sh       (FF × 3 seeds)
#   bash scripts/pipeline/ea_seed_cj_snarks.sh     (CJ + Snarks × seeds 2 & 3)
#   runs/bbh_ea_phase1/                            (CJ/GS/SN/DQ seed 1, pre-existing)
#   runs/variance/ea_seed{2,3}/{geometric_shapes,disambiguation_qa}/  (pre-existing)
#
# What this script does:
#   1. Checks all 15 run dirs exist before launching evals
#   2. Evals all 5 tasks × 3 seeds in parallel (3 eval jobs, each covering 5 tasks)
#      Output: runs/variance/eval_results/ea_combined_seed{1,2,3}_rf.json
#   3. Aggregates into a 3-seed mean
#      Output: runs/variance/ea_combined_3seed_mean.json
#
# Usage:
#   bash scripts/eval/eval_ea_combined.sh
#   bash scripts/eval/eval_ea_combined.sh --dry-run

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
if [ -f "$ROOT/.env" ]; then set -a; source "$ROOT/.env"; set +a; fi

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=true; fi

CONCUR=25
TASKS="causal_judgement geometric_shapes formal_fallacies disambiguation_qa snarks"
EVAL_OUT="runs/variance/eval_results"
LOG_DIR="runs/variance/logs"
COMBINED_OUT="runs/variance/ea_combined_3seed_mean.json"

mkdir -p "$EVAL_OUT" "$LOG_DIR"

# ── Run-dir layout (parallel to ea_seed{2,3} convention) ─────────────────────
# seed 1: original bbh_ea_phase1 output
S1_CJ="runs/bbh_ea_phase1/causal_judgement"
S1_GS="runs/bbh_ea_phase1/geometric_shapes"
S1_FF="runs/bbh_ea_phase1/formal_fallacies"
S1_DQ="runs/bbh_ea_phase1/disambiguation_qa"
S1_SN="runs/bbh_ea_phase1/snarks"

# seed 2: variance ea_seed2 dirs (GS/DQ pre-existing; CJ/SN from ea_seed_cj_snarks.sh;
#          FF from ea_no_oracle_ff.sh)
S2_CJ="runs/variance/ea_seed2/causal_judgement"
S2_GS="runs/variance/ea_seed2/geometric_shapes"
S2_FF="runs/variance/ea_seed2/formal_fallacies"
S2_DQ="runs/variance/ea_seed2/disambiguation_qa"
S2_SN="runs/variance/ea_seed2/snarks"

# seed 3: same pattern
S3_CJ="runs/variance/ea_seed3/causal_judgement"
S3_GS="runs/variance/ea_seed3/geometric_shapes"
S3_FF="runs/variance/ea_seed3/formal_fallacies"
S3_DQ="runs/variance/ea_seed3/disambiguation_qa"
S3_SN="runs/variance/ea_seed3/snarks"

echo "=== EA+no-oracle combined eval: 5 tasks × 3 seeds ==="
echo "  tasks   : $TASKS"
echo "  dry_run : $DRY_RUN"
echo ""

# ── Check all run dirs exist ──────────────────────────────────────────────────
if ! $DRY_RUN; then
    MISSING=()
    for dir in \
        "$S1_CJ" "$S1_GS" "$S1_FF" "$S1_DQ" "$S1_SN" \
        "$S2_CJ" "$S2_GS" "$S2_FF" "$S2_DQ" "$S2_SN" \
        "$S3_CJ" "$S3_GS" "$S3_FF" "$S3_DQ" "$S3_SN"; do
        [[ -d "$dir" ]] || MISSING+=("$dir")
    done
    if [[ ${#MISSING[@]} -gt 0 ]]; then
        echo "ERROR: missing run dirs — pipeline not complete:"
        printf '  %s\n' "${MISSING[@]}"
        echo ""
        echo "Run first:"
        echo "  bash scripts/pipeline/ea_no_oracle_ff.sh"
        echo "  bash scripts/pipeline/ea_seed_cj_snarks.sh"
        exit 1
    fi
fi

# ── Launch 3 eval jobs in parallel ───────────────────────────────────────────
PIDS=()
LABELS=()
LOGS=()

launch_eval() {
    local seed="$1"
    local out="$2"
    local log="$LOG_DIR/ea_combined_seed${seed}_eval.log"
    shift 2   # remaining args are the run-dir-overrides

    LABELS+=("seed${seed}")
    LOGS+=("$log")

    local cmd=(
        python3 scripts/eval/eval_cs_ablation.py
        --tasks $TASKS
        --reasoning-first
        --no-csicl
        --concurrency "$CONCUR"
        --run-dir-overrides "$@"
        --out "$out"
    )

    if $DRY_RUN; then
        echo "[dry-run] seed${seed}: ${cmd[*]}"
        PIDS+=("skip")
    else
        echo "  evaluating seed ${seed} → $log"
        "${cmd[@]}" >> "$log" 2>&1 &
        PIDS+=($!)
    fi
}

launch_eval 1 "$EVAL_OUT/ea_combined_seed1_rf.json" \
    "causal_judgement:$S1_CJ" \
    "geometric_shapes:$S1_GS" \
    "formal_fallacies:$S1_FF" \
    "disambiguation_qa:$S1_DQ" \
    "snarks:$S1_SN"

launch_eval 2 "$EVAL_OUT/ea_combined_seed2_rf.json" \
    "causal_judgement:$S2_CJ" \
    "geometric_shapes:$S2_GS" \
    "formal_fallacies:$S2_FF" \
    "disambiguation_qa:$S2_DQ" \
    "snarks:$S2_SN"

launch_eval 3 "$EVAL_OUT/ea_combined_seed3_rf.json" \
    "causal_judgement:$S3_CJ" \
    "geometric_shapes:$S3_GS" \
    "formal_fallacies:$S3_FF" \
    "disambiguation_qa:$S3_DQ" \
    "snarks:$S3_SN"

if ! $DRY_RUN; then
    echo ""
    echo "Waiting for ${#PIDS[@]} eval jobs..."
    FAILED=0
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo "  OK      ${LABELS[$i]}"
        else
            echo "  FAILED  ${LABELS[$i]}  (see ${LOGS[$i]})"
            FAILED=$((FAILED+1))
        fi
    done
    [[ $FAILED -gt 0 ]] && { echo "$FAILED eval(s) failed"; exit 1; }
fi

# ── Aggregate 3-seed mean ─────────────────────────────────────────────────────
echo ""
cmd=(
    python3 scripts/eval/aggregate_variance.py
    --condition ea_combined
    --seed1 "$EVAL_OUT/ea_combined_seed1_rf.json"
    --seed2 "$EVAL_OUT/ea_combined_seed2_rf.json"
    --seed3 "$EVAL_OUT/ea_combined_seed3_rf.json"
    --out   "$COMBINED_OUT"
)

if $DRY_RUN; then
    echo "[dry-run] aggregate: ${cmd[*]}"
else
    echo "  aggregating 3-seed means → $COMBINED_OUT"
    "${cmd[@]}"
    echo "  OK"
fi

echo ""
echo "=== Done ==="
echo "  3-seed mean: $COMBINED_OUT"
echo ""
echo "Quick summary:"
cat << 'PYEOF'
  python3 -c "
import json, statistics as s
from pathlib import Path
d = json.loads(Path('runs/variance/ea_combined_3seed_mean.json').read_text())
NONTRAIN = [
    ('openai/gpt-4.1',                       'GPT-4.1'),
    ('anthropic/claude-3-7-sonnet-20250219', 'Claude'),
    ('google/gemini-2.0-flash-001',          'Gemini'),
    ('meta-llama/llama-3.3-70b-instruct',    'Llama'),
]
CSICL = {'openai/gpt-4.1': 0.876, 'anthropic/claude-3-7-sonnet-20250219': 0.861,
         'google/gemini-2.0-flash-001': 0.801, 'meta-llama/llama-3.3-70b-instruct': 0.777}
for mid, lbl in NONTRAIN:
    vals = [d[t][mid]['full'] for t in d if mid in d[t]]
    if vals:
        avg = s.mean(vals)
        print(f'{lbl}: EA+no-oracle={avg:.1%}  CS-ICL={CSICL[mid]:.1%}  delta={avg-CSICL[mid]:+.1%}')
"
PYEOF
