#!/usr/bin/env bash
# eval_ablation.sh — Full cheatsheet ablation: baseline / pk_only / full / cs_icl.
#
# Runs all scoring conditions in sequence and writes a combined result file.
# Designed for measuring the contribution of each cheatsheet component across
# tasks and models.
#
# Conditions scored:
#   baseline  — empty cheatsheet (raw model capability)
#   full      — complete cheatsheet (PK + case studies)
#   pk_only   — partition-key section only (no case studies)  [via full run]
#   cs_icl    — CS-ICL cheatsheet baseline (if --cs-icl-dir is available)
#
# Usage:
#   bash scripts/pipeline/eval_ablation.sh [OPTIONS]
#
# Options:
#   -t "TASKS"      space-separated task names  (default: all 11 BBH tasks)
#   -M "MODELS"     space-separated model ids   (default: openai/gpt-4.1-mini)
#   -d RUN_DIR      base pipeline run directory (default: uses TASK_CFG defaults)
#   -o OUTPUT       result JSON path            (default: runs/ablation_results.json)
#   -c CONCUR       API concurrency             (default: 20)
#   --rf            reasoning-first scoring format
#   --no-csicl      skip CS-ICL comparison
#
# Examples:
#   # Full ablation, 3 tasks, 2 models, RF format
#   bash scripts/pipeline/eval_ablation.sh \
#       -t "causal_judgement geometric_shapes disambiguation_qa" \
#       -M "openai/gpt-4.1-mini openai/gpt-4.1" \
#       -d runs/bbh_v3 --rf -o runs/ablation_3tasks.json
#
#   # All tasks, single model
#   bash scripts/pipeline/eval_ablation.sh \
#       -M "openai/gpt-4.1-mini" -d runs/bbh_v3

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# ── Defaults ──────────────────────────────────────────────────────────────────
TASKS=""
MODELS="openai/gpt-4.1-mini"
RUN_DIR=""
OUTPUT="runs/ablation_results.json"
CONCUR=20
RF_FLAG=""
CSICL_FLAG=""

ALL_TASKS="formal_fallacies web_of_lies causal_judgement geometric_shapes \
           boolean_expressions disambiguation_qa logical_deduction_three \
           sports_understanding navigate snarks date_understanding"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t)         TASKS="$2"; shift 2 ;;
        -M)         MODELS="$2"; shift 2 ;;
        -d)         RUN_DIR="$2"; shift 2 ;;
        -o)         OUTPUT="$2"; shift 2 ;;
        -c)         CONCUR="$2"; shift 2 ;;
        --rf)       RF_FLAG="--reasoning-first"; shift ;;
        --no-csicl) CSICL_FLAG="--no-csicl"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

task_list="${TASKS:-$ALL_TASKS}"
TASK_ARGS=""; [[ -n "$TASKS" ]] && TASK_ARGS="--tasks $TASKS"
MODEL_ARGS="--models $MODELS"

# Build --run-dir-overrides if -d given
RD_FLAG=""
if [[ -n "$RUN_DIR" ]]; then
    overrides=""
    for task in $task_list; do
        overrides="$overrides ${task}:${RUN_DIR}/${task}"
    done
    RD_FLAG="--run-dir-overrides $overrides"
fi

BASE_DIR="$(dirname "$OUTPUT")"
STEM="$(basename "$OUTPUT" .json)"

echo "[ablation] tasks=${TASKS:-all}  models=$MODELS"
echo "[ablation] run_dir=${RUN_DIR:-(TASK_CFG defaults)}"
echo ""

# ── Step 1: Baseline (empty cheatsheet) ───────────────────────────────────────
BASELINE_OUT="${BASE_DIR}/${STEM}_baseline.json"
echo "[ablation] 1/2  baseline scoring → $BASELINE_OUT"
# shellcheck disable=SC2086
python3 scripts/eval/eval_cs_ablation.py \
    $TASK_ARGS $MODEL_ARGS $RD_FLAG \
    --concurrency "$CONCUR" \
    --baseline --no-csicl \
    --out "$BASELINE_OUT" \
    $RF_FLAG
echo ""

# ── Step 2: Full ablation (pk_only + full + cs_icl) ───────────────────────────
echo "[ablation] 2/2  full ablation (pk_only + full + cs_icl) → $OUTPUT"
# shellcheck disable=SC2086
python3 scripts/eval/eval_cs_ablation.py \
    $TASK_ARGS $MODEL_ARGS $RD_FLAG \
    --concurrency "$CONCUR" \
    --out "$OUTPUT" \
    $RF_FLAG $CSICL_FLAG
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "[ablation] Done. Merging and printing summary..."
echo ""

OUT_DIR="$BASE_DIR" BASELINE_OUT="$BASELINE_OUT" FULL_OUT="$OUTPUT" \
python3 - <<'PYEOF'
import json, os

baseline_path = os.environ["BASELINE_OUT"]
full_path     = os.environ["FULL_OUT"]

try:
    baseline = json.loads(open(baseline_path).read())
    full     = json.loads(open(full_path).read())
except FileNotFoundError as e:
    print(f"Could not read results: {e}")
    raise SystemExit(0)

print(f"{'Task':<32} {'Model':<28} {'Base':>6} {'PK':>6} {'Full':>6} {'CSICL':>6} {'Lift':>7}")
print("-" * 100)

for task in sorted(set(baseline) & set(full)):
    for model_id in sorted(set(baseline.get(task, {})) & set(full.get(task, {}))):
        base_acc = baseline[task][model_id].get("full", float("nan"))
        pk_acc   = full[task][model_id].get("pk_only", float("nan"))
        full_acc = full[task][model_id].get("full", float("nan"))
        csicl    = full[task][model_id].get("cs_icl", float("nan"))
        lift     = full_acc - base_acc
        sign     = "+" if lift >= 0 else ""
        csicl_s  = f"{csicl:.1%}" if csicl == csicl else "  n/a"
        print(f"{task:<32} {model_id:<28} {base_acc:>6.1%} {pk_acc:>6.1%} {full_acc:>6.1%} {csicl_s:>6} {sign}{lift:>6.1%}")

PYEOF
