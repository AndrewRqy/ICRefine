#!/usr/bin/env bash
# eval.sh — Evaluate cheatsheet quality for one or more tasks and models.
#
# Wraps eval_cs_ablation.py. Supports single-model-multi-task and
# single-task-multi-model workflows via the same flags.
#
# Usage:
#   bash scripts/pipeline/eval.sh [OPTIONS]
#
# Options:
#   -t "TASKS"      space-separated task names  (default: all 11 BBH tasks)
#   -M "MODELS"     space-separated model ids   (default: openai/gpt-4.1-mini)
#   -d RUN_DIR      base pipeline run directory — overrides per-task run_dir
#                   (default: uses TASK_CFG defaults, all pointing to runs/bbh_v3)
#   -o OUTPUT       result JSON path  (default: runs/eval_results.json)
#   -c CONCUR       API concurrency   (default: 20)
#   --rf            reasoning-first scoring format (VERDICT on last line)
#   --baseline      score with empty cheatsheet (no PK); measures raw capability
#   --full-only     skip pk_only pass; only score the combined full cheatsheet
#   --no-csicl      skip CS-ICL comparison
#   --cot           force chain-of-thought scoring for all conditions
#
# Examples:
#   # One model, all tasks, v3 cheatsheets
#   bash scripts/pipeline/eval.sh \
#       -M "openai/gpt-4.1-mini" -d runs/bbh_v3 -o runs/results.json
#
#   # Multiple models, three tasks
#   bash scripts/pipeline/eval.sh \
#       -t "causal_judgement geometric_shapes disambiguation_qa" \
#       -M "openai/gpt-4.1-mini openai/gpt-4.1 google/gemini-2.0-flash-001" \
#       -d runs/bbh_v3 --rf -o runs/rf_results.json
#
#   # Baseline: raw model capability, no cheatsheet
#   bash scripts/pipeline/eval.sh \
#       -t "causal_judgement geometric_shapes" \
#       -M "openai/gpt-4.1-mini openai/gpt-4.1" \
#       --baseline --rf -o runs/baseline.json

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# ── Defaults ──────────────────────────────────────────────────────────────────
TASKS=""
MODELS="openai/gpt-4.1-mini"
RUN_DIR=""
OUTPUT="runs/eval_results.json"
CONCUR=20
EXTRA=""

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t)         TASKS="$2"; shift 2 ;;
        -M)         MODELS="$2"; shift 2 ;;
        -d)         RUN_DIR="$2"; shift 2 ;;
        -o)         OUTPUT="$2"; shift 2 ;;
        -c)         CONCUR="$2"; shift 2 ;;
        --rf)       EXTRA="$EXTRA --reasoning-first"; shift ;;
        --baseline) EXTRA="$EXTRA --baseline"; shift ;;
        --full-only)EXTRA="$EXTRA --full-only"; shift ;;
        --no-csicl) EXTRA="$EXTRA --no-csicl"; shift ;;
        --cot)      EXTRA="$EXTRA --cot"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Build --run-dir-overrides from -d RUN_DIR ─────────────────────────────────
RD_ARGS=""
if [[ -n "$RUN_DIR" ]]; then
    # Determine task list: use specified tasks or all TASK_CFG keys
    if [[ -n "$TASKS" ]]; then
        task_list="$TASKS"
    else
        task_list="formal_fallacies web_of_lies causal_judgement geometric_shapes \
                   boolean_expressions disambiguation_qa logical_deduction_three \
                   sports_understanding navigate snarks date_understanding"
    fi
    for task in $task_list; do
        RD_ARGS="$RD_ARGS ${task}:${RUN_DIR}/${task}"
    done
fi

TASK_ARGS=""; [[ -n "$TASKS"  ]] && TASK_ARGS="--tasks $TASKS"
MODEL_ARGS="";[[ -n "$MODELS" ]] && MODEL_ARGS="--models $MODELS"
RD_FLAG="";   [[ -n "$RD_ARGS" ]] && RD_FLAG="--run-dir-overrides $RD_ARGS"

echo "[eval] tasks=${TASKS:-all}  models=$MODELS"
echo "[eval] run_dir=${RUN_DIR:-(TASK_CFG defaults)}  out=$OUTPUT"

# shellcheck disable=SC2086
python3 scripts/eval/eval_cs_ablation.py \
    $TASK_ARGS \
    $MODEL_ARGS \
    $RD_FLAG \
    --concurrency "$CONCUR" \
    --out "$OUTPUT" \
    $EXTRA
