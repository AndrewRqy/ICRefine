#!/usr/bin/env bash
# gemini_train_v3.sh — Bidirectional transfer: Gemini as the train model.
#
# Runs the standard v3 pipeline (auto-bootstrap → Phase 1 PK patching →
# Phase 2 case study generation) with google/gemini-2.0-flash-001 in all
# three pipeline roles (scoring, rule-patching, case-study generation).
#
# Evaluates the resulting cheatsheet on all 5 models — including mini and
# GPT-4.1 as non-train — to test whether the task-structured transfer gap
# (CJ harmful, GS heterogeneous) holds when the train model changes.
#
# Precedent: E-Phase0-gemini in the experiment log confirmed the pipeline
# infrastructure works with Gemini; this extends to the full v3 pipeline.
#
# Tasks: the same 5 non-ceiling tasks used in the main v3 transfer eval
#   (causal_judgement, geometric_shapes, formal_fallacies,
#    disambiguation_qa, snarks)
#
# Usage:
#   bash scripts/pipeline/gemini_train_v3.sh
#   bash scripts/pipeline/gemini_train_v3.sh --dry-run
#
# Estimated wall time: ~4-6h (pipeline) + ~15min (eval).

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
OUT_BASE="runs/gemini_train_v3"
LOG_DIR="$OUT_BASE/logs"
RF_OUT="runs/gemini_train_v3_rf.json"

TASKS=(causal_judgement geometric_shapes formal_fallacies disambiguation_qa snarks)

mkdir -p "$LOG_DIR"

echo "=== Bidirectional transfer: Gemini as train model ==="
echo "  model   : $MODEL"
echo "  tasks   : ${TASKS[*]}"
echo "  out     : $OUT_BASE"
echo "  dry_run : $DRY_RUN"
echo ""

PIDS=()
LABELS=()

# ── Launch pipeline for each task in parallel ─────────────────────────────────
for task in "${TASKS[@]}"; do
    log="$LOG_DIR/${task}.log"
    out_dir="$OUT_BASE/$task"

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
        --output-dir "$out_dir"
    )

    if $DRY_RUN; then
        echo "[dry-run] $task: ${cmd[*]}"
        PIDS+=("skip")
    else
        mkdir -p "$out_dir"
        echo "  starting $task → $log"
        "${cmd[@]}" >> "$log" 2>&1 &
        PIDS+=($!)
    fi
    LABELS+=("$task")
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
echo "=== Pipeline complete — running RF eval across all 5 models ==="
echo ""

# ── Build --run-dir-overrides string ─────────────────────────────────────────
OVERRIDES=()
for task in "${TASKS[@]}"; do
    OVERRIDES+=("${task}:${OUT_BASE}/${task}")
done

cmd=(
    python3 scripts/eval/eval_cs_ablation.py
    --tasks "${TASKS[@]}"
    --reasoning-first
    --concurrency "$CONCUR"
    --run-dir-overrides "${OVERRIDES[@]}"
    --out "$RF_OUT"
)

if $DRY_RUN; then
    echo "[dry-run] eval: ${cmd[*]}"
else
    echo "  evaluating all 5 models → $RF_OUT"
    "${cmd[@]}" 2>&1 | tee "$LOG_DIR/eval.log"
    echo "  OK  eval"
fi

echo ""
echo "=== Done ==="
echo "Results: $RF_OUT"
echo ""
echo "Key comparisons to run after:"
echo "  python3 -c \""
echo "    import json"
echo "    from pathlib import Path"
echo "    d = json.loads(Path('$RF_OUT').read_text())"
echo "    for task in d:"
echo "        print(task)"
echo "        for mid, v in d[task].items():"
echo "            print(f'  {mid[-20:]}: pk={v.get(\\\"pk_only\\\"):.0%}  full={v.get(\\\"full\\\"):.0%}')"
echo "  \""
