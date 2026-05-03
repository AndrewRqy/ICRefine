#!/usr/bin/env bash
# train.sh — Run ICR pipeline on one or more tasks in parallel.
#
# Each task gets its own background job and log file. The script waits for all
# jobs to finish and prints a pass/fail summary.
#
# Usage:
#   bash scripts/pipeline/train.sh -t TASK1 TASK2 ... [OPTIONS]
#
# Required:
#   -t "TASK1 TASK2 ..."  space-separated task names (quoted)
#
# Options:
#   -m MODEL        model id for all roles  (default: openai/gpt-4.1-mini)
#   -d RUN_DIR      base output directory   (default: runs/train)
#   -p PIPELINE     pipeline module: hybrid | partition  (default: hybrid)
#   -c CONCUR       scoring concurrency     (default: 25)
#   -n MAX_ITERS    max case-study iterations per phase  (default: 5)
#   -O ORACLE_CSV   path to oracle CSV; enables oracle injection
#   --no-oracle     disable oracle (default when -O not given)
#
# Examples:
#   # Three tasks, no oracle
#   bash scripts/pipeline/train.sh \
#       -t "web_of_lies causal_judgement geometric_shapes" \
#       -m openai/gpt-4.1-mini -d runs/bbh_v3
#
#   # Two tasks with oracle
#   bash scripts/pipeline/train.sh \
#       -t "formal_fallacies snarks" \
#       -m openai/gpt-4.1-mini -O gpt5.4_normal_default.csv \
#       -d runs/bbh_oracle

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# ── Defaults ──────────────────────────────────────────────────────────────────
TASKS=""
MODEL="openai/gpt-4.1-mini"
RUN_DIR="runs/train"
PIPELINE="hybrid"
CONCUR=25
MAX_ITERS=5
ORACLE_CSV=""
NO_ORACLE=true

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t) TASKS="$2"; shift 2 ;;
        -m) MODEL="$2"; shift 2 ;;
        -d) RUN_DIR="$2"; shift 2 ;;
        -p) PIPELINE="$2"; shift 2 ;;
        -c) CONCUR="$2"; shift 2 ;;
        -n) MAX_ITERS="$2"; shift 2 ;;
        -O) ORACLE_CSV="$2"; NO_ORACLE=false; shift 2 ;;
        --no-oracle) NO_ORACLE=true; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$TASKS" ]]; then
    echo "Error: -t TASKS is required" >&2; exit 1
fi

# ── Dataset path helper ───────────────────────────────────────────────────────
# Maps pipeline task key → BBH dataset filename (handles aliases)
dataset_file() {
    local task="$1"
    case "$task" in
        logical_deduction_three) echo "datasets/bbh/logical_deduction_three_objects_train.jsonl" ;;
        bbh_boolean)             echo "datasets/bbh/boolean_expressions_train.jsonl" ;;
        *)                       echo "datasets/bbh/${task}_train.jsonl" ;;
    esac
}

# ── Oracle flag ───────────────────────────────────────────────────────────────
oracle_args() {
    if $NO_ORACLE; then
        echo "--no-oracle"
    else
        echo "--oracle-csv $ORACLE_CSV"
    fi
}

# ── Launch jobs ───────────────────────────────────────────────────────────────
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

TASK_LIST=()
PID_LIST=()
LOG_LIST=()

echo "[train] pipeline=ICR_${PIPELINE}  model=$MODEL  run_dir=$RUN_DIR"
echo "[train] tasks: $TASKS"
echo ""

for task in $TASKS; do
    dataset="$(dataset_file "$task")"
    if [[ ! -f "$dataset" ]]; then
        echo "Warning: dataset not found for $task: $dataset" >&2
    fi

    log="$LOG_DIR/${task}.log"
    > "$log"

    # shellcheck disable=SC2046
    python3 -m "ICR_${PIPELINE}.pipeline" \
        --task "$task" \
        --dataset "$dataset" \
        $(oracle_args) \
        --model-score "$MODEL" \
        --model-rule-patch "$MODEL" \
        --model-casestudy "$MODEL" \
        --rule-concurrency "$CONCUR" \
        --cs-concurrency "$CONCUR" \
        --max-cs-iters "$MAX_ITERS" \
        --output-dir "$RUN_DIR/$task" \
        > "$log" 2>&1 &

    TASK_LIST+=("$task")
    PID_LIST+=($!)
    LOG_LIST+=("$log")
    echo "  started $task  (PID $!)  → $log"
done

# ── Wait and report ───────────────────────────────────────────────────────────
echo ""
echo "[train] waiting for ${#TASK_LIST[@]} job(s)..."
FAILED=()

for i in "${!TASK_LIST[@]}"; do
    task="${TASK_LIST[$i]}"
    pid="${PID_LIST[$i]}"
    log="${LOG_LIST[$i]}"
    if wait "$pid"; then
        echo "  OK      $task"
    else
        echo "  FAILED  $task  (see $log)"
        FAILED+=("$task")
    fi
done

echo ""
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "[train] All tasks completed successfully."
else
    echo "[train] ${#FAILED[@]} task(s) failed: ${FAILED[*]}"
    exit 1
fi
