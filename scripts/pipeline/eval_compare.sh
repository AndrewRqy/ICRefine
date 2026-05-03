#!/usr/bin/env bash
# eval_compare.sh — Compare two pipeline run directories on the same tasks/models.
#
# Runs eval_cs_ablation.py twice (once per run dir) and prints a side-by-side
# accuracy diff. Useful for comparing ablation variants, e.g. v3 vs v4,
# or with-oracle vs without-oracle.
#
# Usage:
#   bash scripts/pipeline/eval_compare.sh -A RUN_A -B RUN_B [OPTIONS]
#
# Required:
#   -A RUN_DIR_A    first pipeline output directory  (label: "A")
#   -B RUN_DIR_B    second pipeline output directory (label: "B")
#
# Options:
#   -t "TASKS"      space-separated task names  (default: all 11 BBH tasks)
#   -M "MODELS"     space-separated model ids   (default: openai/gpt-4.1-mini)
#   -o OUTPUT_DIR   directory to write result JSONs (default: runs/compare)
#   -c CONCUR       API concurrency (default: 20)
#   --rf            reasoning-first scoring format
#   --full-only     skip pk_only pass
#   --no-csicl      skip CS-ICL comparison
#
# Examples:
#   # Compare v3 vs v4 on 3 tasks with gpt-4.1-mini
#   bash scripts/pipeline/eval_compare.sh \
#       -A runs/bbh_v3 -B runs/bbh_v4 \
#       -t "causal_judgement geometric_shapes web_of_lies" \
#       -o runs/compare_v3_v4
#
#   # Multi-model comparison
#   bash scripts/pipeline/eval_compare.sh \
#       -A runs/bbh_v3 -B runs/bbh_oracle \
#       -M "openai/gpt-4.1-mini openai/gpt-4.1" \
#       -t "formal_fallacies snarks" --rf

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# ── Defaults ──────────────────────────────────────────────────────────────────
RUN_A=""
RUN_B=""
TASKS=""
MODELS="openai/gpt-4.1-mini"
OUT_DIR="runs/compare"
CONCUR=20
EXTRA=""

ALL_TASKS="formal_fallacies web_of_lies causal_judgement geometric_shapes \
           boolean_expressions disambiguation_qa logical_deduction_three \
           sports_understanding navigate snarks date_understanding"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -A)          RUN_A="$2"; shift 2 ;;
        -B)          RUN_B="$2"; shift 2 ;;
        -t)          TASKS="$2"; shift 2 ;;
        -M)          MODELS="$2"; shift 2 ;;
        -o)          OUT_DIR="$2"; shift 2 ;;
        -c)          CONCUR="$2"; shift 2 ;;
        --rf)        EXTRA="$EXTRA --reasoning-first"; shift ;;
        --full-only) EXTRA="$EXTRA --full-only"; shift ;;
        --no-csicl)  EXTRA="$EXTRA --no-csicl"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$RUN_A" || -z "$RUN_B" ]]; then
    echo "Error: -A and -B are required" >&2; exit 1
fi

mkdir -p "$OUT_DIR"

task_list="${TASKS:-$ALL_TASKS}"
TASK_ARGS=""; [[ -n "$TASKS" ]] && TASK_ARGS="--tasks $TASKS"
MODEL_ARGS="--models $MODELS"

# ── Helper: build --run-dir-overrides for a given base dir ────────────────────
make_overrides() {
    local base="$1"
    local result=""
    for task in $task_list; do
        result="$result ${task}:${base}/${task}"
    done
    echo "$result"
}

# ── Run A ─────────────────────────────────────────────────────────────────────
OUT_A="$OUT_DIR/results_A.json"
echo "[compare] Evaluating A: $RUN_A"
# shellcheck disable=SC2086
python3 scripts/eval/eval_cs_ablation.py \
    $TASK_ARGS \
    $MODEL_ARGS \
    --run-dir-overrides $(make_overrides "$RUN_A") \
    --concurrency "$CONCUR" \
    --out "$OUT_A" \
    $EXTRA
echo ""

# ── Run B ─────────────────────────────────────────────────────────────────────
OUT_B="$OUT_DIR/results_B.json"
echo "[compare] Evaluating B: $RUN_B"
# shellcheck disable=SC2086
python3 scripts/eval/eval_cs_ablation.py \
    $TASK_ARGS \
    $MODEL_ARGS \
    --run-dir-overrides $(make_overrides "$RUN_B") \
    --concurrency "$CONCUR" \
    --out "$OUT_B" \
    $EXTRA
echo ""

# ── Diff ─────────────────────────────────────────────────────────────────────
echo "[compare] Results written to $OUT_DIR/"
echo "[compare] A ($(basename "$RUN_A")) → $OUT_A"
echo "[compare] B ($(basename "$RUN_B")) → $OUT_B"
echo ""

python3 - <<'PYEOF'
import json, sys, os

out_dir = os.environ.get("OUT_DIR", "runs/compare")
a_label = os.environ.get("RUN_A", "A")
b_label = os.environ.get("RUN_B", "B")

try:
    a = json.loads(open(f"{out_dir}/results_A.json").read())
    b = json.loads(open(f"{out_dir}/results_B.json").read())
except FileNotFoundError as e:
    print(f"Could not read results: {e}")
    sys.exit(0)

print(f"{'Task':<32} {'Model':<30} {'A-full':>7} {'B-full':>7} {'Δ(B-A)':>8}")
print("-" * 90)

for task in sorted(set(a) & set(b)):
    for model_id in sorted(set(a[task]) & set(b[task])):
        fa = a[task][model_id].get("full", float("nan"))
        fb = b[task][model_id].get("full", float("nan"))
        delta = fb - fa
        sign = "+" if delta >= 0 else ""
        print(f"{task:<32} {model_id:<30} {fa:>7.1%} {fb:>7.1%} {sign}{delta:>7.1%}")
PYEOF
