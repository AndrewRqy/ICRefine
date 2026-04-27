#!/usr/bin/env bash
# run_bbh_overnight.sh — Overnight hybrid pipeline run for all BBH tasks.
#
# Each task runs Phase 1 (rule patching via bootstrap) + Phase 2 (case study generation).
# Results land in runs/bbh_overnight/<task_name>/
#
# Usage:
#   chmod +x run_bbh_overnight.sh
#   ./run_bbh_overnight.sh
#
# Override model/concurrency:
#   MODEL_SCORE=openai/gpt-4.1-mini MODEL_CS=openai/gpt-4.1-mini ./run_bbh_overnight.sh

set -euo pipefail

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_SCORE="${MODEL_SCORE:-openai/gpt-4.1-mini}"
MODEL_RULE_PATCH="${MODEL_RULE_PATCH:-openai/gpt-4.1-mini}"
MODEL_CS="${MODEL_CS:-openai/gpt-4.1-mini}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_CS_ITERS="${MAX_CS_ITERS:-5}"
MAX_RULE_ITERS="${MAX_RULE_ITERS:-3}"
RULE_ACC_GOAL="${RULE_ACC_GOAL:-0.88}"
BOOTSTRAP_N="${BOOTSTRAP_N:-20}"

DATA_DIR="../cheat-sheet-icl/data/aug_data"
OUT_BASE="runs/bbh_overnight"

# ── Helper ─────────────────────────────────────────────────────────────────────
run_task() {
    local task="$1"
    local dataset="$2"
    local extra_flags="${3:-}"
    local out_dir="${OUT_BASE}/${task}"

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  TASK: ${task}"
    echo "  Output: ${out_dir}"
    echo "  Start: $(date)"
    echo "════════════════════════════════════════════════════════"

    mkdir -p "${out_dir}"

    python3 -m ICR_hybrid.pipeline \
        --task "${task}" \
        --dataset "${dataset}" \
        --no-oracle \
        --model-score "${MODEL_SCORE}" \
        --model-rule-patch "${MODEL_RULE_PATCH}" \
        --model-casestudy "${MODEL_CS}" \
        --rule-concurrency "${CONCURRENCY}" \
        --cs-concurrency "${CONCURRENCY}" \
        --auto-rule-init \
        --bootstrap-n "${BOOTSTRAP_N}" \
        --rule-acc-goal "${RULE_ACC_GOAL}" \
        --max-rule-iters "${MAX_RULE_ITERS}" \
        --max-cs-iters "${MAX_CS_ITERS}" \
        --output-dir "${out_dir}" \
        ${extra_flags} \
        2>&1 | tee "${out_dir}/run.log"

    echo "  Done: $(date)"
}

# ── Main ───────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")"

echo "BBH Overnight Run — $(date)"
echo "Models: score=${MODEL_SCORE}  rule-patch=${MODEL_RULE_PATCH}  cs=${MODEL_CS}"
echo "Output base: ${OUT_BASE}"
mkdir -p "${OUT_BASE}"

run_task "bbh_boolean"          "${DATA_DIR}/bbh_boolean_expressions.jsonl"
run_task "causal_judgement"     "${DATA_DIR}/causal_judgement.jsonl"
run_task "sports_understanding" "${DATA_DIR}/sports_understanding.jsonl"
run_task "disambiguation_qa"    "${DATA_DIR}/disambiguation_qa.jsonl"
run_task "movie_recommendation" "${DATA_DIR}/movie_recommendation.jsonl"
run_task "geometric_shapes"     "${DATA_DIR}/geometric_shapes.jsonl"

echo ""
echo "════════════════════════════════════════════════════════"
echo "ALL TASKS COMPLETE — $(date)"
echo "════════════════════════════════════════════════════════"

# Quick accuracy summary
echo ""
echo "=== Final accuracy summary ==="
for task in bbh_boolean causal_judgement sports_understanding disambiguation_qa movie_recommendation geometric_shapes; do
    log="${OUT_BASE}/${task}/run.log"
    if [[ -f "${log}" ]]; then
        acc=$(grep -oP "train_accuracy\s*:\s*\K[\d.]+%" "${log}" | tail -1)
        cs=$(grep -oP "case_studies_added\s*:\s*\K\d+" "${log}" | tail -1)
        echo "  ${task}: accuracy=${acc:-?}  case_studies=${cs:-?}"
    else
        echo "  ${task}: no log found"
    fi
done
