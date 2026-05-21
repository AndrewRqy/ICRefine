#!/usr/bin/env bash
# run_pipelines_agieval.sh — Faithful v7, SFCR append, and v7sfcr on all 3 AGIEval tasks × 3 seeds.
#
# Pipelines (each in batches of 3):
#   1. icr_holistic_v7      → ICR_paper_ready/holistic_{lq,lsar,lslr}_v7_{seed}
#   2. icr_holistic_sfcr    → runs/sfcr/holistic_{lq,lsar,lslr}_sfcr_append_{seed}
#   3. icr_holistic_v7sfcr  → ICR_paper_ready/holistic_{lq,lsar,lslr}_v7sfcr_gemini_{seed}
#
# Usage:
#   bash scripts/pipeline/run_pipelines_agieval.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
if [ -f "$ROOT/.env" ]; then set -a; source "$ROOT/.env"; set +a; fi

MODEL="openai/gpt-4.1-mini"
PROXY="google/gemini-2.5-flash-lite"
CS_ICL_SUFFIX="gen_gpt-4.1-mini_0"
SEEDS=(1000 2000 3000)

mkdir -p logs/agieval runs/sfcr ICR_paper_ready

# Task definitions: "task_name|abbr|dataset_suffix"
# dataset: datasets/agieval/{suffix}_train_oracle_labeled.jsonl
# cs_dir:  CS_ICL_Initial_Prompt/agieval_{suffix}/
TASK_DEFS=(
    "agieval_logiqa_en|lq|logiqa_en"
    "agieval_lsat_ar|lsar|lsat_ar"
    "agieval_lsat_lr|lslr|lsat_lr"
)

# Build shared job list: "task|abbr|suffix|seed"
ALL_JOBS=()
for TASK_DEF in "${TASK_DEFS[@]}"; do
    IFS='|' read -r TASK ABBR SUFFIX <<< "$TASK_DEF"
    for SEED in "${SEEDS[@]}"; do
        ALL_JOBS+=("${TASK}|${ABBR}|${SUFFIX}|${SEED}")
    done
done

total=${#ALL_JOBS[@]}

# ── Section 1: Faithful v7 ───────────────────────────────────────────────────
echo "=========================================="
echo " SECTION 1: Faithful v7 (icr_holistic_v7)"
echo "=========================================="
echo "=== ${total} v7 jobs queued (3 at a time) ==="
echo ""

i=0; batch=0
while [ $i -lt $total ]; do
    batch=$((batch+1))
    batch_pids=(); batch_labels=()
    for ((b=0; b<3 && i<total; b++, i++)); do
        IFS='|' read -r TASK ABBR SUFFIX SEED <<< "${ALL_JOBS[$i]}"
        OUTDIR="ICR_paper_ready/holistic_${ABBR}_v7_${SEED}"
        LOG="logs/agieval/v7_${ABBR}_${SEED}_$(date +%Y%m%d_%H%M%S).log"
        DATASET="datasets/agieval/${SUFFIX}_train_oracle_labeled.jsonl"
        echo "--- Launching v7 ${ABBR}_${SEED}  →  ${LOG}"
        python3 -m icr_holistic_v7.pipeline \
            --task            "$TASK" \
            --dataset         "$DATASET" \
            --model-score     "$MODEL" \
            --model-gen       "$MODEL" \
            --output-dir      "$OUTDIR" \
            --max-iters 8 \
            --fix-rate 0.10 \
            --regression-pool 100 \
            --bin-threshold 2 \
            --concurrency 30 \
            --bin-retry 2 \
            --rollback \
            --min-pool-for-net-gate 4 \
            --early-stop-patience 3 \
            --beam-size 2 \
            --cs-icl-tokens   "$SEED" \
            > "$LOG" 2>&1 &
        batch_pids+=($!); batch_labels+=("v7_${ABBR}_${SEED}")
    done
    echo "--- Batch ${batch}: waiting for ${batch_labels[*]}..."
    for pid in "${batch_pids[@]}"; do wait "$pid"; done
    echo "[$(date '+%H:%M:%S')] Batch ${batch} done: ${batch_labels[*]}"
    echo ""
done
echo "=== Faithful v7 AGIEval complete ==="
echo ""

# ── Section 2: SFCR append ───────────────────────────────────────────────────
echo "=============================================="
echo " SECTION 2: SFCR append (icr_holistic_sfcr)"
echo "=============================================="
echo "=== ${total} SFCR append jobs queued (3 at a time) ==="
echo ""

i=0; batch=0
while [ $i -lt $total ]; do
    batch=$((batch+1))
    batch_pids=(); batch_labels=()
    for ((b=0; b<3 && i<total; b++, i++)); do
        IFS='|' read -r TASK ABBR SUFFIX SEED <<< "${ALL_JOBS[$i]}"
        OUTDIR="runs/sfcr/holistic_${ABBR}_sfcr_append_${SEED}"
        LOG="logs/agieval/sfcr_append_${ABBR}_${SEED}_$(date +%Y%m%d_%H%M%S).log"
        DATASET="datasets/agieval/${SUFFIX}_train_oracle_labeled.jsonl"
        CS_FILE="CS_ICL_Initial_Prompt/agieval_${SUFFIX}/${CS_ICL_SUFFIX}_${SEED}.txt"
        echo "--- Launching sfcr_append ${ABBR}_${SEED}  →  ${LOG}"
        python3 -m icr_holistic_sfcr.pipeline \
            --task              "$TASK" \
            --dataset           "$DATASET" \
            --init-cheatsheet   "$CS_FILE" \
            --model-score       "$MODEL" \
            --model-gen         "$MODEL" \
            --output-dir        "$OUTDIR" \
            --max-iters 8 \
            --fix-rate 0.10 \
            --regression-pool 100 \
            --bin-threshold 2 \
            --concurrency 30 \
            --bin-retry 2 \
            --rollback \
            --min-pool-for-net-gate 4 \
            --early-stop-patience 3 \
            --rewrite-candidates 2 \
            --proxy-model       "$PROXY" \
            --proxy-concurrency 30 \
            --sfcr-lambda 0.5 \
            --sfcr-mu 0.5 \
            --sfcr-min-private 3 \
            --sfcr-min-easy 8 \
            --append-mode \
            > "$LOG" 2>&1 &
        batch_pids+=($!); batch_labels+=("sfcr_${ABBR}_${SEED}")
    done
    echo "--- Batch ${batch}: waiting for ${batch_labels[*]}..."
    for pid in "${batch_pids[@]}"; do wait "$pid"; done
    echo "[$(date '+%H:%M:%S')] Batch ${batch} done: ${batch_labels[*]}"
    echo ""
done
echo "=== SFCR append AGIEval complete ==="
echo ""

# ── Section 3: v7sfcr ────────────────────────────────────────────────────────
echo "================================================"
echo " SECTION 3: v7sfcr (icr_holistic_v7sfcr)"
echo "================================================"
echo "=== ${total} v7sfcr jobs queued (3 at a time) ==="
echo ""

i=0; batch=0
while [ $i -lt $total ]; do
    batch=$((batch+1))
    batch_pids=(); batch_labels=()
    for ((b=0; b<3 && i<total; b++, i++)); do
        IFS='|' read -r TASK ABBR SUFFIX SEED <<< "${ALL_JOBS[$i]}"
        OUTDIR="ICR_paper_ready/holistic_${ABBR}_v7sfcr_gemini_${SEED}"
        LOG="logs/agieval/v7sfcr_${ABBR}_${SEED}_$(date +%Y%m%d_%H%M%S).log"
        DATASET="datasets/agieval/${SUFFIX}_train_oracle_labeled.jsonl"
        echo "--- Launching v7sfcr ${ABBR}_${SEED}  →  ${LOG}"
        python3 -m icr_holistic_v7sfcr.pipeline \
            --task          "$TASK" \
            --dataset       "$DATASET" \
            --proxy-model   "$PROXY" \
            --model-score   "$MODEL" \
            --model-gen     "$MODEL" \
            --output-dir    "$OUTDIR" \
            --max-iters 8 \
            --fix-rate 0.10 \
            --regression-pool 100 \
            --bin-threshold 2 \
            --concurrency 30 \
            --bin-retry 2 \
            --rollback \
            --min-pool-for-net-gate 4 \
            --early-stop-patience 3 \
            --beam-size 2 \
            --cs-icl-tokens "$SEED" \
            > "$LOG" 2>&1 &
        batch_pids+=($!); batch_labels+=("v7sfcr_${ABBR}_${SEED}")
    done
    echo "--- Batch ${batch}: waiting for ${batch_labels[*]}..."
    for pid in "${batch_pids[@]}"; do wait "$pid"; done
    echo "[$(date '+%H:%M:%S')] Batch ${batch} done: ${batch_labels[*]}"
    echo ""
done
echo "=== v7sfcr AGIEval complete ==="
echo ""

echo "=========================================="
echo " All AGIEval pipeline runs complete."
echo ""
echo " Output dirs (v7 / sfcr_append / v7sfcr):"
for JOB in "${ALL_JOBS[@]}"; do
    IFS='|' read -r TASK ABBR SUFFIX SEED <<< "$JOB"
    echo "  ICR_paper_ready/holistic_${ABBR}_v7_${SEED}"
    echo "  runs/sfcr/holistic_${ABBR}_sfcr_append_${SEED}"
    echo "  ICR_paper_ready/holistic_${ABBR}_v7sfcr_gemini_${SEED}"
done
echo "=========================================="
