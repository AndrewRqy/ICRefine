#!/usr/bin/env bash
# run_v7_oracle_rw_abl.sh — Ablation: oracle injection at the rewrite level.
#
# Runs icr_holistic_v7 with --oracle-rewrite-injection on CJ (seeds 1000+2000)
# and GS (seeds 2000+3000) — selected for high update counts in the label-only run:
#   CJ  seed1000: 7 iters, 4 best-updates
#   CJ  seed2000: 5 iters, 2 best-updates
#   GS  seed2000: 8 iters, 3 best-updates
#   GS  seed3000: 8 iters, 3 best-updates
#
# Compare cheatsheet_best.txt from these runs against label-only equivalents in
# ICR_paper_ready/holistic_{cj,gs}_v7_{seed} to isolate the rewrite oracle effect.
#
# Output: ICR_paper_ready/holistic_{cj,gs}_v7_oracle_rw_{seed}
#
# Usage:
#   bash scripts/pipeline/run_v7_oracle_rw_abl.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
if [ -f "$ROOT/.env" ]; then set -a; source "$ROOT/.env"; set +a; fi

MODEL="openai/gpt-4.1-mini"
mkdir -p logs/abl ICR_paper_ready

TASK_DEFS=(
    "causal_judgement|cj|datasets/bbh/causal_judgement_train_labeled.jsonl|CS_ICL_Initial_Prompt/bbh_causal_judgement|1000 2000"
    "geometric_shapes|gs|datasets/bbh/geometric_shapes_train_labeled.jsonl|CS_ICL_Initial_Prompt/bbh_geometric_shapes|2000 3000"
)

JOBS=()
for TASK_DEF in "${TASK_DEFS[@]}"; do
    IFS='|' read -r TASK ABBR DATASET CS_DIR SEED_LIST <<< "$TASK_DEF"
    for SEED in $SEED_LIST; do
        JOBS+=("${TASK}|${ABBR}|${DATASET}|${CS_DIR}|${SEED}")
    done
done

echo "=== ${#JOBS[@]} oracle-rewrite ablation jobs (all in parallel) ==="
echo "  Tasks: CJ (seeds 1000+2000) + GS (seeds 2000+3000)"
echo "  Flag : --oracle-rewrite-injection"
echo ""

pids=(); labels=()
for JOB in "${JOBS[@]}"; do
    IFS='|' read -r TASK ABBR DATASET CS_DIR SEED <<< "$JOB"
    OUTDIR="ICR_paper_ready/holistic_${ABBR}_v7_oracle_rw_${SEED}"
    LOG="logs/abl/v7_oracle_rw_${ABBR}_${SEED}_$(date +%Y%m%d_%H%M%S).log"
    CS_FILE="${CS_DIR}/gen_gpt-4.1-mini_0_${SEED}.txt"
    echo "--- Launching oracle_rw ${ABBR}_${SEED}  →  ${LOG}"
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
        --cs-icl-tokens "$SEED" \
        --oracle-rewrite-injection \
        > "$LOG" 2>&1 &
    pids+=($!); labels+=("oracle_rw_${ABBR}_${SEED}")
done

echo ""
echo "--- Waiting for all 4 jobs: ${labels[*]}..."
for pid in "${pids[@]}"; do wait "$pid"; done
echo "[$(date '+%H:%M:%S')] All oracle-rewrite ablation runs complete."
echo ""
echo "Output dirs:"
for JOB in "${JOBS[@]}"; do
    IFS='|' read -r TASK ABBR DATASET CS_DIR SEED <<< "$JOB"
    echo "  ICR_paper_ready/holistic_${ABBR}_v7_oracle_rw_${SEED}"
done
echo ""
echo "Label-only baselines to compare against:"
for JOB in "${JOBS[@]}"; do
    IFS='|' read -r TASK ABBR DATASET CS_DIR SEED <<< "$JOB"
    echo "  ICR_paper_ready/holistic_${ABBR}_v7_${SEED}"
done
