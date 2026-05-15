#!/usr/bin/env bash
# run_holistic_cj_rg_comparison_1000.sh
#
# Runs three CJ seed-1000 variants in parallel with the rewrite gate (min_fix=3):
#   1. v7 baseline      → runs/holistic_cj_v7_rg_1000/
#   2. slow-and-steady  → runs/holistic_cj_sas_rg_1000/
#   3. secondary gate   → runs/holistic_cj_sec_rg_1000/
#
# Usage:
#   bash run_holistic_cj_rg_comparison_1000.sh

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

MODEL="openai/gpt-4.1-mini"
SECONDARY="meta-llama/llama-3.3-70b-instruct"
INIT="CS_ICL_Initial_Prompt/bbh_causal_judgement/gen_gpt-4.1-2025-04-14_0_1000.txt"

COMMON=(
  --task causal_judgement
  --dataset datasets/bbh/causal_judgement_train_labeled.jsonl
  --model-score "$MODEL"
  --model-gen   "$MODEL"
  --init-cheatsheet "$INIT"
  --max-iters 8
  --fix-rate 0.10
  --regression-pool 100
  --bin-threshold 2
  --concurrency 30
  --bin-retry 2
  --rollback
  --min-pool-for-net-gate 4
  --early-stop-patience 3
  --beam-size 1
)

python3 -m ICR_holistic.pipeline "${COMMON[@]}" \
  --output-dir runs/holistic_cj_v7_rg_1000 \
  2>&1 | tee logs/holistic_cj_v7_rg_1000.log &

python3 -m ICR_holistic.pipeline "${COMMON[@]}" \
  --output-dir runs/holistic_cj_sas_rg_1000 \
  --slowandsteady \
  2>&1 | tee logs/holistic_cj_sas_rg_1000.log &

python3 -m ICR_holistic.pipeline "${COMMON[@]}" \
  --output-dir runs/holistic_cj_sec_rg_1000 \
  --secondary-model "$SECONDARY" \
  --secondary-tolerance 1 \
  2>&1 | tee logs/holistic_cj_sec_rg_1000.log &

wait
echo ""
echo "All three runs complete."
echo "  v7 baseline:     runs/holistic_cj_v7_rg_1000/"
echo "  slow-and-steady: runs/holistic_cj_sas_rg_1000/"
echo "  secondary gate:  runs/holistic_cj_sec_rg_1000/"
