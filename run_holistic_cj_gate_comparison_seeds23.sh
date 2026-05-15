#!/usr/bin/env bash
# run_holistic_cj_gate_comparison_seeds23.sh
#
# Runs threshold vs net-gain gate comparison on CJ seeds 2000 and 3000 in parallel.
# Same config as run_holistic_cj_gate_comparison_1000.sh.
#
# Output:
#   runs/holistic_cj_v7_threshold_2000/
#   runs/holistic_cj_v7_threshold_3000/
#   runs/holistic_cj_v7_netgain_2000/
#   runs/holistic_cj_v7_netgain_3000/
#
# Usage:
#   bash run_holistic_cj_gate_comparison_seeds23.sh

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

MODEL="openai/gpt-4.1-mini"
INIT_DIR="CS_ICL_Initial_Prompt/bbh_causal_judgement"

COMMON=(
  --task causal_judgement
  --dataset datasets/bbh/causal_judgement_train_labeled.jsonl
  --model-score "$MODEL"
  --model-gen   "$MODEL"
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
  --rewrite-min-fix 3
  --rewrite-gate-retries 3
)

python3 -m ICR_holistic.pipeline "${COMMON[@]}" \
  --init-cheatsheet "${INIT_DIR}/gen_gpt-4.1-2025-04-14_0_2000.txt" \
  --output-dir runs/holistic_cj_v7_threshold_2000 \
  --rewrite-max-broken 10 \
  2>&1 | tee logs/holistic_cj_v7_threshold_2000.log &

python3 -m ICR_holistic.pipeline "${COMMON[@]}" \
  --init-cheatsheet "${INIT_DIR}/gen_gpt-4.1-2025-04-14_0_3000.txt" \
  --output-dir runs/holistic_cj_v7_threshold_3000 \
  --rewrite-max-broken 10 \
  2>&1 | tee logs/holistic_cj_v7_threshold_3000.log &

python3 -m ICR_holistic.pipeline "${COMMON[@]}" \
  --init-cheatsheet "${INIT_DIR}/gen_gpt-4.1-2025-04-14_0_2000.txt" \
  --output-dir runs/holistic_cj_v7_netgain_2000 \
  --rewrite-min-net-gain 2 \
  2>&1 | tee logs/holistic_cj_v7_netgain_2000.log &

python3 -m ICR_holistic.pipeline "${COMMON[@]}" \
  --init-cheatsheet "${INIT_DIR}/gen_gpt-4.1-2025-04-14_0_3000.txt" \
  --output-dir runs/holistic_cj_v7_netgain_3000 \
  --rewrite-min-net-gain 2 \
  2>&1 | tee logs/holistic_cj_v7_netgain_3000.log &

wait
echo ""
echo "All four runs complete."
echo "  threshold seed 2000: runs/holistic_cj_v7_threshold_2000/"
echo "  threshold seed 3000: runs/holistic_cj_v7_threshold_3000/"
echo "  net-gain  seed 2000: runs/holistic_cj_v7_netgain_2000/"
echo "  net-gain  seed 3000: runs/holistic_cj_v7_netgain_3000/"
