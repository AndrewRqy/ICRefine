#!/usr/bin/env bash
# run_holistic_cj_v7_brg_3seeds.sh
#
# Holistic v7 baseline + broken-item rewrite gate across 3 CJ seeds (1000/2000/3000).
# Gate: rewrite must fix ≥3 wrong items AND break ≤3 correct items; retries up to 3x
# with broken cases fed back as caution context.
#
# Output dirs:
#   runs/holistic_cj_v7_brg_1000/
#   runs/holistic_cj_v7_brg_2000/
#   runs/holistic_cj_v7_brg_3000/
#
# Usage:
#   bash run_holistic_cj_v7_brg_3seeds.sh

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
  --rewrite-max-broken 3
  --rewrite-gate-retries 3
)

python3 -m ICR_holistic.pipeline "${COMMON[@]}" \
  --init-cheatsheet "${INIT_DIR}/gen_gpt-4.1-2025-04-14_0_1000.txt" \
  --output-dir runs/holistic_cj_v7_brg_1000 \
  2>&1 | tee logs/holistic_cj_v7_brg_1000.log &

python3 -m ICR_holistic.pipeline "${COMMON[@]}" \
  --init-cheatsheet "${INIT_DIR}/gen_gpt-4.1-2025-04-14_0_2000.txt" \
  --output-dir runs/holistic_cj_v7_brg_2000 \
  2>&1 | tee logs/holistic_cj_v7_brg_2000.log &

python3 -m ICR_holistic.pipeline "${COMMON[@]}" \
  --init-cheatsheet "${INIT_DIR}/gen_gpt-4.1-2025-04-14_0_3000.txt" \
  --output-dir runs/holistic_cj_v7_brg_3000 \
  2>&1 | tee logs/holistic_cj_v7_brg_3000.log &

wait
echo ""
echo "All three seeds complete."
echo "  seed 1000: runs/holistic_cj_v7_brg_1000/"
echo "  seed 2000: runs/holistic_cj_v7_brg_2000/"
echo "  seed 3000: runs/holistic_cj_v7_brg_3000/"
