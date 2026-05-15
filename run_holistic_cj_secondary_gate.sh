#!/usr/bin/env bash
# run_holistic_cj_secondary_gate.sh
#
# Test: secondary-model delta gate on causal_judgement seed1000.
# Secondary model: meta-llama/llama-3.3-70b-instruct  (veto if it regresses
#   more than 1 item on the regression pool when bin content is added).
#
# Matches beam3 v7 config exactly except --secondary-model + --secondary-tolerance.
# Output: runs/holistic_cj_secondary_1000/
#
# Usage:
#   bash run_holistic_cj_secondary_gate.sh

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

MODEL="openai/gpt-4.1-mini"
SECONDARY="meta-llama/llama-3.3-70b-instruct"
INIT_CS="CS_ICL_Initial_Prompt/bbh_causal_judgement/gen_gpt-4.1-2025-04-14_0_1000.txt"

python3 -m ICR_holistic.pipeline \
  --task causal_judgement \
  --dataset datasets/bbh/causal_judgement_train_labeled.jsonl \
  --output-dir runs/holistic_cj_secondary_1000 \
  --model-score "$MODEL" \
  --model-gen   "$MODEL" \
  --init-cheatsheet "$INIT_CS" \
  --max-iters 8 \
  --fix-rate 0.10 \
  --regression-pool 100 \
  --bin-threshold 2 \
  --concurrency 30 \
  --bin-retry 2 \
  --rollback \
  --min-pool-for-net-gate 4 \
  --early-stop-patience 3 \
  --beam-size 3 \
  --cs-icl-tokens 1000 \
  --secondary-model "$SECONDARY" \
  --secondary-tolerance 1 \
  2>&1 | tee logs/holistic_cj_secondary_1000.log

echo ""
echo "Done. Results in runs/holistic_cj_secondary_1000/"
echo "Log:    logs/holistic_cj_secondary_1000.log"
echo "Best:   runs/holistic_cj_secondary_1000/cheatsheet_best.txt"
echo "Train log: runs/holistic_cj_secondary_1000/training_log.json"
