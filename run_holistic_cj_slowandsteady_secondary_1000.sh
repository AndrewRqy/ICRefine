#!/usr/bin/env bash
# run_holistic_cj_slowandsteady_secondary_1000.sh
#
# CJ seed1000 with --slowandsteady + secondary model delta gate.
# Secondary model: meta-llama/llama-3.3-70b-instruct (veto if delta < -1).
# Rewriter selects at most 3 candidates per iteration; deferred candidates
# are re-presented in subsequent iterations.
#
# Matches beam3 v7 config exactly plus --slowandsteady + --secondary-model.
# Output: runs/holistic_cj_slowandsteady_secondary_1000/
#
# Usage:
#   bash run_holistic_cj_slowandsteady_secondary_1000.sh

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

MODEL="openai/gpt-4.1-mini"
SECONDARY="meta-llama/llama-3.3-70b-instruct"
INIT_CS="CS_ICL_Initial_Prompt/bbh_causal_judgement/gen_gpt-4.1-2025-04-14_0_1000.txt"

python3 -m ICR_holistic.pipeline \
  --task causal_judgement \
  --dataset datasets/bbh/causal_judgement_train_labeled.jsonl \
  --output-dir runs/holistic_cj_slowandsteady_secondary_1000 \
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
  --beam-size 1 \
  --slowandsteady \
  --secondary-model "$SECONDARY" \
  --secondary-tolerance 1 \
  2>&1 | tee logs/holistic_cj_slowandsteady_secondary_1000.log

echo ""
echo "Done. Results in runs/holistic_cj_slowandsteady_secondary_1000/"
echo "Log:    logs/holistic_cj_slowandsteady_secondary_1000.log"
echo "Best:   runs/holistic_cj_slowandsteady_secondary_1000/cheatsheet_best.txt"
echo "Train log: runs/holistic_cj_slowandsteady_secondary_1000/training_log.json"
