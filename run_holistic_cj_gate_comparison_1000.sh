#!/usr/bin/env bash
# run_holistic_cj_gate_comparison_1000.sh
#
# Compares two rewrite gate strategies on CJ seed 1000 in parallel:
#
#   threshold  — rewrite must fix ≥3 AND break ≤10 correct items
#                (10 is at the low end of the observed broken range, so the gate
#                 actually fires on bad rewrites rather than never triggering)
#
#   net-gain   — rewrite must achieve net_gain = (fixed − broken) ≥ 2
#                (no hard ceiling on broken; only the balance matters)
#
# Both use the same base v7 config, 3 retries, caution feedback on failure.
# Output:
#   runs/holistic_cj_v7_threshold_1000/
#   runs/holistic_cj_v7_netgain_1000/
#
# Usage:
#   bash run_holistic_cj_gate_comparison_1000.sh

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

MODEL="openai/gpt-4.1-mini"
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
  --rewrite-min-fix 3
  --rewrite-gate-retries 3
)

# Approach A: threshold — gate fires when broken > 10
python3 -m ICR_holistic.pipeline "${COMMON[@]}" \
  --output-dir runs/holistic_cj_v7_threshold_1000 \
  --rewrite-max-broken 10 \
  2>&1 | tee logs/holistic_cj_v7_threshold_1000.log &

# Approach B: net-gain — gate fires when (fixed - broken) < 2
python3 -m ICR_holistic.pipeline "${COMMON[@]}" \
  --output-dir runs/holistic_cj_v7_netgain_1000 \
  --rewrite-min-net-gain 2 \
  2>&1 | tee logs/holistic_cj_v7_netgain_1000.log &

wait
echo ""
echo "Both runs complete."
echo "  threshold (broken≤10): runs/holistic_cj_v7_threshold_1000/"
echo "  net-gain  (net≥2):     runs/holistic_cj_v7_netgain_1000/"
