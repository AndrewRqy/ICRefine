#!/usr/bin/env bash
# run_sfcr_minimized_cj.sh
#
# SFCR on CJ (3 seeds) with minimized generation:
#   - Rule output: RULE + USE WHEN only (no DO NOT USE WHEN, CHECK, MICRO-EXAMPLE)
#   - Subtype-clustered generation with quick-validate (default)
#   - Quick-validate: each candidate is tested on its subtype items before gate scoring;
#     if it fixes nothing, it is discarded and regenerated.
#   - Gate: fixed_shared_count >= 0 (any non-regressing candidate passes the count gate)
#   - repair-attempts 2: rejected candidates get two repair attempts before discard
#   - gpt-4.1-mini source; gpt-4.1 + llama proxies
#
# Output dirs:
#   runs/sfcr_min_cj_{1000,2000,3000}/
#
# Usage:
#   bash run_sfcr_minimized_cj.sh

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

SOURCE="openai/gpt-4.1-mini"
PROXIES="openai/gpt-4.1,meta-llama/llama-3.3-70b-instruct"
CJ_DIR="CS_ICL_Initial_Prompt/bbh_causal_judgement"

COMMON=(
  --task              causal_judgement
  --dataset           datasets/bbh/causal_judgement_train_labeled.jsonl
  --model-source      "$SOURCE"
  --models-proxy      "$PROXIES"
  --oracle-mode       label_only
  --routing-mode      routed
  --memory-format     rule
  --gate-profile      medium
  --repair-attempts   2
  --concurrency       30
)

echo "[$(date '+%H:%M:%S')] SFCR minimized — CJ (3 seeds)"

python3 -m ICR_sfcr.pipeline "${COMMON[@]}" \
  --anchor-cheatsheet "${CJ_DIR}/gen_gpt-4.1-2025-04-14_0_1000.txt" \
  --output-dir runs/sfcr_min_cj_1000 \
  --seed 1000 \
  2>&1 | tee logs/sfcr_min_cj_1000.log &

python3 -m ICR_sfcr.pipeline "${COMMON[@]}" \
  --anchor-cheatsheet "${CJ_DIR}/gen_gpt-4.1-2025-04-14_0_2000.txt" \
  --output-dir runs/sfcr_min_cj_2000 \
  --seed 2000 \
  2>&1 | tee logs/sfcr_min_cj_2000.log &

python3 -m ICR_sfcr.pipeline "${COMMON[@]}" \
  --anchor-cheatsheet "${CJ_DIR}/gen_gpt-4.1-2025-04-14_0_3000.txt" \
  --output-dir runs/sfcr_min_cj_3000 \
  --seed 3000 \
  2>&1 | tee logs/sfcr_min_cj_3000.log &

wait
echo "[$(date '+%H:%M:%S')] All 3 runs complete."
echo "  runs/sfcr_min_cj_{1000,2000,3000}/"
