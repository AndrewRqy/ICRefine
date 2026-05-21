#!/usr/bin/env bash
# run_sfcr_fullcot_cj_dq.sh
#
# SFCR on CJ (3 seeds) + DQ (3 seeds) with full-CoT oracle injection:
#   - oracle-mode full_cot: generator sees Question + correct Answer + full Reasoning
#   - Rule output: RULE + USE WHEN only
#   - Subtype-clustered generation with quick-validate
#   - Gate: fixed_shared_count >= 0 (any non-regressing candidate passes)
#   - repair-attempts 2
#   - gpt-4.1-mini source; gpt-4.1 + llama proxies
#
# Output dirs:
#   runs/sfcr_fcot_cj_{1000,2000,3000}/
#   runs/sfcr_fcot_dq_{1000,2000,3000}/
#
# Usage:
#   bash run_sfcr_fullcot_cj_dq.sh

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

SOURCE="openai/gpt-4.1-mini"
PROXIES="openai/gpt-4.1,meta-llama/llama-3.3-70b-instruct"
CJ_DIR="CS_ICL_Initial_Prompt/bbh_causal_judgement"
DQ_DIR="CS_ICL_Initial_Prompt/bbh_disambiguation_qa"

COMMON=(
  --model-source      "$SOURCE"
  --models-proxy      "$PROXIES"
  --oracle-mode       full_cot
  --routing-mode      routed
  --memory-format     rule
  --gate-profile      medium
  --repair-attempts   2
  --concurrency       30
)

# ── Batch 1: Causal Judgement ─────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Batch 1/2 — Causal Judgement (3 seeds, full_cot)"

python3 -m ICR_sfcr.pipeline "${COMMON[@]}" \
  --task              causal_judgement \
  --dataset           datasets/bbh/causal_judgement_train_labeled.jsonl \
  --anchor-cheatsheet "${CJ_DIR}/gen_gpt-4.1-2025-04-14_0_1000.txt" \
  --output-dir        runs/sfcr_fcot_cj_1000 \
  --seed              1000 \
  2>&1 | tee logs/sfcr_fcot_cj_1000.log &

python3 -m ICR_sfcr.pipeline "${COMMON[@]}" \
  --task              causal_judgement \
  --dataset           datasets/bbh/causal_judgement_train_labeled.jsonl \
  --anchor-cheatsheet "${CJ_DIR}/gen_gpt-4.1-2025-04-14_0_2000.txt" \
  --output-dir        runs/sfcr_fcot_cj_2000 \
  --seed              2000 \
  2>&1 | tee logs/sfcr_fcot_cj_2000.log &

python3 -m ICR_sfcr.pipeline "${COMMON[@]}" \
  --task              causal_judgement \
  --dataset           datasets/bbh/causal_judgement_train_labeled.jsonl \
  --anchor-cheatsheet "${CJ_DIR}/gen_gpt-4.1-2025-04-14_0_3000.txt" \
  --output-dir        runs/sfcr_fcot_cj_3000 \
  --seed              3000 \
  2>&1 | tee logs/sfcr_fcot_cj_3000.log &

wait
echo "[$(date '+%H:%M:%S')] Batch 1/2 done"
echo ""

# ── Batch 2: Disambiguation QA ────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Batch 2/2 — Disambiguation QA (3 seeds, full_cot)"

python3 -m ICR_sfcr.pipeline "${COMMON[@]}" \
  --task              disambiguation_qa \
  --dataset           datasets/bbh/disambiguation_qa_train_labeled.jsonl \
  --anchor-cheatsheet "${DQ_DIR}/gen_gpt-4.1-2025-04-14_0_1000.txt" \
  --output-dir        runs/sfcr_fcot_dq_1000 \
  --seed              1000 \
  2>&1 | tee logs/sfcr_fcot_dq_1000.log &

python3 -m ICR_sfcr.pipeline "${COMMON[@]}" \
  --task              disambiguation_qa \
  --dataset           datasets/bbh/disambiguation_qa_train_labeled.jsonl \
  --anchor-cheatsheet "${DQ_DIR}/gen_gpt-4.1-2025-04-14_0_2000.txt" \
  --output-dir        runs/sfcr_fcot_dq_2000 \
  --seed              2000 \
  2>&1 | tee logs/sfcr_fcot_dq_2000.log &

python3 -m ICR_sfcr.pipeline "${COMMON[@]}" \
  --task              disambiguation_qa \
  --dataset           datasets/bbh/disambiguation_qa_train_labeled.jsonl \
  --anchor-cheatsheet "${DQ_DIR}/gen_gpt-4.1-2025-04-14_0_3000.txt" \
  --output-dir        runs/sfcr_fcot_dq_3000 \
  --seed              3000 \
  2>&1 | tee logs/sfcr_fcot_dq_3000.log &

wait
echo "[$(date '+%H:%M:%S')] Batch 2/2 done"
echo ""
echo "All 6 runs complete."
echo "  CJ: runs/sfcr_fcot_cj_{1000,2000,3000}/"
echo "  DQ: runs/sfcr_fcot_dq_{1000,2000,3000}/"
