#!/usr/bin/env bash
# run_holistic_brg_rwsec_cj_dq_3seeds.sh
#
# brg_v2 + rewrite-level llama secondary gate on CJ (3 seeds) and DQ (3 seeds).
# After each rewrite passes the source-model brg gate, ALL training items are
# scored with llama-3.3-70b. If llama accuracy drops more than 3% vs current,
# the rewrite retries with the regressed llama cases injected as caution.
# Tracks secondary best cheatsheet separately (cheatsheet_sec_best.txt).
#
# Output dirs:
#   runs/holistic_cj_brg_rwsec_{1000,2000,3000}/
#   runs/holistic_dq_brg_rwsec_{1000,2000,3000}/
#
# Usage:
#   bash run_holistic_brg_rwsec_cj_dq_3seeds.sh

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

MODEL="openai/gpt-4.1-mini"
SECONDARY="meta-llama/llama-3.3-70b-instruct"

COMMON=(
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
  --rewrite-secondary-model "$SECONDARY"
  --rewrite-secondary-tolerance 0.03
)

CJ_DIR="CS_ICL_Initial_Prompt/bbh_causal_judgement"
DQ_DIR="CS_ICL_Initial_Prompt/bbh_disambiguation_qa"

# ── Batch 1: Causal Judgement ─────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Batch 1/2 — Causal Judgement (3 seeds)"

python3 -m ICR_holistic.pipeline "${COMMON[@]}" \
  --task causal_judgement \
  --dataset datasets/bbh/causal_judgement_train_labeled.jsonl \
  --init-cheatsheet "${CJ_DIR}/gen_gpt-4.1-2025-04-14_0_1000.txt" \
  --output-dir runs/holistic_cj_brg_rwsec_1000 \
  2>&1 | tee logs/holistic_cj_brg_rwsec_1000.log &

python3 -m ICR_holistic.pipeline "${COMMON[@]}" \
  --task causal_judgement \
  --dataset datasets/bbh/causal_judgement_train_labeled.jsonl \
  --init-cheatsheet "${CJ_DIR}/gen_gpt-4.1-2025-04-14_0_2000.txt" \
  --output-dir runs/holistic_cj_brg_rwsec_2000 \
  2>&1 | tee logs/holistic_cj_brg_rwsec_2000.log &

python3 -m ICR_holistic.pipeline "${COMMON[@]}" \
  --task causal_judgement \
  --dataset datasets/bbh/causal_judgement_train_labeled.jsonl \
  --init-cheatsheet "${CJ_DIR}/gen_gpt-4.1-2025-04-14_0_3000.txt" \
  --output-dir runs/holistic_cj_brg_rwsec_3000 \
  2>&1 | tee logs/holistic_cj_brg_rwsec_3000.log &

wait
echo "[$(date '+%H:%M:%S')] Batch 1/2 done"
echo ""

# ── Batch 2: Disambiguation QA ────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Batch 2/2 — Disambiguation QA (3 seeds)"

python3 -m ICR_holistic.pipeline "${COMMON[@]}" \
  --task disambiguation_qa \
  --dataset datasets/bbh/disambiguation_qa_train_labeled.jsonl \
  --init-cheatsheet "${DQ_DIR}/gen_gpt-4.1-2025-04-14_0_1000.txt" \
  --output-dir runs/holistic_dq_brg_rwsec_1000 \
  2>&1 | tee logs/holistic_dq_brg_rwsec_1000.log &

python3 -m ICR_holistic.pipeline "${COMMON[@]}" \
  --task disambiguation_qa \
  --dataset datasets/bbh/disambiguation_qa_train_labeled.jsonl \
  --init-cheatsheet "${DQ_DIR}/gen_gpt-4.1-2025-04-14_0_2000.txt" \
  --output-dir runs/holistic_dq_brg_rwsec_2000 \
  2>&1 | tee logs/holistic_dq_brg_rwsec_2000.log &

python3 -m ICR_holistic.pipeline "${COMMON[@]}" \
  --task disambiguation_qa \
  --dataset datasets/bbh/disambiguation_qa_train_labeled.jsonl \
  --init-cheatsheet "${DQ_DIR}/gen_gpt-4.1-2025-04-14_0_3000.txt" \
  --output-dir runs/holistic_dq_brg_rwsec_3000 \
  2>&1 | tee logs/holistic_dq_brg_rwsec_3000.log &

wait
echo "[$(date '+%H:%M:%S')] Batch 2/2 done"
echo ""
echo "All 6 runs complete."
echo "  CJ: runs/holistic_cj_brg_rwsec_{1000,2000,3000}/"
echo "  DQ: runs/holistic_dq_brg_rwsec_{1000,2000,3000}/"
