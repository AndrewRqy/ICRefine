#!/usr/bin/env bash
# run_sfcr_tier12.sh — SFCR on Tier 1 & 2 tasks, seed=1000, CS-ICL anchors
#
# Tasks (in order):
#   1. lsat_ar      — highest uniform Jaccard (0.66-0.72); CS-ICL anchor (gpt-4.1-mini)
#   2. logiqa_en    — good Jaccard, 325-item dataset → enlarged splits
#   3. geometric_shapes — highest GPT Jaccard (0.864); CS-ICL anchor (gpt-4.1)
#   4. causal_judgement — CS-ICL anchor (avoids 85% source acc from v7 anchor)
#
# All runs: held-out=gemini, oracle_mode=label_only, routing=routed, seed=1000

set -e
cd "$(dirname "$0")"

PROXY="openai/gpt-4.1,google/gemini-2.0-flash-001,meta-llama/llama-3.3-70b-instruct"

# ── 1. lsat_ar ───────────────────────────────────────────────────────────────
echo "============================================================"
echo " [1/4] lsat_ar  (CS-ICL anchor, default splits 60/40)"
echo "============================================================"
python3 -m ICR_sfcr.pipeline \
  --task              agieval_lsat_ar \
  --dataset           datasets/agieval/lsat_ar_train.jsonl \
  --anchor-cheatsheet CS_ICL_Initial_Prompt/agieval_lsat_ar/gen_gpt-4.1-mini_0_1000.txt \
  --output-dir        runs/sfcr_lsat_ar_1000 \
  --model-source      openai/gpt-4.1-mini \
  --models-proxy      "$PROXY" \
  --held-out-target   gemini \
  --oracle-mode       label_only \
  --routing-mode      routed \
  --seed              1000 \
  --concurrency       50

# ── 2. logiqa_en ─────────────────────────────────────────────────────────────
echo "============================================================"
echo " [2/4] logiqa_en  (CS-ICL anchor, enlarged splits 100/150)"
echo "============================================================"
python3 -m ICR_sfcr.pipeline \
  --task              agieval_logiqa_en \
  --dataset           datasets/agieval/logiqa_en_train.jsonl \
  --anchor-cheatsheet CS_ICL_Initial_Prompt/agieval_logiqa_en/gen_gpt-4.1-mini_0_1000.txt \
  --output-dir        runs/sfcr_logiqa_en_1000 \
  --model-source      openai/gpt-4.1-mini \
  --models-proxy      "$PROXY" \
  --held-out-target   gemini \
  --oracle-mode       label_only \
  --routing-mode      routed \
  --seed              1000 \
  --rule-gen-n        100 \
  --gate-n            150 \
  --concurrency       50

# ── 3. geometric_shapes ──────────────────────────────────────────────────────
echo "============================================================"
echo " [3/4] geometric_shapes  (CS-ICL anchor, enlarged splits 75/60)"
echo "============================================================"
python3 -m ICR_sfcr.pipeline \
  --task              geometric_shapes \
  --dataset           datasets/bbh/geometric_shapes_train.jsonl \
  --anchor-cheatsheet CS_ICL_Initial_Prompt/bbh_geometric_shapes/gen_gpt-4.1-2025-04-14_0_1000.txt \
  --output-dir        runs/sfcr_geometric_shapes_1000 \
  --model-source      openai/gpt-4.1-mini \
  --models-proxy      "$PROXY" \
  --held-out-target   gemini \
  --oracle-mode       label_only \
  --routing-mode      routed \
  --seed              1000 \
  --rule-gen-n        75 \
  --gate-n            60 \
  --concurrency       50

# ── 4. causal_judgement ──────────────────────────────────────────────────────
echo "============================================================"
echo " [4/4] causal_judgement  (CS-ICL anchor, default splits 60/40)"
echo "============================================================"
python3 -m ICR_sfcr.pipeline \
  --task              causal_judgement \
  --dataset           datasets/bbh/causal_judgement_train.jsonl \
  --anchor-cheatsheet CS_ICL_Initial_Prompt/bbh_causal_judgement/gen_gpt-4.1-2025-04-14_0_1000.txt \
  --output-dir        runs/sfcr_cj_csicl_1000 \
  --model-source      openai/gpt-4.1-mini \
  --models-proxy      "$PROXY" \
  --held-out-target   gemini \
  --oracle-mode       label_only \
  --routing-mode      routed \
  --seed              1000 \
  --concurrency       50

echo ""
echo "============================================================"
echo " All 4 SFCR runs complete."
echo "============================================================"
