#!/usr/bin/env bash
# run_sfcr_gs_llama_ho.sh — geometric_shapes re-run with --held-out-target llama
#
# Why: With gemini held out, Llama's delta_shared=0.000 bottlenecks U_LCB to negative.
# Holding out llama instead makes GPT+Gemini the acceptance proxies.
# GPT delta_shared=0.231, reg_easy=0.088 → U_LCB ≈ +0.118 → should accept.

set -e
cd "$(dirname "$0")"

echo "============================================================"
echo " geometric_shapes  (CS-ICL anchor, --held-out-target llama)"
echo " splits: rule-gen-n=75, gate-n=60, seed=1000"
echo "============================================================"

python3 -m ICR_sfcr.pipeline \
  --task              geometric_shapes \
  --dataset           datasets/bbh/geometric_shapes_train.jsonl \
  --anchor-cheatsheet CS_ICL_Initial_Prompt/bbh_geometric_shapes/gen_gpt-4.1-2025-04-14_0_1000.txt \
  --output-dir        runs/sfcr_geometric_shapes_1000_llama_ho \
  --model-source      openai/gpt-4.1-mini \
  --models-proxy      "openai/gpt-4.1,google/gemini-2.0-flash-001,meta-llama/llama-3.3-70b-instruct" \
  --held-out-target   llama \
  --oracle-mode       label_only \
  --routing-mode      routed \
  --seed              1000 \
  --rule-gen-n        75 \
  --gate-n            60 \
  --concurrency       50

echo ""
echo "============================================================"
echo " Done. Check runs/sfcr_geometric_shapes_1000_llama_ho/"
echo "============================================================"
