#!/bin/bash
# run_sfcr_multirun.sh — SFCR pipeline: 5 tasks × 3 seeds = 15 runs
#
# Anchor: CS_ICL_Initial_Prompt/ (per-seed, per-task)
# Source: gpt-4.1-mini
# Proxies: gpt-4.1, gemini-2.0-flash-001, llama-3.3-70b (gemini held out)
# Oracle mode: label_only   Routing mode: routed
#
# Run from ICRefine/:  bash run_sfcr_multirun.sh

set -e
cd "$(dirname "$0")"

CS="CS_ICL_Initial_Prompt"
BBH_DATA="datasets/bbh"
AGI_DATA="datasets/agieval"

COMMON="--model-source openai/gpt-4.1-mini
        --models-proxy openai/gpt-4.1,google/gemini-2.0-flash-001,meta-llama/llama-3.3-70b-instruct
        --held-out-target gemini
        --oracle-mode label_only
        --routing-mode routed
        --concurrency 50"

# ---------------------------------------------------------------------------
# Batch 1: BBH seed 1000
# ---------------------------------------------------------------------------
echo "=== Batch 1: BBH seed 1000 ==="
python -m ICR_sfcr.pipeline \
    --task causal_judgement \
    --dataset $BBH_DATA/causal_judgement_train_labeled.jsonl \
    --anchor-cheatsheet $CS/bbh_causal_judgement/gen_gpt-4.1-2025-04-14_0_1000.txt \
    --output-dir runs/sfcr_cj_1000 --seed 1000 $COMMON &

python -m ICR_sfcr.pipeline \
    --task disambiguation_qa \
    --dataset $BBH_DATA/disambiguation_qa_train_labeled.jsonl \
    --anchor-cheatsheet $CS/bbh_disambiguation_qa/gen_gpt-4.1-2025-04-14_0_1000.txt \
    --output-dir runs/sfcr_dq_1000 --seed 1000 $COMMON &

python -m ICR_sfcr.pipeline \
    --task geometric_shapes \
    --dataset $BBH_DATA/geometric_shapes_train_labeled.jsonl \
    --anchor-cheatsheet $CS/bbh_geometric_shapes/gen_gpt-4.1-2025-04-14_0_1000.txt \
    --output-dir runs/sfcr_gs_1000 --seed 1000 $COMMON &

wait; echo "=== Batch 1 done ==="

# ---------------------------------------------------------------------------
# Batch 2: BBH seed 2000
# ---------------------------------------------------------------------------
echo "=== Batch 2: BBH seed 2000 ==="
python -m ICR_sfcr.pipeline \
    --task causal_judgement \
    --dataset $BBH_DATA/causal_judgement_train_labeled.jsonl \
    --anchor-cheatsheet $CS/bbh_causal_judgement/gen_gpt-4.1-2025-04-14_0_2000.txt \
    --output-dir runs/sfcr_cj_2000 --seed 2000 $COMMON &

python -m ICR_sfcr.pipeline \
    --task disambiguation_qa \
    --dataset $BBH_DATA/disambiguation_qa_train_labeled.jsonl \
    --anchor-cheatsheet $CS/bbh_disambiguation_qa/gen_gpt-4.1-2025-04-14_0_2000.txt \
    --output-dir runs/sfcr_dq_2000 --seed 2000 $COMMON &

python -m ICR_sfcr.pipeline \
    --task geometric_shapes \
    --dataset $BBH_DATA/geometric_shapes_train_labeled.jsonl \
    --anchor-cheatsheet $CS/bbh_geometric_shapes/gen_gpt-4.1-2025-04-14_0_2000.txt \
    --output-dir runs/sfcr_gs_2000 --seed 2000 $COMMON &

wait; echo "=== Batch 2 done ==="

# ---------------------------------------------------------------------------
# Batch 3: BBH seed 3000
# ---------------------------------------------------------------------------
echo "=== Batch 3: BBH seed 3000 ==="
python -m ICR_sfcr.pipeline \
    --task causal_judgement \
    --dataset $BBH_DATA/causal_judgement_train_labeled.jsonl \
    --anchor-cheatsheet $CS/bbh_causal_judgement/gen_gpt-4.1-2025-04-14_0_3000.txt \
    --output-dir runs/sfcr_cj_3000 --seed 3000 $COMMON &

python -m ICR_sfcr.pipeline \
    --task disambiguation_qa \
    --dataset $BBH_DATA/disambiguation_qa_train_labeled.jsonl \
    --anchor-cheatsheet $CS/bbh_disambiguation_qa/gen_gpt-4.1-2025-04-14_0_3000.txt \
    --output-dir runs/sfcr_dq_3000 --seed 3000 $COMMON &

python -m ICR_sfcr.pipeline \
    --task geometric_shapes \
    --dataset $BBH_DATA/geometric_shapes_train_labeled.jsonl \
    --anchor-cheatsheet $CS/bbh_geometric_shapes/gen_gpt-4.1-2025-04-14_0_3000.txt \
    --output-dir runs/sfcr_gs_3000 --seed 3000 $COMMON &

wait; echo "=== Batch 3 done ==="

# ---------------------------------------------------------------------------
# Batch 4: AGIEval seed 1000
# ---------------------------------------------------------------------------
echo "=== Batch 4: AGIEval seed 1000 ==="
python -m ICR_sfcr.pipeline \
    --task agieval_lsat_ar \
    --dataset $AGI_DATA/lsat_ar_train_oracle_labeled.jsonl \
    --anchor-cheatsheet $CS/agieval_lsat_ar/gen_gpt-4.1-mini_0_1000.txt \
    --output-dir runs/sfcr_lsat_ar_1000 --seed 1000 $COMMON &

python -m ICR_sfcr.pipeline \
    --task agieval_logiqa_en \
    --dataset $AGI_DATA/logiqa_en_train_oracle_labeled.jsonl \
    --anchor-cheatsheet $CS/agieval_logiqa_en/gen_gpt-4.1-mini_0_1000.txt \
    --output-dir runs/sfcr_logiqa_en_1000 --seed 1000 $COMMON &

wait; echo "=== Batch 4 done ==="

# ---------------------------------------------------------------------------
# Batch 5: AGIEval seed 2000
# ---------------------------------------------------------------------------
echo "=== Batch 5: AGIEval seed 2000 ==="
python -m ICR_sfcr.pipeline \
    --task agieval_lsat_ar \
    --dataset $AGI_DATA/lsat_ar_train_oracle_labeled.jsonl \
    --anchor-cheatsheet $CS/agieval_lsat_ar/gen_gpt-4.1-mini_0_2000.txt \
    --output-dir runs/sfcr_lsat_ar_2000 --seed 2000 $COMMON &

python -m ICR_sfcr.pipeline \
    --task agieval_logiqa_en \
    --dataset $AGI_DATA/logiqa_en_train_oracle_labeled.jsonl \
    --anchor-cheatsheet $CS/agieval_logiqa_en/gen_gpt-4.1-mini_0_2000.txt \
    --output-dir runs/sfcr_logiqa_en_2000 --seed 2000 $COMMON &

wait; echo "=== Batch 5 done ==="

# ---------------------------------------------------------------------------
# Batch 6: AGIEval seed 3000
# ---------------------------------------------------------------------------
echo "=== Batch 6: AGIEval seed 3000 ==="
python -m ICR_sfcr.pipeline \
    --task agieval_lsat_ar \
    --dataset $AGI_DATA/lsat_ar_train_oracle_labeled.jsonl \
    --anchor-cheatsheet $CS/agieval_lsat_ar/gen_gpt-4.1-mini_0_3000.txt \
    --output-dir runs/sfcr_lsat_ar_3000 --seed 3000 $COMMON &

python -m ICR_sfcr.pipeline \
    --task agieval_logiqa_en \
    --dataset $AGI_DATA/logiqa_en_train_oracle_labeled.jsonl \
    --anchor-cheatsheet $CS/agieval_logiqa_en/gen_gpt-4.1-mini_0_3000.txt \
    --output-dir runs/sfcr_logiqa_en_3000 --seed 3000 $COMMON &

wait; echo "=== Batch 6 done ==="

echo ""
echo "All 15 SFCR runs complete."
