#!/usr/bin/env bash
# v5_cj_seeds.sh — Run v5 (Phase 1 + Phase 2 oracle) on CJ × seeds 2 & 3.
#
# v5 = both phase oracles on (default; no --no-phase1-oracle / --no-phase2-oracle).
# Seed 1 already exists: runs/bbh_v5/causal_judgement/ → runs/v5_full_oracle_rf.json
#
# After both seeds finish:
#   1. Evals each seed (pk_only + full, RF scoring)
#   2. Aggregates seeds 1–3 → runs/variance/v5_3seed_mean.json
#
# Usage:
#   bash scripts/pipeline/v5_cj_seeds.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
if [ -f "$ROOT/.env" ]; then set -a; source "$ROOT/.env"; set +a; fi

MODEL="openai/gpt-4.1-mini"
CONCUR=25
TASK="causal_judgement"
DATASET="datasets/bbh/causal_judgement_train.jsonl"
OUT_BASE="runs/variance"
LOG_DIR="$OUT_BASE/logs"
EVAL_DIR="$OUT_BASE/eval_results"

mkdir -p "$LOG_DIR" "$EVAL_DIR"

echo "=== v5 CJ seeds 2 & 3 ==="

# ── Phase A: run pipeline (seeds 2 & 3 in parallel) ─────────────────────���────
echo "[A] Launching v5 CJ seed 2 ..."
python3 -m ICR_hybrid.pipeline \
    --task "$TASK" \
    --dataset "$DATASET" \
    --no-oracle \
    --model-score       "$MODEL" \
    --model-rule-patch  "$MODEL" \
    --model-casestudy   "$MODEL" \
    --rule-concurrency  "$CONCUR" \
    --cs-concurrency    "$CONCUR" \
    --max-rule-iters 4 \
    --max-cs-iters   5 \
    --cot-first \
    --output-dir "$OUT_BASE/v5_seed2/causal_judgement" \
    >> "$LOG_DIR/v5_seed2_causal_judgement.log" 2>&1 &
PID2=$!

echo "[A] Launching v5 CJ seed 3 ..."
python3 -m ICR_hybrid.pipeline \
    --task "$TASK" \
    --dataset "$DATASET" \
    --no-oracle \
    --model-score       "$MODEL" \
    --model-rule-patch  "$MODEL" \
    --model-casestudy   "$MODEL" \
    --rule-concurrency  "$CONCUR" \
    --cs-concurrency    "$CONCUR" \
    --max-rule-iters 4 \
    --max-cs-iters   5 \
    --cot-first \
    --output-dir "$OUT_BASE/v5_seed3/causal_judgement" \
    >> "$LOG_DIR/v5_seed3_causal_judgement.log" 2>&1 &
PID3=$!

echo "  PIDs: seed2=$PID2  seed3=$PID3"
echo "  Waiting for both pipeline runs to finish ..."
wait $PID2 $PID3
echo "[A] Both seeds done."

# ── Phase B: eval each seed ───────────────────────────────────────────────────
echo ""
echo "[B] Evaluating v5 seed 2 ..."
python3 scripts/eval/eval_cs_ablation.py \
    --tasks causal_judgement \
    --reasoning-first \
    --no-csicl \
    --concurrency "$CONCUR" \
    --run-dir-overrides "causal_judgement:$OUT_BASE/v5_seed2/causal_judgement" \
    --out "$EVAL_DIR/v5_seed2_rf.json" \
    >> "$LOG_DIR/v5_seed2_eval.log" 2>&1 &
EPID2=$!

echo "[B] Evaluating v5 seed 3 ..."
python3 scripts/eval/eval_cs_ablation.py \
    --tasks causal_judgement \
    --reasoning-first \
    --no-csicl \
    --concurrency "$CONCUR" \
    --run-dir-overrides "causal_judgement:$OUT_BASE/v5_seed3/causal_judgement" \
    --out "$EVAL_DIR/v5_seed3_rf.json" \
    >> "$LOG_DIR/v5_seed3_eval.log" 2>&1 &
EPID3=$!

wait $EPID2 $EPID3
echo "[B] Evals done."

# ── Phase C: aggregate ────────────────────────────────────────────────────────
echo ""
echo "[C] Aggregating v5 3-seed mean ..."
python3 scripts/eval/aggregate_variance.py \
    --condition v5 \
    --seed1 runs/v5_full_oracle_rf.json \
    --seed2 "$EVAL_DIR/v5_seed2_rf.json" \
    --seed3 "$EVAL_DIR/v5_seed3_rf.json" \
    --out   "$OUT_BASE/v5_3seed_mean.json"

echo ""
echo "=== Done. Results: $OUT_BASE/v5_3seed_mean.json ==="
