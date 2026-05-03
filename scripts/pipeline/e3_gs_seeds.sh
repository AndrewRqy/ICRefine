#!/usr/bin/env bash
# e3_gs_seeds.sh — Run E3 (no-oracle) on GS × seeds 2 & 3.
#
# E3 = --no-phase1-oracle --no-phase2-oracle (both phase oracles off).
# Seed 1 already exists: runs/bbh_oracle_ablation/no_oracle/geometric_shapes/
#   → included in runs/e3_no_oracle_rf.json (GS entry).
#
# After both seeds finish:
#   1. Evals each seed (pk_only + full, RF scoring)
#   2. Aggregates seeds 1–3 for GS → runs/variance/e3_gs_3seed_mean.json
#
# Usage:
#   bash scripts/pipeline/e3_gs_seeds.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
if [ -f "$ROOT/.env" ]; then set -a; source "$ROOT/.env"; set +a; fi

MODEL="openai/gpt-4.1-mini"
CONCUR=25
TASK="geometric_shapes"
DATASET="datasets/bbh/geometric_shapes_train.jsonl"
OUT_BASE="runs/variance"
LOG_DIR="$OUT_BASE/logs"
EVAL_DIR="$OUT_BASE/eval_results"

mkdir -p "$LOG_DIR" "$EVAL_DIR"

echo "=== E3 GS seeds 2 & 3 ==="

# ── Phase A: run pipeline (seeds 2 & 3 in parallel) ───────────────────���──────
echo "[A] Launching E3 GS seed 2 ..."
python3 -m ICR_hybrid.pipeline \
    --task "$TASK" \
    --dataset "$DATASET" \
    --no-oracle \
    --no-phase1-oracle \
    --no-phase2-oracle \
    --model-score       "$MODEL" \
    --model-rule-patch  "$MODEL" \
    --model-casestudy   "$MODEL" \
    --rule-concurrency  "$CONCUR" \
    --cs-concurrency    "$CONCUR" \
    --max-rule-iters 4 \
    --max-cs-iters   5 \
    --cot-first \
    --output-dir "$OUT_BASE/e3_seed2/geometric_shapes" \
    >> "$LOG_DIR/e3_seed2_geometric_shapes.log" 2>&1 &
PID2=$!

echo "[A] Launching E3 GS seed 3 ..."
python3 -m ICR_hybrid.pipeline \
    --task "$TASK" \
    --dataset "$DATASET" \
    --no-oracle \
    --no-phase1-oracle \
    --no-phase2-oracle \
    --model-score       "$MODEL" \
    --model-rule-patch  "$MODEL" \
    --model-casestudy   "$MODEL" \
    --rule-concurrency  "$CONCUR" \
    --cs-concurrency    "$CONCUR" \
    --max-rule-iters 4 \
    --max-cs-iters   5 \
    --cot-first \
    --output-dir "$OUT_BASE/e3_seed3/geometric_shapes" \
    >> "$LOG_DIR/e3_seed3_geometric_shapes.log" 2>&1 &
PID3=$!

echo "  PIDs: seed2=$PID2  seed3=$PID3"
echo "  Waiting for both pipeline runs to finish ..."
wait $PID2 $PID3
echo "[A] Both seeds done."

# ── Phase B: eval each seed ──────────────────────��─────────────────────��──────
echo ""
echo "[B] Evaluating E3 GS seed 2 ..."
python3 scripts/eval/eval_cs_ablation.py \
    --tasks geometric_shapes \
    --reasoning-first \
    --no-csicl \
    --concurrency "$CONCUR" \
    --run-dir-overrides "geometric_shapes:$OUT_BASE/e3_seed2/geometric_shapes" \
    --out "$EVAL_DIR/e3_seed2_gs_rf.json" \
    >> "$LOG_DIR/e3_seed2_gs_eval.log" 2>&1 &
EPID2=$!

echo "[B] Evaluating E3 GS seed 3 ..."
python3 scripts/eval/eval_cs_ablation.py \
    --tasks geometric_shapes \
    --reasoning-first \
    --no-csicl \
    --concurrency "$CONCUR" \
    --run-dir-overrides "geometric_shapes:$OUT_BASE/e3_seed3/geometric_shapes" \
    --out "$EVAL_DIR/e3_seed3_gs_rf.json" \
    >> "$LOG_DIR/e3_seed3_gs_eval.log" 2>&1 &
EPID3=$!

wait $EPID2 $EPID3
echo "[B] Evals done."

# ── Phase C: aggregate ────────────────────────────���───────────────────────────
# Seed 1 for E3 GS lives in runs/e3_no_oracle_rf.json (contains both CJ and GS).
# aggregate_variance.py merges by task, so passing it as seed1 is fine.
echo ""
echo "[C] Aggregating E3 GS 3-seed mean ..."
python3 scripts/eval/aggregate_variance.py \
    --condition e3_gs \
    --seed1 runs/e3_no_oracle_rf.json \
    --seed2 "$EVAL_DIR/e3_seed2_gs_rf.json" \
    --seed3 "$EVAL_DIR/e3_seed3_gs_rf.json" \
    --out   "$OUT_BASE/e3_gs_3seed_mean.json"

echo ""
echo "=== Done. Results: $OUT_BASE/e3_gs_3seed_mean.json ==="
