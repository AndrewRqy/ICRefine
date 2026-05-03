#!/usr/bin/env bash
# run_ea_and_ablation_evals.sh — Launch all pending RF test evals in parallel.
#
# Jobs launched (7 total, split across primary + backup OpenAI keys):
#
#  Key 1 (OPENAI_API_KEY):
#   1. EA Phase 1 (4 tasks, 5 models, RF)            → runs/bbh_ea_phase1_rf.json
#   2. ablation_size2 p1_unlimited  (4 tasks, mini)   → runs/ablation_size2_p1_unlimited_rf.json
#   3. ablation_size2 p2_1cs        (4 tasks, mini)   → runs/ablation_size2_p2_1cs_rf.json
#   4. ablation_size2 p1_12000chars (4 tasks, mini)   → runs/ablation_size2_p1_12000chars_rf.json
#
#  Key 2 (OPENAI_API_KEY_BACKUP):
#   5. ablation_size2 p1_3000chars  (4 tasks, mini)   → runs/ablation_size2_p1_3000chars_rf.json
#   6. ablation_size2 p1_6000chars  (4 tasks, mini)   → runs/ablation_size2_p1_6000chars_rf.json
#   7. ablation_size2 p2_3cs        (4 tasks, mini)   → runs/ablation_size2_p2_3cs_rf.json
#
# EA uses all 5 models (OpenAI direct + OpenRouter for claude/gemini/llama).
# Ablation_size2 uses gpt-4.1-mini only (size/count study; saves cost + time).
#
# Usage:
#   bash scripts/eval/run_ea_and_ablation_evals.sh

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# Load env file to get OPENAI_API_KEY_BACKUP
if [ -f "$ROOT/.env" ]; then
    set -a; source "$ROOT/.env"; set +a
fi

KEY1="${OPENAI_API_KEY}"
KEY2="${OPENAI_API_KEY_BACKUP}"

if [ -z "$KEY1" ]; then echo "ERROR: OPENAI_API_KEY not set"; exit 1; fi
if [ -z "$KEY2" ]; then
    echo "WARNING: OPENAI_API_KEY_BACKUP not set — all jobs will use KEY1"
    KEY2="$KEY1"
fi

LOG_DIR="$ROOT/runs/eval_logs"
mkdir -p "$LOG_DIR"

ABLATION_TASKS="geometric_shapes formal_fallacies snarks disambiguation_qa"
EA_TASKS="causal_judgement geometric_shapes snarks disambiguation_qa"
CONCUR=20   # 20 concurrent per job; KEY1 runs 4 jobs (80 max), KEY2 runs 3 (60 max)

echo "=== Launching all RF test evals in parallel (dual-key) ==="
echo "  root    : $ROOT"
echo "  key1    : ${KEY1:0:20}..."
echo "  key2    : ${KEY2:0:20}..."
echo "  log dir : $LOG_DIR"
echo ""

# ── KEY 1 jobs ────────────────────────────────────────────────────────────────

# Job 1: EA Phase 1 — all 5 models (OpenAI direct for gpt models; OpenRouter for rest)
echo "  [1/7] EA Phase 1 (all models, RF) — key1 ..."
OPENAI_API_KEY="$KEY1" python3 scripts/eval/eval_cs_ablation.py \
    --tasks $EA_TASKS \
    --reasoning-first \
    --concurrency $CONCUR \
    --run-dir-overrides \
        causal_judgement:runs/bbh_ea_phase1/causal_judgement \
        geometric_shapes:runs/bbh_ea_phase1/geometric_shapes \
        snarks:runs/bbh_ea_phase1/snarks \
        disambiguation_qa:runs/bbh_ea_phase1/disambiguation_qa \
    --out runs/bbh_ea_phase1_rf.json \
    >> "$LOG_DIR/ea_phase1_rf.log" 2>&1 &

# Job 2: p1_unlimited
echo "  [2/7] ablation p1_unlimited (mini, RF) — key1 ..."
OPENAI_API_KEY="$KEY1" python3 scripts/eval/eval_cs_ablation.py \
    --tasks $ABLATION_TASKS \
    --reasoning-first \
    --concurrency $CONCUR \
    --models openai/gpt-4.1-mini \
    --run-dir-overrides \
        geometric_shapes:runs/ablation_size2/p1_unlimited/geometric_shapes \
        formal_fallacies:runs/ablation_size2/p1_unlimited/formal_fallacies \
        snarks:runs/ablation_size2/p1_unlimited/snarks \
        disambiguation_qa:runs/ablation_size2/p1_unlimited/disambiguation_qa \
    --out runs/ablation_size2_p1_unlimited_rf.json \
    >> "$LOG_DIR/ablation_p1_unlimited_rf.log" 2>&1 &

# Job 3: p2_1cs
echo "  [3/7] ablation p2_1cs (mini, RF) — key1 ..."
OPENAI_API_KEY="$KEY1" python3 scripts/eval/eval_cs_ablation.py \
    --tasks $ABLATION_TASKS \
    --reasoning-first \
    --concurrency $CONCUR \
    --models openai/gpt-4.1-mini \
    --run-dir-overrides \
        geometric_shapes:runs/ablation_size2/p2_1cs/geometric_shapes \
        formal_fallacies:runs/ablation_size2/p2_1cs/formal_fallacies \
        snarks:runs/ablation_size2/p2_1cs/snarks \
        disambiguation_qa:runs/ablation_size2/p2_1cs/disambiguation_qa \
    --out runs/ablation_size2_p2_1cs_rf.json \
    >> "$LOG_DIR/ablation_p2_1cs_rf.log" 2>&1 &

# Job 4: p1_12000chars
echo "  [4/7] ablation p1_12000chars (mini, RF) — key1 ..."
OPENAI_API_KEY="$KEY1" python3 scripts/eval/eval_cs_ablation.py \
    --tasks $ABLATION_TASKS \
    --reasoning-first \
    --concurrency $CONCUR \
    --models openai/gpt-4.1-mini \
    --run-dir-overrides \
        geometric_shapes:runs/ablation_size2/p1_12000chars/geometric_shapes \
        formal_fallacies:runs/ablation_size2/p1_12000chars/formal_fallacies \
        snarks:runs/ablation_size2/p1_12000chars/snarks \
        disambiguation_qa:runs/ablation_size2/p1_12000chars/disambiguation_qa \
    --out runs/ablation_size2_p1_12000chars_rf.json \
    >> "$LOG_DIR/ablation_p1_12000chars_rf.log" 2>&1 &

# ── KEY 2 jobs ────────────────────────────────────────────────────────────────

# Job 5: p1_3000chars
echo "  [5/7] ablation p1_3000chars (mini, RF) — key2 ..."
OPENAI_API_KEY="$KEY2" python3 scripts/eval/eval_cs_ablation.py \
    --tasks $ABLATION_TASKS \
    --reasoning-first \
    --concurrency $CONCUR \
    --models openai/gpt-4.1-mini \
    --run-dir-overrides \
        geometric_shapes:runs/ablation_size2/p1_3000chars/geometric_shapes \
        formal_fallacies:runs/ablation_size2/p1_3000chars/formal_fallacies \
        snarks:runs/ablation_size2/p1_3000chars/snarks \
        disambiguation_qa:runs/ablation_size2/p1_3000chars/disambiguation_qa \
    --out runs/ablation_size2_p1_3000chars_rf.json \
    >> "$LOG_DIR/ablation_p1_3000chars_rf.log" 2>&1 &

# Job 6: p1_6000chars
echo "  [6/7] ablation p1_6000chars (mini, RF) — key2 ..."
OPENAI_API_KEY="$KEY2" python3 scripts/eval/eval_cs_ablation.py \
    --tasks $ABLATION_TASKS \
    --reasoning-first \
    --concurrency $CONCUR \
    --models openai/gpt-4.1-mini \
    --run-dir-overrides \
        geometric_shapes:runs/ablation_size2/p1_6000chars/geometric_shapes \
        formal_fallacies:runs/ablation_size2/p1_6000chars/formal_fallacies \
        snarks:runs/ablation_size2/p1_6000chars/snarks \
        disambiguation_qa:runs/ablation_size2/p1_6000chars/disambiguation_qa \
    --out runs/ablation_size2_p1_6000chars_rf.json \
    >> "$LOG_DIR/ablation_p1_6000chars_rf.log" 2>&1 &

# Job 7: p2_3cs
echo "  [7/7] ablation p2_3cs (mini, RF) — key2 ..."
OPENAI_API_KEY="$KEY2" python3 scripts/eval/eval_cs_ablation.py \
    --tasks $ABLATION_TASKS \
    --reasoning-first \
    --concurrency $CONCUR \
    --models openai/gpt-4.1-mini \
    --run-dir-overrides \
        geometric_shapes:runs/ablation_size2/p2_3cs/geometric_shapes \
        formal_fallacies:runs/ablation_size2/p2_3cs/formal_fallacies \
        snarks:runs/ablation_size2/p2_3cs/snarks \
        disambiguation_qa:runs/ablation_size2/p2_3cs/disambiguation_qa \
    --out runs/ablation_size2_p2_3cs_rf.json \
    >> "$LOG_DIR/ablation_p2_3cs_rf.log" 2>&1 &

# ── Wait ──────────────────────────────────────────────────────────────────────
echo ""
echo "All 7 jobs launched — waiting ..."
echo "  tail $LOG_DIR/*.log   to follow progress"
echo ""
wait

echo "=== All evals complete ==="
echo ""
echo "Output files:"
for f in \
    runs/bbh_ea_phase1_rf.json \
    runs/ablation_size2_p1_unlimited_rf.json \
    runs/ablation_size2_p1_3000chars_rf.json \
    runs/ablation_size2_p1_6000chars_rf.json \
    runs/ablation_size2_p1_12000chars_rf.json \
    runs/ablation_size2_p2_1cs_rf.json \
    runs/ablation_size2_p2_3cs_rf.json; do
    if [ -f "$ROOT/$f" ]; then
        echo "  OK   $f"
    else
        echo "  MISS $f"
    fi
done
