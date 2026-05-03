#!/usr/bin/env bash
# gpqa_overnight.sh — First ICRefine run on GPQA Diamond (seed 1).
#
# Runs the standard v3 pipeline (Phase 0 bootstrap → Phase 1 PK patching
# → Phase 2 case study generation) on the 100-item GPQA Diamond train set,
# then evaluates all 5 models (RF scoring) on the 98-item test set against:
#   - CS-ICL baseline
#   - ICRefine PK-only
#   - ICRefine full pipeline
#
# Output:
#   runs/bbh_gpqa/gpqa_diamond/     ← pipeline artefacts
#   runs/gpqa_diamond_rf.json       ← eval results
#
# Usage:
#   bash scripts/pipeline/gpqa_overnight.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
if [ -f "$ROOT/.env" ]; then set -a; source "$ROOT/.env"; set +a; fi

MODEL="openai/gpt-4.1-mini"
CONCUR=25
TASK="gpqa_diamond"
DATASET="datasets/gpqa_diamond/gpqa_diamond_train.jsonl"
OUT_DIR="runs/bbh_gpqa/gpqa_diamond"
LOG_DIR="runs/logs/pipelines/gpqa"
EVAL_OUT="runs/gpqa_diamond_rf.json"

mkdir -p "$LOG_DIR"

echo "=== GPQA Diamond overnight run (seed 1) ==="
echo "  model  : $MODEL"
echo "  out    : $OUT_DIR"
echo "  eval   : $EVAL_OUT"
echo ""

# ── Phase A: pipeline ─────────────────────────────────────────────────────────
echo "[A] Running v3 pipeline ..."
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
    --output-dir "$OUT_DIR" \
    2>&1 | tee "$LOG_DIR/gpqa_pipeline.log"

echo "[A] Pipeline done."

# ── Phase B: eval (all 5 models, RF, with CS-ICL) ────────────────────────────
echo ""
echo "[B] Evaluating all models (RF) ..."
python3 scripts/eval/eval_cs_ablation.py \
    --tasks "$TASK" \
    --reasoning-first \
    --concurrency "$CONCUR" \
    --run-dir-overrides "gpqa_diamond:$OUT_DIR" \
    --out "$EVAL_OUT" \
    2>&1 | tee "$LOG_DIR/gpqa_eval.log"

echo ""
echo "[B] Eval done → $EVAL_OUT"

# ── Phase C: print results ────────────────────────────────────────────────────
echo ""
echo "[C] Results:"
python3 - <<'PYEOF'
import json
from pathlib import Path

p = Path("runs/gpqa_diamond_rf.json")
if not p.exists():
    print("  [warn] eval output not found")
    exit()

d = json.loads(p.read_text())
task = d.get("gpqa_diamond", {})
print(f"  {'Model':<30} {'CS-ICL':>8} {'PK-only':>8} {'Full':>8} {'Δ_cs':>8}")
for mid, r in task.items():
    label   = r.get("label", mid)
    cs_icl  = r.get("cs_icl",  None)
    pk_only = r.get("pk_only", None)
    full    = r.get("full",    None)
    delta   = (full - pk_only) if full is not None and pk_only is not None else None
    fmt = lambda v: f"{v:.1%}" if v is not None else "  ---  "
    dfmt = lambda v: f"{v:+.1%}" if v is not None else "  ---"
    print(f"  {label:<30} {fmt(cs_icl):>8} {fmt(pk_only):>8} {fmt(full):>8} {dfmt(delta):>8}")
PYEOF

echo ""
echo "=== Done. ==="
echo "Next: if results are promising, run seeds 2 & 3 for 3-seed means."
