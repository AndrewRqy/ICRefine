#!/usr/bin/env bash
# rerun_p1_6000_gs.sh — Re-run the anomalous p1_6000chars GS ablation.
#
# The original runs/ablation_size2/p1_6000chars/geometric_shapes run produced
# 18% test accuracy (anomalous — likely a corrupted eval). This script re-runs
# the pipeline into a fresh directory, evals it, then patches the result back
# into runs/ablation_size2_p1_6000chars_rf.json (GS entry only; other tasks
# remain from the original run).
#
# Usage:
#   bash scripts/pipeline/rerun_p1_6000_gs.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
if [ -f "$ROOT/.env" ]; then set -a; source "$ROOT/.env"; set +a; fi

MODEL="openai/gpt-4.1-mini"
CONCUR=25
OUT_DIR="runs/ablation_size2_rerun/p1_6000chars/geometric_shapes"
LOG_DIR="runs/ablation_size2_rerun/logs"
EVAL_OUT="runs/ablation_size2_p1_6000chars_gs_rerun_rf.json"
FINAL_OUT="runs/ablation_size2_p1_6000chars_rf.json"

mkdir -p "$LOG_DIR"

echo "=== Re-run p1_6000chars / geometric_shapes ==="

# ── Phase A: pipeline ─────────────────────────────────────────────────────────
echo "[A] Running pipeline (--max-pk-chars 6000) ..."
python3 -m ICR_hybrid.pipeline \
    --task geometric_shapes \
    --dataset datasets/bbh/geometric_shapes_train.jsonl \
    --no-oracle \
    --model-score       "$MODEL" \
    --model-rule-patch  "$MODEL" \
    --model-casestudy   "$MODEL" \
    --rule-concurrency  "$CONCUR" \
    --cs-concurrency    "$CONCUR" \
    --max-rule-iters 4 \
    --max-cs-iters   5 \
    --cot-first \
    --max-pk-chars 6000 \
    --output-dir "$OUT_DIR" \
    2>&1 | tee "$LOG_DIR/p1_6000_gs_pipeline.log"

echo "[A] Pipeline done."

# ── Phase B: eval ─────────────────────────────────────────────────────────────
echo ""
echo "[B] Evaluating ..."
python3 scripts/eval/eval_cs_ablation.py \
    --tasks geometric_shapes \
    --reasoning-first \
    --no-csicl \
    --concurrency "$CONCUR" \
    --models "$MODEL" \
    --run-dir-overrides "geometric_shapes:$OUT_DIR" \
    --out "$EVAL_OUT" \
    2>&1 | tee "$LOG_DIR/p1_6000_gs_eval.log"

echo "[B] Eval done → $EVAL_OUT"

# ── Phase C: patch GS entry back into the combined results file ───────────────
echo ""
echo "[C] Patching GS result into $FINAL_OUT ..."
python3 - <<'PYEOF'
import json, sys
from pathlib import Path

rerun = Path("runs/ablation_size2_p1_6000chars_gs_rerun_rf.json")
final = Path("runs/ablation_size2_p1_6000chars_rf.json")

if not rerun.exists():
    print(f"[error] rerun file not found: {rerun}", file=sys.stderr)
    sys.exit(1)

new_gs = json.loads(rerun.read_text())["geometric_shapes"]

if final.exists():
    combined = json.loads(final.read_text())
else:
    combined = {}

combined["geometric_shapes"] = new_gs
final.write_text(json.dumps(combined, indent=2))

print(f"Patched geometric_shapes into {final}")
print("New GS result:")
for mid, r in new_gs.items():
    print(f"  {r.get('label', mid)}: full={r.get('full','?'):.1%}  pk={r.get('pk_only','?'):.1%}")
PYEOF

echo ""
echo "=== Done. Updated: $FINAL_OUT ==="
echo "Check the GS result above and update Tab tab:pk_size in findings_draft.tex accordingly."
