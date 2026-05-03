#!/usr/bin/env bash
# run_reasoning_scorer_eval.sh — Evaluate cheatsheets produced by the
# reasoning-scorer pipeline (gpt-oss-120b scorer, gpt-4.1-mini generator).
#
# Tasks: disambiguation_qa, geometric_shapes, formal_fallacies
# Output: runs/reasoning_scorer_rf.json
#
# Usage:
#   bash scripts/eval/run_reasoning_scorer_eval.sh
#   bash scripts/eval/run_reasoning_scorer_eval.sh --dry-run

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
if [ -f "$ROOT/.env" ]; then set -a; source "$ROOT/.env"; set +a; fi

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=true; fi

CMD=(
    python3 scripts/eval/eval_cs_ablation.py
    --tasks disambiguation_qa geometric_shapes formal_fallacies
    --reasoning-first
    --no-csicl
    --concurrency 25
    --run-dir-overrides
        "disambiguation_qa:runs/bbh_reasoning_scorer/disambiguation_qa"
        "geometric_shapes:runs/bbh_reasoning_scorer/geometric_shapes"
        "formal_fallacies:runs/bbh_reasoning_scorer/formal_fallacies"
    --out runs/reasoning_scorer_rf.json
)

if $DRY_RUN; then
    echo "[dry-run] ${CMD[*]}"
    exit 0
fi

echo "=== Reasoning scorer eval ==="
echo "  tasks      : disambiguation_qa  geometric_shapes  formal_fallacies"
echo "  scorer     : gpt-oss-120b (pipeline)  →  all 5 eval models"
echo "  output     : runs/reasoning_scorer_rf.json"
echo ""

"${CMD[@]}"
