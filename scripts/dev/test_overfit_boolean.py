#!/usr/bin/env python3
"""
test_overfit_boolean.py — Test whether pruning reduces overfitting on boolean_expressions.

Protocol:
  1. Load cheatsheet_final from bbh_overnight_v2/boolean_expressions
  2. Eval on train set (150 items) and test set (100 items)
  3. Apply _prune_cs_bank — let the LLM remove redundant CS
  4. Eval again on both sets
  5. Print before/after table: train acc, test acc, and train-test gap
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── Load .env ──────────────────────────────────────────────────────────────
_env = Path(__file__).parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.cheatsheet import Cheatsheet
from ICR_select.training.gates import _prune_cs_bank
from tasks.bbh_boolean import BBH_BOOLEAN_TASK
from utils.scorer import score_batch

# ── Config ─────────────────────────────────────────────────────────────────
CHEATSHEET_PATH = (
    Path(__file__).parent
    / "runs/bbh_overnight_v2/boolean_expressions/cheatsheet_final"
)
TRAIN_PATH = Path(__file__).parent / "datasets/bbh/boolean_expressions_train.jsonl"
TEST_PATH  = Path(__file__).parent / "datasets/bbh/boolean_expressions_test.jsonl"

MODEL     = "openai/gpt-4.1-mini"
API_KEY   = os.environ.get("OPENROUTER_API_KEY", "")
CONCURRENCY = 20


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def eval_cheatsheet(cs: Cheatsheet, items: list[dict], label: str) -> float:
    text = cs.render()
    correct, wrong = score_batch(
        items, text, MODEL, API_KEY,
        concurrency=CONCURRENCY,
        reasoning_effort="low",
        cot_first=True,
        progress_label=label,
        task_spec=BBH_BOOLEAN_TASK,
    )
    acc = len(correct) / len(items) if items else 0.0
    print(f"  {label}: {acc:.1%}  ({len(correct)}/{len(items)} correct)", flush=True)
    return acc


def main() -> None:
    print(f"\nLoading cheatsheet from {CHEATSHEET_PATH}.*", flush=True)
    cs = Cheatsheet.load(str(CHEATSHEET_PATH))
    n_orig = len(cs.case_studies)
    print(f"  {n_orig} case studies loaded.", flush=True)
    print("\nCase study titles (original):")
    for i, c in enumerate(cs.case_studies, 1):
        print(f"  [{i}] {c.title}")

    train_items = load_jsonl(TRAIN_PATH)
    test_items  = load_jsonl(TEST_PATH)
    print(
        f"\nDatasets: train={len(train_items)} items, test={len(test_items)} items\n",
        flush=True,
    )

    # ── Baseline eval ──────────────────────────────────────────────────────
    print("=" * 55)
    print("BASELINE (original cheatsheet)")
    print("=" * 55)
    train_acc_before = eval_cheatsheet(cs, train_items, "train")
    test_acc_before  = eval_cheatsheet(cs, test_items,  "test ")
    gap_before       = train_acc_before - test_acc_before

    # ── Prune ──────────────────────────────────────────────────────────────
    print(f"\nRunning _prune_cs_bank ...", flush=True)
    n_pruned = _prune_cs_bank(cs, MODEL, API_KEY, log_fn=print)
    n_after  = len(cs.case_studies)
    print(f"  Removed {n_pruned}  ({n_orig} → {n_after} case studies)")
    print("\nCase study titles (after prune):")
    for i, c in enumerate(cs.case_studies, 1):
        print(f"  [{i}] {c.title}")

    # ── Post-prune eval ────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print("AFTER PRUNE")
    print("=" * 55)
    train_acc_after = eval_cheatsheet(cs, train_items, "train")
    test_acc_after  = eval_cheatsheet(cs, test_items,  "test ")
    gap_after       = train_acc_after - test_acc_after

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"{'SUMMARY':^55}")
    print(f"{'=' * 55}")
    print(f"  CS count:        {n_orig} → {n_after}  (-{n_pruned})")
    print(f"  Train acc:       {train_acc_before:.1%} → {train_acc_after:.1%}  "
          f"({train_acc_after - train_acc_before:+.1%})")
    print(f"  Test  acc:       {test_acc_before:.1%} → {test_acc_after:.1%}  "
          f"({test_acc_after - test_acc_before:+.1%})")
    print(f"  Train-test gap:  {gap_before:.1%} → {gap_after:.1%}  "
          f"({'narrowed' if gap_after < gap_before else 'widened'} by "
          f"{abs(gap_after - gap_before):.1%})")


if __name__ == "__main__":
    main()
