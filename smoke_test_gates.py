#!/usr/bin/env python3
"""
smoke_test_gates.py — Verifies the fix_rate_threshold and min_pool_for_regression
gate changes without requiring a live LLM.

Patches score_batch (in both loop and gates modules) and generate_candidates
to return deterministic results, then runs run_training_loop and checks:
  1. fix_rate_threshold default is 0.30
  2. min_pool_for_regression default is 10
  3. Regression gate is skipped when pool < min_pool
  4. Regression gate runs when pool >= min_pool
  5. fix_rate=0.30 accepts candidates that 0.50 would block
  6. best_fix_rate recorded in retry-path discard log
  7. CLI --min-pool-for-regression flag exists in --help
"""

from __future__ import annotations

import inspect
import sys
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from ICR_select.training.loop import run_training_loop
from utils.cheatsheet import Cheatsheet
from utils.case_study import CaseStudy

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS}  {name}")
    else:
        msg = f"{name}" + (f": {detail}" if detail else "")
        print(f"  {FAIL}  {msg}")
        _failures.append(msg)


# Correct patch paths:
#   generate_candidates is imported from ..generators.case_study into loop.py
#   score_batch is imported from utils.scorer into BOTH loop.py and gates.py
_PATCH_GEN   = "ICR_select.training.loop.generate_candidates"
_PATCH_SCORE_LOOP  = "ICR_select.training.loop.score_batch"
_PATCH_SCORE_GATES = "ICR_select.training.gates.score_batch"
_PATCH_REGRESS     = "ICR_select.training.loop._regression_check"
_PATCH_SIMGATE     = "ICR_select.training.loop._similarity_gate"


def _make_items(n: int) -> list[dict]:
    return [
        {
            "id": f"smoke_{i:04d}",
            "index": i,
            "difficulty": "normal",
            "equation1": "x = y * z",
            "equation2": "x * y = z * w",
            "answer": True,
        }
        for i in range(n)
    ]


def _prescore(items: list[dict], n_correct: int) -> dict:
    return {
        item["id"]: {
            "predicted": "TRUE" if i < n_correct else "FALSE",
            "post_think": "", "thinking": "", "raw_response": "",
            "correct": i < n_correct,
        }
        for i, item in enumerate(items)
    }


def _score_fixes_all(batch, *a, **kw):
    """Candidate fixes every item in the batch (fix_rate = 100%)."""
    correct = [{**item, "predicted": "TRUE"} for item in batch]
    return correct, []


def _score_fixes_none(batch, *a, **kw):
    """Candidate fixes nothing (fix_rate = 0%)."""
    return [], [{**item, "predicted": "FALSE"} for item in batch]


def _score_fixes_40pct(batch, *a, **kw):
    """Candidate fixes 40% of the batch."""
    n = max(1, int(len(batch) * 0.40))
    correct = [{**item, "predicted": "TRUE"} for item in batch[:n]]
    wrong   = [{**item, "predicted": "FALSE"} for item in batch[n:]]
    return correct, wrong


def _gen_one(*a, **kw):
    return [CaseStudy(title="Test Candidate", activate_if=["E1 is absorbing"], action="TRUE")]


# ---------------------------------------------------------------------------
# Test 1: Default fix_rate_threshold is 0.30
# ---------------------------------------------------------------------------
def test_default_fix_rate_threshold():
    sig = inspect.signature(run_training_loop)
    default = sig.parameters["fix_rate_threshold"].default
    check("default fix_rate_threshold == 0.30", default == 0.30, f"got {default}")


# ---------------------------------------------------------------------------
# Test 2: Default min_pool_for_regression is 10
# ---------------------------------------------------------------------------
def test_default_min_pool():
    sig = inspect.signature(run_training_loop)
    default = sig.parameters["min_pool_for_regression"].default
    check("default min_pool_for_regression == 10", default == 10, f"got {default}")


# ---------------------------------------------------------------------------
# Test 3: Regression gate SKIPPED when pool < min_pool_for_regression
# ---------------------------------------------------------------------------
def test_regression_gate_skipped_when_pool_small():
    items = _make_items(13)
    ps = _prescore(items, n_correct=7)  # 7 correct → pool=7 < min_pool=10

    regression_called = []

    def fake_regress(*a, **kw):
        regression_called.append(True)
        return 0.0

    with patch(_PATCH_SCORE_LOOP, side_effect=_score_fixes_all), \
         patch(_PATCH_SCORE_GATES, side_effect=_score_fixes_all), \
         patch(_PATCH_REGRESS, side_effect=fake_regress), \
         patch(_PATCH_GEN, side_effect=_gen_one), \
         patch(_PATCH_SIMGATE, return_value=("ADD", None)):

        run_training_loop(
            cheatsheet=Cheatsheet(decision_tree="", case_studies=[]),
            train_items=items, val_items=None,
            model_score="dummy", model_casestudy="dummy", api_key="dummy",
            bin_threshold=3, batch_size=5, concurrency=1,
            n_candidates=1, candidate_rounds=1, flush_strategy="default",
            prescore_map=ps,
            fix_rate_threshold=0.30, regress_threshold=0.15,
            min_pool_for_regression=10,   # pool=7 < 10
            similarity_gate=False, log=False,
        )

    check(
        "regression gate skipped when pool < min_pool",
        len(regression_called) == 0,
        f"called {len(regression_called)} time(s)"
    )


# ---------------------------------------------------------------------------
# Test 4: Regression gate RUNS when pool >= min_pool_for_regression
# ---------------------------------------------------------------------------
def test_regression_gate_runs_when_pool_large():
    items = _make_items(25)
    ps = _prescore(items, n_correct=15)  # pool=15 >= min_pool=10

    regression_called = []

    def fake_regress(*a, **kw):
        regression_called.append(True)
        return 0.0  # passes gate

    with patch(_PATCH_SCORE_LOOP, side_effect=_score_fixes_all), \
         patch(_PATCH_SCORE_GATES, side_effect=_score_fixes_all), \
         patch(_PATCH_REGRESS, side_effect=fake_regress), \
         patch(_PATCH_GEN, side_effect=_gen_one), \
         patch(_PATCH_SIMGATE, return_value=("ADD", None)):

        run_training_loop(
            cheatsheet=Cheatsheet(decision_tree="", case_studies=[]),
            train_items=items, val_items=None,
            model_score="dummy", model_casestudy="dummy", api_key="dummy",
            bin_threshold=3, batch_size=5, concurrency=1,
            n_candidates=1, candidate_rounds=1, flush_strategy="default",
            prescore_map=ps,
            fix_rate_threshold=0.30, regress_threshold=0.15,
            min_pool_for_regression=10,   # pool=15 >= 10
            similarity_gate=False, log=False,
        )

    check(
        "regression gate runs when pool >= min_pool",
        len(regression_called) > 0,
        f"called {len(regression_called)} time(s)"
    )


# ---------------------------------------------------------------------------
# Test 5: fix_rate=0.30 accepts a 40%-fixing candidate
# ---------------------------------------------------------------------------
def test_fix_rate_30_accepts_partial_fix():
    items = _make_items(20)
    ps = _prescore(items, n_correct=10)  # 10 correct, 10 wrong → one flush bin

    with patch(_PATCH_SCORE_LOOP, side_effect=_score_fixes_40pct), \
         patch(_PATCH_SCORE_GATES, side_effect=_score_fixes_40pct), \
         patch(_PATCH_REGRESS, return_value=0.0), \
         patch(_PATCH_GEN, side_effect=_gen_one), \
         patch(_PATCH_SIMGATE, return_value=("ADD", None)):

        result = run_training_loop(
            cheatsheet=Cheatsheet(decision_tree="", case_studies=[]),
            train_items=items, val_items=None,
            model_score="dummy", model_casestudy="dummy", api_key="dummy",
            bin_threshold=5, batch_size=10, concurrency=1,
            n_candidates=1, candidate_rounds=1, flush_strategy="default",
            prescore_map=ps,
            fix_rate_threshold=0.30,   # 40% > 30% → should pass
            regress_threshold=0.15,
            min_pool_for_regression=10,
            similarity_gate=False, log=False,
        )

    check(
        "fix_rate=0.30 accepts 40%-fixing candidate",
        result.n_case_studies_added >= 1,
        f"added={result.n_case_studies_added}, discarded={result.n_bins_discarded}"
    )


# ---------------------------------------------------------------------------
# Test 6: fix_rate=0.50 blocks the same 40%-fixing candidate
# ---------------------------------------------------------------------------
def test_fix_rate_50_blocks_partial_fix():
    items = _make_items(20)
    ps = _prescore(items, n_correct=10)

    with patch(_PATCH_SCORE_LOOP, side_effect=_score_fixes_40pct), \
         patch(_PATCH_SCORE_GATES, side_effect=_score_fixes_40pct), \
         patch(_PATCH_REGRESS, return_value=0.0), \
         patch(_PATCH_GEN, side_effect=_gen_one), \
         patch(_PATCH_SIMGATE, return_value=("ADD", None)):

        result = run_training_loop(
            cheatsheet=Cheatsheet(decision_tree="", case_studies=[]),
            train_items=items, val_items=None,
            model_score="dummy", model_casestudy="dummy", api_key="dummy",
            bin_threshold=5, batch_size=10, concurrency=1,
            n_candidates=1, candidate_rounds=1, flush_strategy="default",
            prescore_map=ps,
            fix_rate_threshold=0.50,   # 40% < 50% → should be blocked
            regress_threshold=0.15,
            min_pool_for_regression=10,
            similarity_gate=False, log=False,
        )

    check(
        "fix_rate=0.50 blocks 40%-fixing candidate",
        result.n_bins_discarded >= 1,
        f"added={result.n_case_studies_added}, discarded={result.n_bins_discarded}"
    )


# ---------------------------------------------------------------------------
# Test 7: best_fix_rate logged on retry-path discard
# ---------------------------------------------------------------------------
def test_best_fix_rate_in_discard_log():
    items = _make_items(10)
    ps = _prescore(items, n_correct=0)  # all wrong

    with patch(_PATCH_SCORE_LOOP, side_effect=_score_fixes_none), \
         patch(_PATCH_SCORE_GATES, side_effect=_score_fixes_none), \
         patch(_PATCH_REGRESS, return_value=0.0), \
         patch(_PATCH_GEN, side_effect=_gen_one):

        result = run_training_loop(
            cheatsheet=Cheatsheet(decision_tree="", case_studies=[]),
            train_items=items, val_items=None,
            model_score="dummy", model_casestudy="dummy", api_key="dummy",
            bin_threshold=3, batch_size=5, concurrency=1,
            n_candidates=1, candidate_rounds=1, flush_strategy="retry",
            prescore_map=ps,
            fix_rate_threshold=0.30, regress_threshold=0.15,
            min_pool_for_regression=10,
            similarity_gate=False, log=False,
        )

    retry_discards = [
        e for e in result.update_log
        if e.get("event") == "bin_discarded" and "_rounds_failed" in e.get("reason", "")
    ]
    has_key = all("best_fix_rate" in e for e in retry_discards)
    check(
        "best_fix_rate in retry-path discard log",
        bool(retry_discards) and has_key,
        f"entries: {retry_discards}"
    )


# ---------------------------------------------------------------------------
# Test 8: CLI --min-pool-for-regression in --help
# ---------------------------------------------------------------------------
def test_pipeline_cli_arg():
    result = subprocess.run(
        [sys.executable, "-m", "ICR_select.pipeline", "--help"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent),
    )
    check(
        "--min-pool-for-regression in pipeline --help",
        "--min-pool-for-regression" in result.stdout,
    )
    check(
        "fix-rate default shows 0.3 in --help",
        "0.3" in result.stdout,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\nSmoke test: gate threshold changes\n")

    print("Test 1: default fix_rate_threshold")
    test_default_fix_rate_threshold()

    print("Test 2: default min_pool_for_regression")
    test_default_min_pool()

    print("Test 3: regression gate skipped when pool too small")
    test_regression_gate_skipped_when_pool_small()

    print("Test 4: regression gate runs when pool large enough")
    test_regression_gate_runs_when_pool_large()

    print("Test 5: fix_rate=0.30 accepts 40%-fixing candidate")
    test_fix_rate_30_accepts_partial_fix()

    print("Test 6: fix_rate=0.50 blocks 40%-fixing candidate")
    test_fix_rate_50_blocks_partial_fix()

    print("Test 7: best_fix_rate in retry-path discard log")
    test_best_fix_rate_in_discard_log()

    print("Test 8: pipeline CLI --min-pool-for-regression")
    test_pipeline_cli_arg()

    print()
    if _failures:
        print(f"\n{len(_failures)} failure(s):")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All tests passed.")
