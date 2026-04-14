#!/usr/bin/env python3
"""
smoke_partition.py — Smoke tests for ICR_partition without live API calls.

Patches score_batch, generate_candidates, _mini_eval_full, _regression_check,
and _similarity_gate to return deterministic results, then verifies:

  1.  item_partition_key: correct (form_e1, form_e2, depth_bucket, polarity)
  2.  build_partitions: bins created only when failures >= bin_threshold
  3.  build_partitions: designated correct pool populated from same key
  4.  PartitionBin.add_correct: reservoir-sampled cap respected
  5.  refresh_partitions: failure lists updated, retired bins set solved=True
  6.  run_partition_loop: case study added when fix_rate passes
  7.  run_partition_loop: bin discarded when fix_rate < threshold
  8.  run_partition_loop: regression gate uses designated correct pool, not global
  9.  run_partition_loop: bins solved concurrently (both get candidates)
  10. run_partition_loop: retired bins excluded from re-score pass
  11. run_partition_loop: prescore_map skips initial scoring API call
  12. run_partition_loop: update_log contains initial_score + iter events
  13. pipeline CLI: --help returns zero exit code
  14. pipeline CLI: --no-oracle flag present in --help
  15. pipeline CLI: --partition-concurrency flag present in --help
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

from ICR_partition.training.partition import (
    PartitionBin,
    PartitionKey,
    build_partitions,
    item_partition_key,
    refresh_partitions,
)
from ICR_partition.training.loop import run_partition_loop
from utils.cheatsheet import Cheatsheet
from utils.case_study import CaseStudy

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS}  {name}")
    else:
        msg = name + (f": {detail}" if detail else "")
        print(f"  {FAIL}  {msg}")
        _failures.append(msg)


# ---------------------------------------------------------------------------
# Patch targets (imported names inside loop.py)
# ---------------------------------------------------------------------------
_PATCH_SCORE_LOOP  = "ICR_partition.training.loop.score_batch"
_PATCH_MINI_EVAL   = "ICR_partition.training.loop._mini_eval_full"
_PATCH_REGRESS     = "ICR_partition.training.loop._regression_check"
_PATCH_SIMGATE     = "ICR_partition.training.loop._similarity_gate"
_PATCH_GEN         = "ICR_partition.training.loop.generate_candidates"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _item(eq1: str, eq2: str, answer: bool, predicted: str, idx: int = 0) -> dict:
    return {
        "id":        f"item_{idx:04d}",
        "equation1": eq1,
        "equation2": eq2,
        "answer":    answer,
        "predicted": predicted,
    }


# Absorbing E1 (rhs var absent from lhs) — form_e1 = ABSORBING
ABSORBING_EQ1 = "x * y = z"     # z is only on rhs  → ABSORBING
GENERAL_EQ2   = "x * y = z * w"

# Standard E1 (var on both sides) — form_e1 = STANDARD
STANDARD_EQ1  = "x = x * y"
STANDARD_EQ2  = "x = y * x"

# Trivial E1 — form_e1 = TRIVIAL
TRIVIAL_EQ1   = "x = x"
TRIVIAL_EQ2   = "y = y"


def _make_absorbing_items(n: int, answer: bool = True, predicted: str = "FALSE") -> list[dict]:
    return [_item(ABSORBING_EQ1, GENERAL_EQ2, answer, predicted, i) for i in range(n)]


def _make_standard_items(n: int, answer: bool = True, predicted: str = "FALSE") -> list[dict]:
    return [_item(STANDARD_EQ1, STANDARD_EQ2, answer, predicted, 1000 + i) for i in range(n)]


def _make_trivial_items(n: int, answer: bool = False, predicted: str = "TRUE") -> list[dict]:
    return [_item(TRIVIAL_EQ1, TRIVIAL_EQ2, answer, predicted, 2000 + i) for i in range(n)]


def _candidate() -> CaseStudy:
    return CaseStudy(
        title="Test Case Study",
        activate_if=["E1 is absorbing"],
        action="TRUE",
        feature_signature="absorbing→general_L0",
    )


def _prescore(items: list[dict], n_correct: int) -> dict:
    return {
        item["id"]: {
            "predicted":    "TRUE" if i < n_correct else "FALSE",
            "post_think":   "",
            "thinking":     "",
            "raw_response": "",
            "correct":      i < n_correct,
        }
        for i, item in enumerate(items)
    }


# ---------------------------------------------------------------------------
# Test 1 — item_partition_key correctness
# ---------------------------------------------------------------------------
def test_partition_key():
    print("Test 1: item_partition_key correctness")
    item_abs = _item(ABSORBING_EQ1, GENERAL_EQ2, True, "FALSE")
    key = item_partition_key(item_abs)
    check("form_e1 == ABSORBING", key[0] == "ABSORBING", f"got {key[0]}")
    check("form_e2 == GENERAL",   key[1] == "GENERAL",   f"got {key[1]}")
    check("polarity == TRUE",     key[3] == "TRUE",       f"got {key[3]}")

    item_std = _item(STANDARD_EQ1, STANDARD_EQ2, False, "TRUE")
    key_std = item_partition_key(item_std)
    check("standard E1 polarity == FALSE", key_std[3] == "FALSE", f"got {key_std[3]}")

    # depth_bucket: ABSORBING_EQ1 "x * y = z" has depth_e1 = 1 (one * on lhs) → bucket 1
    check("depth_bucket == 1 for depth-1 equation", key[2] == 1, f"got {key[2]}")

    # Trivial: "x = x" has no *, depth = 0 → bucket 0
    item_triv = _item(TRIVIAL_EQ1, TRIVIAL_EQ2, False, "TRUE")
    key_triv = item_partition_key(item_triv)
    check("depth_bucket == 0 for trivial (x=x)", key_triv[2] == 0, f"got {key_triv[2]}")


# ---------------------------------------------------------------------------
# Test 2 — build_partitions: bins created only above bin_threshold
# ---------------------------------------------------------------------------
def test_build_partitions_threshold():
    print("\nTest 2: build_partitions — bin_threshold respected")
    wrong   = _make_absorbing_items(5)   # 5 failures → above threshold 3
    wrong  += _make_standard_items(2)    # 2 failures → below threshold 3
    correct = _make_absorbing_items(4, answer=True, predicted="TRUE")

    bins = build_partitions(wrong, correct, bin_threshold=3)

    abs_key = item_partition_key(_item(ABSORBING_EQ1, GENERAL_EQ2, True, "FALSE"))
    std_key = item_partition_key(_item(STANDARD_EQ1,  STANDARD_EQ2, True, "FALSE"))

    check("absorbing bin created (5 >= 3)", abs_key in bins, f"keys={list(bins)}")
    check("standard bin NOT created (2 < 3)", std_key not in bins, f"keys={list(bins)}")
    check("absorbing bin has 5 failures", len(bins[abs_key].failures) == 5,
          f"got {len(bins[abs_key].failures)}")


# ---------------------------------------------------------------------------
# Test 3 — designated correct pool populated from same structural key
# ---------------------------------------------------------------------------
def test_designated_correct_pool():
    print("\nTest 3: designated correct pool populated by key")
    wrong   = _make_absorbing_items(5)
    correct_abs = _make_absorbing_items(6, answer=True, predicted="TRUE")
    correct_std = _make_standard_items(4, answer=True, predicted="TRUE")

    bins = build_partitions(wrong, correct_abs + correct_std, bin_threshold=3)

    abs_key = item_partition_key(_item(ABSORBING_EQ1, GENERAL_EQ2, True, "FALSE"))
    pb = bins[abs_key]
    check("absorbing correct pool has 6 items", len(pb.correct_pool) == 6,
          f"got {len(pb.correct_pool)}")

    # Standard-form correct items should NOT appear in absorbing correct_pool
    absorbing_eqs = {(i["equation1"], i["equation2"]) for i in pb.correct_pool}
    wrong_key_in_pool = any(
        (i["equation1"], i["equation2"]) == (STANDARD_EQ1, STANDARD_EQ2)
        for i in pb.correct_pool
    )
    check("standard items NOT in absorbing correct pool", not wrong_key_in_pool)


# ---------------------------------------------------------------------------
# Test 4 — reservoir sampling cap on correct pool
# ---------------------------------------------------------------------------
def test_reservoir_cap():
    print("\nTest 4: PartitionBin.add_correct reservoir cap")
    from ICR_partition.training.partition import CORRECT_POOL_PER_PARTITION_MAX
    pb = PartitionBin(key=("ABSORBING", "GENERAL", 1, "TRUE"))
    n_add = CORRECT_POOL_PER_PARTITION_MAX + 20
    for i in range(n_add):
        pb.add_correct({"id": f"c_{i}"})
    check(
        f"correct_pool capped at {CORRECT_POOL_PER_PARTITION_MAX}",
        len(pb.correct_pool) == CORRECT_POOL_PER_PARTITION_MAX,
        f"got {len(pb.correct_pool)}",
    )


# ---------------------------------------------------------------------------
# Test 5 — refresh_partitions: failure lists updated, retirement applied
# ---------------------------------------------------------------------------
def test_refresh_partitions():
    print("\nTest 5: refresh_partitions — updates and retires bins")
    wrong   = _make_absorbing_items(6)
    correct = []
    bins = build_partitions(wrong, correct, bin_threshold=3)

    abs_key = item_partition_key(_item(ABSORBING_EQ1, GENERAL_EQ2, True, "FALSE"))
    check("bin created with 6 failures", len(bins[abs_key].failures) == 6)

    # After a re-score, only 1 failure remains → below retirement_threshold=2 → retired
    new_correct = _make_absorbing_items(5, answer=True, predicted="TRUE")
    new_wrong   = _make_absorbing_items(1)
    refresh_partitions(bins, new_wrong, new_correct, retirement_threshold=2)
    check("bin retired when failures < retirement_threshold",
          bins[abs_key].solved, f"solved={bins[abs_key].solved}")

    # If 3 failures remain → stays active
    bins2 = build_partitions(_make_absorbing_items(6), [], bin_threshold=3)
    refresh_partitions(bins2, _make_absorbing_items(3), [], retirement_threshold=2)
    check("bin stays active when failures >= retirement_threshold",
          not bins2[abs_key].solved)


# ---------------------------------------------------------------------------
# Test 6 — run_partition_loop: case study added when fix_rate passes
# ---------------------------------------------------------------------------
def test_loop_adds_case_study():
    print("\nTest 6: run_partition_loop — case study added on fix_rate pass")

    items = _make_absorbing_items(10)
    # All wrong initially; all correct after loop (retire after iter 1)
    ps = _prescore(items, n_correct=0)

    def _score_all_correct(batch, *a, **kw):
        return [{**i, "predicted": "TRUE"} for i in batch], []

    def _mini_eval_fixes_all(cand, failures, cs, *a, **kw):
        return 1.0, []

    with patch(_PATCH_SCORE_LOOP,  side_effect=_score_all_correct), \
         patch(_PATCH_MINI_EVAL,   side_effect=_mini_eval_fixes_all), \
         patch(_PATCH_REGRESS,     return_value=0.0), \
         patch(_PATCH_SIMGATE,     return_value=("ADD", None)), \
         patch(_PATCH_GEN,         return_value=[_candidate()]):

        result = run_partition_loop(
            cheatsheet=Cheatsheet(roadmap="", case_studies=[]),
            train_items=items, val_items=None,
            model_score="dummy", model_casestudy="dummy", api_key="dummy",
            oracle=None,
            bin_threshold=3, retirement_threshold=2, max_outer_iters=3,
            partition_concurrency=4, concurrency=1,
            n_candidates=1, candidate_rounds=1,
            fix_rate_threshold=0.30, regress_threshold=0.15,
            min_pool_for_regression=100,   # skip regression (correct pool too small)
            similarity_gate=False,
            prescore_map=ps,
            output_dir=None, log=False,
        )

    check("case study added", result.n_case_studies_added >= 1,
          f"added={result.n_case_studies_added}")
    check("cheatsheet has ≥1 case study",
          len(result.cheatsheet.case_studies) >= 1,
          f"got {len(result.cheatsheet.case_studies)}")


# ---------------------------------------------------------------------------
# Test 7 — run_partition_loop: bin discarded when fix_rate < threshold
# ---------------------------------------------------------------------------
def test_loop_discards_bin():
    print("\nTest 7: run_partition_loop — bin discarded on fix_rate fail")

    items = _make_absorbing_items(10)
    ps    = _prescore(items, n_correct=0)

    def _score_all_correct(batch, *a, **kw):
        return [{**i, "predicted": "TRUE"} for i in batch], []

    def _mini_eval_fixes_none(cand, failures, cs, *a, **kw):
        return 0.0, list(failures)   # fixes nothing

    with patch(_PATCH_SCORE_LOOP,  side_effect=_score_all_correct), \
         patch(_PATCH_MINI_EVAL,   side_effect=_mini_eval_fixes_none), \
         patch(_PATCH_REGRESS,     return_value=0.0), \
         patch(_PATCH_SIMGATE,     return_value=("ADD", None)), \
         patch(_PATCH_GEN,         return_value=[_candidate()]):

        result = run_partition_loop(
            cheatsheet=Cheatsheet(roadmap="", case_studies=[]),
            train_items=items, val_items=None,
            model_score="dummy", model_casestudy="dummy", api_key="dummy",
            oracle=None,
            bin_threshold=3, retirement_threshold=2, max_outer_iters=2,
            partition_concurrency=4, concurrency=1,
            n_candidates=1, candidate_rounds=1,
            fix_rate_threshold=0.30, regress_threshold=0.15,
            min_pool_for_regression=100,
            similarity_gate=False,
            prescore_map=ps,
            output_dir=None, log=False,
        )

    check("bin discarded (not added)", result.n_case_studies_added == 0,
          f"added={result.n_case_studies_added}")
    check("n_bins_discarded >= 1", result.n_bins_discarded >= 1,
          f"discarded={result.n_bins_discarded}")


# ---------------------------------------------------------------------------
# Test 8 — regression gate uses designated correct pool (not global)
# ---------------------------------------------------------------------------
def test_designated_regression_pool():
    print("\nTest 8: regression gate uses designated correct pool")

    # Two structural classes: absorbing (5 wrong) and standard (5 wrong).
    # Give absorbing a 10-item correct pool; standard gets none.
    # Use distinct ID ranges to avoid prescore dict key collisions.
    abs_wrong   = [_item(ABSORBING_EQ1, GENERAL_EQ2, True,  "FALSE", 100+i) for i in range(5)]
    std_wrong   = [_item(STANDARD_EQ1,  STANDARD_EQ2, True, "FALSE", 200+i) for i in range(5)]
    abs_correct = [_item(ABSORBING_EQ1, GENERAL_EQ2, True,  "TRUE",  300+i) for i in range(10)]

    all_items = abs_wrong + std_wrong + abs_correct

    # Build prescore so abs_correct items are "correct" and the rest are "wrong"
    ps: dict = {}
    for item in abs_wrong + std_wrong:
        ps[item["id"]] = {"predicted": "FALSE", "post_think": "", "thinking": "",
                          "raw_response": "", "correct": False}
    for item in abs_correct:
        ps[item["id"]] = {"predicted": "TRUE",  "post_think": "", "thinking": "",
                          "raw_response": "", "correct": True}

    regression_calls: list[str] = []

    def _tracked_regress(cand, correct_pool, cs, *a, **kw):
        for item in correct_pool:
            regression_calls.append(item_partition_key(item)[0])
        return 0.0   # passes gate

    def _score_all_correct(batch, *a, **kw):
        return [{**i, "predicted": "TRUE"} for i in batch], []

    def _mini_eval_fixes_all(cand, failures, cs, *a, **kw):
        return 1.0, []

    with patch(_PATCH_SCORE_LOOP,  side_effect=_score_all_correct), \
         patch(_PATCH_MINI_EVAL,   side_effect=_mini_eval_fixes_all), \
         patch(_PATCH_REGRESS,     side_effect=_tracked_regress), \
         patch(_PATCH_SIMGATE,     return_value=("ADD", None)), \
         patch(_PATCH_GEN,         return_value=[_candidate()]):

        run_partition_loop(
            cheatsheet=Cheatsheet(roadmap="", case_studies=[]),
            train_items=all_items, val_items=None,
            model_score="dummy", model_casestudy="dummy", api_key="dummy",
            oracle=None,
            bin_threshold=3, retirement_threshold=2, max_outer_iters=2,
            partition_concurrency=4, concurrency=1,
            n_candidates=1, candidate_rounds=1,
            fix_rate_threshold=0.30, regress_threshold=0.15,
            min_pool_for_regression=5,   # 10 absorbing correct items ≥ 5 → regression runs
            similarity_gate=False,
            prescore_map=ps,
            output_dir=None, log=False,
        )

    # Every item passed to _regression_check should be ABSORBING
    # (standard partition has 0 correct → regression skipped for it)
    check("regression only called with ABSORBING correct items",
          all(form == "ABSORBING" for form in regression_calls),
          f"forms seen: {set(regression_calls)}")
    check("regression was called at least once",
          len(regression_calls) > 0,
          "regression never called — check min_pool_for_regression logic")


# ---------------------------------------------------------------------------
# Test 9 — bins solved concurrently (both partitions get candidates)
# ---------------------------------------------------------------------------
def test_concurrent_bins():
    print("\nTest 9: concurrent bin solving — both partitions get candidates")

    abs_wrong = _make_absorbing_items(5)
    std_wrong = _make_standard_items(5)
    all_items = abs_wrong + std_wrong
    ps = _prescore(all_items, n_correct=0)   # all wrong initially

    gen_calls: list[str] = []

    def _tracked_gen(failures, cs, model, api_key, **kw):
        key = item_partition_key(failures[0])
        gen_calls.append(key[0])   # record form_e1
        return [_candidate()]

    def _score_all_correct(batch, *a, **kw):
        return [{**i, "predicted": "TRUE"} for i in batch], []

    def _mini_eval_fixes_all(cand, failures, cs, *a, **kw):
        return 1.0, []

    with patch(_PATCH_SCORE_LOOP,  side_effect=_score_all_correct), \
         patch(_PATCH_MINI_EVAL,   side_effect=_mini_eval_fixes_all), \
         patch(_PATCH_REGRESS,     return_value=0.0), \
         patch(_PATCH_SIMGATE,     return_value=("ADD", None)), \
         patch(_PATCH_GEN,         side_effect=_tracked_gen):

        run_partition_loop(
            cheatsheet=Cheatsheet(roadmap="", case_studies=[]),
            train_items=all_items, val_items=None,
            model_score="dummy", model_casestudy="dummy", api_key="dummy",
            oracle=None,
            bin_threshold=3, retirement_threshold=2, max_outer_iters=2,
            partition_concurrency=4, concurrency=1,
            n_candidates=1, candidate_rounds=1,
            fix_rate_threshold=0.30, regress_threshold=0.15,
            min_pool_for_regression=100,
            similarity_gate=False,
            prescore_map=ps,
            output_dir=None, log=False,
        )

    unique_forms = set(gen_calls)
    check("generate_candidates called for both ABSORBING and STANDARD partitions",
          "ABSORBING" in unique_forms and "STANDARD" in unique_forms,
          f"forms called: {unique_forms}")


# ---------------------------------------------------------------------------
# Test 10 — prescore_map skips initial score_batch call
# ---------------------------------------------------------------------------
def test_prescore_skips_initial_score():
    print("\nTest 10: prescore_map skips initial score_batch call")

    items = _make_absorbing_items(8)
    ps    = _prescore(items, n_correct=2)

    initial_score_calls: list[int] = []

    def _score_tracking(batch, *a, **kw):
        initial_score_calls.append(len(batch))
        return [{**i, "predicted": "TRUE"} for i in batch], []

    def _mini_eval_fixes_all(cand, failures, cs, *a, **kw):
        return 1.0, []

    with patch(_PATCH_SCORE_LOOP,  side_effect=_score_tracking), \
         patch(_PATCH_MINI_EVAL,   side_effect=_mini_eval_fixes_all), \
         patch(_PATCH_REGRESS,     return_value=0.0), \
         patch(_PATCH_SIMGATE,     return_value=("ADD", None)), \
         patch(_PATCH_GEN,         return_value=[_candidate()]):

        run_partition_loop(
            cheatsheet=Cheatsheet(roadmap="", case_studies=[]),
            train_items=items, val_items=None,
            model_score="dummy", model_casestudy="dummy", api_key="dummy",
            oracle=None,
            bin_threshold=3, retirement_threshold=2, max_outer_iters=1,
            partition_concurrency=2, concurrency=1,
            n_candidates=1, candidate_rounds=1,
            fix_rate_threshold=0.30, regress_threshold=0.15,
            min_pool_for_regression=100,
            similarity_gate=False,
            prescore_map=ps,    # ← prescore provided
            output_dir=None, log=False,
        )

    # Initial full-dataset score_batch should NOT have been called (prescore used)
    # Only the re-score of active-bin items (and final pass) should appear.
    full_initial_call = any(n == len(items) for n in initial_score_calls[:1])
    check("initial 8-item score_batch NOT called (prescore used)",
          not full_initial_call,
          f"score_batch call sizes: {initial_score_calls}")


# ---------------------------------------------------------------------------
# Test 11 — update_log contains expected event types
# ---------------------------------------------------------------------------
def test_update_log_structure():
    print("\nTest 11: update_log event structure")

    items = _make_absorbing_items(8)
    ps    = _prescore(items, n_correct=0)

    def _score_all_correct(batch, *a, **kw):
        return [{**i, "predicted": "TRUE"} for i in batch], []

    def _mini_eval_fixes_all(cand, failures, cs, *a, **kw):
        return 1.0, []

    with patch(_PATCH_SCORE_LOOP,  side_effect=_score_all_correct), \
         patch(_PATCH_MINI_EVAL,   side_effect=_mini_eval_fixes_all), \
         patch(_PATCH_REGRESS,     return_value=0.0), \
         patch(_PATCH_SIMGATE,     return_value=("ADD", None)), \
         patch(_PATCH_GEN,         return_value=[_candidate()]):

        result = run_partition_loop(
            cheatsheet=Cheatsheet(roadmap="", case_studies=[]),
            train_items=items, val_items=None,
            model_score="dummy", model_casestudy="dummy", api_key="dummy",
            oracle=None,
            bin_threshold=3, retirement_threshold=2, max_outer_iters=2,
            partition_concurrency=2, concurrency=1,
            n_candidates=1, candidate_rounds=1,
            fix_rate_threshold=0.30, regress_threshold=0.15,
            min_pool_for_regression=100,
            similarity_gate=False,
            prescore_map=ps,
            output_dir=None, log=False,
        )

    events = [e.get("event") for e in result.update_log]
    check("update_log contains initial_score event",
          "initial_score" in events, f"events={events}")
    check("update_log contains bin_added event",
          "bin_added" in events, f"events={events}")


# ---------------------------------------------------------------------------
# Test 12 — pipeline CLI: --help exits 0
# ---------------------------------------------------------------------------
def test_cli_help():
    print("\nTest 12: pipeline CLI --help")
    proc = subprocess.run(
        [sys.executable, "-m", "ICR_partition.pipeline", "--help"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent),
    )
    check("--help exits with code 0", proc.returncode == 0,
          f"exit={proc.returncode}\n{proc.stderr[:300]}")
    check("--no-oracle in --help",              "--no-oracle"              in proc.stdout)
    check("--partition-concurrency in --help",  "--partition-concurrency"  in proc.stdout)
    check("--retirement-threshold in --help",   "--retirement-threshold"   in proc.stdout)
    check("--max-outer-iters in --help",        "--max-outer-iters"        in proc.stdout)
    check("--bin-threshold in --help",          "--bin-threshold"          in proc.stdout)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\nSmoke test: ICR_partition\n")

    test_partition_key()
    test_build_partitions_threshold()
    test_designated_correct_pool()
    test_reservoir_cap()
    test_refresh_partitions()
    test_loop_adds_case_study()
    test_loop_discards_bin()
    test_designated_regression_pool()
    test_concurrent_bins()
    test_prescore_skips_initial_score()
    test_update_log_structure()
    test_cli_help()

    print()
    if _failures:
        print(f"\n{len(_failures)} failure(s):")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All tests passed.")
