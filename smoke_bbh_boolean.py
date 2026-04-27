#!/usr/bin/env python3
"""
smoke_bbh_boolean.py — Smoke tests for BBH boolean_expressions TaskSpec integration.

Tests the end-to-end path:
  BBH_BOOLEAN_TASK → build_partitions → run_partition_loop (all API calls patched)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from tasks.bbh_boolean import BBH_BOOLEAN_TASK
from ICR_partition.training.partition import build_partitions, refresh_partitions
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


_PATCH_SCORE_LOOP = "ICR_partition.training.loop.score_batch"
_PATCH_MINI_EVAL  = "ICR_partition.training.loop._mini_eval_full"
_PATCH_REGRESS    = "ICR_partition.training.loop._regression_check"
_PATCH_SIMGATE    = "ICR_partition.training.loop._similarity_gate"
_PATCH_GEN        = "ICR_partition.training.loop.generate_candidates"


# ---------------------------------------------------------------------------
# BBH item factory
# ---------------------------------------------------------------------------

def _bbh_item(expr: str, answer: str, idx: int = 0) -> dict:
    return {
        "id":         f"bbh_{idx:04d}",
        "input":      expr,
        "answer":     answer,
        "gold_reason": "",
    }


# Partition buckets: (has_not, has_and, has_or)
NOT_AND_ITEMS = [
    _bbh_item(f"not ( True ) and ( False ) is", "False", i)
    for i in range(8)
]
NOT_OR_ITEMS = [
    _bbh_item(f"not True or False is", "False", 100 + i)
    for i in range(8)
]
CORRECT_NOT_AND = [
    {**_bbh_item("not ( True ) and ( True ) is", "False", 200 + i), "predicted": "FALSE"}
    for i in range(6)
]


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


def _candidate() -> CaseStudy:
    return CaseStudy(
        title="NOT-AND Precedence Trap",
        activate_if=["has_not == TRUE", "has_and == TRUE"],
        action="FALSE",
        feature_signature="not_and",
    )


# ---------------------------------------------------------------------------
# Test 1 — partition key correctness for BBH items
# ---------------------------------------------------------------------------
def test_partition_key():
    print("Test 1: BBH_BOOLEAN_TASK.partition_key correctness")

    item_not_and = _bbh_item("not ( True ) and True is", "False")
    item_not_or  = _bbh_item("not True or False is", "False")
    item_all     = _bbh_item("not True and False or True is", "True")
    item_no_not  = _bbh_item("True and False is", "False")  # hypothetical

    k1 = BBH_BOOLEAN_TASK.partition_key(item_not_and)
    k2 = BBH_BOOLEAN_TASK.partition_key(item_not_or)
    k3 = BBH_BOOLEAN_TASK.partition_key(item_all)
    k4 = BBH_BOOLEAN_TASK.partition_key(item_no_not)

    check("not+and item → (True, True, False)",  k1 == (True, True, False),  f"got {k1}")
    check("not+or item  → (True, False, True)",  k2 == (True, False, True),  f"got {k2}")
    check("all ops item → (True, True, True)",   k3 == (True, True, True),   f"got {k3}")
    check("no-not item  → (False, True, False)", k4 == (False, True, False), f"got {k4}")

    conds = BBH_BOOLEAN_TASK.partition_key_to_conditions(k1)
    check("conditions list non-empty",           len(conds) >= 1,             f"got {conds}")
    check("NOT mentioned in conditions",         any("NOT" in c for c in conds), f"{conds}")


# ---------------------------------------------------------------------------
# Test 2 — build_partitions respects bin_threshold with BBH partition_key_fn
# ---------------------------------------------------------------------------
def test_build_partitions_bbh():
    print("\nTest 2: build_partitions with BBH partition_key_fn")

    wrong   = NOT_AND_ITEMS[:8]   # 8 failures → above threshold 3
    wrong2  = NOT_OR_ITEMS[:2]    # 2 failures → below threshold 3
    correct = CORRECT_NOT_AND

    bins = build_partitions(
        wrong + wrong2, correct,
        bin_threshold=3,
        partition_key_fn=BBH_BOOLEAN_TASK.partition_key,
    )

    k_and = BBH_BOOLEAN_TASK.partition_key(NOT_AND_ITEMS[0])
    k_or  = BBH_BOOLEAN_TASK.partition_key(NOT_OR_ITEMS[0])

    check("NOT+AND bin created (8 >= 3)",    k_and in bins, f"keys={list(bins)}")
    check("NOT+OR bin NOT created (2 < 3)",  k_or  not in bins, f"keys={list(bins)}")
    check("NOT+AND bin has 8 failures",
          len(bins[k_and].failures) == 8, f"got {len(bins[k_and].failures)}")
    check("correct pool populated for NOT+AND bin",
          len(bins[k_and].correct_pool) > 0, f"pool={len(bins[k_and].correct_pool)}")


# ---------------------------------------------------------------------------
# Test 3 — refresh_partitions retires BBH bins below threshold
# ---------------------------------------------------------------------------
def test_refresh_bbh():
    print("\nTest 3: refresh_partitions retires BBH bins")

    wrong = NOT_AND_ITEMS[:6]
    bins  = build_partitions(wrong, [], bin_threshold=3,
                             partition_key_fn=BBH_BOOLEAN_TASK.partition_key)
    k = BBH_BOOLEAN_TASK.partition_key(NOT_AND_ITEMS[0])
    check("bin created with 6 failures", len(bins[k].failures) == 6)

    new_wrong   = NOT_AND_ITEMS[:1]   # only 1 failure remains
    new_correct = NOT_AND_ITEMS[1:6]
    refresh_partitions(bins, new_wrong, new_correct, retirement_threshold=2,
                       partition_key_fn=BBH_BOOLEAN_TASK.partition_key)
    check("bin retired when failures < retirement_threshold", bins[k].solved)


# ---------------------------------------------------------------------------
# Test 4 — run_partition_loop end-to-end with BBH_BOOLEAN_TASK
# ---------------------------------------------------------------------------
def test_loop_adds_case_study_bbh():
    print("\nTest 4: run_partition_loop adds case study with BBH_BOOLEAN_TASK")

    items = NOT_AND_ITEMS[:8]
    ps    = _prescore(items, n_correct=0)

    def _score_all_correct(batch, *a, **kw):
        return [{**i, "predicted": "TRUE"} for i in batch], []

    def _mini_eval_fixes_all(cand, failures, cs, *a, **kw):
        return 1.0, []

    with patch(_PATCH_SCORE_LOOP, side_effect=_score_all_correct), \
         patch(_PATCH_MINI_EVAL,  side_effect=_mini_eval_fixes_all), \
         patch(_PATCH_REGRESS,    return_value=0.0), \
         patch(_PATCH_SIMGATE,    return_value=("ADD", None)), \
         patch(_PATCH_GEN,        return_value=[_candidate()]):

        result = run_partition_loop(
            cheatsheet=Cheatsheet(roadmap="", case_studies=[]),
            train_items=items, val_items=None,
            model_score="dummy", model_casestudy="dummy", api_key="dummy",
            oracle=None,
            bin_threshold=3, retirement_threshold=2, max_outer_iters=3,
            partition_concurrency=4, concurrency=1,
            n_candidates=1, candidate_rounds=1,
            fix_rate_threshold=0.30, regress_threshold=0.15,
            min_pool_for_regression=100,
            similarity_gate=False,
            prescore_map=ps,
            task_spec=BBH_BOOLEAN_TASK,
            output_dir=None, log=False,
        )

    check("case study added",
          result.n_case_studies_added >= 1, f"added={result.n_case_studies_added}")
    check("cheatsheet has ≥1 case study",
          len(result.cheatsheet.case_studies) >= 1,
          f"got {len(result.cheatsheet.case_studies)}")
    check("train_accuracy in [0, 1]",
          0.0 <= result.train_accuracy <= 1.0, f"got {result.train_accuracy}")


# ---------------------------------------------------------------------------
# Test 5 — scoring prompt format is valid (no missing placeholders)
# ---------------------------------------------------------------------------
def test_scoring_prompt_format():
    print("\nTest 5: scoring prompt format (no missing placeholders)")
    item = _bbh_item("not ( True ) and True is", "False")
    try:
        prompt = BBH_BOOLEAN_TASK.build_scoring_prompt("CHEATSHEET_TEXT", item, cot_first=False)
        check("prompt contains expression", "not ( True )" in prompt)
        check("prompt contains VERDICT instruction", "VERDICT:" in prompt)
        check("prompt contains cheatsheet", "CHEATSHEET_TEXT" in prompt)
    except Exception as e:
        check("prompt builds without error", False, str(e))

    try:
        prompt_cot = BBH_BOOLEAN_TASK.build_scoring_prompt("CHEATSHEET_TEXT", item, cot_first=True)
        check("cot_first prompt contains REASONING", "REASONING:" in prompt_cot)
    except Exception as e:
        check("cot_first prompt builds without error", False, str(e))


# ---------------------------------------------------------------------------
# Test 6 — generation prompt template has all required placeholders
# ---------------------------------------------------------------------------
def test_generation_prompt_template():
    print("\nTest 6: generation prompt template placeholders")
    template = BBH_BOOLEAN_TASK.generation_prompt_template
    required = ["{roadmap}", "{case_studies}", "{already_covered}",
                "{failure_lines}", "{polarity_instruction}", "{retry_context}"]
    for ph in required:
        check(f"template has {ph}", ph in template)

    try:
        rendered = template.format(
            roadmap="ROADMAP", case_studies="CS", already_covered="COVERED",
            failure_lines="FAILURES", polarity_instruction="POLARITY", retry_context="",
        )
        check("template renders without KeyError", True)
        check("rendered contains ROADMAP", "ROADMAP" in rendered)
    except KeyError as e:
        check("template renders without KeyError", False, str(e))


# ---------------------------------------------------------------------------
# Test 7 — polarity instruction customised by BBH task
# ---------------------------------------------------------------------------
def test_polarity_instruction():
    print("\nTest 7: BBH polarity instruction")
    fn = BBH_BOOLEAN_TASK.build_polarity_instruction
    check("build_polarity_instruction is not None", fn is not None)

    if fn:
        true_instr  = fn("TRUE",  "WRONG_ANSWER", "unknown")
        false_instr = fn("FALSE", "WRONG_ANSWER", "unknown")
        mixed_instr = fn("",      "WRONG_ANSWER", "unknown")

        check("TRUE polarity → mentions TYPE A",    "TYPE A" in true_instr)
        check("FALSE polarity → mentions TYPE B",   "TYPE B" in false_instr)
        check("mixed polarity → mentions both types",
              "TYPE A" in mixed_instr or "TYPE B" in mixed_instr)
        check("ABANDONMENT handled",
              "ABANDONMENT" in fn("TRUE", "ABANDONMENT", "STEP-3"))


# ---------------------------------------------------------------------------
# Test 8 — task_name field
# ---------------------------------------------------------------------------
def test_task_name():
    print("\nTest 8: task_name field")
    check("task_name is bbh_boolean_expressions",
          BBH_BOOLEAN_TASK.task_name == "bbh_boolean_expressions",
          f"got '{BBH_BOOLEAN_TASK.task_name}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\nSmoke test: BBH boolean_expressions TaskSpec\n")

    test_partition_key()
    test_build_partitions_bbh()
    test_refresh_bbh()
    test_loop_adds_case_study_bbh()
    test_scoring_prompt_format()
    test_generation_prompt_template()
    test_polarity_instruction()
    test_task_name()

    print()
    if _failures:
        print(f"\n{len(_failures)} failure(s):")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All tests passed.")
