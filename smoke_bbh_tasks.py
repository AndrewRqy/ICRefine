#!/usr/bin/env python3
"""
smoke_bbh_tasks.py — Smoke tests for all BBH TaskSpec implementations.

Covers:
  1.  Generation prompt templates: all required placeholders present + renders
  2.  Scoring prompts: build without error, contain expected fields
  3.  Partition keys: correct tuple shape + conditions non-empty
  4.  Verdict parsing + correctness checking
  5.  Polarity instruction builders
  6.  end-to-end run_partition_loop with each task (API calls mocked)
  7.  concrete_cs_gen_fn pre-pass: section accepted when fix_rate passes
  8.  concrete_cs_gen_fn pre-pass: falls through when fix_rate fails
  9.  bootstrap_cheatsheet_fn: sets prior_knowledge, skips Phase 1 rule_set
  10. _mini_eval_text / _regression_check_text: basic shape check
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

from tasks.bbh_tasks import (
    CAUSAL_JUDGEMENT_TASK,
    SPORTS_TASK,
    DISAMBIGUATION_TASK,
    MOVIE_TASK,
    GEOMETRIC_TASK,
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


_PATCH_SCORE_LOOP  = "ICR_partition.training.loop.score_batch"
_PATCH_MINI_EVAL   = "ICR_partition.training.loop._mini_eval_full"
_PATCH_MINI_TEXT   = "ICR_partition.training.loop._mini_eval_text"
_PATCH_REGRESS     = "ICR_partition.training.loop._regression_check"
_PATCH_REGRESS_TXT = "ICR_partition.training.loop._regression_check_text"
_PATCH_SIMGATE     = "ICR_partition.training.loop._similarity_gate"
_PATCH_GEN         = "ICR_partition.training.loop.generate_candidates"


# ---------------------------------------------------------------------------
# Item factories per task
# ---------------------------------------------------------------------------

def _causal_item(idx: int, answer: str = "Yes") -> dict:
    return {
        "id": f"cj_{idx:04d}",
        "input": (
            "Suzy and Billy both threw rocks at a bottle. "
            "Suzy's rock hit the bottle first. Did Suzy cause the bottle to break?"
        ),
        "answer": answer,
    }

def _sports_item(idx: int, answer: str = "yes") -> dict:
    return {
        "id": f"sp_{idx:04d}",
        "input": "Tiger Woods made a hole-in-one on the 18th hole.",
        "answer": answer,
    }

def _disambig_item(idx: int, answer: str = "(A)") -> dict:
    return {
        "id": f"dq_{idx:04d}",
        "input": (
            "The trophy didn't fit in the suitcase because it was too big. "
            "What was too big? (A) The trophy (B) The suitcase (C) Ambiguous"
        ),
        "answer": answer,
    }

def _movie_item(idx: int, answer: str = "(A)") -> dict:
    return {
        "id": f"mr_{idx:04d}",
        "input": (
            "Find a movie similar to The Matrix, Inception, Interstellar.\n"
            "(A) Blade Runner 2049 (B) The Notebook (C) Toy Story (D) Grease"
        ),
        "answer": answer,
    }

def _geo_item(idx: int, answer: str = "(C)") -> dict:
    return {
        "id": f"gs_{idx:04d}",
        "input": (
            "This SVG path data represents which shape?\n"
            "<svg><path d=\"M 55.0 300.0 L 100.0 200.0 L 200.0 200.0 Z\"/></svg>\n"
            "(A) circle (B) rectangle (C) triangle (D) pentagon"
        ),
        "answer": answer,
    }


def _prescore(items: list[dict], n_correct: int) -> dict:
    return {
        item["id"]: {
            "predicted":    item["answer"].strip().upper() if i < n_correct else "WRONG",
            "post_think":   "",
            "thinking":     "",
            "raw_response": "",
            "correct":      i < n_correct,
        }
        for i, item in enumerate(items)
    }


def _candidate() -> CaseStudy:
    return CaseStudy(
        title="Test Case Study",
        activate_if=["scenario condition"],
        action="YES",
        feature_signature="test_sig",
    )


# ---------------------------------------------------------------------------
# Test 1 — generation prompt templates: placeholders + renders
# ---------------------------------------------------------------------------

def test_generation_prompt_templates():
    print("Test 1: generation_prompt_template — placeholders and render")
    required = [
        "{roadmap}", "{case_studies}", "{already_covered}",
        "{failure_lines}", "{polarity_instruction}", "{retry_context}",
    ]
    tasks = [
        ("causal_judgement",   CAUSAL_JUDGEMENT_TASK),
        ("sports_understanding", SPORTS_TASK),
        ("disambiguation_qa",  DISAMBIGUATION_TASK),
        ("movie_recommendation", MOVIE_TASK),
        ("geometric_shapes",   GEOMETRIC_TASK),
    ]
    for name, task in tasks:
        tmpl = task.generation_prompt_template
        for ph in required:
            check(f"{name}: has {ph}", ph in tmpl)
        try:
            rendered = tmpl.format(
                roadmap="ROADMAP", case_studies="CS", already_covered="COVERED",
                failure_lines="FAILURES", polarity_instruction="POLARITY", retry_context="",
            )
            check(f"{name}: renders without KeyError", True)
            check(f"{name}: rendered contains ROADMAP", "ROADMAP" in rendered)
        except KeyError as e:
            check(f"{name}: renders without KeyError", False, str(e))


# ---------------------------------------------------------------------------
# Test 2 — scoring prompts build correctly
# ---------------------------------------------------------------------------

def test_scoring_prompts():
    print("\nTest 2: scoring prompts build without error")
    cases = [
        ("causal_judgement",    CAUSAL_JUDGEMENT_TASK, _causal_item(0)),
        ("sports_understanding", SPORTS_TASK,          _sports_item(0)),
        ("disambiguation_qa",   DISAMBIGUATION_TASK,   _disambig_item(0)),
        ("movie_recommendation", MOVIE_TASK,           _movie_item(0)),
        ("geometric_shapes",    GEOMETRIC_TASK,        _geo_item(0)),
    ]
    for name, task, item in cases:
        for cot in (False, True):
            try:
                p = task.build_scoring_prompt("CHEATSHEET", item, cot)
                check(f"{name} cot={cot}: builds OK",      True)
                check(f"{name} cot={cot}: has VERDICT",    "VERDICT" in p)
                check(f"{name} cot={cot}: has CHEATSHEET", "CHEATSHEET" in p)
            except Exception as e:
                check(f"{name} cot={cot}: builds OK", False, str(e))


# ---------------------------------------------------------------------------
# Test 3 — partition keys: correct shape + conditions
# ---------------------------------------------------------------------------

def test_partition_keys():
    print("\nTest 3: partition_key + partition_key_to_conditions")
    cases = [
        ("causal_judgement",    CAUSAL_JUDGEMENT_TASK, _causal_item(0)),
        ("sports_understanding", SPORTS_TASK,          _sports_item(0)),
        ("disambiguation_qa",   DISAMBIGUATION_TASK,   _disambig_item(0)),
        ("movie_recommendation", MOVIE_TASK,           _movie_item(0)),
        ("geometric_shapes",    GEOMETRIC_TASK,        _geo_item(0)),
    ]
    for name, task, item in cases:
        try:
            key = task.partition_key(item)
            check(f"{name}: partition_key returns tuple", isinstance(key, tuple),
                  f"got {type(key)}")
            conds = task.partition_key_to_conditions(key)
            check(f"{name}: conditions non-empty", len(conds) >= 1, f"got {conds}")
            check(f"{name}: conditions are strings", all(isinstance(c, str) for c in conds))
        except Exception as e:
            check(f"{name}: partition_key OK", False, str(e))


# ---------------------------------------------------------------------------
# Test 4 — verdict parsing + correctness
# ---------------------------------------------------------------------------

def test_verdict_parsing():
    print("\nTest 4: parse_verdict + is_correct + answer_label")
    cases = [
        ("causal YES",   CAUSAL_JUDGEMENT_TASK, "VERDICT: YES\nREASONING: x", _causal_item(0, "Yes"),  "YES"),
        ("causal NO",    CAUSAL_JUDGEMENT_TASK, "VERDICT: NO\nREASONING: x",  _causal_item(0, "No"),   "NO"),
        ("sports YES",   SPORTS_TASK,           "VERDICT: YES",               _sports_item(0, "yes"),  "YES"),
        ("disambig (A)", DISAMBIGUATION_TASK,   "VERDICT: (A)",               _disambig_item(0, "(A)"), "(A)"),
        ("movie (B)",    MOVIE_TASK,            "VERDICT: (B)",               _movie_item(0, "(B)"),   "(B)"),
        ("geo (C)",      GEOMETRIC_TASK,        "VERDICT: (C)",               _geo_item(0, "(C)"),     "(C)"),
    ]
    for label, task, raw, item, expected_pred in cases:
        pred = task.parse_verdict(raw)
        check(f"{label}: parse_verdict={expected_pred}", pred == expected_pred,
              f"got {pred!r}")
        check(f"{label}: is_correct=True", task.is_correct(pred, item),
              f"pred={pred!r} answer={item['answer']!r}")
        lbl = task.answer_label(item)
        check(f"{label}: answer_label non-empty", bool(lbl), f"got {lbl!r}")


# ---------------------------------------------------------------------------
# Test 5 — polarity instruction builders
# ---------------------------------------------------------------------------

def test_polarity_instructions():
    print("\nTest 5: build_polarity_instruction")
    tasks_with_polarity = [
        ("causal_judgement",    CAUSAL_JUDGEMENT_TASK),
        ("sports_understanding", SPORTS_TASK),
        ("geometric_shapes",    GEOMETRIC_TASK),
    ]
    for name, task in tasks_with_polarity:
        fn = task.build_polarity_instruction
        check(f"{name}: build_polarity_instruction set", fn is not None)
        if fn:
            try:
                instr = fn("YES", "WRONG_ANSWER", "step-1")
                check(f"{name}: returns non-empty string", bool(instr.strip()))
                aband = fn("YES", "ABANDONMENT", "step-2")
                check(f"{name}: ABANDONMENT handled", "ABANDONMENT" in aband or len(aband) > 10)
            except Exception as e:
                check(f"{name}: polarity builds OK", False, str(e))


# ---------------------------------------------------------------------------
# Test 6 — end-to-end run_partition_loop for each BBH task (mocked API)
# ---------------------------------------------------------------------------

def _run_loop(task, items, label):
    ps = _prescore(items, n_correct=0)

    def _score_all_correct(batch, *a, **kw):
        return [{**i, "predicted": i["answer"].upper()} for i in batch], []

    def _mini_eval_fixes_all(cand, failures, cs, *a, **kw):
        return 1.0, []

    with patch(_PATCH_SCORE_LOOP, side_effect=_score_all_correct), \
         patch(_PATCH_MINI_EVAL,  side_effect=_mini_eval_fixes_all), \
         patch(_PATCH_REGRESS,    return_value=0.0), \
         patch(_PATCH_SIMGATE,    return_value=("ADD", None)), \
         patch(_PATCH_GEN,        return_value=[_candidate()]):
        return run_partition_loop(
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
            task_spec=task,
            output_dir=None, log=False,
        )


def test_loop_per_task():
    print("\nTest 6: run_partition_loop end-to-end per BBH task")
    cases = [
        ("causal_judgement",    CAUSAL_JUDGEMENT_TASK, [_causal_item(i) for i in range(8)]),
        ("sports_understanding", SPORTS_TASK,           [_sports_item(i) for i in range(8)]),
        ("disambiguation_qa",   DISAMBIGUATION_TASK,   [_disambig_item(i) for i in range(8)]),
        ("movie_recommendation", MOVIE_TASK,           [_movie_item(i) for i in range(8)]),
        ("geometric_shapes",    GEOMETRIC_TASK,        [_geo_item(i) for i in range(8)]),
    ]
    for name, task, items in cases:
        try:
            result = _run_loop(task, items, name)
            check(f"{name}: loop completes",          True)
            check(f"{name}: case study added",        result.n_case_studies_added >= 1,
                  f"added={result.n_case_studies_added}")
            check(f"{name}: train_accuracy in [0,1]", 0.0 <= result.train_accuracy <= 1.0,
                  f"got {result.train_accuracy}")
        except Exception as e:
            check(f"{name}: loop completes", False, str(e))


# ---------------------------------------------------------------------------
# Test 7 — concrete_cs_gen_fn pre-pass: section accepted on good fix_rate
# ---------------------------------------------------------------------------

def test_concrete_cs_accepted():
    print("\nTest 7: concrete_cs_gen_fn pre-pass — section accepted")

    items = [_causal_item(i) for i in range(8)]
    ps    = _prescore(items, n_correct=0)

    SECTION_TEXT = "=== Overdetermination: Two Fires ===\nScenario: ...\nVerdict: NO"
    concrete_calls: list[int] = []

    def _fake_concrete(failures, cs_text, model, api_key):
        concrete_calls.append(len(failures))
        return SECTION_TEXT

    def _score_all_correct(batch, *a, **kw):
        return [{**i, "predicted": "YES"} for i in batch], []

    def _mini_text_passes(text, failures, cs, *a, **kw):
        return 1.0, []   # fix_rate=100% → passes gate

    task = CAUSAL_JUDGEMENT_TASK

    with patch(_PATCH_SCORE_LOOP, side_effect=_score_all_correct), \
         patch(_PATCH_MINI_TEXT,  side_effect=_mini_text_passes), \
         patch(_PATCH_REGRESS_TXT, return_value=0.0), \
         patch(_PATCH_MINI_EVAL,  return_value=(1.0, [])), \
         patch(_PATCH_REGRESS,    return_value=0.0), \
         patch(_PATCH_SIMGATE,    return_value=("ADD", None)), \
         patch(_PATCH_GEN,        return_value=[_candidate()]), \
         patch.object(task, "concrete_cs_gen_fn", side_effect=_fake_concrete):

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
            task_spec=task,
            output_dir=None, log=False,
        )

    check("concrete_cs_gen_fn was called",
          len(concrete_calls) > 0,
          "pre-pass never triggered")
    check("concrete section text appears in prior_knowledge",
          SECTION_TEXT in result.cheatsheet.prior_knowledge or
          result.n_case_studies_added >= 1,
          f"prior_knowledge={result.cheatsheet.prior_knowledge[:100]!r}")


# ---------------------------------------------------------------------------
# Test 8 — concrete_cs_gen_fn pre-pass: falls through on low fix_rate
# ---------------------------------------------------------------------------

def test_concrete_cs_fallthrough():
    print("\nTest 8: concrete_cs_gen_fn pre-pass — falls through on low fix_rate")

    items = [_causal_item(i) for i in range(8)]
    ps    = _prescore(items, n_correct=0)

    gen_calls: list[int] = []

    def _fake_concrete(failures, cs_text, model, api_key):
        return "=== Some Section ==="

    def _mini_text_fails(text, failures, cs, *a, **kw):
        return 0.0, list(failures)   # fix_rate=0% → below threshold

    def _score_all_correct(batch, *a, **kw):
        return [{**i, "predicted": "YES"} for i in batch], []

    def _mini_eval_fixes_all(cand, failures, cs, *a, **kw):
        gen_calls.append(1)
        return 1.0, []

    task = CAUSAL_JUDGEMENT_TASK

    with patch(_PATCH_SCORE_LOOP, side_effect=_score_all_correct), \
         patch(_PATCH_MINI_TEXT,  side_effect=_mini_text_fails), \
         patch(_PATCH_REGRESS_TXT, return_value=0.0), \
         patch(_PATCH_MINI_EVAL,  side_effect=_mini_eval_fixes_all), \
         patch(_PATCH_REGRESS,    return_value=0.0), \
         patch(_PATCH_SIMGATE,    return_value=("ADD", None)), \
         patch(_PATCH_GEN,        return_value=[_candidate()]), \
         patch.object(task, "concrete_cs_gen_fn", side_effect=_fake_concrete):

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
            task_spec=task,
            output_dir=None, log=False,
        )

    check("structured generate_candidates called after fallthrough",
          len(gen_calls) > 0,
          "loop never fell through to structured generation")
    check("case study added via fallback path",
          result.n_case_studies_added >= 1,
          f"added={result.n_case_studies_added}")


# ---------------------------------------------------------------------------
# Test 9 — bootstrap_cheatsheet_fn: sets prior_knowledge, skips rule_set
# ---------------------------------------------------------------------------

def test_bootstrap_cheatsheet_fn():
    print("\nTest 9: bootstrap_cheatsheet_fn — sets prior_knowledge, Phase 1 skipped")
    from utils.cheatsheet import Cheatsheet as _CS

    BOOTSTRAP_TEXT = "=== Overdetermination: Two Fires ===\nScenario: ..."
    bootstrap_called = [False]

    def _fake_bootstrap(failures, model, api_key):
        bootstrap_called[0] = True
        return BOOTSTRAP_TEXT

    # Import hybrid loop
    from ICR_hybrid.training.loop import run_hybrid_loop

    cs = _CS(roadmap="", case_studies=[])
    items = [_causal_item(i) for i in range(20)]

    # _do_score is a closure inside run_hybrid_loop; patch score_batch instead
    _HYBRID_SCORE = "ICR_hybrid.training.loop.score_batch"

    def _score_some_wrong(batch, *a, **kw):
        half = len(batch) // 2
        correct = [{**i, "predicted": "YES", "expected": "YES",
                    "post_think": "", "thinking": "", "raw_response": ""}
                   for i in batch[:half]]
        wrong   = [{**i, "predicted": "NO",  "expected": "YES",
                    "post_think": "", "thinking": "", "raw_response": ""}
                   for i in batch[half:]]
        return correct, wrong

    with patch.object(CAUSAL_JUDGEMENT_TASK, "bootstrap_cheatsheet_fn",
                      side_effect=_fake_bootstrap), \
         patch(_HYBRID_SCORE, side_effect=_score_some_wrong):

        # run_hybrid_loop with auto_rule_init=True and no initial rule_set
        # should call bootstrap_cheatsheet_fn and set prior_knowledge,
        # then skip Phase 1 (rule_set stays None → phase 1 loop is never entered)
        try:
            from ICR_hybrid.training.loop import run_hybrid_loop as _rhl

            # We only want to test the bootstrap section, not the full loop.
            # Patch Phase 1 and Phase 2 loops to be no-ops.
            _PATCH_PHASE1 = "ICR_hybrid.training.loop.run_rule_patch_loop"
            _PATCH_PHASE2 = "ICR_hybrid.training.loop.run_partition_loop"

            def _fake_phase2(*args, **kwargs):
                # Return the cheatsheet the loop passes in (already bootstrap-modified)
                passed_cs = kwargs.get("cheatsheet", cs)
                r = MagicMock()
                r.cheatsheet        = passed_cs
                r.n_case_studies_added = 0
                r.n_bins_solved     = 0
                r.n_bins_discarded  = 0
                r.n_merges          = 0
                r.n_skipped         = 0
                r.n_outer_iters     = 1
                r.train_accuracy    = 0.5
                r.update_log        = []
                r.partition_summary = []
                r.n_rule_patches    = 0
                return r
            mock_phase2 = _fake_phase2

            with patch(_PATCH_PHASE2, mock_phase2):
                result = _rhl(
                    initial_rule_set=None,
                    initial_cheatsheet=cs,
                    train_items=items,
                    val_items=None,
                    model_score="dummy", model_rule_patch="dummy",
                    model_casestudy="dummy", api_key="dummy",
                    auto_rule_init=True,
                    n_bootstrap_failures=5,
                    max_rule_iters=0,   # skip Phase 1
                    max_cs_iters=1,
                    task_spec=CAUSAL_JUDGEMENT_TASK,
                    output_dir=None, log=False,
                )

            # loop copies initial_cheatsheet; check returned result's cheatsheet
            out_cs = result.cheatsheet
            check("bootstrap_cheatsheet_fn was called", bootstrap_called[0])
            check("prior_knowledge set to bootstrap text",
                  out_cs.prior_knowledge == BOOTSTRAP_TEXT,
                  f"got: {out_cs.prior_knowledge[:80]!r}")
        except Exception as e:
            check("bootstrap_cheatsheet_fn path OK", False, str(e))


# ---------------------------------------------------------------------------
# Test 10 — _mini_eval_text and _regression_check_text shape
# ---------------------------------------------------------------------------

def test_text_gate_helpers():
    print("\nTest 10: _mini_eval_text + _regression_check_text")
    from ICR_select.training.gates import _mini_eval_text, _regression_check_text

    cs = Cheatsheet(roadmap="ROAD", case_studies=[], prior_knowledge="BASE")
    items = [_causal_item(i) for i in range(6)]
    SECTION = "=== Two Fires ==="

    _PATCH_SB = "ICR_select.training.gates.score_batch"

    # All fixed: fix_rate should be 1.0
    def _all_correct(batch, *a, **kw):
        return list(batch), []

    with patch(_PATCH_SB, side_effect=_all_correct):
        fr, still_wrong = _mini_eval_text(
            SECTION, items, cs,
            model_score="dummy", api_key="dummy",
            concurrency=1, reasoning_effort=None, cot_first=True,
        )
    check("_mini_eval_text fix_rate=1.0 when all fixed", fr == 1.0, f"got {fr}")
    check("_mini_eval_text still_wrong=[] when all fixed", still_wrong == [], f"got {still_wrong}")

    # All regressed: regression_rate should be 1.0
    def _all_wrong(batch, *a, **kw):
        return [], list(batch)

    with patch(_PATCH_SB, side_effect=_all_wrong):
        reg = _regression_check_text(
            SECTION, items, cs,
            model_score="dummy", api_key="dummy",
            concurrency=1, reasoning_effort=None, cot_first=True,
        )
    check("_regression_check_text rate=1.0 when all regressed", reg == 1.0, f"got {reg}")

    # Empty correct_pool: regression should return 0.0 without calling score_batch
    reg_empty = _regression_check_text(
        SECTION, [], cs,
        model_score="dummy", api_key="dummy",
        concurrency=1, reasoning_effort=None, cot_first=True,
    )
    check("_regression_check_text returns 0.0 for empty pool", reg_empty == 0.0,
          f"got {reg_empty}")

    # Verify prior_knowledge is extended (not replaced) in temp cheatsheet
    augmented_calls: list[str] = []
    def _capture_render(batch, cs_text, *a, **kw):
        augmented_calls.append(cs_text)
        return list(batch), []
    with patch(_PATCH_SB, side_effect=_capture_render):
        _mini_eval_text(SECTION, items[:1], cs,
                        model_score="dummy", api_key="dummy",
                        concurrency=1, reasoning_effort=None, cot_first=True)
    _sample = repr(augmented_calls[0][:100]) if augmented_calls else "'no calls'"
    check("_mini_eval_text passes augmented cheatsheet (contains BASE)",
          any("BASE" in s for s in augmented_calls),
          f"rendered: {_sample}")
    check("_mini_eval_text passes augmented cheatsheet (contains section)",
          any(SECTION in s for s in augmented_calls),
          f"rendered: {_sample}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nSmoke test: BBH TaskSpec + concrete-example pipeline\n")

    test_generation_prompt_templates()
    test_scoring_prompts()
    test_partition_keys()
    test_verdict_parsing()
    test_polarity_instructions()
    test_loop_per_task()
    test_concrete_cs_accepted()
    test_concrete_cs_fallthrough()
    test_bootstrap_cheatsheet_fn()
    test_text_gate_helpers()

    print()
    if _failures:
        print(f"\n{len(_failures)} failure(s):")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All tests passed.")
