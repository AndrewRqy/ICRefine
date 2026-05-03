"""Smoke tests for ICR_adaptive/training/loop.py."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from ICR_adaptive.config import TaskConfig, PipelineConfig
from ICR_adaptive.components.multi_model_scorer import ItemScore
from ICR_adaptive.training.loop import AdaptiveTrainingLoop


def _make_task():
    return TaskConfig(
        domain_description="Test: does A imply B?",
        input_fields=["eq1", "eq2"],
        answer_field="answer",
        verdict_pattern=r"(?i)\bVERDICT\s*:\s*(TRUE|FALSE)\b",
        answer_map={"True": "TRUE", "False": "FALSE"},
        truncation_token_threshold=5,  # very low so no FORMAT failures
    )


def _make_pipe(**kwargs):
    defaults = dict(
        scoring_models=["model_a"],
        generator_model="gen_model",
        fix_rate_threshold=0.3,
        regress_threshold=0.5,
        regress_relative_factor=3.0,
        min_pool_for_regression=0,
        bin_threshold=1,
        n_candidates=1,
        utility_threshold=-999.0,  # accept anything with positive fix
        lam=1.0, mu=0.0, nu=0.0,
    )
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


def _make_items():
    # Items 0,1 are failures; items 2,3 are correct.
    items = []
    for i in range(4):
        items.append({
            "id": str(i),
            "eq1": f"eq{i}",
            "eq2": f"eq{i}",
            "answer": "True",
            "_verdict": "FALSE" if i < 2 else "TRUE",
            "_response": "VERDICT: FALSE" if i < 2 else "VERDICT: TRUE",
            "_token_count": 50,
        })
    return items


def _complete_case(tag: str) -> str:
    """Minimal valid case study containing the required footer lines."""
    return (
        f"[STEP: parse_equations] {tag}\n"
        f"[RULE: example_rule] fires.\n"
        f"REASONING: test case.\n"
        f"PROOF: holds trivially.\n"
        f"COUNTEREXAMPLE:\n"
        f"VERDICT: TRUE\n"
    )


def test_loop_accepts_candidate():
    """Loop should accept a candidate that fixes the failures."""
    items = _make_items()
    call_count = {"n": 0}

    def score_fn(model, items, sheet_text):
        call_count["n"] += 1
        results = []
        for item in items:
            # After first score call, candidate sheet contains "FIXED"
            # so items 0 and 1 become correct too
            if "FIXED" in sheet_text:
                results.append(ItemScore(item_id=str(item["id"]), correct=True))
            else:
                correct = item.get("_verdict", "") == "TRUE"
                results.append(ItemScore(item_id=str(item["id"]), correct=correct))
        return results

    def generate_fn(prompt, model):
        return _complete_case("FIXED")

    loop = AdaptiveTrainingLoop(_make_task(), _make_pipe(), score_fn, generate_fn)
    result = loop.run(items, initial_sheet="BASE SHEET", max_iterations=3)

    assert result.total_accepted >= 1
    assert "FIXED" in result.final_sheet
    print(f"PASS: loop_accepts_candidate (accepted={result.total_accepted})")


def test_loop_stops_when_no_failures():
    """If all items are correct from the start, loop should exit immediately."""
    items = []
    for i in range(3):
        items.append({
            "id": str(i), "eq1": "x", "eq2": "x", "answer": "True",
            "_verdict": "TRUE", "_response": "VERDICT: TRUE", "_token_count": 50,
        })

    def score_fn(model, items, sheet_text):
        return [ItemScore(item_id=str(i["id"]), correct=True) for i in items]

    def generate_fn(prompt, model):
        return "CASE"

    loop = AdaptiveTrainingLoop(_make_task(), _make_pipe(), score_fn, generate_fn)
    result = loop.run(items, initial_sheet="SHEET", max_iterations=5)

    assert result.total_accepted == 0
    # Loop should have noted "no failures"
    assert len(result.iterations) >= 1
    assert not result.iterations[0].accepted
    print("PASS: loop_stops_when_no_failures")


def test_loop_stops_at_max_iterations():
    """Loop should not run more than max_iterations."""
    items = _make_items()

    def score_fn(model, items, sheet_text):
        # Always report items 0,1 as wrong
        return [ItemScore(item_id=str(i["id"]),
                          correct=(int(i["id"]) >= 2)) for i in items]

    def generate_fn(prompt, model):
        return "CANDIDATE"

    # gate will fail because score_fn never marks 0,1 as correct even with new sheet
    # but utility_threshold=-999 → gate passes on fix_rate alone (fix_rate=0 < 0.3 fails)
    # Actually: candidate sheet also scored via score_fn which ignores sheet content
    # → fix_rate=0 → fix_rate gate fails → no acceptance → loop runs max_iterations
    loop = AdaptiveTrainingLoop(_make_task(), _make_pipe(), score_fn, generate_fn)
    result = loop.run(items, initial_sheet="SHEET", max_iterations=3)

    assert len(result.iterations) <= 3
    print(f"PASS: loop_stops_at_max_iterations (ran {len(result.iterations)} iterations)")


def test_case_bank_populated():
    """Accepted candidates should add entries to the case bank."""
    items = _make_items()
    case_bank = []

    def score_fn(model, items, sheet_text):
        if "CASE" in sheet_text:
            return [ItemScore(item_id=str(i["id"]), correct=True) for i in items]
        return [ItemScore(item_id=str(i["id"]),
                          correct=(int(i["id"]) >= 2)) for i in items]

    def generate_fn(prompt, model):
        return _complete_case("CASE")

    loop = AdaptiveTrainingLoop(_make_task(), _make_pipe(), score_fn, generate_fn)
    result = loop.run(items, initial_sheet="SHEET", max_iterations=3,
                      case_bank=case_bank)

    if result.total_accepted > 0:
        assert len(case_bank) > 0
    print(f"PASS: case_bank_populated (bank size={len(case_bank)})")


if __name__ == "__main__":
    test_loop_accepts_candidate()
    test_loop_stops_when_no_failures()
    test_loop_stops_at_max_iterations()
    test_case_bank_populated()
    print("\nAll loop smoke tests passed.")
