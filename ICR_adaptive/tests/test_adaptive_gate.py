"""Smoke tests for ICR_adaptive/training/adaptive_gate.py."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from ICR_adaptive.config import PipelineConfig
from ICR_adaptive.components.multi_model_scorer import ScorerResult, ModelResult
from ICR_adaptive.training.adaptive_gate import AdaptiveRegressionGate


def _make_pipe(**kwargs):
    defaults = dict(
        scoring_models=["m"],
        generator_model="g",
        fix_rate_threshold=0.5,
        regress_threshold=0.15,
        regress_relative_factor=2.0,
        min_pool_for_regression=5,
        lam=0.5,
        mu=1.0,
        nu=0.1,
        utility_threshold=0.0,
    )
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


def _make_result(correct_ids, wrong_ids, model="m"):
    return ScorerResult(per_model={
        model: ModelResult(
            model=model,
            n_correct=len(correct_ids),
            n_total=len(correct_ids) + len(wrong_ids),
            correct_ids=correct_ids,
            wrong_ids=wrong_ids,
        )
    })


def test_passes_when_gates_clear():
    pipe = _make_pipe()
    gate = AdaptiveRegressionGate(pipe)
    # bin = [0,1,2], all fixed; pool = [10,11,12], none broken
    result = _make_result(correct_ids=["0","1","2","10","11","12"], wrong_ids=[])
    gr = gate.evaluate(result, bin_ids=["0","1","2"], pool_ids=["10","11","12"],
                       candidate_text="abc", current_text="abc")
    assert gr.passed, f"Expected pass, got reason={gr.reason}"
    print("PASS: passes_when_gates_clear")


def test_fails_fix_rate():
    pipe = _make_pipe(fix_rate_threshold=0.8)
    gate = AdaptiveRegressionGate(pipe)
    # bin=[0,1,2], only 0 fixed → fix_rate=1/3 < 0.8
    result = _make_result(correct_ids=["0"], wrong_ids=["1","2"])
    gr = gate.evaluate(result, bin_ids=["0","1","2"], pool_ids=[],
                       candidate_text="x", current_text="x")
    assert not gr.passed
    assert gr.reason == "fix_rate"
    print("PASS: fails_fix_rate")


def test_fails_regression():
    pipe = _make_pipe(fix_rate_threshold=0.3, regress_threshold=0.1,
                      regress_relative_factor=1.0, min_pool_for_regression=5)
    gate = AdaptiveRegressionGate(pipe)
    # bin=[0,1,2], fix 1 → fix_rate=1/3 ✓
    # pool=[10..19] (10 items), 5 broken → limit = max(ceil(0.1*10), ceil(1*1)) = max(1,1)=1
    # 5 > 1 → regression fail
    pool = [str(i) for i in range(10, 20)]
    correct = ["0"] + [str(i) for i in range(10, 15)]   # pool items 10-14 correct
    wrong = ["1", "2"] + [str(i) for i in range(15, 20)]  # pool items 15-19 broken
    result = _make_result(correct_ids=correct, wrong_ids=wrong)
    gr = gate.evaluate(result, bin_ids=["0","1","2"], pool_ids=pool,
                       candidate_text="x", current_text="x")
    assert not gr.passed
    assert gr.reason == "regression"
    print("PASS: fails_regression")


def test_utility_negative_fails():
    # Very high regression cost → negative utility.
    # Set min_pool_for_regression high so regression gate is skipped;
    # only the utility gate should block.
    pipe = _make_pipe(fix_rate_threshold=0.3, mu=100.0, utility_threshold=0.0,
                      min_pool_for_regression=1000)
    gate = AdaptiveRegressionGate(pipe)
    pool = ["10","11","12","13","14"]
    # fix bin item 0, break all 5 pool items
    result = _make_result(correct_ids=["0"], wrong_ids=["1","2","10","11","12","13","14"])
    gr = gate.evaluate(result, bin_ids=["0","1","2"], pool_ids=pool,
                       candidate_text="x", current_text="x")
    assert not gr.passed, f"Expected fail, got passed=True"
    assert gr.reason == "utility", f"Expected utility, got {gr.reason}"
    print("PASS: utility_negative_fails")


def test_empty_bin_fails():
    pipe = _make_pipe()
    gate = AdaptiveRegressionGate(pipe)
    result = _make_result(correct_ids=["0"], wrong_ids=[])
    gr = gate.evaluate(result, bin_ids=[], pool_ids=["0"],
                       candidate_text="x", current_text="x")
    assert not gr.passed
    print("PASS: empty_bin_fails")


def test_pool_below_min_skips_regression():
    # pool size < min_pool_for_regression → skip regression gate
    pipe = _make_pipe(fix_rate_threshold=0.3, min_pool_for_regression=10)
    gate = AdaptiveRegressionGate(pipe)
    # pool has only 3 items (< 10), and all broken — gate should still pass
    result = _make_result(correct_ids=["0"], wrong_ids=["1","2","p0","p1","p2"])
    gr = gate.evaluate(result, bin_ids=["0","1","2"], pool_ids=["p0","p1","p2"],
                       candidate_text="x", current_text="x")
    # fix_rate = 1/3 ≥ 0.3 ✓; regression skipped; utility might still fail
    # just check regression did not block it
    assert gr.reason != "regression"
    print("PASS: pool_below_min_skips_regression")


if __name__ == "__main__":
    test_passes_when_gates_clear()
    test_fails_fix_rate()
    test_fails_regression()
    test_utility_negative_fails()
    test_empty_bin_fails()
    test_pool_below_min_skips_regression()
    print("\nAll adaptive_gate smoke tests passed.")
