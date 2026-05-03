"""Smoke tests for ICR_adaptive/components/multi_model_scorer.py."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from ICR_adaptive.components.multi_model_scorer import (
    MultiModelScorer, ItemScore, ScorerResult, ModelResult
)


def _make_items(n=5):
    return [{"id": str(i)} for i in range(n)]


def _fixed_score_fn(correct_map):
    """Returns a score_fn that marks items correct/wrong per the map."""
    def score_fn(model, items, sheet_text):
        results = []
        for item in items:
            iid = item["id"]
            results.append(ItemScore(item_id=iid, correct=correct_map.get(iid, False)))
        return results
    return score_fn


def test_single_model_all_correct():
    items = _make_items(3)
    scorer = MultiModelScorer(
        models=["model_a"],
        score_fn=_fixed_score_fn({"0": True, "1": True, "2": True}),
    )
    result = scorer.score(items, "sheet")
    mr = result.per_model["model_a"]
    assert mr.n_correct == 3
    assert mr.n_total == 3
    assert abs(mr.accuracy - 1.0) < 1e-9
    print("PASS: single_model_all_correct")


def test_single_model_partial():
    items = _make_items(4)
    scorer = MultiModelScorer(
        models=["model_a"],
        score_fn=_fixed_score_fn({"0": True, "1": False, "2": True, "3": False}),
    )
    result = scorer.score(items, "sheet")
    mr = result.per_model["model_a"]
    assert mr.n_correct == 2
    assert mr.wrong_ids == ["1", "3"]
    assert abs(mr.accuracy - 0.5) < 1e-9
    print("PASS: single_model_partial")


def test_multi_model():
    items = _make_items(2)
    def score_fn(model, items, sheet):
        if model == "good_model":
            return [ItemScore(i["id"], True) for i in items]
        return [ItemScore(i["id"], False) for i in items]

    scorer = MultiModelScorer(models=["good_model", "bad_model"], score_fn=score_fn)
    result = scorer.score(items, "sheet")
    assert result.per_model["good_model"].n_correct == 2
    assert result.per_model["bad_model"].n_correct == 0
    assert abs(result.worst_accuracy() - 0.0) < 1e-9
    print("PASS: multi_model")


def test_all_pass_fix_rate():
    # bin = items 0, 1; model fixes both
    result = ScorerResult(per_model={
        "m": ModelResult("m", n_correct=2, n_total=4,
                         correct_ids=["0", "1"], wrong_ids=["2", "3"])
    })
    assert result.all_pass_fix_rate(["0", "1"], threshold=1.0)
    assert result.all_pass_fix_rate(["0", "1", "2"], threshold=0.5)
    assert not result.all_pass_fix_rate(["0", "1", "2"], threshold=0.9)
    print("PASS: all_pass_fix_rate")


def test_any_regresses():
    pool = ["10", "11", "12", "13", "14"]
    result = ScorerResult(per_model={
        "m": ModelResult("m", n_correct=3, n_total=8,
                         correct_ids=["10", "11", "12"],
                         wrong_ids=["13", "14", "0", "1", "2"])
    })
    # 2 pool items broken, limit=1 → regresses
    assert result.any_regresses(pool, regress_limit=1)
    # 2 broken, limit=2 → OK
    assert not result.any_regresses(pool, regress_limit=2)
    print("PASS: any_regresses")


def test_empty_models_raises():
    try:
        MultiModelScorer(models=[], score_fn=lambda m, i, s: [])
        assert False, "Should have raised"
    except ValueError:
        pass
    print("PASS: empty_models_raises")


if __name__ == "__main__":
    test_single_model_all_correct()
    test_single_model_partial()
    test_multi_model()
    test_all_pass_fix_rate()
    test_any_regresses()
    test_empty_models_raises()
    print("\nAll multi_model_scorer smoke tests passed.")
