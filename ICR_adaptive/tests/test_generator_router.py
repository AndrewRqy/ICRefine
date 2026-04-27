"""Smoke tests for ICR_adaptive/components/generator_router.py."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from ICR_adaptive.config import TaskConfig
from ICR_adaptive.components.generator_router import GeneratorRouter, _jaccard


def _make_cfg():
    return TaskConfig(
        domain_description="Test",
        input_fields=["eq1", "eq2"],
        answer_field="answer",
        verdict_pattern=r"VERDICT:\s*(TRUE|FALSE)",
        answer_map={"True": "TRUE", "False": "FALSE"},
    )


def _case(features, text="case text", bin_key=("WRONG_ANSWER", "motif_table")):
    return {"features": features, "case_text": text, "bin_key": bin_key}


def test_empty_bank():
    router = GeneratorRouter(_make_cfg())
    assert router.route({"answer": "True"}, []) is None
    print("PASS: empty_bank")


def test_exact_match():
    router = GeneratorRouter(_make_cfg())
    bank = [_case(["answer_TRUE"], "case A"), _case(["answer_FALSE"], "case B")]
    query = {"answer": "True"}  # features → ["answer_TRUE"]
    result = router.route(query, bank)
    assert result is not None
    assert result["case_text"] == "case A"
    print("PASS: exact_match")


def test_best_partial_match():
    router = GeneratorRouter(_make_cfg())
    bank = [
        _case(["answer_TRUE", "domain_math"], "case A"),
        _case(["answer_FALSE"], "case B"),
        _case(["answer_TRUE", "extra"], "case C"),
    ]
    query = {"answer": "True"}
    result = router.route(query, bank)
    # case A: jaccard({"answer_TRUE"}, {"answer_TRUE","domain_math"}) = 1/2
    # case C: jaccard({"answer_TRUE"}, {"answer_TRUE","extra"}) = 1/2
    # Both equal; first wins (stable sort)
    assert result["case_text"] in ("case A", "case C")
    print("PASS: best_partial_match")


def test_min_similarity_filters():
    router = GeneratorRouter(_make_cfg())
    bank = [_case(["unrelated_token"], "case X")]
    query = {"answer": "True"}
    # jaccard = 0 → below 0.5 threshold
    result = router.route(query, bank, min_similarity=0.5)
    assert result is None
    print("PASS: min_similarity_filters")


def test_route_top_k():
    router = GeneratorRouter(_make_cfg())
    bank = [
        _case(["answer_TRUE"], "A"),
        _case(["answer_FALSE"], "B"),
        _case(["answer_TRUE", "x"], "C"),
    ]
    query = {"answer": "True"}
    top2 = router.route_top_k(query, bank, k=2)
    texts = [e["case_text"] for e in top2]
    assert "A" in texts or "C" in texts
    assert "B" not in texts
    print("PASS: route_top_k")


def test_jaccard_helper():
    assert _jaccard(set(), set()) == 1.0
    assert _jaccard({"a"}, {"a"}) == 1.0
    assert _jaccard({"a"}, {"b"}) == 0.0
    assert abs(_jaccard({"a", "b"}, {"b", "c"}) - 1/3) < 1e-9
    print("PASS: jaccard_helper")


if __name__ == "__main__":
    test_empty_bank()
    test_exact_match()
    test_best_partial_match()
    test_min_similarity_filters()
    test_route_top_k()
    test_jaccard_helper()
    print("\nAll generator_router smoke tests passed.")
