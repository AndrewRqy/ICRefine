"""Smoke tests for ICR_adaptive/components/failure_classifier.py."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from ICR_adaptive.config import TaskConfig
from ICR_adaptive.components.failure_classifier import FailureClassifier, FailureType


def _make_cfg():
    return TaskConfig(
        domain_description="Test",
        input_fields=["eq1", "eq2"],
        answer_field="answer",
        verdict_pattern=r"(?i)\bVERDICT\s*:\s*(TRUE|FALSE)\b",
        answer_map={"True": "TRUE", "False": "FALSE"},
        truncation_token_threshold=100,
        abandonment_phrases=["requires further analysis", "without completing each step"],
    )


def _classify(clf, verdict, gt, response, tokens=200):
    return clf.classify(verdict, gt, response, token_count=tokens,
                        truncation_threshold=100)


def test_correct_true():
    clf = FailureClassifier(_make_cfg())
    r = _classify(clf, "TRUE", "True", "VERDICT: TRUE")
    assert r.failure_type == FailureType.CORRECT
    assert r.verdict == "TRUE"
    assert r.expected == "TRUE"
    print("PASS: correct_true")


def test_correct_false():
    clf = FailureClassifier(_make_cfg())
    r = _classify(clf, "FALSE", "False", "VERDICT: FALSE")
    assert r.failure_type == FailureType.CORRECT
    print("PASS: correct_false")


def test_wrong_answer_no_abandonment():
    clf = FailureClassifier(_make_cfg())
    r = _classify(clf, "FALSE", "True", "Some reasoning. VERDICT: FALSE")
    assert r.failure_type == FailureType.WRONG_ANSWER
    assert r.matched_phrase == ""
    print("PASS: wrong_answer_no_abandonment")


def test_abandonment_detected():
    clf = FailureClassifier(_make_cfg())
    response = ("The problem requires further analysis to confirm. "
                "VERDICT: FALSE")
    r = _classify(clf, "FALSE", "True", response)
    assert r.failure_type == FailureType.ABANDONMENT
    assert "requires further analysis" in r.matched_phrase
    print("PASS: abandonment_detected")


def test_format_low_tokens():
    clf = FailureClassifier(_make_cfg())
    r = _classify(clf, "TRUE", "True", "VERDICT: TRUE", tokens=50)
    assert r.failure_type == FailureType.FORMAT
    print("PASS: format_low_tokens")


def test_label_normalisation():
    clf = FailureClassifier(_make_cfg())
    # "true" lower-case label → should map to "TRUE"
    r = _classify(clf, "TRUE", "true", "VERDICT: TRUE")
    assert r.failure_type == FailureType.CORRECT, f"Got {r.failure_type}, expected={r.expected}"
    print("PASS: label_normalisation")


def test_abandonment_not_triggered_on_correct():
    # Even if response contains abandonment phrase, CORRECT takes priority.
    clf = FailureClassifier(_make_cfg())
    response = ("requires further analysis but here we go. VERDICT: TRUE")
    r = _classify(clf, "TRUE", "True", response)
    # correct answer → CORRECT, abandonment check never runs
    assert r.failure_type == FailureType.CORRECT
    print("PASS: abandonment_not_triggered_on_correct")


if __name__ == "__main__":
    test_correct_true()
    test_correct_false()
    test_wrong_answer_no_abandonment()
    test_abandonment_detected()
    test_format_low_tokens()
    test_label_normalisation()
    test_abandonment_not_triggered_on_correct()
    print("\nAll failure_classifier smoke tests passed.")
