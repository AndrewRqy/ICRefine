"""Smoke tests for ICR_adaptive/components/format_filter.py."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from ICR_adaptive.config import TaskConfig
from ICR_adaptive.components.format_filter import FormatFilter


def _make_cfg(threshold=100):
    return TaskConfig(
        domain_description="Test",
        input_fields=["eq1", "eq2"],
        answer_field="answer",
        verdict_pattern=r"(?i)\bVERDICT\s*:\s*(TRUE|FALSE)\b",
        answer_map={"True": "TRUE", "False": "FALSE"},
        truncation_token_threshold=threshold,
    )


def test_empty_response():
    ff = FormatFilter(_make_cfg())
    r = ff.check("", token_count=0)
    assert not r.usable
    assert r.reason == "empty"
    r2 = ff.check("   \n  ", token_count=50)
    assert not r2.usable
    assert r2.reason == "empty"
    print("PASS: empty_response")


def test_truncated():
    ff = FormatFilter(_make_cfg(threshold=100))
    long_text = "Some reasoning without a verdict."
    r = ff.check(long_text, token_count=50)
    assert not r.usable
    assert r.reason == "truncated"
    print("PASS: truncated")


def test_no_verdict():
    ff = FormatFilter(_make_cfg(threshold=10))
    r = ff.check("Some reasoning text without verdict line.", token_count=20)
    assert not r.usable
    assert r.reason == "no_verdict"
    print("PASS: no_verdict")


def test_valid_true():
    ff = FormatFilter(_make_cfg(threshold=10))
    resp = "REASONING: something.\nPROOF: proof.\nCOUNTEREXAMPLE:\nVERDICT: TRUE"
    r = ff.check(resp, token_count=30)
    assert r.usable
    assert r.verdict == "TRUE"
    assert r.reason == "ok"
    print("PASS: valid_true")


def test_valid_false():
    ff = FormatFilter(_make_cfg(threshold=10))
    resp = "REASONING: nope.\nPROOF:\nCOUNTEREXAMPLE: a=1,b=2\nVERDICT: FALSE"
    r = ff.check(resp, token_count=30)
    assert r.usable
    assert r.verdict == "FALSE"
    print("PASS: valid_false")


def test_case_insensitive_verdict():
    ff = FormatFilter(_make_cfg(threshold=10))
    resp = "verdict: true"
    r = ff.check(resp, token_count=15)
    assert r.usable
    assert r.verdict.upper() == "TRUE"
    print("PASS: case_insensitive_verdict")


def test_token_count_stored():
    ff = FormatFilter(_make_cfg(threshold=10))
    r = ff.check("VERDICT: TRUE", token_count=42)
    assert r.token_count == 42
    print("PASS: token_count_stored")


if __name__ == "__main__":
    test_empty_response()
    test_truncated()
    test_no_verdict()
    test_valid_true()
    test_valid_false()
    test_case_insensitive_verdict()
    test_token_count_stored()
    print("\nAll format_filter smoke tests passed.")
