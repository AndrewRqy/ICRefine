"""Smoke tests for ICR_adaptive/components/execution_parser.py."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from ICR_adaptive.config import TaskConfig
from ICR_adaptive.components.execution_parser import ExecutionPathParser


def _make_cfg():
    return TaskConfig(
        domain_description="Test",
        input_fields=["eq1", "eq2"],
        answer_field="answer",
        verdict_pattern=r"VERDICT:\s*(TRUE|FALSE)",
        answer_map={"True": "TRUE", "False": "FALSE"},
    )


def test_no_annotations_returns_unknown():
    parser = ExecutionPathParser(_make_cfg())
    result = parser.parse("Some response with no markers.")
    assert result.step == "unknown"
    assert result.rule == "unknown"
    print("PASS: no_annotations_returns_unknown")


def test_last_step_no_oracle():
    parser = ExecutionPathParser(_make_cfg())
    response = "[STEP: bare_check] did something.\n[STEP: motif_table] fired M2.\nVERDICT: TRUE"
    result = parser.parse(response)
    assert result.step == "motif_table", f"Got {result.step}"
    print("PASS: last_step_no_oracle")


def test_last_rule_no_oracle():
    parser = ExecutionPathParser(_make_cfg())
    response = "[STEP: step2b] probes.\n[RULE: P1] passes.\n[RULE: P3] fails."
    result = parser.parse(response)
    assert result.rule == "P3", f"Got {result.rule}"
    print("PASS: last_rule_no_oracle")


def test_oracle_diverge_on_step():
    parser = ExecutionPathParser(_make_cfg())
    oracle = "[STEP: bare_check] x.\n[STEP: motif_table] M1 fires.\n[STEP: conclude] TRUE."
    model  = "[STEP: bare_check] x.\nI get confused here.\nVERDICT: FALSE"
    result = parser.parse(model, oracle_trace=oracle)
    # motif_table is absent in model → divergence at motif_table
    assert result.step == "motif_table", f"Got {result.step}"
    print("PASS: oracle_diverge_on_step")


def test_oracle_all_steps_present():
    parser = ExecutionPathParser(_make_cfg())
    oracle = "[STEP: bare_check] x.\n[STEP: motif_table] M1."
    model  = "[STEP: bare_check] ok.\n[STEP: motif_table] also ok."
    result = parser.parse(model, oracle_trace=oracle)
    # no divergence → "unknown"
    assert result.step == "unknown", f"Got {result.step}"
    print("PASS: oracle_all_steps_present")


def test_steps_in_helper():
    parser = ExecutionPathParser(_make_cfg())
    text = "[STEP: alpha] ...\n[STEP: beta] ..."
    assert parser.steps_in(text) == ["alpha", "beta"]
    print("PASS: steps_in_helper")


def test_rules_in_helper():
    parser = ExecutionPathParser(_make_cfg())
    text = "[RULE: LP check] ...\n[RULE: RP check] ..."
    assert parser.rules_in(text) == ["LP check", "RP check"]
    print("PASS: rules_in_helper")


if __name__ == "__main__":
    test_no_annotations_returns_unknown()
    test_last_step_no_oracle()
    test_last_rule_no_oracle()
    test_oracle_diverge_on_step()
    test_oracle_all_steps_present()
    test_steps_in_helper()
    test_rules_in_helper()
    print("\nAll execution_parser smoke tests passed.")
