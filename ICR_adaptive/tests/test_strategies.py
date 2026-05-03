"""Smoke tests for ICR_adaptive/prompts/strategies.py."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from ICR_adaptive.config import TaskConfig
from ICR_adaptive.components.failure_classifier import FailureType
from ICR_adaptive.prompts.strategies import (
    GenerationContext, PromptStrategy, build_prompt
)


def _make_cfg():
    return TaskConfig(
        domain_description="Binary decision: does A imply B over all magmas?",
        input_fields=["equation1", "equation2"],
        answer_field="answer",
        verdict_pattern=r"VERDICT:\s*(TRUE|FALSE)",
        answer_map={"True": "TRUE", "False": "FALSE"},
    )


def _make_ctx(**kwargs):
    defaults = dict(
        task_cfg=_make_cfg(),
        cheatsheet_text="STEP 0: BARE CHECK\n[STEP: bare_check] ...",
        item={"equation1": "x=y*x", "equation2": "x*x=x", "answer": "True"},
        model_response="VERDICT: FALSE",
        failure_type=FailureType.WRONG_ANSWER,
        divergence_step="motif_table",
        divergence_rule="M2",
    )
    defaults.update(kwargs)
    return GenerationContext(**defaults)


def test_direct_fix_contains_key_info():
    ctx = _make_ctx()
    prompt = build_prompt(ctx, PromptStrategy.DIRECT_FIX)
    assert "DOMAIN" in prompt
    assert "FAILURE TYPE" in prompt
    assert "WRONG_ANSWER" in prompt
    assert "motif_table" in prompt
    assert "equation1" in prompt
    print("PASS: direct_fix_contains_key_info")


def test_direct_fix_with_related_case():
    ctx = _make_ctx(related_case="EXAMPLE: here is how M2 fires.")
    prompt = build_prompt(ctx, PromptStrategy.DIRECT_FIX)
    assert "RELATED CASE STUDY" in prompt
    assert "M2 fires" in prompt
    print("PASS: direct_fix_with_related_case")


def test_oracle_guided_with_oracle():
    ctx = _make_ctx(oracle_trace="Step 0: bare(A)=TRUE. M2 fires → TRUE.")
    prompt = build_prompt(ctx, PromptStrategy.ORACLE_GUIDED)
    assert "ORACLE TRACE" in prompt
    assert "bare(A)=TRUE" in prompt
    assert "motif_table" in prompt
    print("PASS: oracle_guided_with_oracle")


def test_oracle_guided_without_oracle():
    ctx = _make_ctx()
    prompt = build_prompt(ctx, PromptStrategy.ORACLE_GUIDED)
    assert "not available" in prompt
    print("PASS: oracle_guided_without_oracle")


def test_contrast_prompt():
    ctx = _make_ctx()
    prompt = build_prompt(ctx, PromptStrategy.CONTRAST)
    assert "WRONG PATH" in prompt
    assert "CORRECT PATH" in prompt
    assert "motif_table" in prompt
    print("PASS: contrast_prompt")


def test_unknown_strategy_raises():
    ctx = _make_ctx()
    try:
        build_prompt(ctx, "nonexistent_strategy")
        assert False, "Should have raised"
    except (ValueError, KeyError):
        pass
    print("PASS: unknown_strategy_raises")


if __name__ == "__main__":
    test_direct_fix_contains_key_info()
    test_direct_fix_with_related_case()
    test_oracle_guided_with_oracle()
    test_oracle_guided_without_oracle()
    test_contrast_prompt()
    test_unknown_strategy_raises()
    print("\nAll strategies smoke tests passed.")
