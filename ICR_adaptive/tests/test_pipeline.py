"""Smoke tests for ICR_adaptive/pipeline.py."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from ICR_adaptive.config import TaskConfig, PipelineConfig
from ICR_adaptive.components.multi_model_scorer import ItemScore
from ICR_adaptive.pipeline import AdaptivePipeline


def _make_task():
    return TaskConfig(
        domain_description="Test",
        input_fields=["eq1", "eq2"],
        answer_field="answer",
        verdict_pattern=r"(?i)\bVERDICT\s*:\s*(TRUE|FALSE)\b",
        answer_map={"True": "TRUE", "False": "FALSE"},
        truncation_token_threshold=5,
    )


def _make_pipe():
    return PipelineConfig(
        scoring_models=["model_a"],
        generator_model="gen",
        fix_rate_threshold=0.3,
        regress_threshold=0.5,
        regress_relative_factor=3.0,
        min_pool_for_regression=0,
        bin_threshold=1,
        n_candidates=1,
        utility_threshold=-999.0,
        lam=1.0, mu=0.0, nu=0.0,
    )


def test_pipeline_validate_called():
    """Invalid config should raise before run()."""
    task = TaskConfig(
        domain_description="  ",   # blank → validate raises
        input_fields=["f"],
        answer_field="a",
        verdict_pattern="x",
        answer_map={"a": "A"},
    )
    try:
        AdaptivePipeline(task, _make_pipe(),
                         score_fn=lambda m, i, s: [],
                         generate_fn=lambda p, m: "")
        assert False, "Should have raised"
    except ValueError:
        pass
    print("PASS: pipeline_validate_called")


def test_pipeline_run_end_to_end():
    """Full pipeline run with synthetic score/generate functions."""
    items = [
        {"id": "0", "eq1": "x", "eq2": "y", "answer": "True",
         "_verdict": "FALSE", "_response": "VERDICT: FALSE", "_token_count": 50},
        {"id": "1", "eq1": "a", "eq2": "b", "answer": "True",
         "_verdict": "TRUE", "_response": "VERDICT: TRUE", "_token_count": 50},
    ]

    def score_fn(model, items, sheet_text):
        return [
            ItemScore(item_id=str(i["id"]),
                      correct=("FIXED" in sheet_text or i["_verdict"] == "TRUE"))
            for i in items
        ]

    def generate_fn(prompt, model):
        return (
            "[STEP: parse_equations] FIXED\n"
            "[RULE: example_rule] fires.\n"
            "REASONING: test.\nPROOF: holds.\nCOUNTEREXAMPLE:\nVERDICT: TRUE\n"
        )

    pipeline = AdaptivePipeline(_make_task(), _make_pipe(), score_fn, generate_fn)
    result = pipeline.run(items, initial_sheet="BASE", max_iterations=3)

    assert result.final_sheet is not None
    assert isinstance(result.iterations, list)
    assert result.total_accepted >= 1
    print(f"PASS: pipeline_run_end_to_end (accepted={result.total_accepted})")


if __name__ == "__main__":
    test_pipeline_validate_called()
    test_pipeline_run_end_to_end()
    print("\nAll pipeline smoke tests passed.")
