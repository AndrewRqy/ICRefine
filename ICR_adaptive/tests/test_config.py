"""Smoke tests for ICR_adaptive/config.py."""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from ICR_adaptive.config import TaskConfig, PipelineConfig


def test_task_config_required_only():
    cfg = TaskConfig(
        domain_description="Test domain",
        input_fields=["eq1", "eq2"],
        answer_field="answer",
        verdict_pattern=r"(?i)\bVERDICT\s*:\s*(TRUE|FALSE)\b",
        answer_map={"True": "TRUE", "False": "FALSE"},
    )
    assert cfg.domain_description == "Test domain"
    assert cfg.input_fields == ["eq1", "eq2"]
    assert cfg.answer_field == "answer"
    assert cfg.truncation_token_threshold == 100
    assert len(cfg.abandonment_phrases) > 0
    assert cfg.partition_fn is None
    assert cfg.query_features_fn is None
    print("PASS: task_config_required_only")


def test_task_config_with_partition_fn():
    def my_partition(item):
        return (item.get("domain", "x"),)

    cfg = TaskConfig(
        domain_description="D",
        input_fields=["f1"],
        answer_field="ans",
        verdict_pattern=r"VERDICT:\s*(YES|NO)",
        answer_map={"yes": "YES", "no": "NO"},
        partition_fn=my_partition,
    )
    key = cfg.base_partition_key({"domain": "math", "ans": "yes"})
    assert key == ("math",), f"Expected ('math',) got {key}"
    print("PASS: task_config_with_partition_fn")


def test_task_config_base_partition_key_default():
    cfg = TaskConfig(
        domain_description="D",
        input_fields=["f1"],
        answer_field="ans",
        verdict_pattern=r"VERDICT:\s*(YES|NO)",
        answer_map={"yes": "YES", "no": "NO"},
    )
    key = cfg.base_partition_key({"ans": "yes"})
    assert key == ("yes",), f"Expected ('yes',) got {key}"
    key_missing = cfg.base_partition_key({})
    assert key_missing == ("unknown",), f"Expected ('unknown',) got {key_missing}"
    print("PASS: task_config_base_partition_key_default")


def test_task_config_query_features():
    cfg = TaskConfig(
        domain_description="D",
        input_fields=["f1"],
        answer_field="ans",
        verdict_pattern=r"VERDICT:\s*(YES|NO)",
        answer_map={"yes": "YES", "no": "NO"},
    )
    feats = cfg.query_features({"ans": "True"})
    assert feats == ["answer_TRUE"], f"Got {feats}"
    print("PASS: task_config_query_features")


def test_task_config_positive_verdict():
    cfg = TaskConfig(
        domain_description="D",
        input_fields=["f1"],
        answer_field="ans",
        verdict_pattern=r"VERDICT:\s*(TRUE|FALSE)",
        answer_map={"True": "TRUE", "False": "FALSE"},
    )
    assert cfg.positive_verdict() == "TRUE"

    cfg2 = TaskConfig(
        domain_description="D",
        input_fields=["f1"],
        answer_field="ans",
        verdict_pattern=r"VERDICT:\s*(ENFORCEABLE|NOT ENFORCEABLE)",
        answer_map={"yes": "ENFORCEABLE", "no": "NOT ENFORCEABLE"},
    )
    assert cfg2.positive_verdict() == "ENFORCEABLE"
    print("PASS: task_config_positive_verdict")


def test_task_config_validate_raises():
    # missing domain_description
    try:
        cfg = TaskConfig(
            domain_description="  ",
            input_fields=["f"],
            answer_field="a",
            verdict_pattern="x",
            answer_map={"a": "A"},
        )
        cfg.validate()
        assert False, "Should have raised"
    except ValueError as e:
        assert "domain_description" in str(e)

    # missing input_fields
    try:
        cfg = TaskConfig(
            domain_description="D",
            input_fields=[],
            answer_field="a",
            verdict_pattern="x",
            answer_map={"a": "A"},
        )
        cfg.validate()
        assert False, "Should have raised"
    except ValueError as e:
        assert "input_fields" in str(e)

    print("PASS: task_config_validate_raises")


def test_pipeline_config_required_only():
    pipe = PipelineConfig(
        scoring_models=["openai/gpt-4o"],
        generator_model="openai/gpt-4o",
    )
    assert pipe.primary_model() == "openai/gpt-4o"
    assert pipe.fix_rate_threshold == 0.30
    assert pipe.regress_threshold == 0.15
    assert pipe.min_pool_for_regression == 10
    print("PASS: pipeline_config_required_only")


def test_adaptive_regress_limit():
    pipe = PipelineConfig(
        scoring_models=["m"],
        generator_model="g",
        regress_threshold=0.15,
        regress_relative_factor=2.0,
    )
    # abs_limit = ceil(0.15 * 100) = 15, rel_limit = ceil(5 * 2.0) = 10 → max = 15
    assert pipe.adaptive_regress_limit(pool_size=100, fix_count=5) == 15
    # abs_limit = ceil(0.15 * 20) = 3, rel_limit = ceil(10 * 2.0) = 20 → max = 20
    assert pipe.adaptive_regress_limit(pool_size=20, fix_count=10) == 20
    # abs_limit = ceil(0.15 * 10) = 2, rel_limit = ceil(3 * 2.0) = 6 → max = 6
    assert pipe.adaptive_regress_limit(pool_size=10, fix_count=3) == 6
    print("PASS: adaptive_regress_limit")


def test_pipeline_config_validate_raises():
    try:
        pipe = PipelineConfig(scoring_models=[], generator_model="g")
        pipe.validate()
        assert False, "Should have raised"
    except ValueError as e:
        assert "scoring_models" in str(e)

    try:
        pipe = PipelineConfig(
            scoring_models=["m"],
            generator_model="g",
            fix_rate_threshold=1.5,
        )
        pipe.validate()
        assert False, "Should have raised"
    except ValueError as e:
        assert "fix_rate_threshold" in str(e)

    try:
        pipe = PipelineConfig(
            scoring_models=["m"],
            generator_model="g",
            regress_threshold=0.0,
        )
        pipe.validate()
        assert False, "Should have raised"
    except ValueError as e:
        assert "regress_threshold" in str(e)

    print("PASS: pipeline_config_validate_raises")


if __name__ == "__main__":
    test_task_config_required_only()
    test_task_config_with_partition_fn()
    test_task_config_base_partition_key_default()
    test_task_config_query_features()
    test_task_config_positive_verdict()
    test_task_config_validate_raises()
    test_pipeline_config_required_only()
    test_adaptive_regress_limit()
    test_pipeline_config_validate_raises()
    print("\nAll config smoke tests passed.")
