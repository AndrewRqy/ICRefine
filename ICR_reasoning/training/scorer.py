"""
ICR_reasoning/training/scorer.py — Re-exports from the unified scorer in ICR_naive.

The unified scorer lives in ICR_naive/training/scorer.py and already captures
post_think and thinking per item. Import from there directly in new code.
"""

from ICR_naive.training.scorer import (  # noqa: F401
    score_batch,
    test_cheatsheet,
    TestResult,
)
