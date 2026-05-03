"""
ICR_adaptive/config.py — Task and pipeline configuration for the adaptive mode.

Two dataclasses:

  TaskConfig    — everything that is domain-specific (what the task is,
                  how to parse inputs/outputs, how to detect failures).
                  Five fields are REQUIRED; everything else has a sensible default.

  PipelineConfig — everything that controls how the training loop runs
                  (models, gate thresholds, concurrency, etc.).
                  Two fields are REQUIRED (scoring_models, generator_model).

Minimum viable usage for a new domain
--------------------------------------
    cfg = TaskConfig(
        domain_description="Classify whether a contract clause is enforceable.",
        input_fields=["clause_text", "jurisdiction"],
        answer_field="enforceable",
        verdict_pattern=r"(?i)\\bVERDICT\\s*:\\s*(ENFORCEABLE|NOT ENFORCEABLE)\\b",
        answer_map={"yes": "ENFORCEABLE", "no": "NOT ENFORCEABLE"},
    )
    pipe = PipelineConfig(
        scoring_models=["openai/gpt-4o"],
        generator_model="openai/gpt-4o",
    )

Magma-equation usage (backward-compatible with ICR_select)
-----------------------------------------------------------
    from ICR_partition.training.partition import compute_partition_key
    cfg = TaskConfig(
        domain_description="Binary decision: does equation A imply equation B "
                           "over all magmas (binary operation *)?",
        input_fields=["equation1", "equation2"],
        answer_field="answer",
        verdict_pattern=r"(?i)\\bVERDICT\\s*[:：]\\s*(TRUE|FALSE)\\b",
        answer_map={"True": "TRUE", "False": "FALSE"},
        partition_fn=compute_partition_key,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_ABANDONMENT_PHRASES: List[str] = [
    "given the complexity",
    "without explicit calculation",
    "requires further analysis",
    "the process requires",
    "a detailed analysis would",
    "due to the format",
    "without completing each step",
    "this requires careful consideration",
    "a comprehensive analysis",
    "without more information",
]


# ---------------------------------------------------------------------------
# TaskConfig
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    """
    Describes the task being solved.  Domain-specific; must be set per deployment.

    Required fields
    ---------------
    domain_description : str
        Free-text description injected into case-study generation prompts so the
        LLM generator understands what kind of reasoning is expected.
        e.g. "Binary decision: does equation A imply equation B over all magmas?"

    input_fields : List[str]
        Keys in the JSONL that form the problem input (in display order).
        e.g. ["equation1", "equation2"]

    answer_field : str
        Key in the JSONL that holds the ground-truth label.
        e.g. "answer"

    verdict_pattern : str
        Regex (as a raw string) used to extract the model's answer from its response.
        Must contain exactly one capture group that yields a value present in answer_map.
        e.g. r"(?i)\\bVERDICT\\s*:\\s*(TRUE|FALSE)\\b"

    answer_map : Dict[str, str]
        Maps ground-truth label values (as they appear in the JSONL) to the
        corresponding verdict_pattern capture values.
        e.g. {"True": "TRUE", "False": "FALSE"}

    Optional fields
    ---------------
    step_annotation_pattern / rule_annotation_pattern : str
        Regexes for parsing step and rule labels from an annotated cheatsheet.
        When the cheatsheet contains [STEP: name] / [RULE: name] markers,
        ExecutionPathParser uses these to identify where the model diverged.
        Falls back to "unknown" divergence step when annotations are absent.

    truncation_token_threshold : int
        Responses with fewer output tokens than this are classified as
        FORMAT / TRUNCATED rather than logic failures.  Default 100.

    abandonment_phrases : List[str]
        Phrases that, when found in a full-length response, signal that the
        model abandoned the cheatsheet protocol.  Override or extend with
        domain-specific variants.

    partition_fn : Optional[Callable[[dict], tuple]]
        Function mapping a problem dict to a domain-specific partition key tuple.
        The adaptive loop appends (failure_type, divergence_step) to this tuple
        to form the full bin key.
        If None, the generic key (expected_answer,) is used as the base.

    query_features_fn : Optional[Callable[[dict], List[str]]]
        Function mapping a problem dict to a list of feature tokens used for
        case-study routing at inference time (Jaccard similarity).
        If None, generic tokens ["answer_TRUE", "answer_FALSE"] are used.
    """

    # ── REQUIRED ──────────────────────────────────────────────────────────
    domain_description: str
    input_fields: List[str]
    answer_field: str
    verdict_pattern: str
    answer_map: Dict[str, str]

    # ── OPTIONAL: cheatsheet structure ────────────────────────────────────
    step_annotation_pattern: str = r"\[STEP:\s*([^\]]+)\]"
    rule_annotation_pattern: str = r"\[RULE:\s*([^\]]+)\]"

    # ── OPTIONAL: failure detection ───────────────────────────────────────
    truncation_token_threshold: int = 100
    abandonment_phrases: List[str] = field(
        default_factory=lambda: list(_DEFAULT_ABANDONMENT_PHRASES)
    )

    # ── OPTIONAL: domain-specific routing ─────────────────────────────────
    partition_fn: Optional[Callable[[dict], tuple]] = None
    query_features_fn: Optional[Callable[[dict], List[str]]] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def base_partition_key(self, item: dict) -> tuple:
        """Return the domain-specific component of the partition key."""
        if self.partition_fn is not None:
            return self.partition_fn(item)
        expected = str(item.get(self.answer_field, "unknown"))
        return (expected,)

    def query_features(self, item: dict) -> List[str]:
        """Return routing feature tokens for a problem."""
        if self.query_features_fn is not None:
            return self.query_features_fn(item)
        expected = str(item.get(self.answer_field, "unknown")).upper()
        return [f"answer_{expected}"]

    def positive_verdict(self) -> str:
        """Return the verdict string that corresponds to a 'correct/positive' answer."""
        # Heuristic: look for TRUE/YES/CORRECT/ENFORCEABLE in the values.
        for v in self.answer_map.values():
            if v.upper() in ("TRUE", "YES", "CORRECT", "ENFORCEABLE", "VALID"):
                return v
        # Fall back to the first value.
        return next(iter(self.answer_map.values()))

    def validate(self) -> None:
        """Raise ValueError if the config is inconsistent."""
        if not self.domain_description.strip():
            raise ValueError("domain_description must not be empty")
        if not self.input_fields:
            raise ValueError("input_fields must contain at least one field name")
        if not self.answer_field:
            raise ValueError("answer_field must not be empty")
        if not self.verdict_pattern:
            raise ValueError("verdict_pattern must not be empty")
        if not self.answer_map:
            raise ValueError("answer_map must not be empty")


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Controls how the training loop runs.  Model-/infrastructure-specific.

    Required fields
    ---------------
    scoring_models : List[str]
        All models to evaluate candidates against.  The multi-model gate
        requires a candidate to improve (or hold) on ALL models.
        To preserve ICR_select behaviour, pass a single-element list.
        e.g. ["openai/gpt-oss-120b", "google/gemma-4-31b-it",
               "meta-llama/llama-3.3-70b-instruct"]

    generator_model : str
        Model used to write case studies.
        e.g. "openai/gpt-4o"

    Optional fields
    ---------------
    oracle_model : str or None
        Stronger model used to generate ground-truth reasoning traces.
        If None, no oracle signal is used.

    fix_rate_threshold : float
        A candidate must fix at least this fraction of the failure bin on
        every scoring model.  Default 0.30.

    regress_threshold : float
        Absolute regression ceiling: candidate must not break more than this
        fraction of the correct pool on any scoring model.  Default 0.15.

    regress_relative_factor : float
        Relative regression ceiling multiplier.  The effective allowed regression
        is max(regress_threshold × pool_size, fix_count × regress_relative_factor).
        This prevents the gate from being impossibly tight when the correct pool
        is large relative to the number of failures being fixed.  Default 2.0.

    min_pool_for_regression : int
        Skip the regression check when the correct pool is smaller than this.
        Default 10.

    bin_threshold / n_candidates / concurrency / ablation_every / condense_at :
        Directly mirror the equivalent ICR_select parameters.

    lam / mu / nu :
        Utility function weights (Vgap, regression cost, length penalty).
        Mirror ICR_select UtilityConfig defaults.
    """

    # ── REQUIRED ──────────────────────────────────────────────────────────
    scoring_models: List[str]
    generator_model: str

    # ── OPTIONAL: oracle ──────────────────────────────────────────────────
    oracle_model: Optional[str] = None

    # ── OPTIONAL: gates ───────────────────────────────────────────────────
    fix_rate_threshold: float = 0.30
    regress_threshold: float = 0.15
    regress_relative_factor: float = 2.0
    min_pool_for_regression: int = 10

    # ── OPTIONAL: utility weights ─────────────────────────────────────────
    lam: float = 0.5
    mu: float = 1.0
    nu: float = 0.1
    utility_threshold: float = 0.0

    # ── OPTIONAL: loop settings ───────────────────────────────────────────
    bin_threshold: int = 3
    n_candidates: int = 3
    concurrency: int = 25
    ablation_every: int = 5
    condense_at: int = 6
    flush_remainder: bool = False

    # ── OPTIONAL: scoring behaviour ───────────────────────────────────────
    cot_first: bool = False
    reasoning_effort: Optional[str] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def primary_model(self) -> str:
        """Return the first (primary) scoring model."""
        return self.scoring_models[0]

    def adaptive_regress_limit(self, pool_size: int, fix_count: int) -> int:
        """
        Return the maximum number of correct-pool items a candidate is allowed
        to break, using the adaptive threshold formula.

        adaptive_limit = max(
            ceil(regress_threshold × pool_size),
            ceil(fix_count × regress_relative_factor),
        )
        """
        import math
        abs_limit = math.ceil(self.regress_threshold * pool_size)
        rel_limit = math.ceil(fix_count * self.regress_relative_factor)
        return max(abs_limit, rel_limit)

    def validate(self) -> None:
        """Raise ValueError if the config is inconsistent."""
        if not self.scoring_models:
            raise ValueError("scoring_models must contain at least one model")
        if not self.generator_model:
            raise ValueError("generator_model must not be empty")
        if not (0.0 < self.fix_rate_threshold <= 1.0):
            raise ValueError("fix_rate_threshold must be in (0, 1]")
        if not (0.0 < self.regress_threshold <= 1.0):
            raise ValueError("regress_threshold must be in (0, 1]")
