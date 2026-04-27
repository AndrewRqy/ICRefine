"""
ICR_adaptive/components/failure_classifier.py

FailureClassifier — labels a usable response as one of four failure types:

  CORRECT      — verdict matches ground truth
  WRONG_ANSWER — verdict present but wrong
  ABANDONMENT  — response contains abandonment phrase (model gave up)
  FORMAT       — verdict present but response was suspiciously short
                 (token_count < truncation_token_threshold); kept here for
                 completeness but format issues are normally filtered before
                 reaching this class.

The classifier works purely on text; it does NOT call any LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

from ICR_adaptive.config import TaskConfig


class FailureType(str, Enum):
    CORRECT = "CORRECT"
    WRONG_ANSWER = "WRONG_ANSWER"
    ABANDONMENT = "ABANDONMENT"
    FORMAT = "FORMAT"


@dataclass
class ClassifyResult:
    failure_type: FailureType
    verdict: str        # extracted verdict (normalised to upper-case)
    expected: str       # ground-truth verdict (normalised)
    matched_phrase: str # non-empty when ABANDONMENT


class FailureClassifier:
    """
    Classifies a single (response, ground_truth) pair.

    Parameters
    ----------
    task_cfg : TaskConfig
        Provides answer_map (for ground-truth normalisation) and
        abandonment_phrases.
    """

    def __init__(self, task_cfg: TaskConfig) -> None:
        self._answer_map: dict = {
            str(k).upper(): str(v).upper()
            for k, v in task_cfg.answer_map.items()
        }
        self._abandonment_phrases: List[str] = [
            p.lower() for p in task_cfg.abandonment_phrases
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        verdict: str,
        ground_truth_label: str,
        response: str,
        token_count: int,
        truncation_threshold: int,
    ) -> ClassifyResult:
        """
        Parameters
        ----------
        verdict              : extracted verdict string (from FormatFilter)
        ground_truth_label   : raw label from the dataset (e.g. "True")
        response             : full model response text
        token_count          : output token count
        truncation_threshold : from TaskConfig (used for FORMAT detection)
        """
        norm_verdict = verdict.upper().strip()
        norm_expected = self._normalise_label(ground_truth_label)

        # FORMAT: token count below threshold despite having a verdict pattern
        if token_count < truncation_threshold:
            return ClassifyResult(
                failure_type=FailureType.FORMAT,
                verdict=norm_verdict,
                expected=norm_expected,
                matched_phrase="",
            )

        # CORRECT
        if norm_verdict == norm_expected:
            return ClassifyResult(
                failure_type=FailureType.CORRECT,
                verdict=norm_verdict,
                expected=norm_expected,
                matched_phrase="",
            )

        # ABANDONMENT — wrong answer AND contains a bail-out phrase
        resp_lower = response.lower()
        for phrase in self._abandonment_phrases:
            if phrase in resp_lower:
                return ClassifyResult(
                    failure_type=FailureType.ABANDONMENT,
                    verdict=norm_verdict,
                    expected=norm_expected,
                    matched_phrase=phrase,
                )

        # WRONG_ANSWER — wrong but no bail-out phrase detected
        return ClassifyResult(
            failure_type=FailureType.WRONG_ANSWER,
            verdict=norm_verdict,
            expected=norm_expected,
            matched_phrase="",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalise_label(self, label: str) -> str:
        """Map raw dataset label to its verdict string via answer_map."""
        key = str(label).upper().strip()
        # Try exact match first, then the original-case map values
        if key in self._answer_map:
            return self._answer_map[key]
        # Fall back to the label itself (upper-cased) so callers can compare
        return key
