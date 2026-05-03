"""
ICR_adaptive/components/format_filter.py

FormatFilter — decides whether a model response is usable before attempting
to classify its logic failure.

A response is UNUSABLE (filtered out) if:
  - It is empty or whitespace-only.
  - Token count is below truncation_token_threshold (truncated / no verdict).
  - The verdict regex does not match anywhere in the response.

A response that passes all three checks is USABLE and its extracted verdict
string is returned alongside it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ICR_adaptive.config import TaskConfig


@dataclass
class FilterResult:
    usable: bool
    verdict: Optional[str]       # extracted capture group, or None
    reason: str                  # "ok" | "empty" | "truncated" | "no_verdict"
    token_count: int             # caller-supplied; stored for downstream use


class FormatFilter:
    """
    Wraps a TaskConfig's verdict_pattern to test raw LLM responses.

    Parameters
    ----------
    task_cfg : TaskConfig
        Provides verdict_pattern and truncation_token_threshold.
    """

    def __init__(self, task_cfg: TaskConfig) -> None:
        self._pattern = re.compile(task_cfg.verdict_pattern)
        self._threshold = task_cfg.truncation_token_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, response: str, token_count: int) -> FilterResult:
        """
        Evaluate a single model response.

        Parameters
        ----------
        response    : raw text returned by the model
        token_count : number of output tokens (caller-supplied; typically from
                      the API's usage.completion_tokens field)

        Returns
        -------
        FilterResult with usable=True iff the response should be scored.
        """
        if not response or not response.strip():
            return FilterResult(usable=False, verdict=None, reason="empty",
                                token_count=token_count)

        if token_count < self._threshold:
            return FilterResult(usable=False, verdict=None, reason="truncated",
                                token_count=token_count)

        m = self._pattern.search(response)
        if m is None:
            return FilterResult(usable=False, verdict=None, reason="no_verdict",
                                token_count=token_count)

        return FilterResult(usable=True, verdict=m.group(1).strip(),
                            reason="ok", token_count=token_count)
