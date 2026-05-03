"""
ICR_adaptive/components/execution_parser.py

ExecutionPathParser — locates the step/rule in the cheatsheet where the model
first deviated from the expected execution path.

Approach
--------
The cheatsheet author annotates steps and rules with markers:
    [STEP: name]   — a named reasoning phase
    [RULE: name]   — a named decision rule within a step

When a model response is compared to a reference (oracle) trace, the parser
identifies the first marker whose corresponding segment is absent or differs.

In the common case where no oracle trace is available, the parser returns a
coarser signal: the name of the last [STEP:] marker found in the model
response, treating that as the step where execution stopped (i.e., the model
completed up to here and then diverged or gave up).

Returns
-------
DivergenceResult.step  : name of the divergence step, or "unknown"
DivergenceResult.rule  : name of the divergence rule (within that step),
                         or "unknown" if no rule-level granularity available
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from ICR_adaptive.config import TaskConfig


@dataclass
class DivergenceResult:
    step: str    # last step completed before divergence / "unknown"
    rule: str    # rule within that step, or "unknown"


class ExecutionPathParser:
    """
    Parameters
    ----------
    task_cfg : TaskConfig
        Provides step_annotation_pattern and rule_annotation_pattern.
    """

    def __init__(self, task_cfg: TaskConfig) -> None:
        self._step_re = re.compile(task_cfg.step_annotation_pattern)
        self._rule_re = re.compile(task_cfg.rule_annotation_pattern)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        model_response: str,
        oracle_trace: Optional[str] = None,
    ) -> DivergenceResult:
        """
        Identify the execution-path divergence point.

        Parameters
        ----------
        model_response : full model response text
        oracle_trace   : reference reasoning trace (optional).
                         When provided, the parser returns the first step/rule
                         present in oracle_trace but absent in model_response.
                         When None, returns the last step/rule seen in model_response.
        """
        if oracle_trace is not None:
            return self._compare(model_response, oracle_trace)
        return self._last_step(model_response)

    def steps_in(self, text: str) -> List[str]:
        """Return all [STEP: name] values found in text (in order)."""
        return [m.group(1).strip() for m in self._step_re.finditer(text)]

    def rules_in(self, text: str) -> List[str]:
        """Return all [RULE: name] values found in text (in order)."""
        return [m.group(1).strip() for m in self._rule_re.finditer(text)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _last_step(self, response: str) -> DivergenceResult:
        steps = self.steps_in(response)
        rules = self.rules_in(response)
        return DivergenceResult(
            step=steps[-1] if steps else "unknown",
            rule=rules[-1] if rules else "unknown",
        )

    def _compare(self, model_response: str, oracle_trace: str) -> DivergenceResult:
        oracle_steps = self.steps_in(oracle_trace)
        oracle_rules = self.rules_in(oracle_trace)
        model_steps = set(self.steps_in(model_response))
        model_rules = set(self.rules_in(model_response))

        diverge_step = "unknown"
        diverge_rule = "unknown"

        for step in oracle_steps:
            if step not in model_steps:
                diverge_step = step
                break

        for rule in oracle_rules:
            if rule not in model_rules:
                diverge_rule = rule
                break

        return DivergenceResult(step=diverge_step, rule=diverge_rule)
