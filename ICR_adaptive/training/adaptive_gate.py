"""
ICR_adaptive/training/adaptive_gate.py

AdaptiveRegressionGate — decides whether a candidate cheatsheet clears all
quality gates and should replace the current best sheet.

Gate conditions (all must pass):
  1. fix_rate  — candidate fixes ≥ fix_rate_threshold fraction of the
                 target failure bin on EVERY scoring model.
  2. regression — candidate breaks ≤ adaptive_regress_limit items from
                  the correct pool on EVERY scoring model.
  3. utility   — net utility score ≥ utility_threshold.

Utility formula (mirrors ICR_select UtilityConfig):
    U = lam * Vgap - mu * regress_cost - nu * length_penalty

where:
    Vgap         = (fixed / bin_size) - (broken / pool_size)  (if pool_size > 0)
    regress_cost = broken / pool_size  (capped at 1)
    length_penalty = max(0, len(candidate_text) - len(current_text)) / len(current_text)
                     (0 if current_text is empty)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ICR_adaptive.config import PipelineConfig
from ICR_adaptive.components.multi_model_scorer import ScorerResult


@dataclass
class GateResult:
    passed: bool
    reason: str          # "ok" | "fix_rate" | "regression" | "utility"
    utility: float
    fix_rate: float      # worst fix-rate across models
    regress_count: int   # worst regression count across models


class AdaptiveRegressionGate:
    """
    Parameters
    ----------
    pipeline_cfg : PipelineConfig
        Provides all gate thresholds and utility weights.
    """

    def __init__(self, pipeline_cfg: PipelineConfig) -> None:
        self._cfg = pipeline_cfg

    def evaluate(
        self,
        scorer_result: ScorerResult,
        bin_ids: List[str],
        pool_ids: List[str],
        candidate_text: str,
        current_text: str = "",
    ) -> GateResult:
        """
        Parameters
        ----------
        scorer_result   : output of MultiModelScorer.score()
        bin_ids         : item IDs in the target failure bin
        pool_ids        : item IDs in the current correct pool
        candidate_text  : text of the candidate cheatsheet
        current_text    : text of the current best sheet (for length penalty)
        """
        cfg = self._cfg

        # ── Gate 1: fix rate ─────────────────────────────────────────────
        worst_fix_rate, worst_regress = self._compute_worst(
            scorer_result, bin_ids, pool_ids
        )

        if not bin_ids:
            return GateResult(passed=False, reason="fix_rate",
                              utility=0.0, fix_rate=0.0, regress_count=0)

        if worst_fix_rate < cfg.fix_rate_threshold:
            return GateResult(passed=False, reason="fix_rate",
                              utility=0.0, fix_rate=worst_fix_rate,
                              regress_count=worst_regress)

        # ── Gate 2: regression ───────────────────────────────────────────
        if pool_ids and len(pool_ids) >= cfg.min_pool_for_regression:
            # Number of items that are newly wrong (bin fixing excluded)
            pure_regress = max(0, worst_regress)
            fix_count = int(worst_fix_rate * len(bin_ids))
            limit = cfg.adaptive_regress_limit(
                pool_size=len(pool_ids), fix_count=fix_count
            )
            if pure_regress > limit:
                return GateResult(passed=False, reason="regression",
                                  utility=0.0, fix_rate=worst_fix_rate,
                                  regress_count=worst_regress)

        # ── Utility score ────────────────────────────────────────────────
        u = self._utility(worst_fix_rate, worst_regress,
                          len(bin_ids), len(pool_ids),
                          candidate_text, current_text)

        if u < cfg.utility_threshold:
            return GateResult(passed=False, reason="utility",
                              utility=u, fix_rate=worst_fix_rate,
                              regress_count=worst_regress)

        return GateResult(passed=True, reason="ok",
                          utility=u, fix_rate=worst_fix_rate,
                          regress_count=worst_regress)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_worst(
        self,
        scorer_result: ScorerResult,
        bin_ids: List[str],
        pool_ids: List[str],
    ):
        """Return (worst_fix_rate, worst_regress_count) across all models."""
        worst_fix = 1.0
        worst_regress = 0

        for mr in scorer_result.per_model.values():
            correct_set = set(mr.correct_ids)

            if bin_ids:
                fixed = sum(1 for id_ in bin_ids if id_ in correct_set)
                fr = fixed / len(bin_ids)
                if fr < worst_fix:
                    worst_fix = fr

            if pool_ids:
                broken = sum(1 for id_ in pool_ids if id_ not in correct_set)
                if broken > worst_regress:
                    worst_regress = broken

        return worst_fix, worst_regress

    def _utility(
        self,
        fix_rate: float,
        regress_count: int,
        bin_size: int,
        pool_size: int,
        candidate_text: str,
        current_text: str,
    ) -> float:
        cfg = self._cfg

        vgap = fix_rate
        if pool_size > 0:
            vgap -= regress_count / pool_size

        regress_cost = (regress_count / pool_size) if pool_size > 0 else 0.0

        if current_text:
            length_penalty = max(
                0.0,
                (len(candidate_text) - len(current_text)) / len(current_text),
            )
        else:
            length_penalty = 0.0

        return cfg.lam * vgap - cfg.mu * regress_cost - cfg.nu * length_penalty
