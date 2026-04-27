"""
ICR_adaptive/components/multi_model_scorer.py

MultiModelScorer — evaluates a cheatsheet candidate against a fixed evaluation
set on one or more scoring models and returns structured per-model results.

This module contains only the data structures and aggregation logic.
Actual LLM calls are injected via a `score_fn` callable so the scorer can be
unit-tested without live API access.

score_fn signature
------------------
    score_fn(model: str, items: List[dict], sheet_text: str) -> List[ItemScore]

ItemScore.correct : bool — whether the model answered correctly
ItemScore.item_id : str  — identifier from the item dict (used for regression tracking)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class ItemScore:
    item_id: str
    correct: bool
    verdict: Optional[str] = None       # model's extracted verdict
    token_count: int = 0


@dataclass
class ModelResult:
    model: str
    n_correct: int
    n_total: int
    correct_ids: List[str] = field(default_factory=list)
    wrong_ids: List[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_total if self.n_total else 0.0


@dataclass
class ScorerResult:
    """Aggregated results across all scoring models."""
    per_model: Dict[str, ModelResult] = field(default_factory=dict)

    def worst_accuracy(self) -> float:
        """Minimum accuracy across models (conservative gate metric)."""
        if not self.per_model:
            return 0.0
        return min(r.accuracy for r in self.per_model.values())

    def all_pass_fix_rate(self, bin_ids: List[str], threshold: float) -> bool:
        """
        Return True iff every model fixes at least `threshold` fraction of
        the given failure bin (identified by item IDs).
        """
        if not bin_ids:
            return False
        for result in self.per_model.values():
            fixed = sum(1 for id_ in bin_ids if id_ in result.correct_ids)
            if fixed / len(bin_ids) < threshold:
                return False
        return True

    def any_regresses(self, pool_ids: List[str], regress_limit: int) -> bool:
        """
        Return True iff any model breaks more than `regress_limit` items from
        the correct pool.
        """
        for result in self.per_model.values():
            broken = sum(1 for id_ in pool_ids if id_ in result.wrong_ids)
            if broken > regress_limit:
                return True
        return False


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

ScoreFn = Callable[[str, List[dict], str], List[ItemScore]]


class MultiModelScorer:
    """
    Parameters
    ----------
    models   : list of model identifiers to evaluate
    score_fn : callable that takes (model_id, items, sheet_text) and returns
               a List[ItemScore]
    """

    def __init__(self, models: List[str], score_fn: ScoreFn) -> None:
        if not models:
            raise ValueError("models list must not be empty")
        self._models = models
        self._score_fn = score_fn

    def score(self, items: List[dict], sheet_text: str) -> ScorerResult:
        """
        Score all items on all models.

        Parameters
        ----------
        items      : list of problem dicts (must include an "id" key or index
                     will be used as fallback identifier)
        sheet_text : the cheatsheet text to evaluate
        """
        result = ScorerResult()
        for model in self._models:
            item_scores = self._score_fn(model, items, sheet_text)
            n_correct = 0
            correct_ids: List[str] = []
            wrong_ids: List[str] = []
            for s in item_scores:
                if s.correct:
                    n_correct += 1
                    correct_ids.append(s.item_id)
                else:
                    wrong_ids.append(s.item_id)
            result.per_model[model] = ModelResult(
                model=model,
                n_correct=n_correct,
                n_total=len(item_scores),
                correct_ids=correct_ids,
                wrong_ids=wrong_ids,
            )
        return result
