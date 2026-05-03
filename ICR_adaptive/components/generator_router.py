"""
ICR_adaptive/components/generator_router.py

GeneratorRouter — selects the most relevant case study from the case bank
for a given query problem, using Jaccard similarity on feature tokens.

The router is read-only: it ranks existing cases and returns the best match.
Writing new cases to the bank is the responsibility of the training loop.

Case bank schema
----------------
Each entry is a dict with at least:
    "features"   : List[str]   — feature tokens produced by task_cfg.query_features()
    "case_text"  : str         — the actual case-study text
    "bin_key"    : tuple       — (failure_type, divergence_step, …) for which
                                 this case was designed

Usage
-----
    router = GeneratorRouter(task_cfg)
    best = router.route(query_item, case_bank)
    if best:
        print(best["case_text"])
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ICR_adaptive.config import TaskConfig


class GeneratorRouter:
    """
    Parameters
    ----------
    task_cfg : TaskConfig
        Provides query_features() for computing the query feature set.
    """

    def __init__(self, task_cfg: TaskConfig) -> None:
        self._task_cfg = task_cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        query_item: dict,
        case_bank: List[Dict],
        min_similarity: float = 0.0,
    ) -> Optional[Dict]:
        """
        Return the case-bank entry with the highest Jaccard similarity to
        the query_item's feature set, or None if the bank is empty or no
        entry meets min_similarity.

        Parameters
        ----------
        query_item     : problem dict (same format as training data)
        case_bank      : list of case dicts (see module docstring)
        min_similarity : minimum Jaccard score to accept a match (0.0 = any)
        """
        if not case_bank:
            return None

        query_feats = set(self._task_cfg.query_features(query_item))
        best_entry = None
        best_score = -1.0

        for entry in case_bank:
            entry_feats = set(entry.get("features", []))
            score = _jaccard(query_feats, entry_feats)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score < min_similarity:
            return None
        return best_entry

    def route_top_k(
        self,
        query_item: dict,
        case_bank: List[Dict],
        k: int = 3,
        min_similarity: float = 0.0,
    ) -> List[Dict]:
        """
        Return the top-k case-bank entries by Jaccard similarity.
        Ties are broken by list order (stable).
        """
        if not case_bank:
            return []

        query_feats = set(self._task_cfg.query_features(query_item))
        scored = [
            (entry, _jaccard(query_feats, set(entry.get("features", []))))
            for entry in case_bank
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return [entry for entry, score in scored[:k] if score >= min_similarity]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)
