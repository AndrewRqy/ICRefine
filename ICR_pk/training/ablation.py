"""
ablation.py — Measure each PK section's contribution to accuracy.

For each section S:
  1. Score *items* WITHOUT S  →  ablated accuracy
  2. contribution(S) = baseline_acc - ablated_acc
     Positive  → removing S hurts  → S is helping
     Negative  → removing S helps  → S is hurting (confusing the model)
     Near zero → S has no measurable effect

Ablation is run on a random sample of items (ablation_sample_size) rather
than the full dataset to keep cost manageable; None = use all items.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .section_parser import PKSection, render_pk
from .scorer import score_items, accuracy


@dataclass
class SectionRating:
    section: PKSection
    baseline_acc: float       # accuracy with full PK
    ablated_acc: float        # accuracy without this section
    sample_size: int          # number of items used

    @property
    def contribution(self) -> float:
        """Positive = section helps; negative = section hurts."""
        return self.baseline_acc - self.ablated_acc

    def label(self) -> str:
        c = self.contribution
        if c > 0.03:
            return "HELPFUL"
        if c < -0.02:
            return "HARMFUL"
        return "NEUTRAL"


def rate_sections(
    sections: list[PKSection],
    items: list[dict],
    baseline_scores: list[bool | None],
    model: str,
    api_key: str,
    concurrency: int = 100,
    reasoning_effort: str | None = "low",
    ablation_sample_size: int | None = None,
    seed: int = 42,
    log_fn=print,
) -> list[SectionRating]:
    """
    Ablate each active (non-pruned) section and return SectionRating for each.

    baseline_scores: pre-computed scores with full PK — avoids a full re-score
    and ensures ablation delta is measured against a consistent baseline.
    """
    # Optionally sub-sample items for cheaper ablation
    rng = random.Random(seed)
    if ablation_sample_size is not None and ablation_sample_size < len(items):
        indices = rng.sample(range(len(items)), ablation_sample_size)
        ablation_items   = [items[i] for i in indices]
        ablation_baseline = [baseline_scores[i] for i in indices]
    else:
        ablation_items   = items
        ablation_baseline = baseline_scores

    baseline_acc = accuracy(ablation_baseline, ablation_items)
    active = [s for s in sections if not s.pruned]
    ratings: list[SectionRating] = []

    log_fn(f"  [ablation] Baseline accuracy: {baseline_acc:.1%}  ({len(ablation_items)} items)")

    for section in active:
        pk_without = render_pk(sections, skip_index=section.index)
        ablated_scores = score_items(
            ablation_items, pk_without, model, api_key, concurrency,
            reasoning_effort, progress_label=f"ablate '{section.title[:30]}'",
        )
        abl_acc = accuracy(ablated_scores, ablation_items)
        rating = SectionRating(
            section=section,
            baseline_acc=baseline_acc,
            ablated_acc=abl_acc,
            sample_size=len(ablation_items),
        )
        log_fn(
            f"  [ablation] [{rating.label():7s}] '{section.title}' "
            f"baseline={baseline_acc:.1%} → ablated={abl_acc:.1%} "
            f"contribution={rating.contribution:+.1%}"
        )
        ratings.append(rating)

    return ratings
