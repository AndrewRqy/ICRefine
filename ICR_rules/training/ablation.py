"""
ablation.py — Per-rule ablation pre-pass.

Disables one rule at a time, scores the test set, measures accuracy delta.
Negative delta = rule is helping (removing it hurts accuracy).
Positive delta = rule is actively hurting (removing it helps — a bug).

Run this before the main loop to prioritise which rules to patch first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..rules.rule import RuleSet
from .scorer import score_batch_sair


@dataclass
class AblationResult:
    rule_id: str
    accuracy_baseline: float
    accuracy_without: float
    delta: float          # accuracy_without - accuracy_baseline; positive = rule hurts
    n_items: int


def run_ablation(
    rule_set: RuleSet,
    items: list[dict],
    model: str,
    api_key: str,
    concurrency: int = 50,
    baseline_correct: list[dict] | None = None,
    baseline_wrong: list[dict] | None = None,
    task_spec=None,
) -> Dict[str, AblationResult]:
    """
    Ablate each enabled rule and return a dict of rule_id → AblationResult.

    If baseline_correct/wrong are provided, skips the baseline scoring pass
    (saves one full score_batch_sair call).
    """
    n = len(items)
    if n == 0:
        return {}

    # Baseline
    if baseline_correct is not None and baseline_wrong is not None:
        baseline_acc = len(baseline_correct) / n
    else:
        bc, bw = score_batch_sair(items, rule_set, model, api_key,
                                  concurrency=concurrency, task_spec=task_spec)
        baseline_acc = len(bc) / n

    results: Dict[str, AblationResult] = {}
    enabled_ids = [r.id for r in rule_set.all_rules() if r.enabled]

    print(f"[ablation] baseline accuracy: {baseline_acc:.1%} on {n} items", flush=True)
    print(f"[ablation] ablating {len(enabled_ids)} rules...", flush=True)

    for i, rule_id in enumerate(enabled_ids, 1):
        print(f"[ablation] ({i}/{len(enabled_ids)}) ablating {rule_id}...", flush=True)
        ablated = rule_set.disable_rule(rule_id)
        ac, aw = score_batch_sair(items, ablated, model, api_key, concurrency=concurrency,
                                  label=f"ablate-{rule_id}", task_spec=task_spec)
        acc_without = len(ac) / n
        delta = acc_without - baseline_acc
        results[rule_id] = AblationResult(
            rule_id=rule_id,
            accuracy_baseline=baseline_acc,
            accuracy_without=acc_without,
            delta=delta,
            n_items=n,
        )
        sign = "+" if delta >= 0 else ""
        print(f"  {rule_id:<15} baseline={baseline_acc:.1%}  without={acc_without:.1%}  delta={sign}{delta:.1%}", flush=True)

    return results


def print_ablation_report(results: Dict[str, AblationResult]) -> None:
    """Print ablation results sorted by delta (most harmful rules first)."""
    print("\n=== ABLATION REPORT ===")
    print(f"{'Rule':<15} {'Baseline':>10} {'Without':>10} {'Delta':>8}  Interpretation")
    print("-" * 65)
    for r in sorted(results.values(), key=lambda x: x.delta, reverse=True):
        sign = "+" if r.delta >= 0 else ""
        if r.delta > 0.02:
            interp = "HURTS (actively causing errors)"
        elif r.delta < -0.02:
            interp = "HELPS (removing it breaks things)"
        else:
            interp = "neutral"
        print(f"{r.rule_id:<15} {r.accuracy_baseline:>10.1%} {r.accuracy_without:>10.1%} {sign}{r.delta:>7.1%}  {interp}")
