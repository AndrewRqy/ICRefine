"""
ICR_reasoning/analysis/reasoning_analyzer.py — Analyse captured reasoning traces.

Reads the scored items stored in the bin/checkpoints and computes statistics
motivated by Heddaya et al. (ACL 2026):

  1. Deductive marker density — "therefore", "hence", "thus", "since", "because",
     "implies", "it follows" per 100 tokens. Higher = more logically scaffolded.
     Paper finding: post-think retains these at 25× higher density than summaries.

  2. Token length comparison — thinking (full CoT) vs post_think length in chars
     as a proxy for token count.

  3. Verdict consistency — whether the model's post-think REASONING text contains
     language consistent with the final VERDICT (e.g. does a TRUE verdict's
     reasoning mention "proof" / "holds" / "satisfies"?).

  4. Failure vs correct split — compare marker density between items the model
     got right vs wrong to see if logical scaffolding correlates with accuracy.

Usage
-----
    from ICR_reasoning.analysis.reasoning_analyzer import analyze_items, print_report

    correct, wrong = score_batch(...)
    report = analyze_items(correct, wrong)
    print_report(report)

    # Or from the CLI on a saved update_log / run directory:
    python -m ICR_reasoning.analysis.reasoning_analyzer --run-dir runs/my_run
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Deductive markers (from Heddaya et al.)
# ---------------------------------------------------------------------------

_DEDUCTIVE_MARKERS = re.compile(
    r"\b(therefore|hence|thus|since|because|implies|it follows|consequently|"
    r"so we|we conclude|we get|we have|must be|cannot be|satisfies|violates)\b",
    re.IGNORECASE,
)


def _count_markers(text: str) -> int:
    return len(_DEDUCTIVE_MARKERS.findall(text))


def _marker_density(text: str) -> float:
    """Deductive markers per 100 chars (proxy for per 100 tokens)."""
    if not text:
        return 0.0
    return _count_markers(text) / len(text) * 100


# ---------------------------------------------------------------------------
# Verdict consistency
# ---------------------------------------------------------------------------

_TRUE_SIGNALS  = re.compile(r"\b(proof|holds|satisfies|implies|true|valid|follows)\b", re.IGNORECASE)
_FALSE_SIGNALS = re.compile(r"\b(counterexample|violates|does not|false|fails|not hold)\b", re.IGNORECASE)


def _verdict_consistent(item: dict) -> bool | None:
    """
    Check whether the post-think text is consistent with the final verdict.
    Returns True/False, or None if no post_think is available.
    """
    post_think = item.get("post_think", "").strip()
    verdict    = item.get("predicted", "")
    if not post_think or not verdict:
        return None
    if verdict == "TRUE":
        return bool(_TRUE_SIGNALS.search(post_think))
    if verdict == "FALSE":
        return bool(_FALSE_SIGNALS.search(post_think))
    return None


# ---------------------------------------------------------------------------
# Per-item stats
# ---------------------------------------------------------------------------

@dataclass
class ItemStats:
    id:              str
    verdict:         str | None
    expected:        str
    correct:         bool
    thinking_chars:  int
    post_think_chars: int
    thinking_density:  float   # deductive markers per 100 chars
    post_think_density: float
    verdict_consistent: bool | None


def _item_stats(item: dict, is_correct: bool) -> ItemStats:
    thinking  = item.get("thinking",  "")
    post_think = item.get("post_think", "")
    return ItemStats(
        id               = item.get("id", "?"),
        verdict          = item.get("predicted"),
        expected         = item.get("expected", "?"),
        correct          = is_correct,
        thinking_chars   = len(thinking),
        post_think_chars = len(post_think),
        thinking_density  = _marker_density(thinking),
        post_think_density = _marker_density(post_think),
        verdict_consistent = _verdict_consistent(item),
    )


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

@dataclass
class ReasoningReport:
    n_correct: int
    n_wrong: int
    # Means across correct items
    correct_thinking_chars:    float
    correct_post_think_chars:  float
    correct_thinking_density:  float
    correct_post_think_density: float
    correct_verdict_consistent_rate: float
    # Means across wrong items
    wrong_thinking_chars:    float
    wrong_post_think_chars:  float
    wrong_thinking_density:  float
    wrong_post_think_density: float
    wrong_verdict_consistent_rate: float
    # Per-item detail
    items: list[ItemStats] = field(default_factory=list)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def analyze_items(
    correct: list[dict],
    wrong: list[dict],
) -> ReasoningReport:
    """
    Compute reasoning statistics over scored items.

    Parameters
    ----------
    correct : items the model predicted correctly (each has post_think, thinking)
    wrong   : items the model predicted incorrectly
    """
    c_stats = [_item_stats(it, True)  for it in correct]
    w_stats = [_item_stats(it, False) for it in wrong]
    all_stats = c_stats + w_stats

    def _consistent_rate(stats: list[ItemStats]) -> float:
        rated = [s.verdict_consistent for s in stats if s.verdict_consistent is not None]
        return _mean([float(v) for v in rated])

    return ReasoningReport(
        n_correct=len(correct),
        n_wrong=len(wrong),
        correct_thinking_chars    = _mean([s.thinking_chars    for s in c_stats]),
        correct_post_think_chars  = _mean([s.post_think_chars  for s in c_stats]),
        correct_thinking_density  = _mean([s.thinking_density  for s in c_stats]),
        correct_post_think_density = _mean([s.post_think_density for s in c_stats]),
        correct_verdict_consistent_rate = _consistent_rate(c_stats),
        wrong_thinking_chars    = _mean([s.thinking_chars    for s in w_stats]),
        wrong_post_think_chars  = _mean([s.post_think_chars  for s in w_stats]),
        wrong_thinking_density  = _mean([s.thinking_density  for s in w_stats]),
        wrong_post_think_density = _mean([s.post_think_density for s in w_stats]),
        wrong_verdict_consistent_rate = _consistent_rate(w_stats),
        items=all_stats,
    )


def print_report(report: ReasoningReport) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("REASONING ANALYSIS")
    print(sep)
    print(f"  Items: {report.n_correct} correct, {report.n_wrong} wrong\n")

    print("  CORRECT items:")
    print(f"    thinking  chars (avg)    : {report.correct_thinking_chars:,.0f}")
    print(f"    post_think chars (avg)   : {report.correct_post_think_chars:,.0f}")
    print(f"    thinking  marker density : {report.correct_thinking_density:.3f} / 100 chars")
    print(f"    post_think marker density: {report.correct_post_think_density:.3f} / 100 chars")
    print(f"    verdict consistent rate  : {report.correct_verdict_consistent_rate:.1%}")

    print("\n  WRONG items:")
    print(f"    thinking  chars (avg)    : {report.wrong_thinking_chars:,.0f}")
    print(f"    post_think chars (avg)   : {report.wrong_post_think_chars:,.0f}")
    print(f"    thinking  marker density : {report.wrong_thinking_density:.3f} / 100 chars")
    print(f"    post_think marker density: {report.wrong_post_think_density:.3f} / 100 chars")
    print(f"    verdict consistent rate  : {report.wrong_verdict_consistent_rate:.1%}")

    if report.correct_post_think_density > 0:
        ratio = report.wrong_post_think_density / report.correct_post_think_density
        print(f"\n  wrong/correct post_think density ratio: {ratio:.2f}x")
        print("  (< 1.0 = failures have less logical scaffolding in post-think)")

    print(sep)


def save_report(report: ReasoningReport, path: Path) -> None:
    """Save full per-item stats to a JSON file for offline analysis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "summary": {
            "n_correct": report.n_correct,
            "n_wrong":   report.n_wrong,
            "correct": {
                "thinking_chars_avg":     report.correct_thinking_chars,
                "post_think_chars_avg":   report.correct_post_think_chars,
                "thinking_density":       report.correct_thinking_density,
                "post_think_density":     report.correct_post_think_density,
                "verdict_consistent_rate": report.correct_verdict_consistent_rate,
            },
            "wrong": {
                "thinking_chars_avg":     report.wrong_thinking_chars,
                "post_think_chars_avg":   report.wrong_post_think_chars,
                "thinking_density":       report.wrong_thinking_density,
                "post_think_density":     report.wrong_post_think_density,
                "verdict_consistent_rate": report.wrong_verdict_consistent_rate,
            },
        },
        "items": [
            {
                "id":                  s.id,
                "verdict":             s.verdict,
                "expected":            s.expected,
                "correct":             s.correct,
                "thinking_chars":      s.thinking_chars,
                "post_think_chars":    s.post_think_chars,
                "thinking_density":    s.thinking_density,
                "post_think_density":  s.post_think_density,
                "verdict_consistent":  s.verdict_consistent,
            }
            for s in report.items
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
