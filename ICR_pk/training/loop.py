"""
loop.py — ICR_pk outer training loop.

Each outer iteration:
  1. SCORE    — score all items with current PK; build per-partition failure sets
  2. ABLATION — rate every existing section's contribution (accuracy delta)
  3. GENERATE — for each active failure partition, generate a candidate new section
  4. VALIDATE — accept the candidate if it improves accuracy on the partition
               by at least acceptance_threshold; also check regression on correct items
  5. PRUNE    — optionally remove sections with negative contribution
  6. SAVE     — write PK snapshot and update log to disk

Convergence: stop early when no partitions exceed gen_trigger_failures.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ICR_partition.training.partition import item_partition_key, partition_label as pk_label
from utils.data import is_true

from .section_parser import PKSection, parse_sections, render_pk, append_section
from .scorer import score_items, accuracy, split_by_correctness
from .ablation import rate_sections
from .generator import generate_section


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PKLoopConfig:
    model_score: str
    model_casestudy: str
    api_key: str
    concurrency: int            = 100
    reasoning_effort: str | None = "low"
    max_outer_iters: int        = 3

    # Ablation
    ablation_sample_size: int | None = None   # None = use all items
    contribution_threshold: float    = -0.02  # below → flag as harmful

    # Generation / validation
    gen_trigger_failures: int   = 5     # min failures to attempt generation
    acceptance_threshold: float = 0.15  # min improvement on partition failures to accept
    regression_threshold: float = 0.05  # max allowed regression on correct items (0 = skip check)
    max_gen_attempts: int       = 2     # candidate regeneration rounds per partition per iter

    # Pruning
    prune_harmful: bool = False   # auto-remove harmful sections after ablation


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class PKLoopResult:
    sections: list[PKSection]
    update_log: list[dict]       = field(default_factory=list)
    iter_summaries: list[dict]   = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_failure_sets(
    scores: list[bool | None],
    items: list[dict],
) -> dict[str, list[dict]]:
    """Group failing items by partition label."""
    sets: dict[str, list[dict]] = {}
    for score, item in zip(scores, items):
        correct = score is not None and score == is_true(item.get("answer", False))
        if not correct:
            label = pk_label(item_partition_key(item))
            sets.setdefault(label, []).append(item)
    return sets


def _partition_meta(label: str) -> dict:
    """Extract form_e1, form_e2, polarity, depth_bucket from a partition label string."""
    # e.g. "STANDARD→GENERAL_d2+_FALSE_nested"
    arrow = label.split("_")[0]  # "STANDARD→GENERAL"
    form_e1, _, form_e2 = arrow.partition("→")
    polarity = "TRUE" if "_TRUE" in label else "FALSE"
    depth_bucket = 2
    if "_d0_" in label or label.endswith("_d0"):
        depth_bucket = 0
    elif "_d1_" in label or label.endswith("_d1"):
        depth_bucket = 1
    return {
        "form_e1": form_e1 or "GENERAL",
        "form_e2": form_e2 or "GENERAL",
        "polarity": polarity,
        "depth_bucket": depth_bucket,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_pk_loop(
    items: list[dict],
    initial_pk_text: str,
    cfg: PKLoopConfig,
    output_dir: Path,
) -> PKLoopResult:

    output_dir.mkdir(parents=True, exist_ok=True)
    sections   = parse_sections(initial_pk_text)
    update_log: list[dict] = []
    iter_summaries: list[dict] = []

    _log(f"\n[ICR_pk] Parsed {len(sections)} sections from prior knowledge:")
    for s in sections:
        _log(f"  [{s.index:2d}] {s.title}  ({len(s.content)} chars)")

    def _save_log() -> None:
        (output_dir / "update_log.json").write_text(
            json.dumps(update_log, indent=2), encoding="utf-8"
        )

    for outer_iter in range(1, cfg.max_outer_iters + 1):
        _log(f"\n{'='*65}")
        _log(f"[ICR_pk] OUTER ITERATION {outer_iter}/{cfg.max_outer_iters}")
        _log(f"{'='*65}")

        current_pk = render_pk(sections)

        # ── 1. Score all items ────────────────────────────────────────────
        _log(f"\n[iter {outer_iter}] Scoring {len(items)} items with current PK...")
        baseline_scores = score_items(
            items, current_pk, cfg.model_score, cfg.api_key,
            cfg.concurrency, cfg.reasoning_effort,
            progress_label=f"iter-{outer_iter} score",
        )
        overall_acc = accuracy(baseline_scores, items)
        correct_items, all_failures = split_by_correctness(baseline_scores, items)
        failure_sets = _build_failure_sets(baseline_scores, items)

        _log(f"[iter {outer_iter}] Accuracy: {len(correct_items)}/{len(items)} ({overall_acc:.1%}), "
             f"failures: {len(all_failures)}")

        active_partitions = {
            k: v for k, v in failure_sets.items()
            if len(v) >= cfg.gen_trigger_failures
        }
        _log(f"[iter {outer_iter}] Active partitions: {len(active_partitions)}")
        for label, fails in sorted(active_partitions.items(), key=lambda x: -len(x[1])):
            _log(f"  {label}: {len(fails)} failures")

        update_log.append({
            "outer_iter": outer_iter,
            "event": "iter_score",
            "n_correct": len(correct_items),
            "n_failures": len(all_failures),
            "n_items": len(items),
            "overall_acc": round(overall_acc, 4),
            "active_partitions": len(active_partitions),
        })

        if not active_partitions:
            _log(f"[iter {outer_iter}] No active partitions — converged early.")
            break

        # ── 2. Ablation phase ─────────────────────────────────────────────
        _log(f"\n[iter {outer_iter}] ABLATION PHASE "
             f"({len([s for s in sections if not s.pruned])} sections)...")

        ratings = rate_sections(
            sections=sections,
            items=items,
            baseline_scores=baseline_scores,
            model=cfg.model_score,
            api_key=cfg.api_key,
            concurrency=cfg.concurrency,
            reasoning_effort=cfg.reasoning_effort,
            ablation_sample_size=cfg.ablation_sample_size,
            log_fn=_log,
        )

        for r in ratings:
            update_log.append({
                "outer_iter":    outer_iter,
                "event":         "section_rated",
                "section_index": r.section.index,
                "section_title": r.section.title,
                "contribution":  round(r.contribution, 4),
                "baseline_acc":  round(r.baseline_acc, 4),
                "ablated_acc":   round(r.ablated_acc, 4),
                "label":         r.label(),
                "sample_size":   r.sample_size,
            })
            if r.contribution < cfg.contribution_threshold:
                if cfg.prune_harmful:
                    r.section.pruned = True
                    _log(f"  [ablation] PRUNED harmful section: '{r.section.title}'")
                    update_log.append({
                        "outer_iter": outer_iter,
                        "event": "section_pruned",
                        "section_title": r.section.title,
                        "contribution": round(r.contribution, 4),
                    })

        # ── 3. Generation + Validation phase ─────────────────────────────
        _log(f"\n[iter {outer_iter}] GENERATION PHASE "
             f"({len(active_partitions)} partitions)...")

        existing_titles = [s.title for s in sections if not s.pruned]

        for part_label, failures in sorted(
            active_partitions.items(), key=lambda x: -len(x[1])
        ):
            _log(f"\n  [gen] Partition: {part_label}  ({len(failures)} failures)")
            meta = _partition_meta(part_label)
            accepted = False

            for attempt in range(1, cfg.max_gen_attempts + 1):
                _log(f"  [gen] Attempt {attempt}/{cfg.max_gen_attempts}...")
                gen = generate_section(
                    partition_label=part_label,
                    form_e1=meta["form_e1"],
                    form_e2=meta["form_e2"],
                    polarity=meta["polarity"],
                    depth_bucket=meta["depth_bucket"],
                    failures=failures,
                    existing_titles=existing_titles,
                    model=cfg.model_casestudy,
                    api_key=cfg.api_key,
                    reasoning_effort=cfg.reasoning_effort,
                )

                if gen is None:
                    _log(f"  [gen] Generation failed for '{part_label}'")
                    continue

                _log(f"  [gen] Candidate: '{gen.title}'")

                # ── Validate on partition failures ────────────────────────
                pk_with    = render_pk(sections) + "\n\n" + gen.content
                pk_without = render_pk(sections)

                baseline_on_part = accuracy(
                    score_items(failures, pk_without, cfg.model_score, cfg.api_key,
                                min(cfg.concurrency, 30), cfg.reasoning_effort,
                                progress_label="validate-base"),
                    failures,
                )
                acc_with = accuracy(
                    score_items(failures, pk_with, cfg.model_score, cfg.api_key,
                                min(cfg.concurrency, 30), cfg.reasoning_effort,
                                progress_label="validate-new"),
                    failures,
                )
                improvement = acc_with - baseline_on_part
                _log(f"  [validate] partition: baseline={baseline_on_part:.1%} "
                     f"with_new={acc_with:.1%} improvement={improvement:+.1%}")

                # ── Regression check on a sample of correct items ─────────
                regression_ok = True
                if cfg.regression_threshold > 0 and correct_items:
                    sample = correct_items[:min(50, len(correct_items))]
                    acc_correct_before = accuracy(
                        score_items(sample, pk_without, cfg.model_score, cfg.api_key,
                                    min(cfg.concurrency, 20), cfg.reasoning_effort,
                                    progress_label="regress-base"),
                        sample,
                    )
                    acc_correct_after = accuracy(
                        score_items(sample, pk_with, cfg.model_score, cfg.api_key,
                                    min(cfg.concurrency, 20), cfg.reasoning_effort,
                                    progress_label="regress-new"),
                        sample,
                    )
                    regression = acc_correct_before - acc_correct_after
                    _log(f"  [validate] regression: {acc_correct_before:.1%} → "
                         f"{acc_correct_after:.1%} ({regression:+.1%})")
                    if regression > cfg.regression_threshold:
                        regression_ok = False
                        _log(f"  [validate] REGRESSION too high ({regression:.1%} > "
                             f"{cfg.regression_threshold:.1%}) — rejecting")

                if improvement >= cfg.acceptance_threshold and regression_ok:
                    sections = append_section(sections, gen.title, gen.content)
                    existing_titles.append(gen.title)
                    accepted = True
                    _log(f"  [validate] ACCEPTED: '{gen.title}' "
                         f"(improvement={improvement:+.1%})")
                    update_log.append({
                        "outer_iter":  outer_iter,
                        "event":       "section_added",
                        "section_title": gen.title,
                        "partition":   part_label,
                        "improvement": round(improvement, 4),
                        "baseline_on_partition": round(baseline_on_part, 4),
                    })
                    # Save incremental PK snapshot
                    snap = output_dir / f"pk_iter_{outer_iter:02d}.txt"
                    snap.write_text(render_pk(sections), encoding="utf-8")
                    break
                else:
                    _log(f"  [validate] REJECTED: improvement={improvement:+.1%} "
                         f"(need >={cfg.acceptance_threshold:.0%})")
                    update_log.append({
                        "outer_iter":  outer_iter,
                        "event":       "section_rejected",
                        "section_title": gen.title,
                        "partition":   part_label,
                        "improvement": round(improvement, 4),
                        "regression_ok": regression_ok,
                    })

            if not accepted:
                _log(f"  [gen] No candidate accepted for partition '{part_label}'")

        # ── Summary ───────────────────────────────────────────────────────
        n_active_sections = len([s for s in sections if not s.pruned])
        iter_summaries.append({
            "outer_iter":       outer_iter,
            "overall_acc":      round(overall_acc, 4),
            "n_failures":       len(all_failures),
            "active_partitions": len(active_partitions),
            "n_sections":       n_active_sections,
        })
        _log(f"\n[iter {outer_iter}] Summary: acc={overall_acc:.1%}, "
             f"failures={len(all_failures)}, sections={n_active_sections}")

        _save_log()

    return PKLoopResult(
        sections=sections,
        update_log=update_log,
        iter_summaries=iter_summaries,
    )
