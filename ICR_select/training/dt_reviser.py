"""
ICR_select/training/dt_reviser.py — Validated decision tree revision.

Takes the full accumulated failure set (with COT traces), identifies which DT
steps are most commonly misapplied, generates a targeted rewrite of only those
steps, then validates the revision against the full training set before
accepting it.

Only accepts the revision if it improves (or matches) the current accuracy —
no blind DT changes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from ICR_naive.core.cheatsheet import Cheatsheet, DECISION_TREE_MAX_CHARS
from ICR_reasoning.core.llm_client import call_llm
from ICR_reasoning.training.scorer import score_batch
from ..analysis.step_parser import build_profile, format_profile, MisapplicationProfile
from ..prompts.templates import (
    DT_STEP_ANALYSIS_PROMPT,
    DT_STEP_ANALYSIS_MAX_TOKENS,
    DT_REVISION_PROMPT,
    DT_REVISION_MAX_TOKENS,
)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class DtRevisionResult:
    accepted: bool
    revised_dt: str          # new DT text if accepted, original if not
    profile: MisapplicationProfile
    accuracy_before: float
    accuracy_after: float    # 0.0 if not accepted (revision was discarded)
    step_analysis: str       # formatted profile sent to the LLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_failures_for_analysis(failures: list[dict], max_items: int = 20) -> str:
    """Format a sample of failures with their reasoning traces."""
    lines = []
    sample = failures[:max_items]
    for i, it in enumerate(sample, 1):
        trace = (it.get("post_think") or "").strip()
        if not trace or len(trace) < 30:
            trace = (it.get("thinking") or "").strip()
        trace_display = (trace[:400] + " [...]") if len(trace) > 400 else trace
        lines.append(
            f"--- Failure {i} ---\n"
            f"  E1={it['equation1']}  E2={it['equation2']}\n"
            f"  expected={it.get('expected','?')}  predicted={it.get('predicted','?')}\n"
            f"  Reasoning trace:\n    {trace_display or '(none)'}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Core revision function
# ---------------------------------------------------------------------------

def run_dt_revision(
    failures: list[dict],
    train_seen: list[dict],
    cheatsheet: Cheatsheet,
    model_casestudy: str,
    model_score: str,
    api_key: str,
    concurrency: int = 10,
    reasoning_effort: str | None = "low",
    cot_first: bool = True,
    min_failures: int = 5,
    regress_tolerance: float = 0.10,   # max allowed accuracy drop (fraction of train_seen)
    log: bool = True,
) -> DtRevisionResult:
    """
    Analyse failures, generate a targeted DT revision, validate it.

    Parameters
    ----------
    failures      : items the cheatsheet got wrong (must have post_think/thinking)
    train_seen    : all items scored so far (for before/after comparison)
    cheatsheet    : current cheatsheet (DT + case studies)
    min_failures  : don't attempt revision if fewer than this many failures

    Returns
    -------
    DtRevisionResult — accepted=True only if the revised DT improves accuracy.
    """
    def _log(msg: str) -> None:
        if log:
            print(msg, file=sys.stderr, flush=True)

    if len(failures) < min_failures:
        _log(
            f"  [dt_revise] only {len(failures)} failures — "
            f"need ≥{min_failures} for reliable revision. Skipping."
        )
        return DtRevisionResult(
            accepted=False,
            revised_dt=cheatsheet.decision_tree,
            profile=build_profile([], cheatsheet.decision_tree),
            accuracy_before=0.0,
            accuracy_after=0.0,
            step_analysis="(skipped — too few failures)",
        )

    _log(f"\n  [dt_revise] analysing {len(failures)} failures for DT step misapplications ...")

    # ── Step 1: Build misapplication profile from COT traces ────────────────
    profile = build_profile(failures, cheatsheet.decision_tree)
    step_analysis_text = format_profile(profile)
    _log(f"  [dt_revise] profile:\n{step_analysis_text}")

    if not profile.steps:
        _log("  [dt_revise] no step references found in reasoning traces — skipping.")
        return DtRevisionResult(
            accepted=False,
            revised_dt=cheatsheet.decision_tree,
            profile=profile,
            accuracy_before=0.0,
            accuracy_after=0.0,
            step_analysis=step_analysis_text,
        )

    # ── Step 2: Ask LLM to analyse which steps are broken ───────────────────
    _log("  [dt_revise] generating step analysis via LLM ...")
    analysis_resp = call_llm(
        DT_STEP_ANALYSIS_PROMPT.format(
            decision_tree=cheatsheet.decision_tree.strip(),
            failure_lines=_format_failures_for_analysis(failures),
            n_failures=len(failures),
        ),
        model_casestudy, api_key,
        temperature=0.2,
        max_tokens=DT_STEP_ANALYSIS_MAX_TOKENS,
        reasoning_effort=None,
    )
    llm_analysis = analysis_resp.content.strip()
    _log(f"  [dt_revise] LLM step analysis:\n{llm_analysis[:600]}")

    # ── Step 3: Generate revised DT ─────────────────────────────────────────
    _log("  [dt_revise] generating revised DT ...")
    revision_resp = call_llm(
        DT_REVISION_PROMPT.format(
            step_analysis=llm_analysis,
            decision_tree=cheatsheet.decision_tree.strip(),
        ),
        model_casestudy, api_key,
        temperature=0.2,
        max_tokens=DT_REVISION_MAX_TOKENS,
        reasoning_effort=None,
    )
    revised_dt = revision_resp.content.strip()

    if len(revised_dt) > DECISION_TREE_MAX_CHARS * 1.5:
        _log(
            f"  [dt_revise] revised DT too long ({len(revised_dt)} chars) — truncating."
        )
        revised_dt = revised_dt[:DECISION_TREE_MAX_CHARS]

    # ── Step 4: Validate on train_seen ──────────────────────────────────────
    if not train_seen:
        _log("  [dt_revise] no training items seen yet — accepting without validation.")
        return DtRevisionResult(
            accepted=True,
            revised_dt=revised_dt,
            profile=profile,
            accuracy_before=0.0,
            accuracy_after=0.0,
            step_analysis=step_analysis_text,
        )

    _log(f"  [dt_revise] validating on {len(train_seen)} seen items ...")

    original_cs = Cheatsheet(
        decision_tree=cheatsheet.decision_tree,
        case_studies=cheatsheet.case_studies,
    )
    revised_cs = Cheatsheet(
        decision_tree=revised_dt,
        case_studies=cheatsheet.case_studies,
    )

    correct_before, _ = score_batch(
        train_seen, original_cs.render(), model_score, api_key,
        concurrency=concurrency, reasoning_effort=reasoning_effort, cot_first=cot_first,
    )
    correct_after, _ = score_batch(
        train_seen, revised_cs.render(), model_score, api_key,
        concurrency=concurrency, reasoning_effort=reasoning_effort, cot_first=cot_first,
    )

    acc_before = len(correct_before) / len(train_seen)
    acc_after  = len(correct_after)  / len(train_seen)
    delta      = acc_after - acc_before

    _log(
        f"  [dt_revise] accuracy: before={acc_before:.1%}  after={acc_after:.1%}  "
        f"delta={delta:+.1%}"
    )

    # Accept if accuracy drop is within tolerance (percentage of train_seen)
    accepted = delta >= -regress_tolerance
    if accepted:
        _log("  [dt_revise] revision ACCEPTED.")
    else:
        _log(f"  [dt_revise] revision REJECTED (delta={delta:+.1%} < -{regress_tolerance:.1%}).")

    return DtRevisionResult(
        accepted=accepted,
        revised_dt=revised_dt if accepted else cheatsheet.decision_tree,
        profile=profile,
        accuracy_before=acc_before,
        accuracy_after=acc_after,
        step_analysis=step_analysis_text,
    )
