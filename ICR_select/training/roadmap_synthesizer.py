"""
ICR_select/training/roadmap_synthesizer.py — Reasoning roadmap synthesis.

After ICR_select accumulates case studies from failure bins, this module
synthesises them into a structured REASONING ROADMAP.  The roadmap acts as a
CONTROLLER over two sources of guidance:

  1. Prior knowledge (frozen) — general rules the model already has.
  2. Case bank (routed episodic memory) — case studies indexed by structural
     feature and surfaced per-query via render_for_query().

Design:
  Roadmap as controller, cases as tools.  Each roadmap aspect runs a mechanical
  structural check and routes the model to the case bank for the fine-grained
  reasoning appropriate to that configuration.  Verdicts and detailed reasoning
  live in the case studies, not the roadmap — this prevents the roadmap from
  growing monolithic and overriding case-level guidance on every query.

  Every CHECK must be executable without deep inference (count *, check variable
  membership, compare set sizes).  The roadmap supplements prior knowledge; it
  does not re-state it or replace the case bank.

After synthesis:
  - cheatsheet.roadmap is replaced with the new controller roadmap.
  - cheatsheet.case_studies is LEFT INTACT — the case bank is not cleared.
    The caller should NOT clear case studies after a successful synthesis.
  - cheatsheet.prior_knowledge is never modified.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from utils.cheatsheet import Cheatsheet
from utils.llm_client import call_llm
from utils.scorer import score_batch
from ..prompts.templates import (
    ROADMAP_SYNTHESIS_PROMPT,
    ROADMAP_SYNTHESIS_MAX_TOKENS,
)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class RoadmapSynthesisResult:
    accepted: bool
    roadmap: str             # synthesised roadmap text (empty if not accepted)
    accuracy_before: float
    accuracy_after: float    # 0.0 if not validated / not accepted
    n_case_studies_used: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_case_studies(case_studies: list) -> str:
    if not case_studies:
        return "(none yet)"
    lines = []
    for i, cs in enumerate(case_studies, 1):
        text = cs.render() if hasattr(cs, "render") else str(cs)
        lines.append(f"--- Case Study {i} ---\n{text.strip()}")
    return "\n\n".join(lines)


def _format_failures(failures: list[dict], max_items: int = 15) -> str:
    lines = []
    for i, it in enumerate(failures[:max_items], 1):
        trace = (it.get("post_think") or it.get("thinking") or "").strip()
        trace = (trace[:350] + " [...]") if len(trace) > 350 else trace
        lines.append(
            f"--- Failure {i} ---\n"
            f"  E1={it['equation1']}  E2={it['equation2']}\n"
            f"  expected={it.get('answer', '?')}  predicted={it.get('predicted', '?')}\n"
            f"  Reasoning: {trace or '(none)'}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Core synthesis function
# ---------------------------------------------------------------------------

def run_roadmap_synthesis(
    cheatsheet: Cheatsheet,
    failures: list[dict],
    model_casestudy: str,
    model_score: str,
    api_key: str,
    train_seen: list[dict] | None = None,
    concurrency: int = 8,
    reasoning_effort: str | None = "low",
    cot_first: bool = True,
    min_case_studies: int = 2,
    regress_tolerance: float = 0.10,   # max allowed accuracy drop (fraction of train_seen)
    log: bool = True,
) -> RoadmapSynthesisResult:
    """
    Synthesise accumulated case studies into a reasoning roadmap.

    Parameters
    ----------
    cheatsheet       : current cheatsheet — prior_knowledge is read but not
                       modified; case_studies are the material to synthesise.
    failures         : items still wrong (with reasoning traces) for context.
    train_seen       : items to validate against before/after (optional).
    min_case_studies : skip synthesis if fewer case studies than this.

    Returns
    -------
    RoadmapSynthesisResult — if accepted, cheatsheet.roadmap should be
    replaced with result.roadmap.  The caller must NOT clear case_studies:
    the roadmap is a controller over the case bank, not a replacement for it.
    """
    def _log(msg: str) -> None:
        if log:
            print(msg, file=sys.stderr, flush=True)

    if len(cheatsheet.case_studies) < min_case_studies:
        _log(
            f"  [roadmap] only {len(cheatsheet.case_studies)} case studies — "
            f"need ≥{min_case_studies} to synthesise. Skipping."
        )
        return RoadmapSynthesisResult(
            accepted=False, roadmap="",
            accuracy_before=0.0, accuracy_after=0.0,
            n_case_studies_used=0,
        )

    _log(
        f"\n  [roadmap] synthesising {len(cheatsheet.case_studies)} case studies "
        f"+ {len(failures)} failures into a reasoning roadmap ..."
    )

    prior = cheatsheet.prior_knowledge.strip() or "(none provided)"
    cs_text = _format_case_studies(cheatsheet.case_studies)
    failure_text = _format_failures(failures)

    resp = call_llm(
        ROADMAP_SYNTHESIS_PROMPT.format(
            prior_knowledge=prior,
            case_studies=cs_text,
            failure_lines=failure_text,
        ),
        model_casestudy, api_key,
        temperature=0.2,
        max_tokens=ROADMAP_SYNTHESIS_MAX_TOKENS,
        reasoning_effort=None,
    )
    roadmap = resp.content.strip()

    if not roadmap.startswith("ASPECT"):
        _log(f"  [roadmap] unexpected format — first chars: {roadmap[:80]!r}. Rejecting.")
        return RoadmapSynthesisResult(
            accepted=False, roadmap="",
            accuracy_before=0.0, accuracy_after=0.0,
            n_case_studies_used=len(cheatsheet.case_studies),
        )

    _log(f"  [roadmap] synthesised roadmap ({len(roadmap)} chars):\n{roadmap[:400]} ...")

    # ── Optional validation ──────────────────────────────────────────────────
    if not train_seen:
        _log("  [roadmap] no train_seen — accepting without accuracy validation.")
        return RoadmapSynthesisResult(
            accepted=True, roadmap=roadmap,
            accuracy_before=0.0, accuracy_after=0.0,
            n_case_studies_used=len(cheatsheet.case_studies),
        )

    _log(f"  [roadmap] validating on {len(train_seen)} items ...")

    cs_before = Cheatsheet(
        roadmap=cheatsheet.roadmap,
        case_studies=cheatsheet.case_studies,
        prior_knowledge=cheatsheet.prior_knowledge,
    )
    # Roadmap is a controller over the case bank — case studies are kept intact.
    # We measure whether the new roadmap improves navigation WITH the same case bank.
    cs_after = Cheatsheet(
        roadmap=roadmap,
        case_studies=cheatsheet.case_studies,
        prior_knowledge=cheatsheet.prior_knowledge,
    )

    correct_before, _ = score_batch(
        train_seen, cs_before.render(), model_score, api_key,
        concurrency=concurrency, reasoning_effort=reasoning_effort, cot_first=cot_first,
    )
    correct_after, _ = score_batch(
        train_seen, cs_after.render(), model_score, api_key,
        concurrency=concurrency, reasoning_effort=reasoning_effort, cot_first=cot_first,
    )

    acc_before = len(correct_before) / len(train_seen)
    acc_after  = len(correct_after)  / len(train_seen)
    delta      = acc_after - acc_before

    _log(
        f"  [roadmap] accuracy: before={acc_before:.1%}  after={acc_after:.1%}  "
        f"delta={delta:+.1%}"
    )

    # Accept if accuracy drop is within tolerance (percentage of train_seen)
    accepted = delta >= -regress_tolerance
    if accepted:
        _log("  [roadmap] synthesis ACCEPTED.")
    else:
        _log(f"  [roadmap] synthesis REJECTED (delta={delta:+.1%} < -{regress_tolerance:.1%}).")

    return RoadmapSynthesisResult(
        accepted=accepted,
        roadmap=roadmap if accepted else "",
        accuracy_before=acc_before,
        accuracy_after=acc_after if accepted else acc_before,
        n_case_studies_used=len(cheatsheet.case_studies),
    )
