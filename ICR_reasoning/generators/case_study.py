"""
ICR_reasoning/generators/case_study.py — Case study generator using post-think.

The key difference from ICR_naive: each failure's post_think (the REASONING
section the model wrote in its structured output) is included in the prompt.
This gives the case study LLM direct access to the logical path the model
followed — why it reached the wrong verdict — rather than just the input/output pair.

Per Heddaya et al. (ACL 2026), this post-think text preserves deductive markers
("therefore", "hence", etc.) at 25× higher density than externally prompted
summaries, making it a better signal for identifying and correcting reasoning flaws.

The model is asked to produce TWO sections:
  1. A CASE STUDY for the cheatsheet (appended to case_studies list).
  2. A ROADMAP PATCH with targeted corrections to the roadmap (appended to
     roadmap via Cheatsheet.patch_roadmap()).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

from utils.cheatsheet import Cheatsheet
from utils.case_study import CaseStudy
from utils.data import is_true
from ..core.llm_client import call_llm
from ..core.oracle import OracleDict
from ..prompts.templates import CASE_STUDY_WITH_REASONING_PROMPT, FLUSH_MAX_TOKENS


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CaseStudyResult:
    case_study:    CaseStudy   # structured record, ready for add_case_study()
    dt_patch:      str         # backward-compat alias — same as roadmap_patch
    roadmap_patch: str         # the === ROADMAP PATCH === body (may be empty)


# ---------------------------------------------------------------------------
# Failure formatting — includes post_think
# ---------------------------------------------------------------------------

def _format_failures_with_reasoning(
    failures: list[dict],
    oracle: OracleDict | None = None,
) -> str:
    lines = []
    for i, it in enumerate(failures, 1):
        expected   = "TRUE" if is_true(it["answer"]) else "FALSE"
        predicted  = it.get("predicted", "?")
        post_think = it.get("post_think", "").strip()

        block = (
            f"--- Failure {i} ---\n"
            f"  E1 = {it['equation1']}\n"
            f"  E2 = {it['equation2']}\n"
            f"  expected={expected}  predicted={predicted}\n"
            f"  WRONG reasoning (model's post-think):\n"
            f"    {post_think if post_think else '(not captured)'}"
        )

        # Priority 1: exact-match oracle (same equation pair — strongest signal)
        if oracle:
            key = (it["equation1"].strip(), it["equation2"].strip())
            exact_reasoning = oracle.get(key, "")
            if exact_reasoning:
                block += (
                    f"\n  CORRECT reasoning (oracle — exact same pair):\n"
                    f"    {exact_reasoning}"
                )

        # Priority 2: nearest-neighbour oracle (different pair, similar structure)
        # Added by the disagreement mining router in loop.py
        nearest = it.get("oracle_nearest")
        if nearest and not (oracle and (it["equation1"].strip(), it["equation2"].strip()) in oracle):
            sim   = it.get("oracle_sim", 0.0)
            block += (
                f"\n  STRUCTURALLY SIMILAR correct oracle case (sim={sim:.2f}):\n"
                f"    E1' = {nearest['eq1']}\n"
                f"    E2' = {nearest['eq2']}\n"
                f"    CORRECT reasoning:\n"
                f"      {nearest['reasoning']}"
            )

        lines.append(block)
    return "\n\n".join(lines)


def _render_case_studies_text(cheatsheet: Cheatsheet) -> str:
    """Render just the case studies as a numbered list for the prompt."""
    if not cheatsheet.case_studies:
        return "(none yet)"
    parts = []
    for i, cs in enumerate(cheatsheet.case_studies, 1):
        parts.append(f"--- Case Study {i} ---\n{cs.render()}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_response(text: str) -> CaseStudyResult:
    """
    Split the LLM response into a structured CaseStudy and the ROADMAP PATCH.

    Expected format:
        === CASE STUDY: <title> ===
        ACTIVATE IF: ...
        DO NOT ACTIVATE IF: ...
        COMMON WRONG MOVE: ...
        NEXT CHECK: ...
        WHY THIS WORKS: ...
        SUPPORT: ...
        FEATURE_SIGNATURE: ...
        TARGET_STEP: ...
        === ROADMAP PATCH ===
        ...
        === END PATCH ===
    """
    # Accept both new ("ROADMAP PATCH") and old ("DECISION TREE PATCH") headers
    patch_match = re.search(
        r"=== (?:ROADMAP PATCH|DECISION TREE PATCH) ===\s*\n(.*?)=== END PATCH ===",
        text, re.DOTALL | re.IGNORECASE,
    )
    roadmap_patch = patch_match.group(1).strip() if patch_match else ""

    # Extract CASE STUDY block — everything up to (but not including) the patch section
    cs_end = patch_match.start() if patch_match else len(text)
    cs_text = text[:cs_end].strip() or text.strip()

    case_study = CaseStudy.from_text(cs_text)
    return CaseStudyResult(
        case_study=case_study,
        dt_patch=roadmap_patch,       # backward-compat alias
        roadmap_patch=roadmap_patch,
    )


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def generate_case_study_with_reasoning(
    failures: list[dict],
    cheatsheet: Cheatsheet,
    model: str,
    api_key: str,
    temperature: float = 0.3,
    oracle: OracleDict | None = None,
) -> CaseStudyResult:
    """
    Generate a new case study AND a roadmap patch using each failure's
    post-think as the teaching signal.

    Parameters
    ----------
    failures   : items the cheatsheet predicted incorrectly;
                 each must have post_think (from scorer.py)
    cheatsheet : current Cheatsheet object (used to extract roadmap and case studies
                 separately for the prompt)
    model      : model ID for case study generation
    api_key    : API key
    temperature: generation temperature
    oracle     : optional dict mapping (eq1, eq2) -> correct reasoning text;
                 when provided, the correct reasoning is shown alongside the
                 wrong model reasoning as a contrast signal

    Returns
    -------
    CaseStudyResult with case_study (for add_case_study) and roadmap_patch (for
    patch_roadmap).
    """
    if not failures:
        raise ValueError("generate_case_study_with_reasoning called with empty failures list.")

    n_with_oracle = sum(
        1 for it in failures
        if oracle and (it["equation1"].strip(), it["equation2"].strip()) in oracle
    ) if oracle else 0
    print(
        f"  [bin flush] Generating reasoning-aware case study + DT patch "
        f"from {len(failures)} failures with {model} "
        f"({n_with_oracle} have oracle contrast) ...",
        file=sys.stderr,
    )

    prompt = CASE_STUDY_WITH_REASONING_PROMPT.format(
        roadmap=cheatsheet.roadmap.strip(),
        case_studies=_render_case_studies_text(cheatsheet),
        failure_lines=_format_failures_with_reasoning(failures, oracle=oracle),
    )

    resp = call_llm(
        prompt,
        model, api_key,
        temperature=temperature,
        max_tokens=FLUSH_MAX_TOKENS,
        reasoning_effort=None,   # case study generation doesn't need reasoning mode
    )
    return _parse_response(resp.content)
