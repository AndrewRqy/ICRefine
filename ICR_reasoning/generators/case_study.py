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
  2. A DECISION TREE PATCH with targeted corrections to the DT (appended to
     decision_tree via Cheatsheet.patch_decision_tree()).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

from ICR_naive.core.cheatsheet import Cheatsheet
from ICR_naive.core.data import _is_true
from ..core.llm_client import call_llm
from ..prompts.templates import CASE_STUDY_WITH_REASONING_PROMPT, FLUSH_MAX_TOKENS


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CaseStudyResult:
    case_study: str   # the === CASE STUDY === section, ready for add_case_study()
    dt_patch:   str   # the === DECISION TREE PATCH === body (may be empty)


# ---------------------------------------------------------------------------
# Failure formatting — includes post_think
# ---------------------------------------------------------------------------

def _format_failures_with_reasoning(failures: list[dict]) -> str:
    lines = []
    for i, it in enumerate(failures, 1):
        expected   = "TRUE" if _is_true(it["answer"]) else "FALSE"
        predicted  = it.get("predicted", "?")
        post_think = it.get("post_think", "").strip()

        lines.append(
            f"--- Failure {i} ---\n"
            f"  E1 = {it['equation1']}\n"
            f"  E2 = {it['equation2']}\n"
            f"  expected={expected}  predicted={predicted}\n"
            f"  Model's reasoning (post-think):\n"
            f"    {post_think if post_think else '(not captured)'}"
        )
    return "\n\n".join(lines)


def _render_case_studies_text(cheatsheet: Cheatsheet) -> str:
    """Render just the case studies as a numbered list for the prompt."""
    if not cheatsheet.case_studies:
        return "(none yet)"
    parts = []
    for i, cs in enumerate(cheatsheet.case_studies, 1):
        parts.append(f"--- Case Study {i} ---\n{cs.strip()}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_response(text: str) -> CaseStudyResult:
    """
    Split the LLM response into the CASE STUDY block and the DECISION TREE PATCH block.

    Expected format:
        === CASE STUDY: <title> ===
        ...
        === DECISION TREE PATCH ===
        ...
        === END PATCH ===
    """
    # Extract DECISION TREE PATCH body (between header and END PATCH)
    dt_match = re.search(
        r"=== DECISION TREE PATCH ===\s*\n(.*?)=== END PATCH ===",
        text, re.DOTALL | re.IGNORECASE,
    )
    dt_patch = dt_match.group(1).strip() if dt_match else ""

    # Extract CASE STUDY block — everything up to (but not including) the DT patch section
    cs_end = dt_match.start() if dt_match else len(text)
    case_study = text[:cs_end].strip()

    # Fallback: if we couldn't identify a case study, use the full response
    if not case_study:
        case_study = text.strip()

    return CaseStudyResult(case_study=case_study, dt_patch=dt_patch)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def generate_case_study_with_reasoning(
    failures: list[dict],
    cheatsheet: Cheatsheet,
    model: str,
    api_key: str,
    temperature: float = 0.3,
) -> CaseStudyResult:
    """
    Generate a new case study AND a decision tree patch using each failure's
    post-think as the teaching signal.

    Parameters
    ----------
    failures   : items the cheatsheet predicted incorrectly;
                 each must have post_think (from scorer.py)
    cheatsheet : current Cheatsheet object (used to extract DT and case studies
                 separately for the prompt)
    model      : model ID for case study generation
    api_key    : API key
    temperature: generation temperature

    Returns
    -------
    CaseStudyResult with case_study (for add_case_study) and dt_patch (for
    patch_decision_tree).
    """
    if not failures:
        raise ValueError("generate_case_study_with_reasoning called with empty failures list.")

    print(
        f"  [bin flush] Generating reasoning-aware case study + DT patch "
        f"from {len(failures)} failures with {model} ...",
        file=sys.stderr,
    )

    prompt = CASE_STUDY_WITH_REASONING_PROMPT.format(
        decision_tree=cheatsheet.decision_tree.strip(),
        case_studies=_render_case_studies_text(cheatsheet),
        failure_lines=_format_failures_with_reasoning(failures),
    )

    resp = call_llm(
        prompt,
        model, api_key,
        temperature=temperature,
        max_tokens=FLUSH_MAX_TOKENS,
        reasoning_effort=None,   # case study generation doesn't need reasoning mode
    )
    return _parse_response(resp.content)
