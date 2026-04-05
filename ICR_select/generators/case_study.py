"""
ICR_select/generators/case_study.py — Multi-candidate case study generator.

Generates N candidates per bin flush in parallel at different temperatures,
so the selection loop has real diversity to choose from rather than running
the same prompt twice and hoping the output differs.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from ICR_naive.core.cheatsheet import Cheatsheet
from ICR_naive.core.data import _is_true
from ICR_reasoning.core.llm_client import call_llm
from ICR_reasoning.core.oracle import OracleDict
from ICR_reasoning.generators.case_study import _format_failures_with_reasoning, _parse_response, _render_case_studies_text
from ..prompts.templates import (
    CASE_STUDY_WITH_REASONING_PROMPT,
    FLUSH_MAX_TOKENS,
    N_CANDIDATES,
    CANDIDATE_TEMPS,
)


def generate_candidates(
    failures: list[dict],
    cheatsheet: Cheatsheet,
    model: str,
    api_key: str,
    n: int = N_CANDIDATES,
    temperatures: list[float] | None = None,
    oracle: OracleDict | None = None,
) -> list[str]:
    """
    Generate *n* candidate case study strings in parallel at different temperatures.

    Returns a list of case study strings (the CASE STUDY section only, without
    the DT patch), ordered by temperature ascending.  Failures are silently
    dropped — a failed generation produces an empty string, which the caller
    filters out.

    oracle: optional (eq1, eq2) -> correct_reasoning dict; when provided, each
            failure that has a matching oracle entry will show the correct
            reasoning as a contrast signal alongside the wrong model reasoning.
    """
    temps = (temperatures or CANDIDATE_TEMPS)[:n]
    prompt = CASE_STUDY_WITH_REASONING_PROMPT.format(
        decision_tree=cheatsheet.decision_tree.strip(),
        case_studies=_render_case_studies_text(cheatsheet),
        failure_lines=_format_failures_with_reasoning(failures, oracle=oracle),
    )

    def _call(temp: float) -> str:
        try:
            resp = call_llm(
                prompt, model, api_key,
                temperature=temp,
                max_tokens=FLUSH_MAX_TOKENS,
                reasoning_effort=None,
            )
            result = _parse_response(resp.content)
            return result.case_study.strip()
        except Exception as exc:
            print(f"  [candidate gen] temp={temp} failed: {exc}", file=sys.stderr)
            return ""

    candidates: list[str] = [""] * len(temps)
    with ThreadPoolExecutor(max_workers=len(temps)) as pool:
        futures = {pool.submit(_call, t): i for i, t in enumerate(temps)}
        for fut in as_completed(futures):
            candidates[futures[fut]] = fut.result()

    valid = [c for c in candidates if c]
    if not valid:
        raise RuntimeError("All candidate generations failed.")

    print(
        f"  [candidates] generated {len(valid)}/{n} valid candidates",
        file=sys.stderr,
    )
    return valid
