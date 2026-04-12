"""
ICR_select/generators/case_study.py — Multi-candidate case study generator.

Generates N candidates per bin flush in parallel at different temperatures,
so the selection loop has real diversity to choose from rather than running
the same prompt twice and hoping the output differs.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.cheatsheet import Cheatsheet, extract_query_features
from utils.case_study import CaseStudy
from utils.llm_client import call_llm
from ICR_reasoning.core.oracle import OracleDict
from ICR_reasoning.generators.case_study import _format_failures_with_reasoning, _parse_response, _render_case_studies_text
from ..prompts.templates import (
    CASE_STUDY_WITH_REASONING_PROMPT,
    RETRY_CONTEXT_TEMPLATE,
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
    prev_attempt: dict | None = None,
) -> list[CaseStudy]:
    """
    Generate *n* candidate case study strings in parallel at different temperatures.

    Returns a list of case study strings (the CASE STUDY section only, without
    the DT patch), ordered by temperature ascending.  Failures are silently
    dropped — a failed generation produces an empty string, which the caller
    filters out.

    oracle: optional (eq1, eq2) -> correct_reasoning dict; when provided, each
            failure that has a matching oracle entry will show the correct
            reasoning as a contrast signal alongside the wrong model reasoning.

    prev_attempt: optional dict with keys "candidate" (str), "still_wrong" (list[dict]),
            and "reason" (str — "fix_rate" or "regression"). When provided (retry
            flush strategy), the previous candidate and its still-wrong items are
            appended to the prompt so the model knows what was tried and what failed.
    """
    temps = (temperatures or CANDIDATE_TEMPS)[:n]

    failure_lines = _format_failures_with_reasoning(failures, oracle=oracle)
    if prev_attempt:
        reason_desc = (
            "it fixed too few failures (fix-rate gate)"
            if prev_attempt["reason"] == "fix_rate"
            else "it broke too many previously-correct items (regression gate)"
        )
        still_wrong = prev_attempt["still_wrong"]
        still_wrong_lines = _format_failures_with_reasoning(still_wrong)
        # prev_attempt["candidate"] is a CaseStudy — render it for the prompt
        prev_cand = prev_attempt["candidate"]
        prev_cand_text = prev_cand.render() if isinstance(prev_cand, CaseStudy) else str(prev_cand).strip()
        prev_section = RETRY_CONTEXT_TEMPLATE.format(
            reason_desc=reason_desc,
            prev_candidate=prev_cand_text,
            n_still_wrong=len(still_wrong),
            still_wrong_lines=still_wrong_lines,
        )
        failure_lines = failure_lines + "\n\n" + prev_section

    prompt = CASE_STUDY_WITH_REASONING_PROMPT.format(
        roadmap=cheatsheet.roadmap.strip(),
        case_studies=_render_case_studies_text(cheatsheet),
        failure_lines=failure_lines,
    )

    _COMPLETION_RETRY_PROMPT = """\
The case study below is INCOMPLETE — it is missing the following required fields: {missing}.

Please rewrite it in full, adding the missing fields. Keep all existing content intact.
Required format:
=== CASE STUDY: [title] ===
FAILURE_TYPE: A or B
ACTIVATE IF:
  - [condition 1]
  - [condition 2]
DO NOT ACTIVATE IF: [boundary case where this should not fire]
COMMON WRONG MOVE: [what the weaker model does wrong]
NEXT CHECK: [mechanical check to perform instead — end with "If yes → TRUE/FALSE."]
WHY THIS WORKS: [1–2 sentence justification]
SUPPORT:
  • E1 = ...  |  E2 = ...  |  Answer: TRUE/FALSE  — [brief note]
  • E1 = ...  |  E2 = ...  |  Answer: TRUE/FALSE  — [brief note]
TARGET_STEP: [roadmap aspect this corrects]

=== INCOMPLETE CASE STUDY ===
{incomplete_text}
=== END ===

Output ONLY the completed case study starting with === CASE STUDY: ..."""

    def _call(temp: float) -> CaseStudy | None:
        try:
            resp = call_llm(
                prompt, model, api_key,
                temperature=temp,
                max_tokens=FLUSH_MAX_TOKENS,
                reasoning_effort=None,
            )
            result = _parse_response(resp.content)
            cs = result.case_study
            if cs is None:
                return None

            ok, missing = cs.is_complete()
            if ok:
                return cs

            # Retry once with the incomplete output fed back to the LLM
            print(
                f"  [candidate gen] temp={temp} incomplete (missing: {', '.join(missing)}) — retrying",
                file=sys.stderr,
            )
            retry_prompt = _COMPLETION_RETRY_PROMPT.format(
                missing=", ".join(missing),
                incomplete_text=cs.raw_text.strip(),
            )
            retry_resp = call_llm(
                retry_prompt, model, api_key,
                temperature=0.3,
                max_tokens=FLUSH_MAX_TOKENS,
                reasoning_effort=None,
            )
            retry_result = _parse_response(retry_resp.content)
            retry_cs = retry_result.case_study
            if retry_cs is None:
                return None
            ok2, missing2 = retry_cs.is_complete()
            if not ok2:
                print(
                    f"  [candidate gen] temp={temp} still incomplete after retry "
                    f"(missing: {', '.join(missing2)}) — dropping",
                    file=sys.stderr,
                )
                return None
            return retry_cs

        except Exception as exc:
            print(f"  [candidate gen] temp={temp} failed: {exc}", file=sys.stderr)
            return None

    candidates: list[CaseStudy | None] = [None] * len(temps)
    with ThreadPoolExecutor(max_workers=len(temps)) as pool:
        futures = {pool.submit(_call, t): i for i, t in enumerate(temps)}
        for fut in as_completed(futures):
            candidates[futures[fut]] = fut.result()

    # Auto-compute feature_signature from failure structural features when the
    # LLM didn't write a FEATURE_SIGNATURE: line.  Without this, build_vmatch
    # returns [] for every candidate → utility gate always falls back.
    #
    # TYPE A (missing knowledge): the lemma applies to an entire E1 form class,
    # not a specific structural pair.  Use only E1's form token so the case study
    # routes broadly to every query where that lemma condition holds.
    # TYPE B (wrong reasoning pattern): mistake is configuration-specific;
    # keep the full pair signature so routing stays narrow and precise.
    failure_qfs = []
    for item in failures:
        try:
            failure_qfs.append(extract_query_features(item))
        except Exception:
            pass

    full_pair_sig = failure_qfs[0].signature() if failure_qfs else ""
    e1_form_sig   = failure_qfs[0].form_e1.lower() if failure_qfs else ""

    valid = [c for c in candidates if c is not None]
    if not valid:
        raise RuntimeError("All candidate generations failed or were incomplete after retry.")

    for c in valid:
        if not c.feature_signature:
            # TYPE A → broad scope (E1 form only); TYPE B or unknown → full pair
            c.feature_signature = e1_form_sig if c.failure_type == "A" else full_pair_sig

    print(
        f"  [candidates] {len(valid)}/{n} candidates complete and valid",
        file=sys.stderr,
    )
    return valid
