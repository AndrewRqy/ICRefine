"""
ICR_select/training/gates.py — Quality gate helpers for the selection loop.

Gates (called by loop.py in order):
  _mini_eval         — score failure batch with candidate injected → fix_rate
  _mini_eval_full    — same but also returns still-wrong items (retry path)
  _replace_eval      — score failures with CS[merge_idx] replaced → fix_rate
  _regression_check  — score correct pool with candidate → regression_rate
  _similarity_gate   — LLM dedup: ADD / SKIP / MERGE
  _merge_case_studies — LLM merge of two case studies into one
  _apply_prescore    — split batch using pre-computed SAIR eval results
"""

from __future__ import annotations

import re

from utils.cheatsheet import Cheatsheet
from utils.case_study import CaseStudy
from utils.data import is_true
from utils.llm_client import call_llm
from utils.scorer import score_batch
from ..prompts.templates import (
    SIMILARITY_CHECK_PROMPT, SIMILARITY_MAX_TOKENS,
    MERGE_PROMPT, MERGE_MAX_TOKENS,
)

_MIN_CS_FOR_SIMILARITY = 3   # skip similarity gate until this many CSes exist


# ---------------------------------------------------------------------------
# Mini-eval helpers
# ---------------------------------------------------------------------------

def _mini_eval(
    candidate_cs: CaseStudy,
    failures: list[dict],
    cheatsheet: Cheatsheet,
    model_score: str,
    api_key: str,
    concurrency: int,
    reasoning_effort: str | None,
    cot_first: bool,
    label: str = "mini-eval",
) -> float:
    """Score the failure batch with candidate injected. Returns fix_rate."""
    temp = Cheatsheet(
        decision_tree=cheatsheet.decision_tree,
        case_studies=cheatsheet.case_studies + [candidate_cs],
    )
    correct, _ = score_batch(
        failures, temp.render(), model_score, api_key,
        concurrency=concurrency, reasoning_effort=reasoning_effort, cot_first=cot_first,
        progress_label=label,
    )
    return len(correct) / len(failures) if failures else 0.0


def _mini_eval_full(
    candidate_cs: CaseStudy,
    failures: list[dict],
    cheatsheet: Cheatsheet,
    model_score: str,
    api_key: str,
    concurrency: int,
    reasoning_effort: str | None,
    cot_first: bool,
    label: str = "mini-eval",
) -> tuple[float, list[dict]]:
    """Score the failure batch with candidate injected. Returns (fix_rate, still_wrong)."""
    temp = Cheatsheet(
        decision_tree=cheatsheet.decision_tree,
        case_studies=cheatsheet.case_studies + [candidate_cs],
    )
    correct, wrong = score_batch(
        failures, temp.render(), model_score, api_key,
        concurrency=concurrency, reasoning_effort=reasoning_effort, cot_first=cot_first,
        progress_label=label,
    )
    return len(correct) / len(failures) if failures else 0.0, wrong


def _replace_eval(
    merged_cs: CaseStudy,
    merge_idx: int,
    failures: list[dict],
    cheatsheet: Cheatsheet,
    model_score: str,
    api_key: str,
    concurrency: int,
    reasoning_effort: str | None,
    cot_first: bool,
) -> float:
    """Score failures with CS at merge_idx replaced by merged_cs. Returns fix_rate."""
    new_studies = cheatsheet.case_studies[:]
    new_studies[merge_idx] = merged_cs
    temp = Cheatsheet(decision_tree=cheatsheet.decision_tree, case_studies=new_studies)
    correct, _ = score_batch(
        failures, temp.render(), model_score, api_key,
        concurrency=concurrency, reasoning_effort=reasoning_effort, cot_first=cot_first,
        progress_label="merge-eval-merged",
    )
    return len(correct) / len(failures) if failures else 0.0


def _regression_check(
    candidate_cs: CaseStudy,
    correct_pool: list[dict],
    cheatsheet: Cheatsheet,
    model_score: str,
    api_key: str,
    concurrency: int,
    reasoning_effort: str | None,
    cot_first: bool,
) -> float:
    """Score the correct pool with candidate injected. Returns regression_rate."""
    if not correct_pool:
        return 0.0
    temp = Cheatsheet(
        decision_tree=cheatsheet.decision_tree,
        case_studies=cheatsheet.case_studies + [candidate_cs],
    )
    _, wrong = score_batch(
        correct_pool, temp.render(), model_score, api_key,
        concurrency=concurrency, reasoning_effort=reasoning_effort, cot_first=cot_first,
        progress_label="regression-check",
    )
    return len(wrong) / len(correct_pool)


# ---------------------------------------------------------------------------
# Similarity gate
# ---------------------------------------------------------------------------

def _format_existing(case_studies: list[CaseStudy]) -> str:
    parts = [f"[{i}]\n{cs.render()}" for i, cs in enumerate(case_studies, 1)]
    return "\n\n".join(parts) if parts else "(none yet)"


def _similarity_gate(
    candidate_cs: CaseStudy,
    cheatsheet: Cheatsheet,
    model_casestudy: str,
    api_key: str,
) -> tuple[str, int | None]:
    """
    Check if candidate duplicates an existing case study.

    Returns:
        ('ADD',   None)  — new pattern, add it
        ('SKIP',  None)  — duplicate, discard
        ('MERGE', N)     — merge into case study at index N (0-based)
    """
    if not cheatsheet.case_studies:
        return "ADD", None

    resp = call_llm(
        SIMILARITY_CHECK_PROMPT.format(
            existing=_format_existing(cheatsheet.case_studies),
            candidate=candidate_cs.render(),
        ),
        model_casestudy, api_key,
        temperature=0.0,
        max_tokens=SIMILARITY_MAX_TOKENS,
        reasoning_effort=None,
    )
    raw = resp.content.strip().upper()

    if raw.startswith("SKIP"):
        return "SKIP", None
    if raw.startswith("MERGE"):
        m = re.search(r"\d+", raw)
        if m:
            idx = int(m.group()) - 1   # prompt uses 1-based numbering
            if 0 <= idx < len(cheatsheet.case_studies):
                return "MERGE", idx
    return "ADD", None


def _merge_case_studies(cs_a: CaseStudy, cs_b: CaseStudy, model_casestudy: str, api_key: str) -> CaseStudy:
    """Merge two case studies into one via LLM. Returns a new CaseStudy."""
    from utils.case_study import CaseStudy as _CS
    resp = call_llm(
        MERGE_PROMPT.format(cs_a=cs_a.render(), cs_b=cs_b.render()),
        model_casestudy, api_key,
        temperature=0.2,
        max_tokens=MERGE_MAX_TOKENS,
        reasoning_effort=None,
    )
    merged = _CS.from_text(resp.content.strip())
    # Carry forward the higher fix rate from the two sources
    merged.creation_fix_rate = max(cs_a.creation_fix_rate, cs_b.creation_fix_rate)
    merged.historical_fix_rate = merged.creation_fix_rate
    return merged


# ---------------------------------------------------------------------------
# Prescore helper
# ---------------------------------------------------------------------------

def _apply_prescore(
    batch: list[dict],
    prescore_map: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """
    Split a batch into (correct, wrong) using pre-computed eval scores,
    avoiding a redundant scoring pass when SAIR eval results are available.

    Items not found in prescore_map are treated as wrong so they surface
    in the failure bin.
    """
    correct, wrong = [], []
    for item in batch:
        pre = prescore_map.get(item.get("id", ""))
        ground_truth = is_true(item["answer"])
        base = {
            **item,
            "expected":     "TRUE" if ground_truth else "FALSE",
            "post_think":   "",
            "thinking":     "",
            "raw_response": "",
        }
        if pre is None:
            wrong.append({**base, "predicted": None})
            continue
        annotated = {
            **base,
            "predicted":    pre.get("predicted"),
            "post_think":   pre.get("post_think", ""),
            "thinking":     pre.get("thinking", ""),
            "raw_response": pre.get("raw_response", ""),
        }
        if pre.get("correct"):
            correct.append(annotated)
        else:
            wrong.append(annotated)
    return correct, wrong
