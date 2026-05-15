"""
ICR_holistic/cheatsheet_rewriter.py — End-of-iteration holistic cheatsheet rewrite.

One LLM call per iteration that sees:
  1. The current cheatsheet
  2. All accepted bin outputs (RULE or EXAMPLE, one per structural bin)
  3. All cases that were correct before but flipped wrong after a bin rule was added

The model analyses the tensions between new rules and regressions, then rewrites
the full cheatsheet to incorporate the new content with appropriate scope limits.

Output is parsed from <CHEATSHEET>…</CHEATSHEET> tags.
The analysis (Step 1) is saved separately as analysis_{tag}.txt for inspection.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from utils.llm_client import call_llm
from .bin_generator import BinGeneratorOutput
from .prompts import (
    HOLISTIC_REWRITE_PROMPT,
    HOLISTIC_REWRITE_PROMPT_CONSERVATIVE,
    HOLISTIC_REWRITE_PROMPT_SELECTIVE,
)


# Holds the raw response from the last failed rewrite attempt so the loop can save it.
_last_failed_raw: list[str] = [""]


@dataclass
class RewriteOutput:
    cheatsheet: str          # complete rewritten cheatsheet text
    analysis: str            # Step 1 analysis (everything before <CHEATSHEET>)
    raw_response: str        # full LLM response (kept for inspection)
    deferred_labels: list    # bin labels the model chose NOT to merge this iter


def _format_new_content_block(accepted: list[tuple[str, BinGeneratorOutput]]) -> str:
    """Format accepted bin outputs for the rewrite prompt."""
    if not accepted:
        return "(none)"
    parts = []
    for bin_label, out in accepted:
        rationale = out.reasoning[:200].strip() if out.reasoning else "see content"
        parts.append(
            f"[{bin_label}]\n"
            f"TYPE: {out.content_type}\n"
            f"{out.content}\n"
            f"(Rationale: {rationale})"
        )
    return "\n\n---\n\n".join(parts)


def _format_pending_pool_block(pending: list[tuple[str, BinGeneratorOutput]]) -> str:
    """Format pending (deferred) bin candidates for the selective rewrite prompt."""
    if not pending:
        return "(none — no candidates pending from previous iterations)"
    parts = []
    for bin_label, out in pending:
        rationale = out.reasoning[:200].strip() if out.reasoning else "see content"
        parts.append(
            f"[{bin_label}]\n"
            f"TYPE: {out.content_type}\n"
            f"{out.content}\n"
            f"(Rationale: {rationale})"
        )
    return "\n\n---\n\n".join(parts)


def _format_caution_block(caution_cases: list[dict]) -> str:
    """Format items broken by the previous rewrite attempt."""
    lines = []
    for i, item in enumerate(caution_cases[:10], 1):
        eq1 = item.get("equation1", item.get("input", item.get("question", "?")))
        eq2 = item.get("equation2", "")
        answer = item.get("answer", "?")
        if eq2:
            q = f'Does "{eq1}" imply "{eq2}"?  (correct: {answer})'
        else:
            q = f'{eq1}  (correct: {answer})'
        lines.append(f"[{i}] {q}")
    return "\n".join(lines)


def _format_regression_block(regressed: list[dict]) -> str:
    """Format regressed cases for the rewrite prompt."""
    if not regressed:
        return "(none — no regressions detected)"
    lines = []
    for i, item in enumerate(regressed[:15], 1):
        eq1 = item.get("equation1", item.get("input", "?"))
        eq2 = item.get("equation2", "")
        answer = item.get("answer", "?")
        source_bin = item.get("_regression_source_bin", "unknown")
        if eq2:
            q = f'Does "{eq1}" imply "{eq2}"?  (correct: {answer})'
        else:
            q = f'{eq1}  (correct: {answer})'
        lines.append(f"[{i}] {q}\n     Caused by rule from bin: [{source_bin}]")
    return "\n\n".join(lines)


def rewrite_cheatsheet(
    current_cheatsheet: str,
    accepted_bin_outputs: list[tuple[str, BinGeneratorOutput]],
    regressed_cases: list[dict],
    model: str,
    api_key: str,
    cheatsheet_max_chars: int = 4000,
    max_retries: int = 2,
    prompt_template: str | None = None,
    temperature: float = 0.0,
    rewriter_max_tokens: int = 3000,
    pending_pool: list[tuple[str, BinGeneratorOutput]] | None = None,
    caution_cases: list[dict] | None = None,
) -> Optional[RewriteOutput]:
    """
    One holistic LLM rewrite call, with up to max_retries attempts.

    Returns None if all attempts fail or produce unparseable output.
    The raw LLM response is always returned inside RewriteOutput so the caller
    can save it for inspection even on partial failures.

    pending_pool: candidates deferred from previous iterations (used with
    HOLISTIC_REWRITE_PROMPT_SELECTIVE / --slowandsteady mode). The model
    selects at most 3 to merge and lists the rest in <DEFERRED> tags.

    caution_cases: items that were previously correct but broken by the last
    rewrite attempt. Injected into the prompt so the model avoids repeating
    the same regression.
    """
    new_content_block  = _format_new_content_block(accepted_bin_outputs)
    regression_block   = _format_regression_block(regressed_cases)

    if pending_pool is not None:
        template = (prompt_template
                    if prompt_template is not None
                    else HOLISTIC_REWRITE_PROMPT_SELECTIVE)
        pending_pool_block = _format_pending_pool_block(pending_pool)
        prompt = template.format(
            current_cheatsheet=current_cheatsheet[:cheatsheet_max_chars],
            new_content_block=new_content_block,
            pending_pool_block=pending_pool_block,
            regression_block=regression_block,
        )
    else:
        template = prompt_template if prompt_template is not None else HOLISTIC_REWRITE_PROMPT
        prompt = template.format(
            current_cheatsheet=current_cheatsheet[:cheatsheet_max_chars],
            new_content_block=new_content_block,
            regression_block=regression_block,
        )

    if caution_cases:
        caution_text = _format_caution_block(caution_cases)
        caution_section = (
            "\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "⚠  RETRY CAUTION — BROKEN BY YOUR PREVIOUS REWRITE ATTEMPT\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Your previous rewrite caused these CORRECT items to flip WRONG.\n"
            "Your revised cheatsheet must NOT break them:\n\n"
            f"{caution_text}\n"
        )
        prompt = prompt.replace("\n<CHEATSHEET>", caution_section + "\n<CHEATSHEET>")

    for attempt in range(1, max_retries + 1):
        _temp = temperature if attempt == 1 else 0.3
        try:
            resp = call_llm(
                prompt, model=model, api_key=api_key,
                max_tokens=rewriter_max_tokens, temperature=_temp, reasoning_effort=None,
            )
            raw = resp.content.strip()
        except Exception as e:
            print(f"[rewriter] LLM call failed (attempt {attempt}): {e}")
            continue

        result = _parse_rewrite_output(raw)
        if result is not None:
            return result
        print(f"[rewriter] Parse failed (attempt {attempt}) — "
              f"no <CHEATSHEET> tags found; raw saved for inspection")
        # Store raw on the result object so caller can save it
        _last_failed_raw[0] = raw

    return None


def _parse_rewrite_output(raw: str) -> Optional[RewriteOutput]:
    cs_m = re.search(r'<CHEATSHEET>\s*(.*?)\s*</CHEATSHEET>', raw, re.DOTALL)
    if not cs_m:
        print("[rewriter] Could not find <CHEATSHEET>…</CHEATSHEET> in response")
        return None

    cheatsheet = cs_m.group(1).strip()
    analysis   = raw[:cs_m.start()].strip()

    # Parse deferred bin labels (only present in selective / slowandsteady mode).
    deferred_labels: list[str] = []
    def_m = re.search(r'<DEFERRED>\s*(.*?)\s*</DEFERRED>', raw, re.DOTALL)
    if def_m:
        for line in def_m.group(1).splitlines():
            line = line.strip().strip("[]")
            if line and line.lower() != "(none)":
                deferred_labels.append(line)

    return RewriteOutput(
        cheatsheet=cheatsheet,
        analysis=analysis,
        raw_response=raw,
        deferred_labels=deferred_labels,
    )
