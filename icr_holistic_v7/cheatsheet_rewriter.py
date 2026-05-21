"""
icr_holistic_v7/cheatsheet_rewriter.py — End-of-iteration holistic cheatsheet rewrite.

Copied from icr_holistic_old/cheatsheet_rewriter.py (the true original v7 rewriter).
Regression block is label-only (question + source bin), no wrong_cot or oracle.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from utils.llm_client import call_llm
from ICR_holistic.bin_generator import BinGeneratorOutput
from ICR_holistic.prompts import HOLISTIC_REWRITE_PROMPT, HOLISTIC_REWRITE_PROMPT_CONSERVATIVE


# Holds the raw response from the last failed rewrite attempt so the loop can save it.
_last_failed_raw: list[str] = [""]


@dataclass
class RewriteOutput:
    cheatsheet: str     # complete rewritten cheatsheet text
    analysis: str       # Step 1 analysis (everything before <CHEATSHEET>)
    raw_response: str   # full LLM response (kept for inspection)


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


def _format_regression_block(regressed: list[dict]) -> str:
    """Format regressed cases for the rewrite prompt (label-only, no wrong_cot/oracle)."""
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


def _format_regression_block_oracle(regressed: list[dict]) -> str:
    """Enriched regression block: label + wrong model CoT + oracle reasoning."""
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
        entry = f"[{i}] {q}\n     Caused by rule from bin: [{source_bin}]"
        wrong_cot = item.get("post_think", item.get("raw_response", "")).strip()
        if wrong_cot:
            entry += f"\n     Model's wrong reasoning: {wrong_cot[:400]}"
        oracle = (item.get("reason") or item.get("oracle_reasoning") or "").strip()
        if oracle:
            entry += f"\n     Oracle reasoning: {oracle[:400]}"
        lines.append(entry)
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
    oracle_injection: bool = False,
) -> Optional[RewriteOutput]:
    """
    One holistic LLM rewrite call, with up to max_retries attempts.

    Returns None if all attempts fail or produce unparseable output.
    """
    new_content_block = _format_new_content_block(accepted_bin_outputs)
    regression_block  = (
        _format_regression_block_oracle(regressed_cases)
        if oracle_injection
        else _format_regression_block(regressed_cases)
    )

    template = prompt_template if prompt_template is not None else HOLISTIC_REWRITE_PROMPT
    prompt = template.format(
        current_cheatsheet=current_cheatsheet[:cheatsheet_max_chars],
        new_content_block=new_content_block,
        regression_block=regression_block,
    )

    for attempt in range(1, max_retries + 1):
        _temp = temperature if attempt == 1 else 0.3
        try:
            resp = call_llm(
                prompt, model=model, api_key=api_key,
                max_tokens=3000, temperature=_temp, reasoning_effort=None,
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
        _last_failed_raw[0] = raw

    return None


def _parse_rewrite_output(raw: str) -> Optional[RewriteOutput]:
    cs_m = re.search(r'<CHEATSHEET>\s*(.*?)\s*</CHEATSHEET>', raw, re.DOTALL)
    if not cs_m:
        print("[rewriter] Could not find <CHEATSHEET>…</CHEATSHEET> in response")
        return None

    cheatsheet = cs_m.group(1).strip()
    analysis   = raw[:cs_m.start()].strip()

    return RewriteOutput(
        cheatsheet=cheatsheet,
        analysis=analysis,
        raw_response=raw,
    )
