"""
generators/case_study.py — Generate a new case study from the failure bin.

Called by the training loop whenever the bin reaches its threshold.
The LLM is shown the current cheatsheet + the failure examples and asked to
write a single new case study identifying the pattern behind those failures.
"""

from __future__ import annotations

import sys

from ..core.data import is_true
from ..core.llm_client import call_llm
from ..prompts.templates import CASE_STUDY_PROMPT, CS_MAX_TOKENS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_failures(failures: list[dict]) -> str:
    lines = []
    for i, it in enumerate(failures, 1):
        expected  = "TRUE" if is_true(it["answer"]) else "FALSE"
        predicted = it.get("predicted", "?")
        lines.append(
            f"  {i:3d}. E1 = {it['equation1']}"
            f"  |  E2 = {it['equation2']}"
            f"  |  expected={expected}  predicted={predicted}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def generate_case_study(
    failures: list[dict],
    cheatsheet_text: str,
    model: str,
    api_key: str,
    temperature: float = 0.3,
) -> str:
    """
    Generate a new case study from a batch of failure examples.

    Parameters
    ----------
    failures        : items the cheatsheet predicted incorrectly
                      (each has equation1, equation2, answer, predicted)
    cheatsheet_text : current rendered cheatsheet text (Cheatsheet.render())
    model           : OpenRouter model ID
    api_key         : OpenRouter API key
    temperature     : generation temperature

    Returns
    -------
    A new case study string ready to be passed to Cheatsheet.add_case_study().
    """
    if not failures:
        raise ValueError("generate_case_study called with empty failures list.")

    print(
        f"  [bin flush] Generating case study from {len(failures)} failures "
        f"with {model} ...",
        file=sys.stderr,
    )
    resp = call_llm(
        CASE_STUDY_PROMPT.format(
            cheatsheet=cheatsheet_text,
            failure_lines=_format_failures(failures),
        ),
        model, api_key,
        temperature=temperature,
        max_tokens=CS_MAX_TOKENS,
    )
    return resp.content
