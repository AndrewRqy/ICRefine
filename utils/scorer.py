"""
scorer.py — Unified scorer for all ICRefine pipelines.

Each scored item carries:
  predicted    : "TRUE" | "FALSE" | None
  expected     : "TRUE" | "FALSE"
  post_think   : REASONING section extracted from the model's structured output
  thinking     : full internal CoT trace (empty for non-reasoning models)
  raw_response : the full content string

Per Heddaya et al. (ACL 2026), post_think preserves deductive markers at
25× higher density than externally prompted summaries — it is the right
signal for identifying what went wrong in a failure.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from .data import is_true
from .llm_client import LLMResponse, call_llm_batch
from .parser import parse_response as _parse, normalize as _normalize
from ICR_naive.prompts.templates import SCORING_PROMPT, SCORING_PROMPT_COT_FIRST, SCORING_MAX_TOKENS


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_scoring_prompt(cheatsheet_text: str, item: dict, cot_first: bool = False) -> str:
    template = SCORING_PROMPT_COT_FIRST if cot_first else SCORING_PROMPT
    return template.format(
        cheatsheet=cheatsheet_text,
        equation1=item["equation1"],
        equation2=item["equation2"],
    )


# ---------------------------------------------------------------------------
# Verdict + post-think extraction
# ---------------------------------------------------------------------------

def _parse_verdict(content: str) -> str | None:
    return _parse(_normalize(content))["verdict"]


def _extract_post_think(content: str) -> str:
    """Extract REASONING section. Falls back to full content if absent."""
    parsed = _parse(_normalize(content))
    return parsed["reasoning"] or content.strip()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    accuracy: float
    correct: list[dict] = field(default_factory=list)
    wrong:   list[dict] = field(default_factory=list)
    errors:  list[dict] = field(default_factory=list)
    n_total: int = 0

    def summary(self) -> str:
        return (
            f"accuracy={self.accuracy:.1%}  "
            f"correct={len(self.correct)}  "
            f"wrong={len(self.wrong)}  "
            f"parse_errors={len(self.errors)}  "
            f"total={self.n_total}"
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_batch(
    items: list[dict],
    cheatsheet_text: str,
    model: str,
    api_key: str,
    concurrency: int = 10,
    temperature: float = 0.0,
    progress_label: str = "scoring",
    reasoning_effort: str | None = "low",
    cot_first: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Score items against the current cheatsheet in parallel.

    Returns (correct_items, wrong_items) — both annotated with predicted,
    expected, post_think, thinking, and raw_response.
    Parse errors are counted as wrong.

    cot_first: use SCORING_PROMPT_COT_FIRST (REASONING before VERDICT) to
               force a genuine reasoning trace before the verdict is stated.
    """
    prompts   = [_build_scoring_prompt(cheatsheet_text, item, cot_first) for item in items]
    responses = call_llm_batch(
        prompts,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=SCORING_MAX_TOKENS,
        concurrency=concurrency,
        progress_label=progress_label,
        reasoning_effort=reasoning_effort,
    )

    correct, wrong = [], []
    n_parse_errors = 0

    for item, resp in zip(items, responses):
        ground_truth = is_true(item["answer"])

        if resp is None:
            annotated = {
                **item,
                "predicted":    None,
                "expected":     "TRUE" if ground_truth else "FALSE",
                "post_think":   "",
                "thinking":     "",
                "raw_response": "",
            }
            wrong.append(annotated)
            n_parse_errors += 1
            continue

        predicted  = _parse_verdict(resp.content)
        post_think = _extract_post_think(resp.content)

        annotated = {
            **item,
            "predicted":    predicted,
            "expected":     "TRUE" if ground_truth else "FALSE",
            "post_think":   post_think,
            "thinking":     resp.thinking,
            "raw_response": resp.content,
        }

        if predicted is None:
            n_parse_errors += 1
            wrong.append(annotated)
        elif (predicted == "TRUE") != ground_truth:
            wrong.append(annotated)
        else:
            correct.append(annotated)

    if n_parse_errors:
        print(
            f"\n  [scorer] {n_parse_errors} parse errors (no VERDICT: line) — "
            f"counted as wrong.",
            file=sys.stderr,
        )

    return correct, wrong


def test_cheatsheet(
    cheatsheet_text: str,
    val_items: list[dict],
    model: str,
    api_key: str,
    concurrency: int = 10,
    temperature: float = 0.0,
    reasoning_effort: str | None = "low",
    cot_first: bool = False,
) -> TestResult:
    """Score cheatsheet_text on the full val_items set. Returns a TestResult."""
    print(f"  Testing on {len(val_items)} items with {model} ...", file=sys.stderr)
    correct, wrong = score_batch(
        val_items, cheatsheet_text, model, api_key,
        concurrency, temperature,
        reasoning_effort=reasoning_effort, cot_first=cot_first,
    )
    scored   = len(correct) + len(wrong)
    accuracy = len(correct) / scored if scored > 0 else 0.0
    return TestResult(accuracy=accuracy, correct=correct, wrong=wrong, n_total=len(val_items))
