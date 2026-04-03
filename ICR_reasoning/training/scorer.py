"""
ICR_reasoning/training/scorer.py — Scorer that captures post-think per item.

Key difference from ICR_naive: each scored item carries two extra fields:
  post_think : the REASONING section extracted from content (the model's
               structured explanation after internal reasoning)
  thinking   : the full raw CoT trace from the reasoning API field

Per Heddaya et al. (ACL 2026), post_think preserves deductive markers at
25× higher density than externally prompted summaries — it is the right
distillation signal for identifying what went wrong in a failure.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from ICR_naive.core.cheatsheet import Cheatsheet
from ICR_naive.core.data import _is_true
from ..core.llm_client import LLMResponse, call_llm_batch
from ..prompts.templates import SCORING_PROMPT, SCORING_PROMPT_COT_FIRST, SCORING_MAX_TOKENS

# Re-use SAIR's battle-tested parser: anchored ^VERDICT: with re.MULTILINE,
# multi-line _extract_section for REASONING, and None for unparseable verdicts.
import re as _re
import sys as _sys
import os as _os
_sair_path = str(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", "SAIR_evaluation_pipeline"))
if _sair_path not in _sys.path:
    _sys.path.insert(0, _sair_path)
from pipeline.parser import parse_response as _sair_parse

# Strips markdown bold/italic markers that some models wrap around headers,
# e.g. "**VERDICT:** TRUE" → "VERDICT: TRUE"
_MD_BOLD_RE = _re.compile(r"\*{1,2}(VERDICT|REASONING|PROOF|COUNTEREXAMPLE):\*{0,2}", _re.IGNORECASE)


def _normalize(content: str) -> str:
    """Remove markdown bold/italic from section headers so the SAIR parser can find them."""
    return _MD_BOLD_RE.sub(lambda m: m.group(1).upper() + ":", content)


# ---------------------------------------------------------------------------
# Post-think extraction — delegates to SAIR parser for REASONING section
# ---------------------------------------------------------------------------

def _extract_post_think(content: str) -> str:
    """
    Extract the REASONING section using SAIR's multi-line section extractor.
    Falls back to full content if no REASONING header is found.
    """
    parsed = _sair_parse(_normalize(content))
    return parsed["reasoning"] or content.strip()


def _parse_verdict(content: str) -> str | None:
    """
    Extract VERDICT using SAIR's anchored regex (^VERDICT: with re.MULTILINE).
    Returns None for unparseable responses — these are NOT counted as wrong.
    """
    return _sair_parse(_normalize(content))["verdict"]


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
    temperature: float = 1.0,
    progress_label: str = "scoring",
    reasoning_effort: str | None = "low",
    cot_first: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Score items and return (correct, wrong).
    Each item is annotated with: predicted, expected, post_think, thinking, raw_response.

    cot_first: use SCORING_PROMPT_COT_FIRST (REASONING before VERDICT) so the model
               cannot anchor on a verdict and reverse-engineer justification.
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
        ground_truth = _is_true(item["answer"])

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
            # Unparseable verdict — send to wrong bin so the failure
            # is visible to the case study generator, but log it separately.
            n_parse_errors += 1
            wrong.append(annotated)
        elif (predicted == "TRUE") != ground_truth:
            wrong.append(annotated)
        else:
            correct.append(annotated)

    if n_parse_errors:
        print(
            f"\n  [scorer] {n_parse_errors} parse errors (no VERDICT: line found) — "
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
    temperature: float = 1.0,
    reasoning_effort: str | None = "low",
    cot_first: bool = False,
) -> TestResult:
    print(f"  Testing on {len(val_items)} items with {model} ...", file=sys.stderr)
    correct, wrong = score_batch(
        val_items, cheatsheet_text, model, api_key,
        concurrency, temperature, reasoning_effort=reasoning_effort, cot_first=cot_first,
    )
    scored   = len(correct) + len(wrong)
    accuracy = len(correct) / scored if scored > 0 else 0.0
    return TestResult(accuracy=accuracy, correct=correct, wrong=wrong, n_total=len(val_items))
