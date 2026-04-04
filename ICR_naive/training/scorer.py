"""
training/scorer.py — Score a cheatsheet against a set of labeled items.

For each (equation1, equation2) item, the LLM is shown the rendered cheatsheet
and asked to predict TRUE/FALSE.  The prediction is compared to the ground-truth
answer to produce an accuracy score and per-item breakdowns.

Uses call_llm_batch() so all items are scored in parallel.
"""

from __future__ import annotations

import sys
import os as _os
from dataclasses import dataclass, field

from ..core.data import _is_true
from ..core.llm_client import call_llm_batch
from ..prompts.templates import SCORING_PROMPT, SCORING_MAX_TOKENS

# Re-use SAIR's battle-tested parser so verdict extraction is consistent
# across ICRefine and the SAIR evaluation pipeline.
import re as _re
_sair_path = str(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", "SAIR_evaluation_pipeline"))
if _sair_path not in sys.path:
    sys.path.insert(0, _sair_path)
from pipeline.parser import parse_response as _sair_parse

_MD_BOLD_RE = _re.compile(r"\*{1,2}(VERDICT|REASONING|PROOF|COUNTEREXAMPLE):\*{0,2}", _re.IGNORECASE)

def _normalize(text: str) -> str:
    return _MD_BOLD_RE.sub(lambda m: m.group(1).upper() + ":", text)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_scoring_prompt(cheatsheet_text: str, item: dict) -> str:
    return SCORING_PROMPT.format(
        cheatsheet=cheatsheet_text,
        equation1=item["equation1"],
        equation2=item["equation2"],
    )


# ---------------------------------------------------------------------------
# Verdict parsing — delegates to SAIR's anchored multiline parser
# ---------------------------------------------------------------------------

def _parse_verdict(text: str | None) -> str | None:
    """Return 'TRUE', 'FALSE', or None if the response cannot be parsed."""
    if not text:
        return None
    return _sair_parse(_normalize(text))["verdict"]


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
# Scoring functions
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
) -> tuple[list[dict], list[dict]]:
    """
    Score a list of items against the current cheatsheet in parallel.

    Returns (correct_items, wrong_items) — both annotated with 'predicted'.
    Parse errors are counted as wrong.
    """
    prompts = [_build_scoring_prompt(cheatsheet_text, item) for item in items]
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
    for item, response in zip(items, responses):
        predicted    = _parse_verdict(response)
        ground_truth = _is_true(item["answer"])
        annotated = {
            **item,
            "predicted":    predicted,
            "expected":     "TRUE" if ground_truth else "FALSE",
            "raw_response": response or "",
        }
        if predicted is None or (predicted == "TRUE") != ground_truth:
            wrong.append(annotated)
        else:
            correct.append(annotated)

    return correct, wrong


def test_cheatsheet(
    cheatsheet_text: str,
    val_items: list[dict],
    model: str,
    api_key: str,
    concurrency: int = 10,
    temperature: float = 0.0,
    reasoning_effort: str | None = "low",
) -> TestResult:
    """
    Score *cheatsheet_text* on the full *val_items* set.
    Returns a TestResult with accuracy and per-item breakdowns.
    """
    print(
        f"  Testing on {len(val_items)} items with {model} ...",
        file=sys.stderr,
    )
    correct, wrong = score_batch(
        val_items, cheatsheet_text, model, api_key, concurrency, temperature,
        reasoning_effort=reasoning_effort,
    )
    scored   = len(correct) + len(wrong)
    accuracy = len(correct) / scored if scored > 0 else 0.0
    return TestResult(
        accuracy=accuracy,
        correct=correct,
        wrong=wrong,
        n_total=len(val_items),
    )
