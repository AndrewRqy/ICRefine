"""
scorer.py — Score a batch of items with a given prior knowledge string.

Uses the same prompt format as evaluation.jinja2 so that ablation results
are directly comparable to the eval pipeline's accuracy numbers.
"""

from __future__ import annotations

from utils.llm_client import call_llm_batch, LLMResponse
from utils.data import is_true

# Must match evaluation.jinja2 exactly (PK injected between equation declaration
# and the CRITICAL INSTRUCTION block).
_SCORE_TEMPLATE = """\
You are a mathematician specializing in equational theories of magmas.
Your task is to determine whether Equation 1 ({equation1}) implies Equation 2 ({equation2}) over all magmas.

{prior_knowledge}

CRITICAL INSTRUCTION: The VERY FIRST LINE of your response must be either:
  VERDICT: TRUE
  VERDICT: FALSE
Do NOT write anything before this line. Not a single word. Start with VERDICT immediately.
Even if you are uncertain, you MUST commit to a verdict — write VERDICT: TRUE or VERDICT: FALSE based on your best assessment. Never leave the verdict blank or say "I don't know".

Output format:
VERDICT: TRUE or FALSE  ← THIS MUST BE YOUR FIRST LINE, NO EXCEPTIONS.
REASONING: must be non-empty.
PROOF: required if VERDICT is TRUE, empty otherwise.
COUNTEREXAMPLE: required if VERDICT is FALSE, empty otherwise.\
"""


def _build_prompt(item: dict, prior_knowledge: str) -> str:
    return _SCORE_TEMPLATE.format(
        equation1=item["equation1"],
        equation2=item["equation2"],
        prior_knowledge=prior_knowledge,
    )


def score_items(
    items: list[dict],
    prior_knowledge: str,
    model: str,
    api_key: str,
    concurrency: int = 100,
    reasoning_effort: str | None = "low",
    progress_label: str = "",
) -> list[bool | None]:
    """
    Score items with the given prior_knowledge.
    Returns list of True / False / None (None = parse failure or API error).
    """
    prompts = [_build_prompt(item, prior_knowledge) for item in items]
    responses: list[LLMResponse | None] = call_llm_batch(
        prompts=prompts,
        model=model,
        api_key=api_key,
        concurrency=concurrency,
        reasoning_effort=reasoning_effort,
        max_tokens=1024,
        progress_label=progress_label,
    )

    results: list[bool | None] = []
    for resp in responses:
        if resp is None:
            results.append(None)
            continue
        text = resp.content.upper()
        if "VERDICT: TRUE" in text:
            results.append(True)
        elif "VERDICT: FALSE" in text:
            results.append(False)
        else:
            results.append(None)
    return results


def accuracy(scores: list[bool | None], items: list[dict]) -> float:
    """Fraction of non-None scores that match ground truth."""
    n_correct = n_answered = 0
    for score, item in zip(scores, items):
        if score is None:
            continue
        n_answered += 1
        if score == is_true(item.get("answer", False)):
            n_correct += 1
    return n_correct / n_answered if n_answered > 0 else 0.0


def split_by_correctness(
    scores: list[bool | None],
    items: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Return (correct_items, failure_items) based on scores vs ground truth."""
    correct, failures = [], []
    for score, item in zip(scores, items):
        if score is None:
            failures.append(item)
        elif score == is_true(item.get("answer", False)):
            correct.append(item)
        else:
            failures.append(item)
    return correct, failures
