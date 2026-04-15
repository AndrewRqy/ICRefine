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
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, Iterator

from .data import is_true
from .llm_client import LLMResponse, call_llm, call_llm_batch
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
        # Debug: show first 3 failed raw responses so we can diagnose the format
        shown = 0
        for it in wrong:
            if it.get("predicted") is None and shown < 3:
                raw = it.get("raw_response", "")
                print(
                    f"\n  [parse-debug] raw_response (first 300 chars):\n"
                    f"  {repr(raw[:300])}",
                    file=sys.stderr,
                )
                shown += 1

    return correct, wrong


def _score_batch_ordered(
    items: list[dict],
    cheatsheet_text: str,
    model: str,
    api_key: str,
    concurrency: int = 10,
    temperature: float = 0.0,
    reasoning_effort: str | None = "low",
    cot_first: bool = False,
    progress_label: str = "scoring",
) -> list[tuple[bool | None, dict]]:
    """
    Like score_batch but returns results in the original item order as
    list[(is_correct, annotated_item)].  is_correct is None on parse error.
    Internal helper — used by score_batch_ensemble.
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
    results = []
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
            results.append((None, annotated))
            continue
        predicted  = _parse_verdict(resp.content)
        post_think = _extract_post_think(resp.content)
        annotated  = {
            **item,
            "predicted":    predicted,
            "expected":     "TRUE" if ground_truth else "FALSE",
            "post_think":   post_think,
            "thinking":     resp.thinking,
            "raw_response": resp.content,
        }
        is_correct = (predicted is not None) and ((predicted == "TRUE") == ground_truth)
        results.append((is_correct if predicted is not None else None, annotated))
    return results


def score_batch_ensemble(
    items: list[dict],
    cheatsheet_text: str,
    models: list[str],
    weights: list[float],
    api_key: str,
    concurrency: int = 10,
    temperature: float = 0.0,
    reasoning_effort: str | None = "low",
    cot_first: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Score items with multiple models in parallel and return weighted (correct, wrong).

    An item is "correct" only if ALL models agree it is correct.
    An item is "wrong" if ANY model fails it; the item carries a ``_wrong_weight``
    field ∈ (0, 1] — the normalised sum of weights of models that failed it.

    This propagates into weighted fix_rate inside _mini_eval_full:
      • weight=1.0 — both models wrong  (consensus failure, highest priority)
      • weight=0.5 — one model wrong    (single-model failure, lower priority)

    Post-think traces from all failing models are concatenated with a divider
    so the case-study generator sees richer failure reasoning.
    Structured fields (predicted, expected) come from models[0] (primary).

    weights: relative contribution of each model (normalised internally to sum=1).
    """
    from concurrent.futures import ThreadPoolExecutor as _TPE

    assert len(models) == len(weights) >= 1, "models and weights must be same length ≥ 1"
    total_w      = sum(weights)
    norm_weights = [w / total_w for w in weights]

    # Run all models in parallel — each spawns its own inner thread pool over items.
    with _TPE(max_workers=len(models)) as pool:
        futures = [
            pool.submit(
                _score_batch_ordered,
                items, cheatsheet_text, m, api_key,
                concurrency, temperature, reasoning_effort, cot_first,
                f"scoring[{m.split('/')[-1]}]",
            )
            for m in models
        ]
        all_ordered: list[list[tuple]] = [f.result() for f in futures]

    correct: list[dict] = []
    wrong:   list[dict] = []
    n_parse_errors = 0

    for i in range(len(items)):
        wrong_weight    = 0.0
        reasoning_parts: list[str] = []
        primary_ann: dict | None   = None

        for j, (model_results, nw) in enumerate(zip(all_ordered, norm_weights)):
            is_correct, ann = model_results[i]
            if j == 0:
                primary_ann = ann
            if is_correct is None:
                wrong_weight   += nw
                n_parse_errors += 1
            elif not is_correct:
                wrong_weight += nw
            think = ann.get("post_think", "")
            if think:
                label = models[j].split("/")[-1]
                reasoning_parts.append(f"[{label}]\n{think}")

        merged = {
            **primary_ann,
            "_wrong_weight": round(wrong_weight, 4),
            "post_think":    "\n\n---\n\n".join(reasoning_parts),
        }

        if wrong_weight == 0.0:
            correct.append(merged)
        else:
            wrong.append(merged)

    if n_parse_errors:
        print(
            f"\n  [ensemble] {n_parse_errors} parse errors across {len(models)} models — "
            f"counted as wrong.",
            file=sys.stderr,
        )

    return correct, wrong


def score_items_streaming(
    items: list[dict],
    get_cheatsheet: Callable[[], str],
    model: str,
    api_key: str,
    concurrency: int = 10,
    temperature: float = 0.0,
    reasoning_effort: str | None = "low",
    cot_first: bool = False,
    max_tokens: int = SCORING_MAX_TOKENS,
    seed: int | None = 42,
) -> Iterator[dict]:
    """
    Sliding-window scorer: yields one annotated item dict as each request
    completes. Always keeps `concurrency` requests in-flight so vLLM never
    idles between batches or during case-study generation.

    get_cheatsheet() is called immediately before each new submission, so
    any cheatsheet update made during a yield (e.g. adding a case study)
    is automatically picked up for the next queued request.

    Yielded dict keys: predicted, expected, post_think, thinking,
    raw_response — plus all original item fields.
    """
    items_iter = iter(items)
    pending: dict[Future, dict] = {}

    def _submit_next(pool: ThreadPoolExecutor) -> bool:
        try:
            item = next(items_iter)
        except StopIteration:
            return False
        prompt = _build_scoring_prompt(get_cheatsheet(), item, cot_first)
        f = pool.submit(
            call_llm, prompt, model, api_key, temperature, max_tokens, reasoning_effort, seed
        )
        pending[f] = item
        return True

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for _ in range(concurrency):
            if not _submit_next(pool):
                break

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                item = pending.pop(f)
                ground_truth = is_true(item["answer"])
                try:
                    resp = f.result()
                    predicted  = _parse_verdict(resp.content)
                    post_think = _extract_post_think(resp.content)
                    thinking   = resp.thinking
                    raw        = resp.content
                except Exception:
                    predicted = post_think = thinking = raw = ""
                    predicted = None

                yield {
                    **item,
                    "predicted":    predicted,
                    "expected":     "TRUE" if ground_truth else "FALSE",
                    "post_think":   post_think,
                    "thinking":     thinking,
                    "raw_response": raw,
                }

                # Submit next AFTER yield so get_cheatsheet() sees any update
                # the caller made while processing the yielded item.
                _submit_next(pool)


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
