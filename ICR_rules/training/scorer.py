"""
scorer.py — SAIR-style scoring for ICR_rules.

Unlike ICR_partition's scorer, this does NOT inject pre-computed features.
The model must compute everything itself — exactly as in real deployment.
This measures true SAIR accuracy, not ICR upper-bound accuracy.

Prompt construction uses Jinja2 rendering to exactly match the SAIR eval
pipeline output. RuleSet.render() is only used when a rule has been patched
(i.e. the RuleSet diverges from its source file).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from jinja2 import Template as _Jinja2Template
    _JINJA2_AVAILABLE = True
except ImportError:
    _JINJA2_AVAILABLE = False

from utils.llm_client import call_llm
from utils.parser import parse_response, compute_correct
from ..rules.rule import RuleSet


def _build_jinja2_prompt(template_text: str, equation1: str, equation2: str) -> str:
    """Render a jinja2 template string with equation substitutions."""
    if _JINJA2_AVAILABLE:
        return _Jinja2Template(template_text).render(equation1=equation1, equation2=equation2)
    # Fallback: simple string replace
    return (template_text
            .replace("{{ equation1 }}", equation1)
            .replace("{{ equation2 }}", equation2))


def _get_template_text(rule_set: RuleSet) -> str:
    """
    Return the jinja2 template text for this RuleSet.
    Prefers loading from source_path (exact original file) when the RuleSet
    has not been patched; falls back to rule_set.render() for patched sets.
    """
    if rule_set.source_path and Path(rule_set.source_path).exists():
        # Check if the RuleSet has been patched by comparing rule count/ids
        # against the original file. Simple heuristic: if source_path exists
        # and is readable, use it as the base and apply any rule text changes.
        # For now: use source file only if NO rules have been disabled or patched.
        # A patched RuleSet has source_path="" (set by apply_patch caller).
        return Path(rule_set.source_path).read_text(encoding="utf-8")
    return rule_set.render()


def score_batch_sair(
    items: list[dict],
    rule_set: RuleSet,
    model: str,
    api_key: str,
    concurrency: int = 50,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    label: str = "",
    task_spec=None,
) -> tuple[list[dict], list[dict]]:
    """
    Score *items* using the SAIR prompt (no pre-computed features injected).

    Returns (correct_items, wrong_items), each annotated with:
      predicted, reasoning, raw_response, correct
    """
    import sys
    template_text = _get_template_text(rule_set)
    n = len(items)
    prefix = f"[score{' ' + label if label else ''}]"

    _rule_prompt_fn = (
        task_spec.build_rule_scoring_prompt
        if task_spec is not None and task_spec.build_rule_scoring_prompt is not None
        else None
    )

    def _build_prompt(item: dict) -> str:
        if _rule_prompt_fn is not None:
            return _rule_prompt_fn(template_text, item)
        return _build_jinja2_prompt(template_text, item["equation1"], item["equation2"])

    prompts = [_build_prompt(item) for item in items]

    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                call_llm, prompt,
                model=model, api_key=api_key,
                max_tokens=max_tokens, temperature=temperature,
            ): i
            for i, prompt in enumerate(prompts)
        }
        responses: dict[int, object] = {}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                responses[idx] = future.result()
            except Exception as e:
                responses[idx] = _ErrorResponse(str(e))
            done += 1
            if done % 25 == 0 or done == n:
                print(f"  {prefix} {done}/{n}", flush=True)

    correct: list[dict] = []
    wrong: list[dict] = []

    _parse_v  = task_spec.parse_verdict       if task_spec is not None else None
    _is_corr  = task_spec.is_correct          if task_spec is not None else None
    _post_th  = task_spec.extract_post_think  if task_spec is not None else None

    for i, item in enumerate(items):
        resp = responses.get(i)
        if isinstance(resp, _ErrorResponse):
            annotated = {**item, "predicted": None, "reasoning": "", "raw_response": "", "correct": False, "error": resp.msg}
            wrong.append(annotated)
            continue

        if _parse_v is not None:
            predicted  = _parse_v(resp.content)
            reasoning  = _post_th(resp.content) if _post_th else resp.content
            is_correct = _is_corr(predicted, item) if _is_corr else False
        else:
            parsed     = parse_response(resp.content)
            predicted  = parsed.get("verdict")
            reasoning  = parsed.get("reasoning", "")
            is_correct = compute_correct(parsed, item)

        annotated = {
            **item,
            "predicted": predicted,
            "reasoning": reasoning,
            "raw_response": resp.content,
            "correct": is_correct,
            "error": None,
        }
        (correct if is_correct else wrong).append(annotated)

    return correct, wrong


class _ErrorResponse:
    def __init__(self, msg: str):
        self.msg = msg
