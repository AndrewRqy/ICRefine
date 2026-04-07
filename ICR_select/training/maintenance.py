"""
ICR_select/training/maintenance.py — Periodic cheatsheet maintenance operations.

  _ablation_prune — remove case studies with zero contribution to train accuracy
  _condense       — rewrite a large set of case studies into a denser set
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.cheatsheet import Cheatsheet
from utils.parser import split_case_studies
from utils.llm_client import call_llm
from utils.scorer import score_batch
from ..prompts.templates import CONDENSATION_PROMPT, CONDENSATION_MAX_TOKENS
from .gates import _format_existing

ABLATION_SAMPLE_MAX = 40   # cap ablation scoring to this many items


def _ablation_prune(
    cheatsheet: Cheatsheet,
    train_seen: list[dict],
    model_score: str,
    api_key: str,
    concurrency: int,
    reasoning_effort: str | None,
    cot_first: bool,
    log_fn,
) -> tuple[Cheatsheet, int]:
    """
    Remove case studies with zero contribution to train accuracy.
    Uses a random sample of seen items (capped at ABLATION_SAMPLE_MAX).
    Returns (pruned_cheatsheet, n_pruned).
    """
    if len(cheatsheet.case_studies) <= 1:
        return cheatsheet, 0

    sample = train_seen
    if len(train_seen) > ABLATION_SAMPLE_MAX:
        sample = random.sample(train_seen, ABLATION_SAMPLE_MAX)

    log_fn(
        f"  [ablation] scoring baseline on {len(sample)} items "
        f"(sampled from {len(train_seen)} seen) ..."
    )
    correct_base, _ = score_batch(
        sample, cheatsheet.render(), model_score, api_key,
        concurrency=concurrency, reasoning_effort=reasoning_effort, cot_first=cot_first,
        progress_label="ablation-baseline",
    )
    base_acc = len(correct_base) / len(sample)

    def _score_without(i: int) -> tuple[int, float]:
        temp = Cheatsheet(
            decision_tree=cheatsheet.decision_tree,
            case_studies=[c for j, c in enumerate(cheatsheet.case_studies) if j != i],
        )
        correct_without, _ = score_batch(
            sample, temp.render(), model_score, api_key,
            concurrency=concurrency, reasoning_effort=reasoning_effort, cot_first=cot_first,
            progress_label=f"ablation-CS{i+1}",
        )
        return i, len(correct_without) / len(sample)

    contributions: dict[int, float] = {}
    with ThreadPoolExecutor(max_workers=len(cheatsheet.case_studies)) as ex:
        for i, acc_without in ex.map(_score_without, range(len(cheatsheet.case_studies))):
            contributions[i] = base_acc - acc_without

    to_keep = []
    n_pruned = 0
    for i, cs in enumerate(cheatsheet.case_studies):
        if contributions[i] > 0:
            to_keep.append(cs)
            log_fn(f"  [ablation] CS {i+1}: contribution={contributions[i]:+.1%} — KEPT")
        else:
            n_pruned += 1
            log_fn(f"  [ablation] CS {i+1}: contribution={contributions[i]:+.1%} — PRUNED")

    return Cheatsheet(decision_tree=cheatsheet.decision_tree, case_studies=to_keep), n_pruned


def _condense(
    cheatsheet: Cheatsheet,
    train_seen: list[dict],
    model_casestudy: str,
    model_score: str,
    api_key: str,
    concurrency: int,
    reasoning_effort: str | None,
    cot_first: bool,
    log_fn,
) -> Cheatsheet:
    """
    Condense case studies into a denser set. Validates before swapping:
    only replaces if the condensed version doesn't hurt train accuracy.
    """
    n_current = len(cheatsheet.case_studies)
    n_target  = max(2, n_current // 2)
    log_fn(f"  [condense] rewriting {n_current} case studies → {n_target} ...")

    resp = call_llm(
        CONDENSATION_PROMPT.format(
            decision_tree=cheatsheet.decision_tree.strip(),
            case_studies=_format_existing(cheatsheet.case_studies),
            n_current=n_current,
            n_target=n_target,
        ),
        model_casestudy, api_key,
        temperature=0.2,
        max_tokens=CONDENSATION_MAX_TOKENS,
        reasoning_effort=None,
    )
    condensed_list = split_case_studies(resp.content)
    if not condensed_list:
        log_fn("  [condense] parse failed — keeping original.")
        return cheatsheet

    condensed_cs = Cheatsheet(
        decision_tree=cheatsheet.decision_tree,
        case_studies=condensed_list,
    )

    if train_seen:
        sample = train_seen
        if len(train_seen) > ABLATION_SAMPLE_MAX:
            sample = random.sample(train_seen, ABLATION_SAMPLE_MAX)

        def _score_version(args: tuple[str, Cheatsheet]) -> tuple[str, int]:
            label, cs = args
            correct, _ = score_batch(
                sample, cs.render(), model_score, api_key,
                concurrency=concurrency, reasoning_effort=reasoning_effort, cot_first=cot_first,
                progress_label=label,
            )
            return label, len(correct)

        results: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=2) as ex:
            for label, n in ex.map(_score_version, [
                ("condense-validate-new", condensed_cs),
                ("condense-validate-old", cheatsheet),
            ]):
                results[label] = n

        delta = results["condense-validate-new"] - results["condense-validate-old"]
        if delta >= -1:
            log_fn(
                f"  [condense] validated: {len(condensed_list)} entries, "
                f"delta={delta:+d} items. Swapping in condensed version."
            )
            return condensed_cs
        else:
            log_fn(
                f"  [condense] validation failed (delta={delta:+d}). "
                f"Keeping original {n_current} entries."
            )
            return cheatsheet
    else:
        log_fn("  [condense] no training items seen yet — accepting condensed without validation.")
        return condensed_cs
