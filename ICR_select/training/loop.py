"""
ICR_select/training/loop.py — Selective training loop with five quality gates.

Every candidate case study must pass all active gates before entering the
cheatsheet.  Useless entries are pruned periodically.  The cheatsheet is
condensed when it grows too large.

Gates (in order):
  1. Candidate competition  — generate N candidates, score each on the failure
                              batch, keep the best.
  2. Fix-rate gate          — best candidate must fix >= fix_rate_threshold of
                              the failure batch.
  3. Regression gate        — candidate must not break > regress_threshold of
                              the correct pool.
  4. Similarity gate        — LLM dedup: skip if duplicate, merge if overlap,
                              add if genuinely new.

Periodic maintenance:
  5. Ablation pruning       — every ablation_every flushes, remove case studies
                              with zero contribution.
  6. Condensation           — when len(case_studies) >= condense_at, rewrite to
                              a denser set and validate before swapping.
"""

from __future__ import annotations

import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from ICR_naive.core.cheatsheet import Cheatsheet
from ICR_naive.generators.initial import _split_case_studies
from ICR_reasoning.core.llm_client import call_llm
from ICR_reasoning.core.oracle import OracleDict
from ICR_reasoning.training.scorer import score_batch, test_cheatsheet
from ..generators.case_study import generate_candidates
from ..prompts.templates import (
    SIMILARITY_CHECK_PROMPT, SIMILARITY_MAX_TOKENS,
    MERGE_PROMPT, MERGE_MAX_TOKENS,
    CONDENSATION_PROMPT, CONDENSATION_MAX_TOKENS,
    CORRECT_POOL_MAX,
)


# ---------------------------------------------------------------------------
# Failure bin (same as ICR_naive/reasoning)
# ---------------------------------------------------------------------------

@dataclass
class FailureBin:
    threshold: int
    _items: list[dict] = field(default_factory=list, init=False)

    def add(self, item: dict) -> None:
        self._items.append(item)

    def is_full(self) -> bool:
        return len(self._items) >= self.threshold

    def flush(self) -> list[dict]:
        items = self._items[:]
        self._items.clear()
        return items

    def __len__(self) -> int:
        return len(self._items)


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    cheatsheet: Cheatsheet
    n_case_studies_added: int
    n_bins_discarded: int
    n_bins_skipped: int
    n_merges: int
    n_ablation_pruned: int
    n_condensations: int
    train_accuracy: float
    val_accuracy: float | None
    update_log: list[dict]


# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------

def _mini_eval(
    candidate_cs: str,
    failures: list[dict],
    cheatsheet: Cheatsheet,
    model_score: str,
    api_key: str,
    concurrency: int,
    reasoning_effort: str | None,
    cot_first: bool,
    label: str = "mini-eval",
) -> float:
    """Score the failure batch with candidate injected. Returns fix_rate."""
    temp = Cheatsheet(
        decision_tree=cheatsheet.decision_tree,
        case_studies=cheatsheet.case_studies + [candidate_cs],
    )
    correct, _ = score_batch(
        failures, temp.render(), model_score, api_key,
        concurrency=concurrency, reasoning_effort=reasoning_effort, cot_first=cot_first,
        progress_label=label,
    )
    return len(correct) / len(failures) if failures else 0.0


def _replace_eval(
    merged_cs: str,
    merge_idx: int,
    failures: list[dict],
    cheatsheet: Cheatsheet,
    model_score: str,
    api_key: str,
    concurrency: int,
    reasoning_effort: str | None,
    cot_first: bool,
) -> float:
    """Score the failure batch with CS at merge_idx replaced by merged_cs. Returns fix_rate."""
    new_studies = cheatsheet.case_studies[:]
    new_studies[merge_idx] = merged_cs
    temp = Cheatsheet(
        decision_tree=cheatsheet.decision_tree,
        case_studies=new_studies,
    )
    correct, _ = score_batch(
        failures, temp.render(), model_score, api_key,
        concurrency=concurrency, reasoning_effort=reasoning_effort, cot_first=cot_first,
        progress_label="merge-eval-merged",
    )
    return len(correct) / len(failures) if failures else 0.0


def _regression_check(
    candidate_cs: str,
    correct_pool: list[dict],
    cheatsheet: Cheatsheet,
    model_score: str,
    api_key: str,
    concurrency: int,
    reasoning_effort: str | None,
    cot_first: bool,
) -> float:
    """Score the correct pool with candidate injected. Returns regression_rate."""
    if not correct_pool:
        return 0.0
    temp = Cheatsheet(
        decision_tree=cheatsheet.decision_tree,
        case_studies=cheatsheet.case_studies + [candidate_cs],
    )
    _, wrong = score_batch(
        correct_pool, temp.render(), model_score, api_key,
        concurrency=concurrency, reasoning_effort=reasoning_effort, cot_first=cot_first,
        progress_label="regression-check",
    )
    return len(wrong) / len(correct_pool)


def _format_existing(case_studies: list[str]) -> str:
    parts = []
    for i, cs in enumerate(case_studies, 1):
        parts.append(f"[{i}]\n{cs.strip()}")
    return "\n\n".join(parts) if parts else "(none yet)"


def _similarity_gate(
    candidate_cs: str,
    cheatsheet: Cheatsheet,
    model_casestudy: str,
    api_key: str,
) -> tuple[str, int | None]:
    """
    Check if candidate duplicates an existing case study.

    Returns:
        ('ADD',   None)  — new pattern, add it
        ('SKIP',  None)  — duplicate, discard
        ('MERGE', N)     — merge into case study at index N (0-based)
    """
    if not cheatsheet.case_studies:
        return "ADD", None

    resp = call_llm(
        SIMILARITY_CHECK_PROMPT.format(
            existing=_format_existing(cheatsheet.case_studies),
            candidate=candidate_cs,
        ),
        model_casestudy, api_key,
        temperature=0.0,
        max_tokens=SIMILARITY_MAX_TOKENS,
        reasoning_effort=None,
    )
    raw = resp.content.strip().upper()

    if raw.startswith("SKIP"):
        return "SKIP", None
    if raw.startswith("MERGE"):
        m = re.search(r"\d+", raw)
        if m:
            idx = int(m.group()) - 1   # prompt uses 1-based numbering
            if 0 <= idx < len(cheatsheet.case_studies):
                return "MERGE", idx
    return "ADD", None


def _merge_case_studies(
    cs_a: str,
    cs_b: str,
    model_casestudy: str,
    api_key: str,
) -> str:
    """Merge two case studies into one via LLM."""
    resp = call_llm(
        MERGE_PROMPT.format(cs_a=cs_a, cs_b=cs_b),
        model_casestudy, api_key,
        temperature=0.2,
        max_tokens=MERGE_MAX_TOKENS,
        reasoning_effort=None,
    )
    return resp.content.strip()


# ---------------------------------------------------------------------------
# Periodic maintenance
# ---------------------------------------------------------------------------

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
    Uses a random sample of seen items (capped at ABLATION_SAMPLE_MAX) so the
    ablation pass stays fast regardless of how many items have been scored.
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

    def _score_without(i):
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

    contributions = {}
    with ThreadPoolExecutor(max_workers=len(cheatsheet.case_studies)) as ex:
        futures = {ex.submit(_score_without, i): i for i in range(len(cheatsheet.case_studies))}
        for fut in as_completed(futures):
            i, acc_without = fut.result()
            contributions[i] = base_acc - acc_without

    to_keep = []
    n_pruned = 0
    for i, cs in enumerate(cheatsheet.case_studies):
        contribution = contributions[i]
        if contribution > 0:
            to_keep.append(cs)
            log_fn(f"  [ablation] CS {i+1}: contribution={contribution:+.1%} — KEPT")
        else:
            n_pruned += 1
            log_fn(f"  [ablation] CS {i+1}: contribution={contribution:+.1%} — PRUNED")

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
    only replaces if condensed version doesn't hurt train accuracy.
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
    condensed = _split_case_studies(resp.content)
    if not condensed:
        log_fn("  [condense] parse failed — keeping original.")
        return cheatsheet

    condensed_cs = Cheatsheet(
        decision_tree=cheatsheet.decision_tree,
        case_studies=condensed,
    )

    # Validate: condensed must match or beat current accuracy on seen items
    if train_seen:
        sample = train_seen
        if len(train_seen) > ABLATION_SAMPLE_MAX:
            sample = random.sample(train_seen, ABLATION_SAMPLE_MAX)

        def _score_version(args):
            label, cs = args
            correct, _ = score_batch(
                sample, cs.render(), model_score, api_key,
                concurrency=concurrency, reasoning_effort=reasoning_effort, cot_first=cot_first,
                progress_label=label,
            )
            return label, len(correct)

        results = {}
        with ThreadPoolExecutor(max_workers=2) as ex:
            for label, n in ex.map(_score_version, [
                ("condense-validate-new", condensed_cs),
                ("condense-validate-old", cheatsheet),
            ]):
                results[label] = n
        delta = results["condense-validate-new"] - results["condense-validate-old"]
        if delta >= -1:   # allow at most 1-item regression
            log_fn(
                f"  [condense] validated: {len(condensed)} entries, "
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


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _save_checkpoint(
    cheatsheet: Cheatsheet,
    update_log: list[dict],
    output_dir: Path,
    tag: int | str = "",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cheatsheet.save(output_dir / "cheatsheet_current")
    if tag != "":
        name = f"cheatsheet_update_{tag:02d}" if isinstance(tag, int) \
               else f"cheatsheet_{tag}"
        cheatsheet.save(output_dir / name)
    (output_dir / "update_log.json").write_text(
        json.dumps(update_log, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_training_loop(
    cheatsheet: Cheatsheet,
    train_items: list[dict],
    val_items: list[dict] | None,
    model_score: str,
    model_casestudy: str,
    api_key: str,
    # Bin
    bin_threshold: int = 5,
    batch_size: int = 10,
    concurrency: int = 10,
    # Candidate generation
    n_candidates: int = 3,
    oracle: OracleDict | None = None,
    # Gates
    fix_rate_threshold: float = 0.5,
    regress_threshold: float = 0.15,
    similarity_gate: bool = True,
    validate_merge: bool = False,
    # Maintenance
    ablation_every: int = 5,
    condense_at: int = 6,
    # Other
    flush_remainder: bool = True,
    cot_first: bool = True,
    reasoning_effort: str | None = "low",
    output_dir: Path | None = None,
    log: bool = True,
) -> TrainingResult:

    def _log(msg: str) -> None:
        if log:
            print(msg, file=sys.stderr, flush=True)

    bin_           = FailureBin(threshold=bin_threshold)
    correct_pool   : list[dict] = []   # rolling window of correct items for regression
    train_seen     : list[dict] = []   # all scored items for ablation
    update_log     : list[dict] = []

    n_added        = 0
    n_discarded    = 0
    n_skipped      = 0
    n_merges       = 0
    n_pruned_total = 0
    n_condensed    = 0
    flush_count    = 0

    total_correct = 0
    total_scored  = 0
    total_batches = (len(train_items) + batch_size - 1) // batch_size

    _log(
        f"\n{'='*60}\n"
        f"ICR_select Training loop\n"
        f"  items={len(train_items)}  batch={batch_size}  bin={bin_threshold}\n"
        f"  n_candidates={n_candidates}  fix_rate≥{fix_rate_threshold}  "
        f"regress≤{regress_threshold}\n"
        f"  validate_merge={validate_merge}  "
        f"ablation_every={ablation_every}  condense_at={condense_at}\n"
        f"  model_score={model_score}\n"
        f"  model_casestudy={model_casestudy}\n"
        f"{'='*60}"
    )

    def _process_flush(failures: list[dict], batch_num: int) -> None:
        nonlocal n_added, n_discarded, n_skipped, n_merges, flush_count

        _log(f"\n  [flush] {len(failures)} failures → generating {n_candidates} candidates ...")

        # ── Step 1: candidate competition ──────────────────────────────────
        try:
            candidates = generate_candidates(
                failures, cheatsheet, model_casestudy, api_key,
                n=n_candidates, oracle=oracle,
            )
        except RuntimeError as exc:
            _log(f"  [flush] candidate generation failed: {exc} — discarding bin.")
            n_discarded += 1
            return

        # ── Step 2: mini-eval all candidates in parallel, pick best ──────────
        scored = [None] * len(candidates)

        def _eval_candidate(args):
            i, cand = args
            fix_rate = _mini_eval(
                cand, failures, cheatsheet, model_score, api_key,
                concurrency, reasoning_effort, cot_first,
                label=f"mini-eval cand {i+1}/{len(candidates)}",
            )
            return i, fix_rate, cand

        with ThreadPoolExecutor(max_workers=len(candidates)) as ex:
            futures = {ex.submit(_eval_candidate, (i, c)): i for i, c in enumerate(candidates)}
            for fut in as_completed(futures):
                i, fix_rate, cand = fut.result()
                scored[i] = (fix_rate, cand)
                _log(f"  [mini-eval] candidate {i+1}: fix_rate={fix_rate:.0%}")

        scored.sort(key=lambda x: x[0], reverse=True)
        best_fix_rate, best_cand = scored[0]

        # ── Step 3: fix-rate gate ───────────────────────────────────────────
        if best_fix_rate < fix_rate_threshold:
            _log(
                f"  [gate:fix_rate] best={best_fix_rate:.0%} < "
                f"threshold={fix_rate_threshold:.0%} — discarding bin."
            )
            n_discarded += 1
            update_log.append({
                "event": "bin_discarded", "batch": batch_num,
                "reason": "fix_rate_below_threshold",
                "best_fix_rate": best_fix_rate,
            })
            return

        # ── Step 4: regression gate ─────────────────────────────────────────
        if correct_pool:
            reg_rate = _regression_check(
                best_cand, correct_pool, cheatsheet, model_score, api_key,
                concurrency, reasoning_effort, cot_first,
            )
            _log(f"  [gate:regression] regression_rate={reg_rate:.0%}")
            if reg_rate > regress_threshold:
                _log(
                    f"  [gate:regression] {reg_rate:.0%} > "
                    f"threshold={regress_threshold:.0%} — discarding."
                )
                n_discarded += 1
                update_log.append({
                    "event": "bin_discarded", "batch": batch_num,
                    "reason": "regression_above_threshold",
                    "regression_rate": reg_rate,
                })
                return
        else:
            reg_rate = 0.0

        # ── Step 5: similarity gate (optional) ─────────────────────────────
        if similarity_gate and cheatsheet.case_studies:
            action, merge_idx = _similarity_gate(
                best_cand, cheatsheet, model_casestudy, api_key,
            )
            _log(f"  [gate:similarity] action={action}" +
                 (f" (merge into CS {merge_idx+1})" if merge_idx is not None else ""))

            if action == "SKIP":
                n_skipped += 1
                update_log.append({
                    "event": "bin_skipped", "batch": batch_num, "reason": "duplicate",
                })
                return

            if action == "MERGE" and merge_idx is not None:
                existing = cheatsheet.case_studies[merge_idx]
                merged   = _merge_case_studies(existing, best_cand, model_casestudy, api_key)

                if validate_merge:
                    # Score failures with existing CS in place (baseline)
                    correct_base, _ = score_batch(
                        failures, cheatsheet.render(), model_score, api_key,
                        concurrency=concurrency, reasoning_effort=reasoning_effort,
                        cot_first=cot_first, progress_label="merge-eval-baseline",
                    )
                    existing_fix_rate = len(correct_base) / len(failures) if failures else 0.0

                    # Score failures with merged CS replacing the existing entry
                    merged_fix_rate = _replace_eval(
                        merged, merge_idx, failures, cheatsheet, model_score, api_key,
                        concurrency, reasoning_effort, cot_first,
                    )
                    _log(
                        f"  [gate:merge_validate] existing_fix_rate={existing_fix_rate:.0%}  "
                        f"merged_fix_rate={merged_fix_rate:.0%}"
                    )

                    if merged_fix_rate < existing_fix_rate:
                        # Merge would hurt — fall through to ADD instead
                        _log(
                            f"  [gate:merge_validate] merged ({merged_fix_rate:.0%}) < "
                            f"existing ({existing_fix_rate:.0%}) — rejecting merge, "
                            f"adding candidate as new entry."
                        )
                        update_log.append({
                            "event": "merge_rejected", "batch": batch_num,
                            "merge_idx": merge_idx + 1,
                            "existing_fix_rate": existing_fix_rate,
                            "merged_fix_rate": merged_fix_rate,
                        })
                        # Fall through to ADD below
                    else:
                        cheatsheet.case_studies[merge_idx] = merged
                        n_merges += 1
                        flush_count += 1
                        _log(
                            f"  [merge] validated and updated CS {merge_idx+1} in-place "
                            f"({existing_fix_rate:.0%} → {merged_fix_rate:.0%})."
                        )
                        update_log.append({
                            "event": "bin_merged", "batch": batch_num,
                            "merged_into": merge_idx + 1,
                            "fix_rate": best_fix_rate, "regression_rate": reg_rate,
                            "existing_fix_rate": existing_fix_rate,
                            "merged_fix_rate": merged_fix_rate,
                        })
                        return
                else:
                    cheatsheet.case_studies[merge_idx] = merged
                    n_merges += 1
                    flush_count += 1
                    _log(f"  [merge] updated CS {merge_idx+1} in-place.")
                    update_log.append({
                        "event": "bin_merged", "batch": batch_num,
                        "merged_into": merge_idx + 1,
                        "fix_rate": best_fix_rate, "regression_rate": reg_rate,
                    })
                    return

        # ── ADD ─────────────────────────────────────────────────────────────
        cheatsheet.add_case_study(best_cand)
        n_added   += 1
        flush_count += 1
        _log(
            f"  [added] CS {len(cheatsheet.case_studies)} — "
            f"fix_rate={best_fix_rate:.0%}  regress={reg_rate:.0%}"
        )
        update_log.append({
            "event": "bin_added", "batch": batch_num,
            "fix_rate": best_fix_rate, "regression_rate": reg_rate,
            "n_case_studies_total": len(cheatsheet.case_studies),
        })

    def _maybe_maintain(batch_num: int) -> None:
        nonlocal n_pruned_total, n_condensed

        # Ablation
        if flush_count > 0 and flush_count % ablation_every == 0 and train_seen:
            _log(f"\n  [ablation] running after flush #{flush_count} ...")
            pruned_cs, n_pruned = _ablation_prune(
                cheatsheet, train_seen, model_score, api_key,
                concurrency, reasoning_effort, cot_first, _log,
            )
            cheatsheet.decision_tree = pruned_cs.decision_tree
            cheatsheet.case_studies  = pruned_cs.case_studies
            n_pruned_total += n_pruned
            update_log.append({
                "event": "ablation", "batch": batch_num,
                "n_pruned": n_pruned,
                "n_remaining": len(cheatsheet.case_studies),
            })

        # Condensation
        if len(cheatsheet.case_studies) >= condense_at:
            _log(
                f"\n  [condense] {len(cheatsheet.case_studies)} case studies "
                f"≥ condense_at={condense_at} ..."
            )
            condensed = _condense(
                cheatsheet, train_seen, model_casestudy, model_score, api_key,
                concurrency, reasoning_effort, cot_first, _log,
            )
            if condensed.case_studies != cheatsheet.case_studies:
                cheatsheet.decision_tree = condensed.decision_tree
                cheatsheet.case_studies  = condensed.case_studies
                n_condensed += 1
                update_log.append({
                    "event": "condensation", "batch": batch_num,
                    "n_after": len(cheatsheet.case_studies),
                })

    # ── Main training loop ────────────────────────────────────────────────

    for batch_start in range(0, len(train_items), batch_size):
        batch     = train_items[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1

        _log(
            f"\n[batch {batch_num}/{total_batches}]  "
            f"items {batch_start+1}–{min(batch_start+len(batch), len(train_items))}  "
            f"bin={len(bin_)}/{bin_threshold}"
        )

        correct, wrong = score_batch(
            batch, cheatsheet.render(), model_score, api_key, concurrency,
            reasoning_effort=reasoning_effort, cot_first=cot_first,
        )
        total_correct += len(correct)
        total_scored  += len(batch)
        train_seen    += batch
        running_acc    = total_correct / total_scored

        # Update correct pool (reservoir — keep up to CORRECT_POOL_MAX)
        for item in correct:
            if len(correct_pool) < CORRECT_POOL_MAX:
                correct_pool.append(item)
            else:
                # Replace random element to avoid staleness
                correct_pool[random.randrange(CORRECT_POOL_MAX)] = item

        _log(
            f"  correct={len(correct)}  wrong={len(wrong)}  "
            f"running_accuracy={running_acc:.1%}  "
            f"correct_pool={len(correct_pool)}"
        )

        for item in wrong:
            bin_.add(item)

        while bin_.is_full():
            failures = bin_.flush()
            _process_flush(failures, batch_num)
            _maybe_maintain(batch_num)
            if output_dir:
                _save_checkpoint(cheatsheet, update_log, output_dir, n_added)

    # ── Remainder flush ───────────────────────────────────────────────────

    if flush_remainder and len(bin_) > 0:
        _log(f"\n[remainder] flushing {len(bin_)} remaining failures ...")
        failures = bin_.flush()
        _process_flush(failures, batch_num=-1)
        _maybe_maintain(batch_num=-1)
        if output_dir:
            _save_checkpoint(cheatsheet, update_log, output_dir, "remainder")

    train_accuracy = total_correct / total_scored if total_scored > 0 else 0.0
    _log(
        f"\n{'='*60}\n"
        f"Training complete.\n"
        f"  train_accuracy={train_accuracy:.1%}\n"
        f"  case_studies_added={n_added}  merges={n_merges}\n"
        f"  bins_discarded={n_discarded}  bins_skipped={n_skipped}\n"
        f"  ablation_pruned={n_pruned_total}  condensations={n_condensed}\n"
        f"  {cheatsheet.summary()}\n"
        f"{'='*60}"
    )

    val_accuracy: float | None = None
    if val_items:
        _log(f"\nValidating on {len(val_items)} held-out items ...")
        result = test_cheatsheet(
            cheatsheet_text=cheatsheet.render(),
            val_items=val_items,
            model=model_score,
            api_key=api_key,
            concurrency=concurrency,
            reasoning_effort=reasoning_effort,
            cot_first=cot_first,
        )
        val_accuracy = result.accuracy
        _log(f"  {result.summary()}")

    if output_dir:
        _save_checkpoint(cheatsheet, update_log, output_dir, "final")

    return TrainingResult(
        cheatsheet=cheatsheet,
        n_case_studies_added=n_added,
        n_bins_discarded=n_discarded,
        n_bins_skipped=n_skipped,
        n_merges=n_merges,
        n_ablation_pruned=n_pruned_total,
        n_condensations=n_condensed,
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
        update_log=update_log,
    )
