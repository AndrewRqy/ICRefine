"""
ICR_select/training/loop.py — Selective training loop with quality gates.

Every candidate case study must pass all active gates before entering the
cheatsheet. Useless entries are pruned periodically, and the cheatsheet is
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

Periodic maintenance (see maintenance.py):
  5. Ablation pruning       — every ablation_every flushes, remove case studies
                              with zero contribution.
  6. Condensation           — when len(case_studies) >= condense_at, rewrite
                              to a denser set and validate before swapping.

Gate helpers live in gates.py; maintenance functions in maintenance.py.
"""

from __future__ import annotations

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from utils.cheatsheet import Cheatsheet
from utils.data import FailureBin
from utils.scorer import score_batch, test_cheatsheet
from ICR_reasoning.core.oracle import OracleDict
from ..generators.case_study import generate_candidates
from ..prompts.templates import CORRECT_POOL_MAX
from .gates import (
    _MIN_CS_FOR_SIMILARITY,
    _apply_prescore,
    _mini_eval,
    _mini_eval_full,
    _regression_check,
    _replace_eval,
    _similarity_gate,
    _merge_case_studies,
)
from .maintenance import _ablation_prune, _condense


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
    candidate_rounds: int = 3,
    flush_strategy: str = "default",   # "default" | "retry"
    oracle: OracleDict | None = None,
    prescore_map: dict | None = None,
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
    correct_pool   : list[dict] = []
    train_seen     : list[dict] = []
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

    # ---- shared gate kwargs ------------------------------------------------
    _gkw = dict(
        model_score=model_score, api_key=api_key,
        concurrency=concurrency, reasoning_effort=reasoning_effort, cot_first=cot_first,
    )

    def _process_flush(failures: list[dict], batch_num: int) -> None:
        nonlocal n_added, n_discarded, n_skipped, n_merges, flush_count

        _log(f"\n  [flush] {len(failures)} failures → generating {n_candidates} candidates ...")

        # Step 1: candidate competition
        try:
            candidates = generate_candidates(
                failures, cheatsheet, model_casestudy, api_key,
                n=n_candidates, oracle=oracle,
            )
        except RuntimeError as exc:
            _log(f"  [flush] candidate generation failed: {exc} — discarding bin.")
            n_discarded += 1
            return

        # Step 2: mini-eval all candidates in parallel, pick best
        scored: list[tuple[float, str] | None] = [None] * len(candidates)

        def _eval_candidate(args: tuple[int, str]) -> tuple[int, float, str]:
            i, cand = args
            fix_rate = _mini_eval(
                cand, failures, cheatsheet,
                label=f"mini-eval cand {i+1}/{len(candidates)}", **_gkw,
            )
            return i, fix_rate, cand

        with ThreadPoolExecutor(max_workers=len(candidates)) as ex:
            futures = {ex.submit(_eval_candidate, (i, c)): i for i, c in enumerate(candidates)}
            for fut in as_completed(futures):
                i, fix_rate, cand = fut.result()
                scored[i] = (fix_rate, cand)
                _log(f"  [mini-eval] candidate {i+1}: fix_rate={fix_rate:.0%}")

        scored_valid = [s for s in scored if s is not None]
        scored_valid.sort(key=lambda x: x[0], reverse=True)
        best_fix_rate, best_cand = scored_valid[0]

        # Step 3: fix-rate gate
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

        # Step 4: regression gate
        reg_rate = 0.0
        if correct_pool:
            reg_rate = _regression_check(best_cand, correct_pool, cheatsheet, **_gkw)
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

        # Step 5: similarity gate (optional)
        if similarity_gate and len(cheatsheet.case_studies) >= _MIN_CS_FOR_SIMILARITY:
            action, merge_idx = _similarity_gate(best_cand, cheatsheet, model_casestudy, api_key)
            _log(f"  [gate:similarity] action={action}" +
                 (f" (merge into CS {merge_idx+1})" if merge_idx is not None else ""))

            if action == "SKIP":
                n_skipped += 1
                update_log.append({"event": "bin_skipped", "batch": batch_num, "reason": "duplicate"})
                return

            if action == "MERGE" and merge_idx is not None:
                existing = cheatsheet.case_studies[merge_idx]
                merged   = _merge_case_studies(existing, best_cand, model_casestudy, api_key)

                if validate_merge:
                    correct_base, _ = score_batch(
                        failures, cheatsheet.render(), model_score, api_key,
                        concurrency=concurrency, reasoning_effort=reasoning_effort,
                        cot_first=cot_first, progress_label="merge-eval-baseline",
                    )
                    existing_fix_rate = len(correct_base) / len(failures) if failures else 0.0
                    merged_fix_rate   = _replace_eval(merged, merge_idx, failures, cheatsheet, **_gkw)
                    _log(
                        f"  [gate:merge_validate] existing={existing_fix_rate:.0%}  "
                        f"merged={merged_fix_rate:.0%}"
                    )
                    if merged_fix_rate < existing_fix_rate:
                        _log("  [gate:merge_validate] merged worse — adding as new entry.")
                        update_log.append({
                            "event": "merge_rejected", "batch": batch_num,
                            "merge_idx": merge_idx + 1,
                            "existing_fix_rate": existing_fix_rate,
                            "merged_fix_rate": merged_fix_rate,
                        })
                        # fall through to ADD
                    else:
                        cheatsheet.case_studies[merge_idx] = merged
                        n_merges += 1
                        flush_count += 1
                        _log(f"  [merge] validated and updated CS {merge_idx+1}.")
                        update_log.append({
                            "event": "bin_merged", "batch": batch_num,
                            "merged_into": merge_idx + 1,
                            "fix_rate": best_fix_rate, "regression_rate": reg_rate,
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

        # ADD
        cheatsheet.add_case_study(best_cand)
        n_added     += 1
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

    def _process_flush_retry(failures: list[dict], batch_num: int) -> None:
        """Retry flush — like _process_flush but retries up to candidate_rounds times."""
        nonlocal n_added, n_discarded, n_skipped, n_merges, flush_count

        prev_attempt: dict | None = None

        for attempt in range(1, candidate_rounds + 1):
            if attempt == 1:
                _log(f"\n  [flush/retry] {len(failures)} failures → generating {n_candidates} candidates ...")
            else:
                _log(f"\n  [flush/retry] attempt {attempt}/{candidate_rounds} (prev failed: {prev_attempt['reason']}) ...")

            try:
                candidates = generate_candidates(
                    failures, cheatsheet, model_casestudy, api_key,
                    n=n_candidates, oracle=oracle, prev_attempt=prev_attempt,
                )
            except RuntimeError as exc:
                _log(f"  [flush/retry] generation failed: {exc} — {'retrying' if attempt < candidate_rounds else 'discarding'}.")
                continue

            scored: list[tuple[float, list[dict], str] | None] = [None] * len(candidates)

            def _eval_full(args: tuple[int, str]) -> tuple[int, float, list[dict], str]:
                i, cand = args
                fix_rate, still_wrong = _mini_eval_full(
                    cand, failures, cheatsheet,
                    label=f"mini-eval cand {i+1}/{len(candidates)}", **_gkw,
                )
                return i, fix_rate, still_wrong, cand

            with ThreadPoolExecutor(max_workers=len(candidates)) as ex:
                futures = {ex.submit(_eval_full, (i, c)): i for i, c in enumerate(candidates)}
                for fut in as_completed(futures):
                    i, fix_rate, still_wrong, cand = fut.result()
                    scored[i] = (fix_rate, still_wrong, cand)
                    _log(f"  [mini-eval] candidate {i+1}: fix_rate={fix_rate:.0%}")

            scored_valid = [s for s in scored if s is not None]
            scored_valid.sort(key=lambda x: x[0], reverse=True)
            best_fix_rate, best_still_wrong, best_cand = scored_valid[0]

            if best_fix_rate < fix_rate_threshold:
                _log(
                    f"  [gate:fix_rate] best={best_fix_rate:.0%} < threshold={fix_rate_threshold:.0%} — "
                    f"{'retrying' if attempt < candidate_rounds else 'discarding'}."
                )
                prev_attempt = {"candidate": best_cand, "still_wrong": best_still_wrong, "reason": "fix_rate"}
                continue

            reg_rate = 0.0
            if correct_pool:
                reg_rate = _regression_check(best_cand, correct_pool, cheatsheet, **_gkw)
                _log(f"  [gate:regression] regression_rate={reg_rate:.0%}")
                if reg_rate > regress_threshold:
                    _log(
                        f"  [gate:regression] {reg_rate:.0%} > threshold={regress_threshold:.0%} — "
                        f"{'retrying' if attempt < candidate_rounds else 'discarding'}."
                    )
                    prev_attempt = {"candidate": best_cand, "still_wrong": best_still_wrong, "reason": "regression"}
                    continue

            if similarity_gate and len(cheatsheet.case_studies) >= _MIN_CS_FOR_SIMILARITY:
                action, merge_idx = _similarity_gate(best_cand, cheatsheet, model_casestudy, api_key)
                _log(f"  [gate:similarity] action={action}" +
                     (f" (merge into CS {merge_idx+1})" if merge_idx is not None else ""))

                if action == "SKIP":
                    n_skipped += 1
                    update_log.append({"event": "bin_skipped", "batch": batch_num, "reason": "duplicate"})
                    return

                if action == "MERGE" and merge_idx is not None:
                    existing = cheatsheet.case_studies[merge_idx]
                    merged   = _merge_case_studies(existing, best_cand, model_casestudy, api_key)
                    cheatsheet.case_studies[merge_idx] = merged
                    n_merges += 1
                    flush_count += 1
                    _log(f"  [merge] updated CS {merge_idx+1} in-place.")
                    update_log.append({
                        "event": "bin_merged", "batch": batch_num,
                        "merged_into": merge_idx + 1,
                        "fix_rate": best_fix_rate, "regression_rate": reg_rate,
                        "attempt": attempt,
                    })
                    return

            # ADD
            cheatsheet.add_case_study(best_cand)
            n_added     += 1
            flush_count += 1
            _log(
                f"  [added] CS {len(cheatsheet.case_studies)} — "
                f"fix_rate={best_fix_rate:.0%}  regress={reg_rate:.0%}  attempt={attempt}"
            )
            update_log.append({
                "event": "bin_added", "batch": batch_num,
                "fix_rate": best_fix_rate, "regression_rate": reg_rate,
                "n_case_studies_total": len(cheatsheet.case_studies),
                "attempt": attempt,
            })
            return

        # All rounds exhausted
        n_discarded += 1
        _log(f"  [flush/retry] all {candidate_rounds} rounds failed — discarding bin.")
        update_log.append({
            "event": "bin_discarded", "batch": batch_num,
            "reason": f"all_{candidate_rounds}_rounds_failed",
            "last_reason": prev_attempt["reason"] if prev_attempt else None,
        })

    def _maybe_maintain(batch_num: int) -> None:
        nonlocal n_pruned_total, n_condensed

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

        if len(cheatsheet.case_studies) >= condense_at:
            _log(f"\n  [condense] {len(cheatsheet.case_studies)} case studies ≥ condense_at={condense_at} ...")
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

        if prescore_map is not None:
            correct, wrong = _apply_prescore(batch, prescore_map)
            _log(f"  [prescore] {len(correct)} correct  {len(wrong)} wrong  (no API call)")
        else:
            correct, wrong = score_batch(
                batch, cheatsheet.render(), model_score, api_key, concurrency,
                reasoning_effort=reasoning_effort, cot_first=cot_first,
            )
        total_correct += len(correct)
        total_scored  += len(batch)
        train_seen    += batch
        running_acc    = total_correct / total_scored

        for item in correct:
            if len(correct_pool) < CORRECT_POOL_MAX:
                correct_pool.append(item)
            else:
                correct_pool[random.randrange(CORRECT_POOL_MAX)] = item

        _log(
            f"  correct={len(correct)}  wrong={len(wrong)}  "
            f"running_accuracy={running_acc:.1%}  "
            f"correct_pool={len(correct_pool)}"
        )

        for item in wrong:
            bin_.add(item)

        _flush_fn = _process_flush_retry if flush_strategy == "retry" else _process_flush

        while bin_.is_full():
            failures = bin_.flush()
            _flush_fn(failures, batch_num)
            _maybe_maintain(batch_num)
            if output_dir:
                _save_checkpoint(cheatsheet, update_log, output_dir, n_added)

    # ── Remainder flush ───────────────────────────────────────────────────

    if flush_remainder and len(bin_) > 0:
        _log(f"\n[remainder] flushing {len(bin_)} remaining failures ...")
        failures = bin_.flush()
        _flush_fn(failures, batch_num=-1)
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
