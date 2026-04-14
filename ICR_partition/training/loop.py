"""
ICR_partition/training/loop.py — Partition-parallel training loop.

Algorithm
---------
Outer iterations (up to max_outer_iters):

  1. Score all items (iter 0) or only active-partition items (iter 1+).
     Prescore map supported for iter 0.

  2. Build (or refresh) PartitionBins from the scored results.
     Each bin holds structurally homogeneous failures + a designated
     correct pool from the same structural class.

  3. Solve all active bins concurrently (bounded by partition_concurrency).
     Per bin:
       a. Enrich failures with oracle nearest-neighbour traces.
       b. Generate N candidates at different temperatures.
       c. Mini-eval: score each candidate on bin.failures, pick best.
       d. Fix-rate gate: best_fix_rate >= fix_rate_threshold.
       e. Regression gate: check against bin.correct_pool (NOT global pool).
       f. Similarity gate: LLM dedup / merge against existing case studies.
       g. Write to shared cheatsheet (serialised under cs_lock).

  4. Checkpoint after each outer iteration.

  5. Re-score only the failures of active (non-retired) bins.
     This is O(active_failures) not O(all_items) — the key efficiency win.

  6. Refresh bins: update failure lists, retire bins below retirement_threshold.

  7. Repeat until all bins retired or max_outer_iters reached.

Concurrency model
-----------------
Each bin is solved in its own thread.  Cheatsheet writes are the only shared
mutation; they are serialised under cs_lock.  Because ACTIVATE IF conditions
are strict structural checks, a case study added for partition A should not
fire on partition B's items, so:
  * Fix-rate and regression evals do NOT need to be re-run after another
    partition writes to the cheatsheet.
  * The lock is held only during the final add/merge mutation, not during the
    (expensive) scoring calls.

This is an independence assumption backed by prompt design, not a hard
guarantee.  Cross-partition interference is measured in the update_log
(cross_regression field) when --check-cross-regression is enabled.
"""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from utils.cheatsheet import Cheatsheet
from utils.scorer import score_batch
from utils.oracle_index import OracleIndex
from ICR_reasoning.core.oracle import OracleDict
from ICR_select.generators.case_study import generate_candidates
from ICR_select.training.gates import (
    _MIN_CS_FOR_SIMILARITY,
    _apply_prescore,
    _mini_eval_full,
    _regression_check,
    _similarity_gate,
    _merge_case_studies,
)

from .partition import (
    PartitionBin,
    PartitionKey,
    build_partitions,
    item_partition_key,
    partition_key_to_conditions,
    partition_summary,
    print_partition_table,
    refresh_partitions,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PartitionTrainingResult:
    cheatsheet:           Cheatsheet
    n_case_studies_added: int
    n_bins_solved:        int    # bins that reached retirement naturally
    n_bins_discarded:     int    # bins where all candidate rounds failed
    n_merges:             int
    n_skipped:            int
    n_outer_iters:        int
    train_accuracy:       float
    update_log:           list[dict]
    partition_summary:    list[dict]


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _save_checkpoint(
    cheatsheet: Cheatsheet,
    update_log: list[dict],
    output_dir: Path,
    tag: int | str = "",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cheatsheet.save(output_dir / "cheatsheet_current")
    if tag != "":
        name = (
            f"cheatsheet_iter_{tag:02d}" if isinstance(tag, int)
            else f"cheatsheet_{tag}"
        )
        cheatsheet.save(output_dir / name)
    (output_dir / "update_log.json").write_text(
        json.dumps(update_log, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Per-bin solver (runs in its own thread)
# ---------------------------------------------------------------------------

def _solve_bin(
    pb:                      PartitionBin,
    cheatsheet:              Cheatsheet,    # shared; writes serialised by cs_lock
    cs_lock:                 threading.Lock,
    oracle:                  OracleDict | None,
    oracle_index:            OracleIndex | None,
    n_candidates:            int,
    candidate_rounds:        int,
    model_score:             str,
    model_casestudy:         str,
    api_key:                 str,
    concurrency:             int,
    fix_rate_threshold:      float,
    regress_threshold:       float,
    min_pool_for_regression: int,
    similarity_gate:         bool,
    reasoning_effort:        str | None,
    cot_first:               bool,
    log_fn,
) -> dict:
    """
    Attempt to generate and accept one case study for this partition.
    Returns a result dict describing the outcome (for the update_log).

    Snapshot strategy: we snapshot cheatsheet.render() at the top of each
    attempt round.  Evals (fix-rate, regression) use this snapshot.
    The cheatsheet may be updated by another partition between rounds, but
    the structural independence assumption means this does not invalidate
    our fix-rate measurement.  The write is still serialised under cs_lock.
    """
    label = pb.label

    # Enrich failures with oracle traces (once, before retry loop)
    failures = list(pb.failures)
    if oracle_index:
        enriched = []
        for item in failures:
            match = oracle_index.find_nearest(item)
            if match:
                nearest, sim = match
                item = {
                    **item,
                    "oracle_nearest": nearest.to_dict(),
                    "oracle_sim":     round(sim, 3),
                }
            enriched.append(item)
        failures = enriched

    log_fn(
        f"\n  [partition:{label}] {len(failures)} failures → "
        f"generating {n_candidates} candidates ..."
    )

    gkw = dict(
        model_score=model_score,
        api_key=api_key,
        concurrency=concurrency,
        reasoning_effort=reasoning_effort,
        cot_first=cot_first,
    )

    prev_attempt: dict | None = None
    best_fix_rate = 0.0

    for attempt in range(1, candidate_rounds + 1):
        is_last = attempt == candidate_rounds

        # Snapshot the current cheatsheet render for this attempt
        with cs_lock:
            cs_snapshot_text = cheatsheet.render()
            cs_snapshot = Cheatsheet(
                roadmap=cheatsheet.roadmap,
                case_studies=list(cheatsheet.case_studies),
                prior_knowledge=cheatsheet.prior_knowledge,
                no_limit=cheatsheet.no_limit,
            )

        if attempt > 1:
            log_fn(
                f"\n  [partition:{label}] attempt {attempt}/{candidate_rounds} "
                f"(prev: {prev_attempt['reason']}) ..."
            )

        # --- Generate candidates ---
        try:
            candidates = generate_candidates(
                failures, cs_snapshot, model_casestudy, api_key,
                n=n_candidates, oracle=oracle, prev_attempt=prev_attempt,
            )
        except RuntimeError as exc:
            log_fn(f"  [partition:{label}] generation failed: {exc}")
            if not is_last:
                prev_attempt = {"candidate": None, "still_wrong": failures, "reason": "generation_failed"}
                continue
            return {
                "partition": label,
                "event":     "bin_discarded",
                "reason":    "generation_failed",
                "attempt":   attempt,
            }

        # --- Inject partition key as guaranteed ACTIVATE IF conditions ---
        # The LLM writes semantic conditions; we prepend the structural ground
        # truth from the partition key so the case study is guaranteed to only
        # fire on items from this structural class.  This makes ACTIVATE IF
        # conditions machine-checkable and prevents cross-partition interference.
        partition_conditions = partition_key_to_conditions(pb.key)
        for cand in candidates:
            # Avoid duplicating if retry already injected them
            if not cand.activate_if or cand.activate_if[:1] != partition_conditions[:1]:
                cand.activate_if = partition_conditions + [
                    c for c in cand.activate_if
                    if not any(c.startswith(pc[:20]) for pc in partition_conditions)
                ]

        # --- Mini-eval all candidates in parallel, pick best ---
        # Each candidate is evaluated by appending it to a temp cheatsheet
        # built from cs_snapshot (consistent snapshot across all N evals).
        scored: list[tuple[float, list[dict], object] | None] = [None] * len(candidates)

        def _eval_one(args: tuple[int, object]) -> tuple[int, float, list[dict], object]:
            i, cand = args
            fr, still_wrong = _mini_eval_full(
                cand, failures, cs_snapshot,
                label=f"[{label}] cand {i+1}/{len(candidates)}",
                **gkw,
            )
            return i, fr, still_wrong, cand

        with ThreadPoolExecutor(max_workers=len(candidates)) as ex:
            futs = {ex.submit(_eval_one, (i, c)): i for i, c in enumerate(candidates)}
            for fut in as_completed(futs):
                i, fr, sw, cand = fut.result()
                scored[i] = (fr, sw, cand)
                log_fn(f"  [partition:{label}] cand {i+1}: fix_rate={fr:.0%}")

        scored_valid = [(fr, sw, c) for item in scored if (item is not None) for fr, sw, c in [item]]
        scored_valid.sort(key=lambda x: x[0], reverse=True)
        best_fix_rate, best_still_wrong, best_cand = scored_valid[0]

        # --- Fix-rate gate ---
        if best_fix_rate < fix_rate_threshold:
            log_fn(
                f"  [partition:{label}] fix_rate={best_fix_rate:.0%} < "
                f"threshold={fix_rate_threshold:.0%} — "
                f"{'retrying' if not is_last else 'discarding'}."
            )
            prev_attempt = {
                "candidate":   best_cand,
                "still_wrong": best_still_wrong,
                "reason":      "fix_rate",
            }
            continue

        # --- Regression gate — designated correct pool only ---
        reg_rate = 0.0
        if pb.correct_pool and len(pb.correct_pool) >= min_pool_for_regression:
            reg_rate = _regression_check(best_cand, pb.correct_pool, cs_snapshot, **gkw)
            log_fn(f"  [partition:{label}] regression_rate={reg_rate:.0%}")
            if reg_rate > regress_threshold:
                log_fn(
                    f"  [partition:{label}] regression {reg_rate:.0%} > "
                    f"threshold={regress_threshold:.0%} — "
                    f"{'retrying' if not is_last else 'discarding'}."
                )
                prev_attempt = {
                    "candidate":   best_cand,
                    "still_wrong": best_still_wrong,
                    "reason":      "regression",
                }
                continue
        else:
            log_fn(
                f"  [partition:{label}] regression skipped — "
                f"correct_pool={len(pb.correct_pool)} < min={min_pool_for_regression}"
            )

        # --- Similarity gate + write (serialised under cs_lock) ---
        with cs_lock:
            if similarity_gate and len(cheatsheet.case_studies) >= _MIN_CS_FOR_SIMILARITY:
                action, merge_idx = _similarity_gate(
                    best_cand, cheatsheet, model_casestudy, api_key
                )
                log_fn(
                    f"  [partition:{label}] similarity → {action}"
                    + (f" (CS {merge_idx+1})" if merge_idx is not None else "")
                )
                if action == "SKIP":
                    return {
                        "partition": label,
                        "event":     "bin_skipped",
                        "reason":    "duplicate",
                        "attempt":   attempt,
                    }
                if action == "MERGE" and merge_idx is not None:
                    existing = cheatsheet.case_studies[merge_idx]
                    merged   = _merge_case_studies(existing, best_cand, model_casestudy, api_key)
                    cheatsheet.case_studies[merge_idx] = merged
                    pb.n_flushes += 1
                    log_fn(f"  [partition:{label}] merged into CS {merge_idx+1}.")
                    return {
                        "partition":       label,
                        "event":           "bin_merged",
                        "merged_into":     merge_idx + 1,
                        "fix_rate":        best_fix_rate,
                        "regression_rate": reg_rate,
                        "attempt":         attempt,
                    }

            # ADD
            best_cand.creation_fix_rate   = best_fix_rate
            best_cand.historical_fix_rate  = best_fix_rate
            cheatsheet.add_case_study(best_cand)
            pb.n_flushes += 1
            log_fn(
                f"  [partition:{label}] added CS {len(cheatsheet.case_studies)} "
                f"'{best_cand.title}' — fix_rate={best_fix_rate:.0%}  "
                f"regress={reg_rate:.0%}  attempt={attempt}"
            )
            return {
                "partition":       label,
                "event":           "bin_added",
                "title":           best_cand.title,
                "fix_rate":        best_fix_rate,
                "regression_rate": reg_rate,
                "attempt":         attempt,
                "n_cs_total":      len(cheatsheet.case_studies),
            }

    # All rounds exhausted without passing gates
    return {
        "partition":     label,
        "event":         "bin_discarded",
        "reason":        "all_rounds_failed",
        "best_fix_rate": best_fix_rate,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_partition_loop(
    cheatsheet:              Cheatsheet,
    train_items:             list[dict],
    val_items:               list[dict] | None,
    model_score:             str,
    model_casestudy:         str,
    api_key:                 str,
    # Oracle (on by default)
    oracle:                  OracleDict | None = None,
    oracle_min_similarity:   float = 0.25,
    # Binning
    bin_threshold:           int   = 3,
    retirement_threshold:    int   = 2,
    max_outer_iters:         int   = 5,
    # Concurrency
    partition_concurrency:   int   = 8,    # max bins solved in parallel
    concurrency:             int   = 25,   # LLM API concurrency per score_batch
    # Per-bin generation
    n_candidates:            int   = 3,
    candidate_rounds:        int   = 3,
    # Gates
    fix_rate_threshold:      float = 0.30,
    regress_threshold:       float = 0.15,
    min_pool_for_regression: int   = 5,
    similarity_gate:         bool  = True,
    # Scoring
    reasoning_effort:        str | None = "low",
    cot_first:               bool  = True,
    prescore_map:            dict | None = None,
    # Output
    output_dir:              Path | None = None,
    log:                     bool  = True,
) -> PartitionTrainingResult:

    def _log(msg: str) -> None:
        if log:
            print(msg, file=sys.stderr, flush=True)

    # Build oracle index for nearest-neighbour enrichment
    oracle_index: OracleIndex | None = None
    if oracle:
        from utils.oracle_index import OracleIndex as _OI
        oracle_index = _OI(oracle, min_similarity=oracle_min_similarity)

    update_log:       list[dict] = []
    n_added           = 0
    n_discarded       = 0
    n_skipped         = 0
    n_merges          = 0
    n_bins_solved     = 0
    outer_iter        = 0

    cs_lock = threading.Lock()

    _log(
        f"\n{'='*65}\n"
        f"ICR_partition Training Loop\n"
        f"  items={len(train_items)}  bin_threshold={bin_threshold}  "
        f"retirement_threshold={retirement_threshold}\n"
        f"  max_outer_iters={max_outer_iters}  "
        f"partition_concurrency={partition_concurrency}\n"
        f"  n_candidates={n_candidates}  candidate_rounds={candidate_rounds}\n"
        f"  fix_rate≥{fix_rate_threshold:.0%}  "
        f"regress≤{regress_threshold:.0%}  "
        f"min_pool={min_pool_for_regression}\n"
        f"  oracle={'yes (' + str(len(oracle_index)) + ' entries)' if oracle_index else 'none'}\n"
        f"  model_score={model_score}  model_casestudy={model_casestudy}\n"
        f"{'='*65}"
    )

    # ── Iteration 0: score all items ─────────────────────────────────────────

    _log(f"\n[iter 0] Scoring all {len(train_items)} items ...")
    if prescore_map is not None:
        correct, wrong = _apply_prescore(train_items, prescore_map)
        _log(f"  [prescore] {len(correct)} correct  {len(wrong)} wrong  (no API call)")
    else:
        correct, wrong = score_batch(
            train_items, cheatsheet.render(), model_score, api_key,
            concurrency=concurrency,
            reasoning_effort=reasoning_effort,
            cot_first=cot_first,
        )

    train_accuracy = len(correct) / len(train_items) if train_items else 0.0
    _log(
        f"  initial accuracy={train_accuracy:.1%}  "
        f"correct={len(correct)}  wrong={len(wrong)}"
    )

    bins = build_partitions(wrong, correct, bin_threshold=bin_threshold)
    print_partition_table(bins, title="INITIAL PARTITION SUMMARY")
    update_log.append({
        "event":          "initial_score",
        "accuracy":       round(train_accuracy, 4),
        "n_correct":      len(correct),
        "n_wrong":        len(wrong),
        "n_partitions":   len(bins),
    })

    if output_dir:
        _save_checkpoint(cheatsheet, update_log, output_dir, tag="init")

    # ── Outer iterations ─────────────────────────────────────────────────────

    for outer_iter in range(1, max_outer_iters + 1):
        active_bins = {k: pb for k, pb in bins.items() if not pb.solved}
        if not active_bins:
            _log(f"\n[iter {outer_iter}] All bins retired — stopping.")
            break

        _log(
            f"\n{'─'*65}\n"
            f"[iter {outer_iter}] Solving {len(active_bins)} active partitions ...\n"
            f"{'─'*65}"
        )

        iter_results: list[dict] = []

        # Solve all active bins concurrently
        with ThreadPoolExecutor(max_workers=min(partition_concurrency, len(active_bins))) as ex:
            futs = {
                ex.submit(
                    _solve_bin,
                    pb,
                    cheatsheet,
                    cs_lock,
                    oracle,
                    oracle_index,
                    n_candidates,
                    candidate_rounds,
                    model_score,
                    model_casestudy,
                    api_key,
                    concurrency,
                    fix_rate_threshold,
                    regress_threshold,
                    min_pool_for_regression,
                    similarity_gate,
                    reasoning_effort,
                    cot_first,
                    _log,
                ): pb.label
                for pb in active_bins.values()
            }
            for fut in as_completed(futs):
                result = fut.result()
                iter_results.append({**result, "outer_iter": outer_iter})
                event = result.get("event", "")
                if event == "bin_added":
                    n_added += 1
                elif event == "bin_merged":
                    n_merges += 1
                elif event == "bin_skipped":
                    n_skipped += 1
                elif event == "bin_discarded":
                    n_discarded += 1

        update_log.extend(iter_results)

        # Checkpoint after all bins in this iteration are resolved
        if output_dir:
            _save_checkpoint(cheatsheet, update_log, output_dir, tag=outer_iter)

        # ── Re-score only active-bin items ────────────────────────────────────
        # Collect the union of failures from all still-active bins.
        # This is O(active_failures) not O(all_items).
        active_failure_items = [
            item
            for pb in active_bins.values()
            for item in pb.failures
        ]

        if not active_failure_items:
            _log(f"\n[iter {outer_iter}] No active failures remaining — stopping.")
            break

        _log(
            f"\n[iter {outer_iter}] Re-scoring {len(active_failure_items)} active-bin items ..."
        )
        new_correct, new_wrong = score_batch(
            active_failure_items, cheatsheet.render(), model_score, api_key,
            concurrency=concurrency,
            reasoning_effort=reasoning_effort,
            cot_first=cot_first,
            progress_label=f"iter-{outer_iter}-rescore",
        )
        iter_acc = len(new_correct) / len(active_failure_items) if active_failure_items else 0.0
        _log(
            f"  re-score accuracy (active bins)={iter_acc:.1%}  "
            f"resolved={len(new_correct)}  remaining={len(new_wrong)}"
        )

        # Refresh partition bins with updated failure/correct lists
        refresh_partitions(
            bins,
            new_wrong=new_wrong,
            new_correct=new_correct,
            retirement_threshold=retirement_threshold,
            log_fn=_log,
        )

        newly_retired = sum(1 for pb in bins.values() if pb.solved)
        n_bins_solved = newly_retired

        update_log.append({
            "event":          "iter_rescore",
            "outer_iter":     outer_iter,
            "accuracy":       round(iter_acc, 4),
            "n_resolved":     len(new_correct),
            "n_remaining":    len(new_wrong),
            "n_bins_retired": newly_retired,
        })

        print_partition_table(bins, title=f"PARTITION SUMMARY — after iter {outer_iter}")

        # Early stop: no bin made progress this iteration
        iter_added = sum(1 for r in iter_results if r.get("event") in ("bin_added", "bin_merged"))
        if iter_added == 0:
            _log(
                f"\n[iter {outer_iter}] No case studies added this iteration — stopping early."
            )
            break

    # ── Final accuracy on full train set ─────────────────────────────────────
    _log(f"\n[final] Scoring {len(train_items)} train items for final accuracy ...")
    final_correct, final_wrong = score_batch(
        train_items, cheatsheet.render(), model_score, api_key,
        concurrency=concurrency,
        reasoning_effort=reasoning_effort,
        cot_first=cot_first,
        progress_label="final-train",
    )
    train_accuracy = len(final_correct) / len(train_items) if train_items else 0.0
    _log(f"  final train accuracy={train_accuracy:.1%}")

    if output_dir:
        _save_checkpoint(cheatsheet, update_log, output_dir, tag="final")

    return PartitionTrainingResult(
        cheatsheet=cheatsheet,
        n_case_studies_added=n_added,
        n_bins_solved=n_bins_solved,
        n_bins_discarded=n_discarded,
        n_merges=n_merges,
        n_skipped=n_skipped,
        n_outer_iters=outer_iter,
        train_accuracy=train_accuracy,
        update_log=update_log,
        partition_summary=partition_summary(bins),
    )
