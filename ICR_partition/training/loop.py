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
import os
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from utils.cheatsheet import Cheatsheet
from utils.scorer import score_batch, score_batch_ensemble
from utils.oracle_index import OracleIndex
from ICR_reasoning.core.oracle import OracleDict
from ICR_select.generators.case_study import generate_candidates, generate_crossover
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

    # ── EA: Archive re-evaluation ─────────────────────────────────────────────
    # Candidates that failed gates in a previous outer iteration are re-scored
    # against the CURRENT failure set.  If the failure set has shifted enough
    # (e.g. a previously-broken item was fixed by another case study), an
    # archived candidate may now pass both the fix-rate and regression gates.
    # This avoids discarding work that was "close" in an earlier iteration.
    if pb.candidate_archive:
        log_fn(
            f"  [partition:{label}] re-evaluating {len(pb.candidate_archive)} "
            f"archived candidate(s) against updated failure set ..."
        )
        with cs_lock:
            cs_arch_snapshot = Cheatsheet(
                roadmap=cheatsheet.roadmap,
                case_studies=list(cheatsheet.case_studies),
                prior_knowledge=cheatsheet.prior_knowledge,
                no_limit=cheatsheet.no_limit,
            )
        for arch_fr_prev, arch_cand in pb.candidate_archive:
            arch_fr, arch_still_wrong = _mini_eval_full(
                arch_cand, failures, cs_arch_snapshot,
                label=f"[{label}] archive",
                **gkw,
            )
            log_fn(
                f"  [partition:{label}] archive '{arch_cand.title}': "
                f"fix_rate={arch_fr:.0%} (was {arch_fr_prev:.0%})"
            )
            if arch_fr < fix_rate_threshold:
                continue
            # Passes fix-rate — check regression
            arch_reg = 0.0
            if pb.correct_pool and len(pb.correct_pool) >= min_pool_for_regression:
                arch_reg = _regression_check(arch_cand, pb.correct_pool, cs_arch_snapshot, **gkw)
                log_fn(f"  [partition:{label}] archive regression_rate={arch_reg:.0%}")
                if arch_reg > regress_threshold:
                    log_fn(
                        f"  [partition:{label}] archive regression {arch_reg:.0%} > "
                        f"threshold={regress_threshold:.0%} — skipping"
                    )
                    continue
            # Passes both gates — accept and return immediately.
            # Remove from archive so it isn't re-added on future iterations.
            pb.candidate_archive = [
                (fr, cs) for fr, cs in pb.candidate_archive if cs is not arch_cand
            ]
            with cs_lock:
                arch_cand.creation_fix_rate  = arch_fr
                arch_cand.historical_fix_rate = arch_fr
                cheatsheet.add_case_study(arch_cand)
                pb.n_flushes += 1
                log_fn(
                    f"  [partition:{label}] archive candidate accepted: "
                    f"'{arch_cand.title}' fix_rate={arch_fr:.0%}  regress={arch_reg:.0%}"
                )
                return {
                    "partition":       label,
                    "event":           "bin_added",
                    "title":           arch_cand.title,
                    "fix_rate":        arch_fr,
                    "regression_rate": arch_reg,
                    "attempt":         0,   # 0 = accepted from archive (no new generation)
                    "n_cs_total":      len(cheatsheet.case_studies),
                    "source":          "archive",
                }

    # ── EA: Crossover candidate ───────────────────────────────────────────────
    # If the archive has two or more entries, combine the top-2 into a single
    # crossover candidate that is prepended to the pool for attempt 1.
    # This is cheap (one LLM call at temperature=0.5) and gives the loop a
    # structurally novel starting point before falling back to fresh generation.
    crossover_cand = None
    if len(pb.candidate_archive) >= 2:
        try:
            fr1, cs1 = pb.candidate_archive[0]
            fr2, cs2 = pb.candidate_archive[1]
            crossover_cand = generate_crossover(cs1, fr1, cs2, fr2, model_casestudy, api_key)
            if crossover_cand:
                log_fn(f"  [partition:{label}] crossover candidate generated from archive top-2")
        except Exception as exc:
            log_fn(f"  [partition:{label}] crossover generation failed: {exc}")

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
        # pb.key[3] is expected_answer ("TRUE" or "FALSE") — the bin polarity.
        # Passing it lets generate_candidates weight case study type appropriately:
        # TRUE bins → TYPE A (missing lemma/proof); FALSE bins → TYPE B (wrong pattern).
        try:
            candidates = generate_candidates(
                failures, cs_snapshot, model_casestudy, api_key,
                n=n_candidates, oracle=oracle, prev_attempt=prev_attempt,
                polarity=pb.key[3],   # index 3 = expected_answer (unchanged)
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

        # Prepend crossover candidate on the first attempt (consumed once only).
        if crossover_cand is not None and attempt == 1:
            candidates = [crossover_cand] + candidates
            crossover_cand = None   # don't reuse in subsequent attempts

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

        # Archive all evaluated candidates for future re-evaluation and crossover.
        # This is safe to do unconditionally: the archive keeps top-ARCHIVE_MAX by
        # fix_rate and silently discards duplicates via the sort-and-cap logic.
        for fr_i, _, cand_i in scored_valid:
            pb.archive_candidate(fr_i, cand_i)

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
    # Ensemble scoring — optional second scoring model
    model_score_2:           str | None = None,
    model_score_weights:     list[float] | None = None,  # [w1, w2]; default [1,1]
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

    # ── Ensemble scorer ───────────────────────────────────────────────────────
    # _do_score is a drop-in for score_batch throughout this loop.
    # When model_score_2 is provided it calls score_batch_ensemble, scoring both
    # models in parallel.  Each wrong item carries _wrong_weight ∈ (0,1] —
    # 1.0 if both models failed, 0.5 if only one failed — which flows into the
    # weighted fix_rate inside _mini_eval_full without any extra API calls.
    if model_score_2:
        _models  = [model_score, model_score_2]
        _weights = model_score_weights or [1.0, 1.0]
        def _do_score(
            items, cheatsheet_text, *, concurrency, reasoning_effort, cot_first,
            progress_label="scoring",
        ) -> tuple[list[dict], list[dict]]:
            return score_batch_ensemble(
                items, cheatsheet_text, _models, _weights, api_key,
                concurrency=concurrency, reasoning_effort=reasoning_effort,
                cot_first=cot_first,
            )
        _log(
            f"\n[ensemble] scoring with 2 models: {model_score} (w={_weights[0]}) "
            f"+ {model_score_2} (w={_weights[1]})"
        )
    else:
        def _do_score(
            items, cheatsheet_text, *, concurrency, reasoning_effort, cot_first,
            progress_label="scoring",
        ) -> tuple[list[dict], list[dict]]:
            return score_batch(
                items, cheatsheet_text, model_score, api_key,
                concurrency=concurrency, reasoning_effort=reasoning_effort,
                cot_first=cot_first, progress_label=progress_label,
            )

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

    # ── Graceful-shutdown via SIGINT / SIGTERM ────────────────────────────────
    # Writing the PID file lets the user do:  kill -TERM <pid>  or Ctrl-C
    # The loop checks _stop_event before each outer iteration and after each
    # bin completes; it always saves a checkpoint before exiting.
    _stop_event = threading.Event()

    def _handle_signal(signum, frame):
        _log(
            f"\n[shutdown] Signal {signum} received — "
            f"finishing current iteration then saving checkpoint ..."
        )
        _stop_event.set()

    _prev_sigint  = signal.signal(signal.SIGINT,  _handle_signal)
    _prev_sigterm = signal.signal(signal.SIGTERM, _handle_signal)

    _log(
        f"\n{'='*65}\n"
        f"ICR_partition Training Loop  (PID {os.getpid()})\n"
        f"  items={len(train_items)}  bin_threshold={bin_threshold}  "
        f"retirement_threshold={retirement_threshold}\n"
        f"  max_outer_iters={max_outer_iters}  "
        f"partition_concurrency={partition_concurrency}\n"
        f"  concurrency={concurrency} (divided across active bins)\n"
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
        correct, wrong = _do_score(
            train_items, cheatsheet.render(),
            concurrency=concurrency,
            reasoning_effort=reasoning_effort,
            cot_first=cot_first,
            progress_label="scoring",
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
        if _stop_event.is_set():
            _log(f"\n[iter {outer_iter}] Stop requested — exiting loop.")
            break

        active_bins = {k: pb for k, pb in bins.items() if not pb.solved}
        if not active_bins:
            _log(f"\n[iter {outer_iter}] All bins retired — stopping.")
            break

        # Divide total concurrency across all simultaneous LLM calls.
        # Each bin evaluates n_candidates in parallel, each using score_batch
        # with per_bin_concurrency workers.  The true multiplication is:
        #   n_parallel_bins × n_candidates × per_bin_concurrency
        # We solve for per_bin_concurrency so the product ≤ concurrency.
        n_parallel = min(partition_concurrency, len(active_bins))
        per_bin_concurrency = max(1, concurrency // (n_parallel * n_candidates))
        _log(
            f"\n{'─'*65}\n"
            f"[iter {outer_iter}] Solving {len(active_bins)} active partitions "
            f"({n_parallel} parallel × {n_candidates} cands × "
            f"{per_bin_concurrency} concurrency = "
            f"{n_parallel * n_candidates * per_bin_concurrency} max requests) ...\n"
            f"{'─'*65}"
        )

        iter_results: list[dict] = []

        # Solve all active bins concurrently
        with ThreadPoolExecutor(max_workers=n_parallel) as ex:
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
                    per_bin_concurrency,
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

        if _stop_event.is_set():
            _log(f"\n[iter {outer_iter}] Stop requested — checkpoint saved, exiting.")
            break

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
        new_correct, new_wrong = _do_score(
            active_failure_items, cheatsheet.render(),
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

    # Restore original signal handlers
    signal.signal(signal.SIGINT,  _prev_sigint)
    signal.signal(signal.SIGTERM, _prev_sigterm)

    # ── Final accuracy on full train set ─────────────────────────────────────
    _log(f"\n[final] Scoring {len(train_items)} train items for final accuracy ...")
    final_correct, final_wrong = _do_score(
        train_items, cheatsheet.render(),
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
