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
from utils.task_spec import TaskSpec
from ICR_reasoning.core.oracle import OracleDict
from ICR_select.generators.case_study import (
    generate_candidates,
    generate_crossover,
    _detect_failure_type,
    _parse_divergence_step,
)
from ICR_select.training.gates import (
    _MIN_CS_FOR_SIMILARITY,
    _apply_prescore,
    _mini_eval_full,
    _mini_eval_text,
    _regression_check,
    _regression_check_text,
    _similarity_gate,
    _merge_case_studies,
    _prune_cs_bank,
)
from ICR_rules.rules.rule import RuleSet
from ICR_rules.rules.parser import parse_cheatsheet_text, identify_triggered_rule
from ICR_rules.generators.rule_patch import generate_rule_patch
from ICR_rules.training.scorer import score_batch_sair

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
    n_rule_patches:       int = 0   # rule patches applied (ICR_rules mode)


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
    task_spec:               TaskSpec | None = None,
    inject_gold_oracle:      bool = True,
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
        task_spec=task_spec,
    )

    # ── Concrete-example generation pre-pass ─────────────────────────────────
    # When task_spec.concrete_cs_gen_fn is set, try generating a named-scenario
    # text section (CS-ICL style) before the standard archive/crossover path.
    # If the section passes fix-rate and regression gates, it is appended to
    # cheatsheet.prior_knowledge and we return immediately.  Otherwise we fall
    # through to the normal structured CaseStudy generation path.
    if task_spec is not None and task_spec.concrete_cs_gen_fn is not None:
        with cs_lock:
            cs_text = cheatsheet.render()
            cs_pre_snap = Cheatsheet(
                roadmap=cheatsheet.roadmap,
                case_studies=list(cheatsheet.case_studies),
                prior_knowledge=cheatsheet.prior_knowledge,
                no_limit=cheatsheet.no_limit,
            )
        log_fn(f"  [partition:{label}] trying concrete-example generation ...")
        try:
            section = task_spec.concrete_cs_gen_fn(
                failures, cs_text, model_casestudy, api_key
            )
        except Exception as exc:
            log_fn(f"  [partition:{label}] concrete_cs_gen_fn failed: {exc}")
            section = None
        if section:
            fr_txt, _ = _mini_eval_text(
                section, failures, cs_pre_snap,
                label=f"[{label}] concrete-gen",
                **gkw,
            )
            log_fn(f"  [partition:{label}] concrete-gen fix_rate={fr_txt:.0%}")
            if fr_txt >= fix_rate_threshold:
                reg_txt = 0.0
                if pb.correct_pool and len(pb.correct_pool) >= min_pool_for_regression:
                    reg_txt = _regression_check_text(
                        section, pb.correct_pool, cs_pre_snap, **gkw
                    )
                    log_fn(f"  [partition:{label}] concrete-gen regression_rate={reg_txt:.0%}")
                if reg_txt <= regress_threshold:
                    with cs_lock:
                        cheatsheet.prior_knowledge = (
                            cheatsheet.prior_knowledge.rstrip() + "\n\n" + section
                        ).lstrip()
                        pb.n_flushes += 1
                    log_fn(
                        f"  [partition:{label}] concrete-gen accepted: "
                        f"fix_rate={fr_txt:.0%}  regress={reg_txt:.0%}"
                    )
                    return {
                        "partition":       label,
                        "event":           "bin_added",
                        "title":           f"concrete:{label}",
                        "fix_rate":        fr_txt,
                        "regression_rate": reg_txt,
                        "attempt":         1,
                        "n_cs_total":      len(cheatsheet.case_studies),
                        "source":          "concrete_cs_gen",
                    }
                else:
                    log_fn(
                        f"  [partition:{label}] concrete-gen regression {reg_txt:.0%} > "
                        f"threshold={regress_threshold:.0%} — falling back to structured generation."
                    )
            else:
                log_fn(
                    f"  [partition:{label}] concrete-gen fix_rate {fr_txt:.0%} < "
                    f"threshold={fix_rate_threshold:.0%} — falling back to structured generation."
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
                existing_titles = {cs.title.strip().lower() for cs in cheatsheet.case_studies}
                if arch_cand.title.strip().lower() in existing_titles:
                    log_fn(
                        f"  [partition:{label}] archived CS '{arch_cand.title}' "
                        f"is a title duplicate — skipping"
                    )
                    return {
                        "partition": label,
                        "event":     "bin_skipped",
                        "reason":    "title_duplicate",
                        "attempt":   0,
                    }
                arch_cand.creation_fix_rate  = arch_fr
                arch_cand.historical_fix_rate = arch_fr
                cheatsheet.add_case_study(arch_cand)
                if arch_cand.roadmap_patch:
                    cheatsheet.patch_roadmap(arch_cand.roadmap_patch)
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

        # --- Detect failure type and divergence step (from ICR_adaptive) ---
        # ABANDONMENT failures → CONTRAST strategy (show wrong path + correct path).
        # WRONG_ANSWER failures → DIRECT_FIX or ORACLE_GUIDED (existing behaviour).
        # divergence_step_hint focuses the NEXT CHECK on the specific step where the
        # model went wrong, rather than a generic instruction.
        failure_type_hint  = _detect_failure_type(failures)
        divergence_step_hint = _parse_divergence_step(failures)
        log_fn(
            f"  [partition:{label}] failure_type={failure_type_hint}  "
            f"diverge_step={divergence_step_hint}"
        )

        # --- Generate candidates ---
        # Polarity = majority expected answer ("TRUE" / "FALSE") — tells the generator
        # whether to weight TYPE A (missing proof, TRUE bins) or TYPE B (wrong pattern,
        # FALSE bins).  For the 8-element magma key, index 3 stores this directly.
        # For generic keys (e.g. BBH boolean (bool, bool, bool)), derive it from items.
        if len(pb.key) >= 4 and isinstance(pb.key[3], str) and pb.key[3] in ("TRUE", "FALSE"):
            bin_polarity = pb.key[3]
        else:
            from collections import Counter as _Counter
            _lc = _Counter(task_spec.answer_label(it) for it in failures)
            bin_polarity = _lc.most_common(1)[0][0] if _lc else "FALSE"
        try:
            candidates = generate_candidates(
                failures, cs_snapshot, model_casestudy, api_key,
                n=n_candidates, oracle=oracle, prev_attempt=prev_attempt,
                polarity=bin_polarity,
                failure_type_hint=failure_type_hint,
                divergence_step_hint=divergence_step_hint,
                task_spec=task_spec,
                inject_gold_oracle=inject_gold_oracle,
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
        _pk_to_cond = (task_spec.partition_key_to_conditions
                       if task_spec is not None else partition_key_to_conditions)
        partition_conditions = _pk_to_cond(pb.key)
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

            # MODIFY or ADD case study + apply any accompanying roadmap patch.
            best_cand.creation_fix_rate   = best_fix_rate
            best_cand.historical_fix_rate  = best_fix_rate
            if best_cand.modification_target:
                replaced = cheatsheet.replace_case_study(best_cand.modification_target, best_cand)
                if replaced:
                    pb.n_flushes += 1
                    log_fn(
                        f"  [partition:{label}] modified CS "
                        f"'{best_cand.modification_target}' → '{best_cand.title}' "
                        f"fix_rate={best_fix_rate:.0%}  regress={reg_rate:.0%}  attempt={attempt}"
                    )
                    if best_cand.roadmap_patch:
                        cheatsheet.patch_roadmap(best_cand.roadmap_patch)
                        log_fn(f"  [partition:{label}] roadmap patch applied ({len(best_cand.roadmap_patch)} chars)")
                    return {
                        "partition":       label,
                        "event":           "bin_modified",
                        "title":           best_cand.title,
                        "modified_from":   best_cand.modification_target,
                        "fix_rate":        best_fix_rate,
                        "regression_rate": reg_rate,
                        "attempt":         attempt,
                        "n_cs_total":      len(cheatsheet.case_studies),
                    }
                else:
                    log_fn(
                        f"  [partition:{label}] MODIFY target '{best_cand.modification_target}' "
                        f"not found — falling back to ADD NEW"
                    )
                    best_cand.modification_target = ""

            # ADD path (new case study, or fallback from failed MODIFY)
            existing_titles = {cs.title.strip().lower() for cs in cheatsheet.case_studies}
            if best_cand.title.strip().lower() in existing_titles:
                log_fn(
                    f"  [partition:{label}] CS '{best_cand.title}' "
                    f"is a title duplicate — skipping"
                )
                return {
                    "partition": label,
                    "event":     "bin_skipped",
                    "reason":    "title_duplicate",
                    "attempt":   attempt,
                }
            cheatsheet.add_case_study(best_cand)
            if best_cand.roadmap_patch:
                cheatsheet.patch_roadmap(best_cand.roadmap_patch)
                log_fn(
                    f"  [partition:{label}] roadmap patch applied "
                    f"({len(best_cand.roadmap_patch)} chars)"
                )
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
# Rule-patch pre-step (ICR_rules mode integrated into ICR_partition)
# ---------------------------------------------------------------------------

def _dominant_triggered_rule(failures: list[dict], task_spec=None) -> str | None:
    """Return the rule ID that fired most often across failure reasoning traces."""
    from collections import Counter
    _identify = (
        task_spec.identify_triggered_rule
        if task_spec is not None and task_spec.identify_triggered_rule is not None
        else identify_triggered_rule
    )
    counts: Counter = Counter()
    for item in failures:
        reasoning = item.get("reasoning") or item.get("post_think") or ""
        rule_id = _identify(reasoning)
        if rule_id:
            counts[rule_id] += 1
    return counts.most_common(1)[0][0] if counts else None


def _rule_patch_pass(
    active_bins:       dict,
    rule_set:          RuleSet,
    cheatsheet:        Cheatsheet,
    model_patch:       str,
    model_score:       str,
    api_key:           str,
    oracle:            OracleDict | None,
    fix_rate_threshold: float,
    regress_threshold:  float,
    concurrency:        int,
    partition_concurrency: int,
    log_fn,
    task_spec=None,
) -> tuple[RuleSet, list[dict], int]:
    """
    For each active bin, try to identify the dominant misfiring rule and
    generate a surgical patch (TIGHTEN / SPLIT / REPLACE / ADD_GUARD).

    Patches are staged (one per target rule ID, highest fix_rate wins) and
    applied atomically after all bins are processed — mirroring ICR_rules'
    conflict-resolution strategy.

    Returns (updated_rule_set, patch_log_entries, n_applied).
    Bins that receive a patch have their failure lists trimmed in-place so
    the subsequent case-study pass only sees items not already fixed.
    """
    staged: list[tuple[str, object, RuleSet, float, str]] = []  # (bin_label, patch, patched_rs, fix_rate, bin_key_str)
    patch_log: list[dict] = []

    def _solve_one(pb_label_key):
        pb, label, bin_key_str = pb_label_key
        triggered = _dominant_triggered_rule(pb.failures, task_spec=task_spec)
        if triggered is None:
            log_fn(f"  [rule-patch:{label}] no dominant rule identified — skipping")
            return None

        target_rule = rule_set.get_rule(triggered)
        if target_rule is None:
            log_fn(f"  [rule-patch:{label}] rule {triggered} not in RuleSet — skipping")
            return None

        log_fn(f"  [rule-patch:{label}] targeting rule {triggered}")

        patch = generate_rule_patch(
            target_rule=target_rule,
            rule_set=rule_set,
            failures=pb.failures,
            correct_pool=list(pb.correct_pool)[:30],
            oracle=oracle or {},
            model=model_patch,
            api_key=api_key,
            task_spec=task_spec,
        )
        if patch is None:
            log_fn(f"  [rule-patch:{label}] patch generation failed")
            return None

        try:
            patched_rs = rule_set.apply_patch(patch)
        except Exception as exc:
            log_fn(f"  [rule-patch:{label}] patch apply error: {exc}")
            return None

        # Score bin failures with the patched RuleSet
        bin_items = pb.failures + list(pb.correct_pool)[:len(pb.failures)]
        bc, bw = score_batch_sair(
            bin_items, patched_rs, model_score, api_key,
            concurrency=min(concurrency, len(bin_items) + 1),
            task_spec=task_spec,
        )
        failure_ids = {item["id"] for item in pb.failures}
        fix_rate = sum(1 for it in bc if it["id"] in failure_ids) / len(pb.failures)
        patch.bin_fix_rate = fix_rate

        if fix_rate < fix_rate_threshold:
            log_fn(f"  [rule-patch:{label}] fix_rate={fix_rate:.0%} < threshold — discarding")
            return None

        # Regression check on correct pool
        pool = list(pb.correct_pool)[:30]
        if pool:
            pc, pw = score_batch_sair(
                pool, patched_rs, model_score, api_key,
                concurrency=min(concurrency, len(pool) + 1),
                task_spec=task_spec,
            )
            reg_rate = len(pw) / len(pool)
            if reg_rate > regress_threshold:
                log_fn(f"  [rule-patch:{label}] regression={reg_rate:.0%} > threshold — discarding")
                return None

        log_fn(
            f"  [rule-patch:{label}] ACCEPTED  rule={triggered}  "
            f"type={patch.patch_type}  fix_rate={fix_rate:.0%}"
        )
        return (label, patch, patched_rs, fix_rate, bin_key_str, triggered, pb)

    n_parallel = min(partition_concurrency, len(active_bins))
    work = [(pb, pb.label, str(k)) for k, pb in active_bins.items()]

    with ThreadPoolExecutor(max_workers=n_parallel) as ex:
        futs = {ex.submit(_solve_one, item): item for item in work}
        for fut in as_completed(futs):
            result = fut.result()
            if result is not None:
                label, patch, patched_rs, fix_rate, bin_key_str, triggered, pb = result
                staged.append((label, patch, patched_rs, fix_rate, bin_key_str, triggered, pb))

    if not staged:
        return rule_set, [], 0

    # Apply one patch per target rule (highest fix_rate wins) — conflict resolution
    staged.sort(key=lambda x: x[3], reverse=True)
    applied_targets: set[str] = set()
    current_rs = rule_set
    n_applied = 0

    for label, patch, patched_rs, fix_rate, bin_key_str, triggered, pb in staged:
        if triggered in applied_targets:
            continue
        current_rs = patched_rs
        applied_targets.add(triggered)
        n_applied += 1
        patch_log.append({
            "event":       "rule_patch_applied",
            "partition":   bin_key_str,
            "target_rule": triggered,
            "patch_type":  patch.patch_type,
            "fix_rate":    fix_rate,
            "reasoning":   patch.reasoning,
        })
        log_fn(
            f"  [rule-patch] applied: {triggered} → {patch.patch_type}  "
            f"(fix_rate={fix_rate:.0%})"
        )

        # Mark newly-fixed failures so the case-study pass skips them.
        # Re-use the patched RuleSet to score only this bin's failures.
        bc, _ = score_batch_sair(
            pb.failures, current_rs, model_score, api_key,
            concurrency=min(concurrency, len(pb.failures) + 1),
            task_spec=task_spec,
        )
        fixed_ids = {it["id"] for it in bc}
        pb.failures = [f for f in pb.failures if f["id"] not in fixed_ids]

    # Update prior_knowledge in the cheatsheet from the patched RuleSet.
    # render_decision_guide() strips the jinja2 preamble so it's pure rule text.
    cheatsheet.prior_knowledge = current_rs.render_decision_guide()

    return current_rs, patch_log, n_applied


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
    prune_every_n:           int   = 5,   # run LLM prune pass every N accepted CS (0 = disabled)
    # Scoring
    reasoning_effort:        str | None = "low",
    cot_first:               bool  = True,
    prescore_map:            dict | None = None,
    # ICR_rules integration — optional rule-patch pre-step
    rule_set:                RuleSet | None = None,
    model_patch:             str = "",     # model used for rule-patch generation
    # Task specification — defaults to MAGMA_TASK for backward compat
    task_spec:               TaskSpec | None = None,
    # Output
    cs_static_iters:         int   = 2,   # stop Phase 2 after this many consecutive idle iters
    output_dir:              Path | None = None,
    save_scored:             bool  = False,
    log:                     bool  = True,
    # Prior-knowledge regression guard
    # When enabled, a pk-only baseline is scored at iter 0.  At the end of
    # Phase 2, if the full cheatsheet accuracy drops more than pk_regression_tolerance
    # below that baseline, all case studies are reverted and the pk-only cheatsheet
    # is restored.
    pk_regression_guard:     bool  = False,
    pk_regression_tolerance: float = 0.03,
    # Gold oracle injection in Phase 2 case study generation.
    # When True (default), item["reason"] is shown as correct-reasoning contrast
    # alongside the model's wrong reasoning when generating case studies.
    # Set False to ablate this signal.
    inject_gold_oracle:      bool  = True,
    max_case_studies:        int | None = None,  # hard cap on CS added; None = unlimited
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
    # Resolve task_spec — default to MAGMA_TASK for backward compat
    if task_spec is None:
        from tasks.magma import MAGMA_TASK
        task_spec = MAGMA_TASK

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
                cot_first=cot_first, task_spec=task_spec,
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
                task_spec=task_spec,
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
    n_rule_patches    = 0
    outer_iter        = 0
    static_cs_iters   = 0   # consecutive outer iters with no CS added

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

    # When best-of-N mode is active, loosen the collection gates so more
    # candidates enter the pool; the top N are selected after the loop ends.
    # Floors/ceilings prevent the effective thresholds from becoming meaningless.
    _best_of_n = max_case_studies is not None
    _pool_fix_rate  = max(0.15, fix_rate_threshold * 0.67) if _best_of_n else fix_rate_threshold
    _pool_regress   = min(0.25, regress_threshold  * 1.33) if _best_of_n else regress_threshold

    _log(
        f"\n{'='*65}\n"
        f"ICR_partition Training Loop  (PID {os.getpid()})\n"
        f"  items={len(train_items)}  bin_threshold={bin_threshold}  "
        f"retirement_threshold={retirement_threshold}\n"
        f"  max_outer_iters={max_outer_iters}  "
        f"partition_concurrency={partition_concurrency}\n"
        f"  concurrency={concurrency} (divided across active bins)\n"
        f"  n_candidates={n_candidates}  candidate_rounds={candidate_rounds}\n"
        f"  fix_rate≥{_pool_fix_rate:.0%}  "
        f"regress≤{_pool_regress:.0%}  "
        f"min_pool={min_pool_for_regression}\n"
        + (f"  best-of-N={max_case_studies} (pool thresholds loosened from "
           f"{fix_rate_threshold:.0%}/{regress_threshold:.0%})\n" if _best_of_n else "")
        + f"  oracle={'yes (' + str(len(oracle_index)) + ' entries)' if oracle_index else 'none'}\n"
        f"  model_score={model_score}  model_casestudy={model_casestudy}\n"
        f"  rule_patch={'yes (model=' + model_patch + ')' if rule_set is not None else 'disabled'}\n"
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

    # ── PK regression guard — baseline ───────────────────────────────────────
    pk_baseline_acc: float | None = None
    if pk_regression_guard and cheatsheet.prior_knowledge.strip():
        _log(f"\n[pk-guard] Scoring prior_knowledge-only baseline ...")
        _pk_only = Cheatsheet(
            roadmap=cheatsheet.roadmap,
            case_studies=[],
            prior_knowledge=cheatsheet.prior_knowledge,
            no_limit=cheatsheet.no_limit,
        )
        _pk_correct, _ = _do_score(
            train_items, _pk_only.render(),
            concurrency=concurrency, reasoning_effort=reasoning_effort,
            cot_first=cot_first, progress_label="pk-baseline",
        )
        pk_baseline_acc = len(_pk_correct) / len(train_items) if train_items else 0.0
        _log(f"  [pk-guard] baseline acc = {pk_baseline_acc:.1%}  tolerance = {pk_regression_tolerance:.0%}")

    if save_scored and output_dir is not None:
        scored_path = Path(output_dir) / "scored_failures.jsonl"
        with open(scored_path, "w", encoding="utf-8") as _sf:
            for _item in wrong:
                _sf.write(json.dumps(_item, ensure_ascii=False) + "\n")
        _log(f"  [scored] {len(wrong)} failures saved to {scored_path}")

    bins = build_partitions(wrong, correct, bin_threshold=bin_threshold,
                            partition_key_fn=task_spec.partition_key)
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

        # ── Rule-patch pre-step (ICR_rules mode) ─────────────────────────────
        # If a RuleSet was provided, attempt to patch misfiring named rules
        # before the case-study generation pass.  Patches that pass the fix-rate
        # and regression gates are applied and the updated RuleSet is used for
        # all subsequent iterations.  Bins whose failures are resolved by a patch
        # are trimmed in-place so the case-study pass only sees remaining items.
        if rule_set is not None:
            _log(
                f"\n[iter {outer_iter}] Running rule-patch pre-step on "
                f"{len(active_bins)} active bins ..."
            )
            rule_set, rp_log, n_rp = _rule_patch_pass(
                active_bins=active_bins,
                rule_set=rule_set,
                cheatsheet=cheatsheet,
                model_patch=model_patch or model_casestudy,
                model_score=model_score,
                api_key=api_key,
                oracle=oracle,
                fix_rate_threshold=fix_rate_threshold,
                regress_threshold=regress_threshold,
                concurrency=concurrency,
                partition_concurrency=partition_concurrency,
                log_fn=_log,
                task_spec=task_spec,
            )
            n_rule_patches += n_rp
            update_log.extend([{**e, "outer_iter": outer_iter} for e in rp_log])
            if n_rp:
                _log(f"  [rule-patch] {n_rp} patch(es) applied this iteration.")

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
        _n_cs_before_iter = len(cheatsheet.case_studies)

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
                    _pool_fix_rate,
                    _pool_regress,
                    min_pool_for_regression,
                    similarity_gate,
                    reasoning_effort,
                    cot_first,
                    _log,
                    task_spec,
                    inject_gold_oracle,
                ): pb.label
                for pb in active_bins.values()
            }
            for fut in as_completed(futs):
                result = fut.result()
                iter_results.append({**result, "outer_iter": outer_iter})
                event = result.get("event", "")
                if event in ("bin_added", "bin_modified"):
                    n_added += 1
                elif event == "bin_merged":
                    n_merges += 1
                elif event == "bin_skipped":
                    n_skipped += 1
                elif event == "bin_discarded":
                    n_discarded += 1

        update_log.extend(iter_results)

        # Periodic CS bank prune — fires when the total count crosses a multiple of prune_every_n
        if prune_every_n and (
            len(cheatsheet.case_studies) // prune_every_n
            > _n_cs_before_iter // prune_every_n
        ):
            _log(
                f"\n[iter {outer_iter}] CS bank prune triggered "
                f"({_n_cs_before_iter}→{len(cheatsheet.case_studies)} CS, "
                f"every_n={prune_every_n}) ..."
            )
            n_pruned = _prune_cs_bank(cheatsheet, model_casestudy, api_key, _log)
            if n_pruned:
                update_log.append({
                    "event":      "cs_pruned",
                    "outer_iter": outer_iter,
                    "n_pruned":   n_pruned,
                    "n_cs_total": len(cheatsheet.case_studies),
                })

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
            partition_key_fn=task_spec.partition_key,
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

        # Early stop: exit after cs_static_iters consecutive idle iterations
        iter_added = sum(1 for r in iter_results if r.get("event") in ("bin_added", "bin_modified", "bin_merged"))
        if iter_added == 0:
            static_cs_iters += 1
            _log(
                f"\n[iter {outer_iter}] No case studies added "
                f"({static_cs_iters}/{cs_static_iters} consecutive idle) — "
                + ("stopping early." if static_cs_iters >= cs_static_iters
                   else "continuing.")
            )
            if static_cs_iters >= cs_static_iters:
                break
        else:
            static_cs_iters = 0

    # Restore original signal handlers
    signal.signal(signal.SIGINT,  _prev_sigint)
    signal.signal(signal.SIGTERM, _prev_sigterm)

    # ── Best-of-N post-hoc pruning ────────────────────────────────────────────
    if _best_of_n and len(cheatsheet.case_studies) > max_case_studies:
        ranked = sorted(
            cheatsheet.case_studies,
            key=lambda cs: cs.historical_fix_rate or cs.creation_fix_rate,
            reverse=True,
        )
        kept   = ranked[:max_case_studies]
        dropped = ranked[max_case_studies:]
        _log(
            f"\n[best-of-N] Pool has {len(ranked)} CS — selecting top {max_case_studies} "
            f"by fix_rate, dropping {len(dropped)}:"
        )
        for cs in kept:
            fr = cs.historical_fix_rate or cs.creation_fix_rate
            _log(f"  KEEP  '{cs.title}'  fix_rate={fr:.0%}")
        for cs in dropped:
            fr = cs.historical_fix_rate or cs.creation_fix_rate
            _log(f"  DROP  '{cs.title}'  fix_rate={fr:.0%}")
        cheatsheet.case_studies = kept
        n_added = len(kept)
        if output_dir:
            _save_checkpoint(cheatsheet, update_log, output_dir, tag="best_of_n")

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

    # ── PK regression guard — revert check ───────────────────────────────────
    if pk_baseline_acc is not None:
        _log(
            f"\n[pk-guard] final={train_accuracy:.1%}  "
            f"baseline={pk_baseline_acc:.1%}  tol={pk_regression_tolerance:.0%}"
        )
        if train_accuracy < pk_baseline_acc - pk_regression_tolerance:
            n_reverted = len(cheatsheet.case_studies)
            reverted_from_acc = train_accuracy
            _log(
                f"  [pk-guard] REGRESSION DETECTED — reverting {n_reverted} case "
                f"studies and restoring prior_knowledge-only cheatsheet."
            )
            cheatsheet.case_studies = []
            _revert_correct, _ = _do_score(
                train_items, cheatsheet.render(),
                concurrency=concurrency, reasoning_effort=reasoning_effort,
                cot_first=cot_first, progress_label="pk-fallback",
            )
            train_accuracy = len(_revert_correct) / len(train_items) if train_items else 0.0
            _log(f"  [pk-guard] reverted accuracy = {train_accuracy:.1%}")
            update_log.append({
                "event":                         "pk_guard_revert",
                "n_case_studies_reverted":       n_reverted,
                "train_accuracy_before_revert":  round(reverted_from_acc, 4),
                "pk_baseline_acc":               round(pk_baseline_acc, 4),
                "train_accuracy_after_revert":   round(train_accuracy, 4),
            })
        else:
            _log(f"  [pk-guard] OK — no regression detected.")

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
        n_rule_patches=n_rule_patches,
        partition_summary=partition_summary(bins),
    )
