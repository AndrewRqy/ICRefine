"""
ICR_hybrid/training/loop.py — Hybrid rule-then-case-study refinement loop.

Algorithm
---------
Phase 1  (rule patching, skipped when initial_rule_set is None):
  For each outer iteration (up to max_rule_iters, or until rule_acc_goal):
    1. Score train_items with the current cheatsheet (ICR-style scorer).
    2. Build partition bins from failures.
    3. _rule_patch_pass: for each active bin, find the dominant misfiring rule,
       generate a TIGHTEN/SPLIT/REPLACE/ADD_GUARD patch, gate it (fix-rate +
       regression), stage it, and apply at the end of the iteration.
    4. Checkpoint.  Re-score active failures.  Refresh bins.
    5. Exit early when acc >= rule_acc_goal OR static_iters >= rule_static_iters.

  NOTE: Phase 1 is currently magma-only. _rule_patch_pass uses score_batch_sair
  internally (SAIR-style rendering) to evaluate patches, which requires the
  scoring model to consume the NeuriCo jinja2 template directly.  For generic
  tasks (BBH, etc.), set initial_rule_set=None to skip Phase 1 entirely.

Optional ablation (both default OFF):
  run_initial_ablation  — ablate the initial RuleSet before Phase 1.
  run_midpoint_ablation — ablate the refined RuleSet on residual failures
                          between Phase 1 and Phase 2.

Phase 2  (case study generation, always runs):
  run_partition_loop() on the full training set with the final cheatsheet
  (whose prior_knowledge has been updated to the refined rule text by Phase 1).
  Generates structural case studies to patch failure modes that rules couldn't fix.

Bridge:
  _rule_patch_pass already sets cheatsheet.prior_knowledge = rule_set.render_decision_guide()
  after each accepted patch, so the ICR scorer always sees the current rules.
  Phase 2 inherits this state automatically — no extra conversion step needed.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from utils.cheatsheet import Cheatsheet
from utils.scorer import score_batch
from utils.task_spec import TaskSpec
from ICR_reasoning.core.oracle import OracleDict
from ICR_partition.training.partition import (
    build_partitions, refresh_partitions,
    print_partition_table, partition_summary,
)
from ICR_partition.training.loop import (
    _rule_patch_pass,
    _save_checkpoint,
    run_partition_loop,
    PartitionTrainingResult,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HybridLoopResult:
    # Phase 1 — rule patching
    rule_set:             object          # RuleSet | None  (typed as object to avoid hard import)
    n_rule_patches:       int
    n_outer_iters_rule:   int
    accuracy_after_rules: float

    # Phase 2 — case study generation
    cheatsheet:           Cheatsheet
    n_case_studies_added: int
    n_merges:             int
    n_outer_iters_cs:     int
    accuracy_final:       float

    # Ablation reports (empty dicts when not run)
    ablation_initial:     dict           # rule_id → AblationResult.__dict__
    ablation_midpoint:    dict

    # Diagnostics
    update_log:           list[dict]
    partition_summary_rule: list[dict]   # partition state at end of Phase 1
    partition_summary_cs:   list[dict]   # partition state at end of Phase 2


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_hybrid_loop(
    # ── Data ──────────────────────────────────────────────────────────────────
    train_items:                list[dict],
    val_items:                  list[dict] | None,

    # ── Starting state ────────────────────────────────────────────────────────
    # Provide initial_rule_set to enable Phase 1 (rule patching).
    # Provide initial_cheatsheet to start Phase 2 from a non-empty cheatsheet.
    # Both default to None (start from scratch, skip Phase 1).
    initial_rule_set:           object | None,      # RuleSet | None
    initial_cheatsheet:         Cheatsheet | None,

    # ── Models ────────────────────────────────────────────────────────────────
    model_score:                str,
    model_rule_patch:           str,       # model for rule patch generation (Phase 1)
    model_casestudy:            str,       # model for case study generation (Phase 2)
    api_key:                    str,

    # ── Oracle ────────────────────────────────────────────────────────────────
    oracle:                     OracleDict | None = None,
    oracle_min_similarity:      float = 0.25,

    # ── Phase 1 parameters ────────────────────────────────────────────────────
    max_rule_iters:             int   = 4,
    rule_acc_goal:              float = 0.85,   # exit Phase 1 early if acc >= this
    rule_static_iters:          int   = 2,      # exit if no patches for this many iters
    rule_bin_threshold:         int   = 3,
    rule_fix_rate_threshold:    float = 0.20,
    rule_regress_threshold:     float = 0.20,
    rule_concurrency:           int   = 25,
    rule_partition_concurrency: int   = 8,

    # ── Ablation options (both default OFF) ───────────────────────────────────
    run_initial_ablation:       bool  = False,  # ablate initial RuleSet before Phase 1
    run_midpoint_ablation:      bool  = False,  # ablate refined RuleSet between phases

    # ── Phase 2 parameters ────────────────────────────────────────────────────
    max_cs_iters:               int   = 5,
    cs_static_iters:            int   = 2,   # stop after this many consecutive idle CS iters
    cs_bin_threshold:           int   = 3,
    cs_retirement_threshold:    int   = 2,
    cs_fix_rate_threshold:      float = 0.30,
    cs_regress_threshold:       float = 0.15,
    cs_n_candidates:            int   = 3,
    cs_candidate_rounds:        int   = 3,
    cs_similarity_gate:         bool  = True,
    cs_concurrency:             int   = 25,
    cs_partition_concurrency:   int   = 8,
    cs_min_pool_for_regression: int   = 5,

    # ── Shared scoring ────────────────────────────────────────────────────────
    reasoning_effort:           str | None = "low",
    cot_first:                  bool = True,
    task_spec:                  TaskSpec | None = None,
    prescore_map:               dict | None = None,

    # ── Rule bootstrap ────────────────────────────────────────────────────────
    # When True and initial_rule_set is None, score a sample of train_items,
    # pass the failures to task_spec.bootstrap_ruleset to generate a seed RuleSet,
    # then proceed with Phase 1 rule patching as normal.
    auto_rule_init:             bool = False,
    n_bootstrap_failures:       int  = 20,   # how many failures to seed the bootstrap

    # ── Phase 2 regression guard ──────────────────────────────────────────────
    pk_regression_guard:        bool  = False,
    pk_regression_tolerance:    float = 0.03,

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir:                 Path | None = None,
    log:                        bool = True,
) -> HybridLoopResult:

    def _log(msg: str) -> None:
        if log:
            print(msg, file=sys.stderr, flush=True)

    # ── Resolve task_spec ─────────────────────────────────────────────────────
    if task_spec is None:
        from tasks.magma import MAGMA_TASK
        task_spec = MAGMA_TASK

    # ── Initialise state ──────────────────────────────────────────────────────
    cheatsheet = (
        Cheatsheet(
            roadmap=initial_cheatsheet.roadmap,
            case_studies=list(initial_cheatsheet.case_studies),
            prior_knowledge=initial_cheatsheet.prior_knowledge,
            no_limit=getattr(initial_cheatsheet, "no_limit", False),
        )
        if initial_cheatsheet is not None
        else Cheatsheet(roadmap="", case_studies=[])
    )
    rule_set = initial_rule_set

    update_log:           list[dict] = []
    ablation_initial:     dict = {}
    ablation_midpoint:    dict = {}
    n_rule_patches        = 0
    rule_iters_done       = 0
    accuracy_after_rules  = 0.0
    rule_ps:              list[dict] = []
    bins:                 dict = {}

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ICR-style scorer shared by both phases
    def _do_score(items, cs_text, *, concurrency, progress_label="scoring"):
        return score_batch(
            items, cs_text, model_score, api_key,
            concurrency=concurrency,
            reasoning_effort=reasoning_effort,
            cot_first=cot_first,
            progress_label=progress_label,
            task_spec=task_spec,
        )

    _log(
        f"\n{'='*65}\n"
        f"ICR_hybrid Training Loop\n"
        f"  items         : {len(train_items)}\n"
        f"  Phase 1       : {'rule-patch (' + str(max_rule_iters) + ' iters, goal=' + f'{rule_acc_goal:.0%}' + ')' if rule_set is not None else 'SKIPPED (no initial_rule_set)'}\n"
        f"  Phase 2       : case-study ({max_cs_iters} iters)\n"
        f"  ablation init : {'yes' if run_initial_ablation else 'no'}\n"
        f"  ablation mid  : {'yes' if run_midpoint_ablation else 'no'}\n"
        f"  model_score   : {model_score}\n"
        f"  model_patch   : {model_rule_patch}\n"
        f"  model_cs      : {model_casestudy}\n"
        f"{'='*65}"
    )

    # ── Rule bootstrap ────────────────────────────────────────────────────────
    # When auto_rule_init=True and no rule set was provided, score a sample of
    # training items to get failures, then call task_spec.bootstrap_ruleset to
    # generate a seed RuleSet from those failures via LLM.  Phase 1 then refines
    # that seed through the normal rule-patch loop.
    if auto_rule_init and rule_set is None:
        if task_spec.bootstrap_cheatsheet_fn is not None:
            # Concrete-example bootstrap: generates CS-ICL-style named scenarios and
            # populates cheatsheet.prior_knowledge directly.  Phase 1 is skipped.
            _log("\n[bootstrap] Scoring sample to collect seed failures (concrete-example bootstrap) ...")
            sample = train_items[:min(60, len(train_items))]
            _, sample_wrong = _do_score(
                sample, cheatsheet.render(),
                concurrency=min(rule_concurrency, len(sample)),
                progress_label="bootstrap-score",
            )
            seed_failures = sample_wrong[:n_bootstrap_failures]
            if not seed_failures:
                _log("  [bootstrap] no failures in sample — concrete bootstrap skipped.")
            else:
                _log(
                    f"  [bootstrap] {len(seed_failures)} failures → "
                    f"calling bootstrap_cheatsheet_fn ..."
                )
                text = task_spec.bootstrap_cheatsheet_fn(seed_failures, model_rule_patch, api_key)
                cheatsheet.prior_knowledge = text
                _log(f"  [bootstrap] prior_knowledge set ({len(text)} chars). Phase 1 will be skipped.")
                if output_dir:
                    (Path(output_dir) / "ruleset_bootstrap.txt").write_text(text, encoding="utf-8")
        elif task_spec.bootstrap_ruleset is None:
            _log(
                "\n[bootstrap] auto_rule_init=True but task_spec has no bootstrap_ruleset — "
                "Phase 1 will be skipped."
            )
        else:
            _log("\n[bootstrap] Scoring sample to collect seed failures ...")
            sample = train_items[:min(60, len(train_items))]
            _, sample_wrong = _do_score(
                sample, cheatsheet.render(),
                concurrency=min(rule_concurrency, len(sample)),
                progress_label="bootstrap-score",
            )
            seed_failures = sample_wrong[:n_bootstrap_failures]
            if not seed_failures:
                _log("  [bootstrap] no failures in sample — Phase 1 will be skipped.")
            else:
                _log(
                    f"  [bootstrap] {len(seed_failures)} failures → "
                    f"calling LLM to generate initial rule set ..."
                )
                rule_set = task_spec.bootstrap_ruleset(
                    seed_failures, model_rule_patch, api_key
                )
                _log(f"  [bootstrap] {rule_set.summary()}")
                if output_dir:
                    (Path(output_dir) / "ruleset_bootstrap.txt").write_text(
                        rule_set.render(), encoding="utf-8"
                    )

    # ── Optional initial ablation ─────────────────────────────────────────────
    if run_initial_ablation and rule_set is not None:
        _log("\n[hybrid] Running initial ablation pre-pass ...")
        from ICR_rules.training.scorer import score_batch_sair
        from ICR_rules.training.ablation import run_ablation, print_ablation_report
        bc, bw = score_batch_sair(
            train_items, rule_set, model_score, api_key,
            concurrency=rule_concurrency,
            task_spec=task_spec,
        )
        ablation_initial_results = run_ablation(
            rule_set, train_items, model_score, api_key,
            concurrency=rule_concurrency,
            baseline_correct=bc, baseline_wrong=bw,
            task_spec=task_spec,
        )
        print_ablation_report(ablation_initial_results)
        ablation_initial = {
            k: {"rule_id": v.rule_id, "accuracy_baseline": v.accuracy_baseline,
                "accuracy_without": v.accuracy_without, "delta": v.delta}
            for k, v in ablation_initial_results.items()
        }
        if output_dir:
            (Path(output_dir) / "ablation_initial.json").write_text(
                json.dumps(ablation_initial, indent=2), encoding="utf-8"
            )

    # =========================================================================
    # PHASE 1 — Rule patching
    # =========================================================================
    if rule_set is not None:
        _log("\n[hybrid] ══════════════════ PHASE 1: Rule Patching ══════════════════")

        # Seed cheatsheet.prior_knowledge from the initial rule set so the
        # ICR scorer already sees the rules before any patch is applied.
        cheatsheet.prior_knowledge = rule_set.render_decision_guide()

        # Initial score
        _log(f"\n[Phase 1 / iter 0] Scoring all {len(train_items)} items ...")
        if prescore_map is not None:
            from ICR_select.training.gates import _apply_prescore
            correct, wrong = _apply_prescore(train_items, prescore_map)
            _log(f"  [prescore] {len(correct)} correct  {len(wrong)} wrong")
        else:
            correct, wrong = _do_score(
                train_items, cheatsheet.render(),
                concurrency=rule_concurrency,
                progress_label="phase1-init",
            )
        accuracy_after_rules = len(correct) / len(train_items) if train_items else 0.0
        _log(f"  initial accuracy = {accuracy_after_rules:.1%}")

        bins = build_partitions(
            wrong, correct, rule_bin_threshold,
            partition_key_fn=task_spec.partition_key,
        )
        print_partition_table(bins, title="PHASE 1 INITIAL PARTITIONS")
        update_log.append({
            "event":    "phase1_initial_score",
            "phase":    1,
            "accuracy": round(accuracy_after_rules, 4),
            "n_correct": len(correct),
            "n_wrong":   len(wrong),
            "n_bins":    len(bins),
        })
        if output_dir:
            _save_checkpoint(cheatsheet, update_log, output_dir, tag="phase1_init")

        static_rule_iters = 0

        for rule_iter in range(1, max_rule_iters + 1):
            _log(
                f"\n[Phase 1 / iter {rule_iter}/{max_rule_iters}]  "
                f"acc={accuracy_after_rules:.1%}  goal={rule_acc_goal:.1%}"
            )

            if accuracy_after_rules >= rule_acc_goal:
                _log(f"  acc_goal reached — exiting Phase 1 after iter {rule_iter - 1}")
                break

            active_bins = {k: pb for k, pb in bins.items() if not pb.solved}
            if not active_bins:
                _log("  All bins retired — exiting Phase 1")
                break

            # ── Rule-patch pass ───────────────────────────────────────────────
            _log(f"  Running rule-patch on {len(active_bins)} active bin(s) ...")
            rule_set, rp_log, n_applied = _rule_patch_pass(
                active_bins=active_bins,
                rule_set=rule_set,
                cheatsheet=cheatsheet,
                model_patch=model_rule_patch,
                model_score=model_score,
                api_key=api_key,
                oracle=oracle,
                fix_rate_threshold=rule_fix_rate_threshold,
                regress_threshold=rule_regress_threshold,
                concurrency=rule_concurrency,
                partition_concurrency=rule_partition_concurrency,
                log_fn=_log,
                task_spec=task_spec,
            )
            n_rule_patches += n_applied
            update_log.extend(
                [{**e, "phase": 1, "rule_iter": rule_iter} for e in rp_log]
            )

            if n_applied == 0:
                static_rule_iters += 1
                _log(
                    f"  No patches accepted this iter "
                    f"({static_rule_iters}/{rule_static_iters} static)"
                )
                if static_rule_iters >= rule_static_iters:
                    _log(f"  max_static_iters reached — exiting Phase 1")
                    break
            else:
                static_rule_iters = 0
                _log(f"  {n_applied} patch(es) applied")

            # ── Re-score active failures ──────────────────────────────────────
            active_failures = [
                item for pb in active_bins.values() for item in pb.failures
            ]
            if not active_failures:
                _log("  No active failures remain — exiting Phase 1")
                rule_iters_done = rule_iter
                break

            _log(f"  Re-scoring {len(active_failures)} active-bin items ...")
            new_correct, new_wrong = _do_score(
                active_failures, cheatsheet.render(),
                concurrency=rule_concurrency,
                progress_label=f"phase1-iter{rule_iter}-rescore",
            )
            refresh_partitions(
                bins, new_wrong, new_correct,
                retirement_threshold=rule_bin_threshold,
                partition_key_fn=task_spec.partition_key,
            )

            rule_iters_done = rule_iter
            if output_dir:
                _save_checkpoint(cheatsheet, update_log, output_dir,
                                 tag=f"phase1_iter{rule_iter:02d}")

        # Final Phase 1 accuracy (full rescore)
        _log(f"\n[Phase 1] Complete — rescoring full training set ...")
        correct, wrong = _do_score(
            train_items, cheatsheet.render(),
            concurrency=rule_concurrency,
            progress_label="phase1-final",
        )
        accuracy_after_rules = len(correct) / len(train_items) if train_items else 0.0
        rule_ps = partition_summary(bins)

        _log(
            f"  Phase 1 accuracy = {accuracy_after_rules:.1%}  "
            f"patches = {n_rule_patches}  iters = {rule_iters_done}"
        )
        update_log.append({
            "event":          "phase1_complete",
            "phase":          1,
            "accuracy":       round(accuracy_after_rules, 4),
            "n_rule_patches": n_rule_patches,
            "n_iters":        rule_iters_done,
        })
        if output_dir:
            _save_checkpoint(cheatsheet, update_log, output_dir, tag="phase1_final")

    # ── Optional mid-point ablation ───────────────────────────────────────────
    if run_midpoint_ablation and rule_set is not None:
        residual = [it for pb in bins.values() for it in pb.failures]
        if residual:
            _log(
                f"\n[hybrid] Running mid-point ablation on {len(residual)} "
                f"residual failures ..."
            )
            from ICR_rules.training.scorer import score_batch_sair
            from ICR_rules.training.ablation import run_ablation, print_ablation_report
            ablation_mid_results = run_ablation(
                rule_set, residual, model_score, api_key,
                concurrency=rule_concurrency,
            )
            print_ablation_report(ablation_mid_results)
            ablation_midpoint = {
                k: {"rule_id": v.rule_id, "accuracy_baseline": v.accuracy_baseline,
                    "accuracy_without": v.accuracy_without, "delta": v.delta}
                for k, v in ablation_mid_results.items()
            }
            if output_dir:
                (Path(output_dir) / "ablation_midpoint.json").write_text(
                    json.dumps(ablation_midpoint, indent=2), encoding="utf-8"
                )
        else:
            _log("\n[hybrid] Mid-point ablation skipped — no residual failures.")

    # =========================================================================
    # PHASE 2 — Case study generation
    # =========================================================================
    _log("\n[hybrid] ══════════════════ PHASE 2: Case Study Generation ══════════════════")

    cs_output_dir = Path(output_dir) / "phase2" if output_dir else None

    cs_result: PartitionTrainingResult = run_partition_loop(
        cheatsheet=cheatsheet,
        train_items=train_items,
        val_items=val_items,
        model_score=model_score,
        model_casestudy=model_casestudy,
        api_key=api_key,
        oracle=oracle,
        oracle_min_similarity=oracle_min_similarity,
        bin_threshold=cs_bin_threshold,
        retirement_threshold=cs_retirement_threshold,
        max_outer_iters=max_cs_iters,
        cs_static_iters=cs_static_iters,
        partition_concurrency=cs_partition_concurrency,
        concurrency=cs_concurrency,
        n_candidates=cs_n_candidates,
        candidate_rounds=cs_candidate_rounds,
        fix_rate_threshold=cs_fix_rate_threshold,
        regress_threshold=cs_regress_threshold,
        min_pool_for_regression=cs_min_pool_for_regression,
        similarity_gate=cs_similarity_gate,
        reasoning_effort=reasoning_effort,
        cot_first=cot_first,
        task_spec=task_spec,
        output_dir=cs_output_dir,
        log=log,
        pk_regression_guard=pk_regression_guard,
        pk_regression_tolerance=pk_regression_tolerance,
    )

    update_log.extend([{**e, "phase": 2} for e in cs_result.update_log])

    _log(
        f"\n[hybrid] ══════════════════ COMPLETE ══════════════════\n"
        f"  Phase 1: {n_rule_patches} rule patches  acc={accuracy_after_rules:.1%}\n"
        f"  Phase 2: {cs_result.n_case_studies_added} case studies  "
        f"acc={cs_result.train_accuracy:.1%}\n"
        f"{'='*65}"
    )

    return HybridLoopResult(
        rule_set=rule_set,
        n_rule_patches=n_rule_patches,
        n_outer_iters_rule=rule_iters_done,
        accuracy_after_rules=accuracy_after_rules,
        cheatsheet=cs_result.cheatsheet,
        n_case_studies_added=cs_result.n_case_studies_added,
        n_merges=cs_result.n_merges,
        n_outer_iters_cs=cs_result.n_outer_iters,
        accuracy_final=cs_result.train_accuracy,
        ablation_initial=ablation_initial,
        ablation_midpoint=ablation_midpoint,
        update_log=update_log,
        partition_summary_rule=rule_ps,
        partition_summary_cs=cs_result.partition_summary,
    )
