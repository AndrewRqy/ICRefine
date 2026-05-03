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

  NOTE: _rule_patch_pass (RuleSet mode) is magma-only — it requires the SAIR
  jinja2 template.  For generic tasks (BBH, etc.) there are two Phase 1 modes:
    • _pk_patch_phase (text mode): activated when cheatsheet.prior_knowledge is
      non-empty but no RuleSet is provided.  Iterates on the PK text directly,
      feeding failure post_think traces to the model to produce improved text.
    • SKIPPED: when no RuleSet and no prior_knowledge exist.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from utils.cheatsheet import Cheatsheet
from utils.llm_client import call_llm
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
# Phase 1 alternative: iterative prior_knowledge text patching
# ---------------------------------------------------------------------------

def _pk_patch_phase(
    cheatsheet:          Cheatsheet,
    train_items:         list[dict],
    model_patch:         str,
    model_score:         str,
    api_key:             str,
    oracle:              "OracleDict | None",
    max_iters:           int,
    acc_goal:            float,
    static_iters:        int,
    fix_rate_threshold:  float,
    regress_threshold:   float,
    concurrency:         int,
    log_fn,
    task_spec:           "TaskSpec",
    reasoning_effort:    "str | None",
    cot_first:           bool,
    n_failures_to_show:  int = 15,
    inject_oracle:       bool = True,
    max_pk_chars:        int | None = None,
) -> tuple[int, float, int]:
    """
    Iteratively improve cheatsheet.prior_knowledge using failure post_think traces.

    Each iteration:
      1. Sample up to n_failures_to_show failures (with wrong CoT from post_think).
      2. Call model_patch to produce an improved version of the PK text.
      3. Gate: re-score failures → fix_rate; re-score correct sample → regression.
      4. Accept if both gates pass; full-rescore for next iter.

    Returns (n_patches_applied, final_accuracy, iters_done).
    """
    def _do_score(items, cs_text, label):
        return score_batch(
            items, cs_text, model_score, api_key,
            concurrency=min(concurrency, len(items)),
            reasoning_effort=reasoning_effort,
            cot_first=cot_first,
            progress_label=label,
            task_spec=task_spec,
        )

    n_patches = 0
    static_count = 0
    iters_done = 0

    log_fn(f"\n[pk_patch] Scoring {len(train_items)} items for Phase 1 PK patching ...")
    correct, wrong = _do_score(train_items, cheatsheet.render(), "pk-init")
    final_acc = len(correct) / len(train_items) if train_items else 0.0
    log_fn(f"  initial accuracy = {final_acc:.1%}  ({len(wrong)} failures, {len(correct)} correct)")

    for pk_iter in range(1, max_iters + 1):
        iters_done = pk_iter
        if final_acc >= acc_goal:
            log_fn(f"  [pk_patch] acc_goal {acc_goal:.0%} reached after iter {pk_iter - 1}")
            break
        if not wrong:
            log_fn("  [pk_patch] no failures remain")
            break

        log_fn(
            f"\n[pk_patch / iter {pk_iter}/{max_iters}]  "
            f"acc={final_acc:.1%}  failures={len(wrong)}"
        )

        shown = wrong[:n_failures_to_show]
        pk_text = cheatsheet.prior_knowledge.strip()

        # Build failure block (input + expected/got + post_think + oracle if available)
        failure_blocks = []
        for idx, item in enumerate(shown, 1):
            inp = str(item.get("input", "")).strip()[:300]
            exp = str(item.get("expected", item.get("answer", "?"))).strip()
            got = str(item.get("predicted", "?"))
            pt  = (item.get("post_think") or item.get("thinking") or "").strip()

            lines = [
                f"[{idx}]",
                f"  Input:    {inp}",
                f"  Expected: {exp}  |  Got: {got}",
            ]
            if pt:
                lines.append(f"  Wrong reasoning:\n    {pt[:500]}")

            # Oracle contrast — show correct reasoning so the patcher knows what good looks like.
            # Priority: pre-baked _oracle_exact → item gold reason → external oracle dict.
            # Controlled by inject_oracle; set False to ablate this signal.
            if inject_oracle:
                oracle_text = (
                    item.get("_oracle_exact", "")
                    or item.get("reason", "")
                    or item.get("gold_reason", "")
                )
                if not oracle_text and oracle is not None:
                    oracle_text = (oracle.get(str(item.get("id", ""))) or {}).get("explanation", "")
                if oracle_text:
                    lines.append(f"  Correct reasoning:\n    {oracle_text[:600]}")

            failure_blocks.append("\n".join(lines))

        _size_instruction = (
            f"HARD LIMIT: the improved guide must be at most {max_pk_chars:,} characters "
            f"(current: {len(pk_text):,} chars). Trim or remove existing rules if needed to stay under this limit."
            if max_pk_chars is not None
            else "Keep the improved guide focused and concise (no more than 50% longer than the original)"
        )
        prompt = (
            f"You are refining a knowledge guide that helps a model answer questions correctly.\n\n"
            f"=== CURRENT KNOWLEDGE GUIDE ({len(pk_text)} chars) ===\n"
            f"{pk_text}\n"
            f"=== END KNOWLEDGE GUIDE ===\n\n"
            f"The model is making the following {len(shown)} errors "
            f"(out of {len(wrong)} total failures):\n\n"
            + "\n\n".join(failure_blocks)
            + "\n\n---\n"
            "Produce an IMPROVED version of the knowledge guide that helps avoid these mistakes.\n"
            "You may:\n"
            "  - ADD a new rule, clarification, or concrete example\n"
            "  - MODIFY an existing rule to be more precise or correct\n"
            "  - REMOVE a rule that is actively causing errors\n\n"
            f"Requirements:\n"
            f"  - {_size_instruction}\n"
            "  - Preserve rules that are working correctly\n"
            "  - Focus on ABSTRACT REASONING PRINCIPLES: do not reproduce specific input text, "
            "entity names, or numbers from the failure examples above\n"
            "  - Write principles that would help any capable model on new similar questions, "
            "not just correct these specific items\n"
            "  - Output ONLY the improved knowledge guide text — "
            "no preamble, no commentary, no markdown fences"
        )

        log_fn(f"  Calling {model_patch} to patch prior_knowledge ({len(pk_text)} chars) ...")
        # Cap max_tokens to ~1.3 chars/token headroom above the size limit, or default heuristic.
        if max_pk_chars is not None:
            max_tok = min(4000, max(400, int(max_pk_chars / 3)))
        else:
            max_tok = min(4000, max(800, int(len(pk_text) * 1.6)))
        resp = call_llm(prompt, model=model_patch, api_key=api_key,
                        max_tokens=max_tok, temperature=0.3)
        candidate_pk = resp.content.strip()

        if not candidate_pk:
            log_fn("  [pk_patch] empty response — skipping")
            static_count += 1
            if static_count >= static_iters:
                log_fn("  max_static_iters reached — exiting pk_patch")
                break
            continue

        log_fn(f"  Candidate PK: {len(candidate_pk)} chars")

        if max_pk_chars is not None and len(candidate_pk) > max_pk_chars:
            log_fn(f"  [pk_size_cap] {len(candidate_pk):,} chars > {max_pk_chars:,} — rejecting oversized candidate")
            static_count += 1
            if static_count >= static_iters:
                log_fn("  max_static_iters reached — exiting pk_patch")
                break
            continue

        cand = Cheatsheet(
            roadmap=cheatsheet.roadmap,
            case_studies=list(cheatsheet.case_studies),
            prior_knowledge=candidate_pk,
            no_limit=getattr(cheatsheet, "no_limit", False),
        )

        # Fix-rate gate — re-score the shown failures with candidate PK
        log_fn(f"  Fix-rate gate: scoring {len(shown)} failures ...")
        new_correct_from_wrong, _ = _do_score(shown, cand.render(), "pk-fix-gate")
        fix_rate = len(new_correct_from_wrong) / len(shown)
        log_fn(f"  fix_rate = {fix_rate:.1%}  (threshold {fix_rate_threshold:.1%})")

        if fix_rate < fix_rate_threshold:
            log_fn("  fix_rate below threshold — rejecting candidate")
            static_count += 1
            if static_count >= static_iters:
                log_fn("  max_static_iters reached — exiting pk_patch")
                break
            continue

        # Regression gate — re-score a sample of the currently-correct items
        regression = 0.0
        regress_sample = correct[:min(40, len(correct))]
        if regress_sample:
            log_fn(f"  Regression gate: scoring {len(regress_sample)} correct items ...")
            still_correct, regressed = _do_score(
                regress_sample, cand.render(), "pk-regress-gate"
            )
            regression = len(regressed) / len(regress_sample)
            log_fn(f"  regression = {regression:.1%}  (threshold {regress_threshold:.1%})")
            if regression > regress_threshold:
                log_fn("  regression too high — rejecting candidate")
                static_count += 1
                if static_count >= static_iters:
                    log_fn("  max_static_iters reached — exiting pk_patch")
                    break
                continue

        # Accept
        cheatsheet.prior_knowledge = candidate_pk
        n_patches += 1
        static_count = 0
        log_fn(
            f"  [pk_patch] ACCEPTED patch #{n_patches}  "
            f"fix={fix_rate:.1%}  regress={regression:.1%}  "
            f"PK={len(candidate_pk)} chars"
        )

        # Full rescore after each accepted patch for accurate failure list next iter
        log_fn("  Full rescore after acceptance ...")
        correct, wrong = _do_score(
            train_items, cheatsheet.render(), f"pk-iter{pk_iter}-rescore"
        )
        final_acc = len(correct) / len(train_items) if train_items else 0.0
        log_fn(f"  new accuracy = {final_acc:.1%}  ({len(wrong)} failures remain)")

    log_fn(
        f"\n[pk_patch] Complete — {n_patches} patch(es) applied  "
        f"iters={iters_done}  final_acc={final_acc:.1%}"
    )
    return n_patches, final_acc, iters_done


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

    # ── CS-ICL auto-bootstrap ─────────────────────────────────────────────────
    # When True (default) and no initial cheatsheet / prior_knowledge was
    # provided, generate a CS-ICL style prior_knowledge from the first
    # bootstrap_n_items training examples before Phase 1 runs.
    # This ensures _pk_patch_phase always has a non-trivial starting point
    # regardless of whether the task defines a bootstrap_ruleset.
    auto_bootstrap:             bool = True,
    bootstrap_n_items:          int  = 75,

    # ── Phase 2 regression guard ──────────────────────────────────────────────
    pk_regression_guard:        bool  = False,
    pk_regression_tolerance:    float = 0.03,

    # ── Oracle injection per phase ────────────────────────────────────────────
    # Phase 2 oracle injection (item["reason"] as correct-reasoning contrast in
    # case study generation) was present from the beginning.  Phase 1 oracle
    # injection was added in v5.  Both default True; set False to ablate.
    phase1_inject_oracle:       bool = True,
    phase2_inject_oracle:       bool = True,

    # ── Size ablation ─────────────────────────────────────────────────────────
    # max_pk_chars: reject Phase 1 PK candidates that exceed this char count.
    # max_case_studies: stop Phase 2 once this many CS have been added.
    # Both default to None (unlimited).
    max_pk_chars:               int | None = None,
    max_case_studies:           int | None = None,

    # ── EA Phase 1 ────────────────────────────────────────────────────────────
    # When True, replaces _pk_patch_phase with the evolutionary algorithm.
    # EA params map directly to ea_pk_phase(); max_rule_iters becomes
    # max_generations and rule_static_iters becomes static_gens_limit.
    use_ea:                     bool  = False,
    ea_population_size:         int   = 3,
    ea_n_survivors:             int   = 2,
    ea_lambda_min:              float = 1.0,
    ea_lambda_max:              float = 2.0,
    ea_regress_hard_cap:        int   = 3,
    ea_pk_size_budget:          int   = 12_000,
    ea_val_fraction:            float = 0.20,
    ea_failure_sample_frac:     float = 0.60,

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
        f"  Phase 1       : "
        f"{'rule-patch (' + str(max_rule_iters) + ' iters, goal=' + f'{rule_acc_goal:.0%}' + ')' if rule_set is not None else 'pk-text-patch (' + str(max_rule_iters) + ' iters)' if (initial_cheatsheet is not None and initial_cheatsheet.prior_knowledge.strip()) or auto_rule_init else 'SKIPPED (no rule set or prior knowledge)'}\n"
        f"  auto-bootstrap: {'yes (' + str(bootstrap_n_items) + ' items)' if auto_bootstrap else 'no'}\n"
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
            # When --init-cheatsheet already provides prior_knowledge, skip entirely —
            # auto_rule_init would overwrite the bootstrap content with a lower-quality
            # replacement generated from a different failure distribution.
            if initial_cheatsheet is not None and cheatsheet.prior_knowledge.strip():
                _log(
                    "\n[bootstrap] --init-cheatsheet provides prior_knowledge — "
                    "skipping auto_rule_init to preserve bootstrap content."
                )
            else:
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
            # When --init-cheatsheet already provides prior_knowledge (e.g. from a
            # CS-ICL bootstrap run), skip the ruleset bootstrap entirely and let
            # _pk_patch_phase iteratively improve the existing text in Phase 1.
            # Generating a RuleSet here would overwrite cheatsheet.prior_knowledge with
            # auto-generated rules, discarding the higher-quality bootstrap content.
            if initial_cheatsheet is not None and cheatsheet.prior_knowledge.strip():
                _log(
                    "\n[bootstrap] --init-cheatsheet provides prior_knowledge — "
                    "skipping bootstrap_ruleset. _pk_patch_phase will refine it in Phase 1."
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

    # ── CS-ICL auto-bootstrap ─────────────────────────────────────────────────
    # Fires when no init_cheatsheet / prior_knowledge was supplied AND neither
    # the concrete bootstrap nor the ruleset bootstrap produced prior_knowledge.
    # Mirrors CS-ICL's two-step pipeline: first enrich items with LLM-generated
    # reasoning (if not already present), then summarise into a prior_knowledge
    # cheat sheet so Phase 1 always has a non-trivial start.
    if auto_bootstrap and rule_set is None and not cheatsheet.prior_knowledge.strip():
        _log(
            f"\n[auto-bootstrap] No prior knowledge — generating CS-ICL style "
            f"cheatsheet from first {bootstrap_n_items} train items ..."
        )
        sample = list(train_items[:bootstrap_n_items])

        # Step 1 — generate reasoning for items that lack it, mirroring
        # CS-ICL's generate_reason_api.py (zero-shot explanation prompt).
        needs_reason = [
            (i, it) for i, it in enumerate(sample)
            if not (it.get("reason") or it.get("gold_reason"))
        ]
        if needs_reason:
            _log(
                f"  [auto-bootstrap] {len(needs_reason)} of {len(sample)} items "
                f"lack reasoning — generating via LLM ..."
            )

            def _gen_reason(idx: int, item: dict) -> tuple[int, str]:
                prompt = (
                    "Given the following question and its correct answer, "
                    "provide a clear step-by-step explanation of how to arrive at the answer.\n\n"
                    f"Question: {item.get('input', '')}\n"
                    f"Answer: {item.get('answer', '')}\n"
                    "Explanation:"
                )
                r = call_llm(
                    prompt, model_rule_patch, api_key,
                    temperature=0.0, max_tokens=500, reasoning_effort=None,
                )
                return idx, r.content.strip()

            reason_map: dict[int, str] = {}
            with ThreadPoolExecutor(max_workers=min(20, len(needs_reason))) as ex:
                futs = {ex.submit(_gen_reason, i, it): i for i, it in needs_reason}
                for fut in as_completed(futs):
                    idx, reason = fut.result()
                    reason_map[idx] = reason

            # Shallow-copy items so train_items is not mutated
            sample = [
                {**it, "reason": reason_map[i]} if i in reason_map else it
                for i, it in enumerate(sample)
            ]
            _log(f"  [auto-bootstrap] reasoning generated for {len(reason_map)} items.")

        # Step 2 — format items with reasoning when available, then summarise
        dataset_str = "\n\n".join(
            (
                f"Question: {it.get('input', '')}\n"
                f"Reasoning: {it.get('reason') or it.get('gold_reason')}\n"
                f"Answer: {it.get('answer', '')}"
            )
            if (it.get("reason") or it.get("gold_reason"))
            else f"Question: {it.get('input', '')}\nAnswer: {it.get('answer', '')}"
            for it in sample
        )
        _size_clause = (
            f"\n\nIMPORTANT: Your cheat sheet MUST be at most {max_pk_chars:,} characters "
            f"(including spaces). Be concise — prioritise the most critical points only."
            if max_pk_chars is not None
            else ""
        )
        bootstrap_prompt = (
            "Create a cheat sheet based on the examples below. "
            "You will be asked to answer questions similar to these examples during the test, "
            "without being allowed to refer to the examples at that time. "
            "Your task here is to make a cheat sheet that will help you answer such problems correctly. "
            "First, carefully read the examples below and identify which ones you find most difficult to answer.\n\n"
            f"{dataset_str}\n\n"
            "Now, create a cheat sheet to help you solve the difficult examples. "
            "Exclude any content that is easy for you, and only include specific, detailed points "
            "to address the challenging ones."
            f"{_size_clause}\n\n"
        )
        _bootstrap_max_tok = (
            min(4000, max(400, int(max_pk_chars / 3))) if max_pk_chars is not None else 4000
        )
        _bootstrap_retries = 3
        pk_text = ""
        for _attempt in range(_bootstrap_retries):
            _retry_clause = (
                f"\n\nSTRICT LIMIT: your output must be ≤{max_pk_chars:,} characters. "
                f"Your previous attempt was {len(pk_text):,} chars — too long. "
                f"Cut aggressively: keep only the top rules, merge related points."
                if (_attempt > 0 and max_pk_chars is not None)
                else ""
            )
            resp = call_llm(
                bootstrap_prompt + _retry_clause, model_rule_patch, api_key,
                temperature=0.0, max_tokens=_bootstrap_max_tok, reasoning_effort=None,
            )
            pk_text = resp.content.strip()
            if max_pk_chars is None or len(pk_text) <= max_pk_chars:
                break
            _log(
                f"  [auto-bootstrap] attempt {_attempt + 1}: {len(pk_text):,} chars "
                f"> {max_pk_chars:,} — retrying ..."
            )
        else:
            # All retries exhausted — hard-truncate as last resort
            _log(
                f"  [auto-bootstrap] {_bootstrap_retries} retries exhausted — "
                f"hard-truncating to {max_pk_chars:,} chars."
            )
            pk_text = pk_text[:max_pk_chars]
        cheatsheet.prior_knowledge = pk_text
        _log(
            f"  [auto-bootstrap] prior_knowledge generated "
            f"({len(cheatsheet.prior_knowledge)} chars)"
        )
        if output_dir:
            (Path(output_dir) / "ruleset_bootstrap.txt").write_text(
                cheatsheet.prior_knowledge, encoding="utf-8"
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

    elif cheatsheet.prior_knowledge.strip():
        # Phase 1 alternative: iteratively improve prior_knowledge text directly.
        # Used when init-cheatsheet provides a CS-ICL bootstrap (no RuleSet available).
        _log("\n[hybrid] ══════════════════ PHASE 1: Prior-Knowledge Text Patching ══════════════════")
        if use_ea:
            from ICR_hybrid.training.ea_phase1 import ea_pk_phase
            _log("[hybrid] Using EA Phase 1 (evolutionary PK refinement)")
            n_rule_patches, accuracy_after_rules, rule_iters_done = ea_pk_phase(
                cheatsheet=cheatsheet,
                train_items=train_items,
                model_patch=model_rule_patch,
                model_score=model_score,
                api_key=api_key,
                oracle=oracle,
                max_generations=max_rule_iters,
                acc_goal=rule_acc_goal,
                static_gens_limit=rule_static_iters,
                concurrency=rule_concurrency,
                log_fn=_log,
                task_spec=task_spec,
                reasoning_effort=reasoning_effort,
                cot_first=cot_first,
                inject_oracle=phase1_inject_oracle,
                val_fraction=ea_val_fraction,
                population_size=ea_population_size,
                n_survivors=ea_n_survivors,
                lambda_min=ea_lambda_min,
                lambda_max=ea_lambda_max,
                regress_hard_cap=ea_regress_hard_cap,
                pk_size_budget=ea_pk_size_budget,
                failure_sample_frac=ea_failure_sample_frac,
            )
        else:
            n_rule_patches, accuracy_after_rules, rule_iters_done = _pk_patch_phase(
                cheatsheet=cheatsheet,
                train_items=train_items,
                model_patch=model_rule_patch,
                model_score=model_score,
                api_key=api_key,
                oracle=oracle,
                max_iters=max_rule_iters,
                acc_goal=rule_acc_goal,
                static_iters=rule_static_iters,
                fix_rate_threshold=rule_fix_rate_threshold,
                regress_threshold=rule_regress_threshold,
                concurrency=rule_concurrency,
                log_fn=_log,
                task_spec=task_spec,
                reasoning_effort=reasoning_effort,
                cot_first=cot_first,
                inject_oracle=phase1_inject_oracle,
                max_pk_chars=max_pk_chars,
            )
        update_log.append({
            "event":        "phase1_pk_patch_complete",
            "phase":        1,
            "accuracy":     round(accuracy_after_rules, 4),
            "n_pk_patches": n_rule_patches,
            "n_iters":      rule_iters_done,
        })
        if output_dir:
            _save_checkpoint(cheatsheet, update_log, output_dir, tag="phase1_pk_final")

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
        inject_gold_oracle=phase2_inject_oracle,
        max_case_studies=max_case_studies,
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
