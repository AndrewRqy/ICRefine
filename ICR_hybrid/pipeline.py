"""
ICR_hybrid/pipeline.py — Hybrid rule-patch → case-study refinement pipeline.

Phase 1 (rule patching) is enabled by providing --rule-set.
Phase 2 (case study generation) always runs.

Usage — magma with NeuriCo rule set:
    python -m ICR_hybrid.pipeline \\
        --task magma \\
        --dataset datasets/hard2.jsonl \\
        --oracle-csv gpt5.4_normal_default.csv \\
        --rule-set path/to/neurico.jinja2 \\
        --model-score deepseek-r1-32b \\
        --model-rule-patch openai/gpt-4o \\
        --model-casestudy openai/gpt-4o \\
        --rule-acc-goal 0.85 \\
        --max-rule-iters 4 \\
        --max-cs-iters 5 \\
        --output-dir runs/hybrid_run

Usage — BBH boolean (Phase 1 skipped, no rule set):
    python -m ICR_hybrid.pipeline \\
        --task bbh_boolean \\
        --dataset ../cheat-sheet-icl/data/aug_data/bbh_boolean_expressions.jsonl \\
        --no-oracle \\
        --model-score openai/gpt-4.1-2025-04-14 \\
        --model-casestudy openai/gpt-4.1-2025-04-14 \\
        --max-cs-iters 5 \\
        --output-dir runs/bbh_hybrid_run
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from utils.cheatsheet import Cheatsheet
from utils.data import load_jsonl
from utils.llm_client import get_api_key
from ICR_reasoning.core.oracle import load_oracle_csv
from .training.loop import run_hybrid_loop

load_dotenv(Path(__file__).parent.parent / "ICR_partition" / ".env")


_TASK_MAP = {
    "magma":                         ("tasks.magma",          "MAGMA_TASK"),
    "bbh_boolean":                   ("tasks.bbh_boolean",    "BBH_BOOLEAN_TASK"),
    "causal_judgement":              ("tasks.bbh_tasks",      "CAUSAL_JUDGEMENT_TASK"),
    "sports_understanding":          ("tasks.bbh_tasks",      "SPORTS_TASK"),
    "disambiguation_qa":             ("tasks.bbh_tasks",      "DISAMBIGUATION_TASK"),
    "movie_recommendation":          ("tasks.bbh_tasks",      "MOVIE_TASK"),
    "geometric_shapes":              ("tasks.bbh_tasks",      "GEOMETRIC_TASK"),
    "formal_fallacies":              ("tasks.bbh_tasks_ext",  "FORMAL_FALLACIES_TASK"),
    "logical_deduction_three":       ("tasks.bbh_tasks_ext",  "LOGICAL_DEDUCTION_3_TASK"),
    "web_of_lies":                   ("tasks.bbh_tasks_ext",  "WEB_OF_LIES_TASK"),
    "date_understanding":            ("tasks.bbh_tasks_ext",  "DATE_UNDERSTANDING_TASK"),
    "navigate":                      ("tasks.bbh_tasks_ext",  "NAVIGATE_TASK"),
    "snarks":                        ("tasks.bbh_tasks_ext",  "SNARKS_TASK"),
}


def _load_task(name: str):
    module_path, attr = _TASK_MAP[name]
    return getattr(importlib.import_module(module_path), attr)


def _load_rule_set(path: str):
    from ICR_rules.rules.parser import parse_cheatsheet_file
    return parse_cheatsheet_file(path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ICR_hybrid — rule-patch then case-study refinement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Task")
    g.add_argument("--task", default="magma", choices=list(_TASK_MAP),
                   help="Task domain.")

    g = p.add_argument_group("Data")
    g.add_argument("--dataset",       required=True, metavar="FILE")
    g.add_argument("--limit",         type=int, default=None, metavar="N",
                   help="Cap training items to first N.")
    g.add_argument("--prescore-file", default=None, metavar="FILE",
                   help="JSON of pre-computed scores (id → result) — skips initial scoring.")

    g = p.add_argument_group("Oracle")
    g.add_argument("--oracle-csv",            default=None, metavar="FILE")
    g.add_argument("--no-oracle",             action="store_true")
    g.add_argument("--oracle-min-similarity", type=float, default=0.25, metavar="F")

    g = p.add_argument_group("Starting state")
    g.add_argument("--rule-set",         default=None, metavar="FILE",
                   help="Jinja2 rule file.  Enables Phase 1 (rule patching). "
                        "If omitted, use --auto-rule-init to bootstrap from failures, "
                        "or omit both to skip Phase 1 entirely.")
    g.add_argument("--auto-rule-init",   action="store_true", default=False,
                   help="Bootstrap an initial rule set from scored failures before Phase 1. "
                        "Requires task_spec.bootstrap_ruleset to be implemented.")
    g.add_argument("--bootstrap-n",      type=int, default=20, metavar="N",
                   help="Number of failure examples used for rule bootstrap (default: 20).")
    g.add_argument("--init-cheatsheet", default=None, metavar="PATH",
                   help="Load a cheatsheet JSON as the starting point for Phase 2.")
    g.add_argument("--prior-knowledge", default=None, metavar="FILE",
                   help="Frozen knowledge prefix injected into every scoring call.")
    g.add_argument("--ablate-prior-segments", default=None, metavar="IDS",
                   help="Comma-separated segment IDs to disable from prior_knowledge_segments "
                        "(e.g. 'seg_0,seg_3'). Requires --init-cheatsheet with segments.")
    g.add_argument("--no-render-limit", action="store_true", default=False)

    g = p.add_argument_group("Models")
    g.add_argument("--model",            default="openai/gpt-4.1-2025-04-14", metavar="MODEL")
    g.add_argument("--model-score",      default=None, metavar="MODEL",
                   help="Scoring model (default: --model).")
    g.add_argument("--model-rule-patch", default=None, metavar="MODEL",
                   help="Rule-patch generation model (default: --model).")
    g.add_argument("--model-casestudy",  default=None, metavar="MODEL",
                   help="Case study generation model (default: --model).")

    g = p.add_argument_group("Phase 1 — Rule patching")
    g.add_argument("--max-rule-iters",          type=int,   default=4,    metavar="N")
    g.add_argument("--rule-acc-goal",           type=float, default=0.85, metavar="F",
                   help="Exit Phase 1 early when training accuracy reaches this.")
    g.add_argument("--rule-static-iters",       type=int,   default=2,    metavar="N",
                   help="Exit Phase 1 if no patch accepted for this many consecutive iters.")
    g.add_argument("--rule-bin-threshold",      type=int,   default=3,    metavar="N")
    g.add_argument("--rule-fix-rate-threshold", type=float, default=0.20, metavar="F")
    g.add_argument("--rule-regress-threshold",  type=float, default=0.20, metavar="F")
    g.add_argument("--rule-concurrency",        type=int,   default=25,   metavar="N")
    g.add_argument("--rule-partition-concurrency", type=int, default=8,   metavar="N")

    g = p.add_argument_group("Ablation (both disabled by default)")
    g.add_argument("--initial-ablation",  action="store_true", default=False,
                   help="Ablate initial RuleSet before Phase 1.")
    g.add_argument("--midpoint-ablation", action="store_true", default=False,
                   help="Ablate refined RuleSet on residual failures between phases.")

    g = p.add_argument_group("Phase 2 — Case study generation")
    g.add_argument("--max-cs-iters",               type=int,   default=5,    metavar="N")
    g.add_argument("--cs-static-iters",           type=int,   default=2,    metavar="N",
                   help="Stop Phase 2 after this many consecutive iterations with no CS added.")
    g.add_argument("--cs-bin-threshold",           type=int,   default=3,    metavar="N")
    g.add_argument("--cs-retirement-threshold",    type=int,   default=2,    metavar="N")
    g.add_argument("--cs-fix-rate-threshold",      type=float, default=0.30, metavar="F")
    g.add_argument("--cs-regress-threshold",       type=float, default=0.15, metavar="F")
    g.add_argument("--cs-n-candidates",            type=int,   default=3,    metavar="N")
    g.add_argument("--cs-candidate-rounds",        type=int,   default=3,    metavar="N")
    g.add_argument("--cs-concurrency",             type=int,   default=25,   metavar="N")
    g.add_argument("--cs-partition-concurrency",   type=int,   default=8,    metavar="N")
    g.add_argument("--cs-min-pool-for-regression", type=int,   default=5,    metavar="N")
    g.add_argument("--no-similarity-gate",         action="store_true")

    g = p.add_argument_group("Phase 2 — PK regression guard")
    g.add_argument("--pk-regression-guard",     action="store_true", default=False,
                   help="Revert Phase 2 case studies if they degrade below prior_knowledge-only baseline.")
    g.add_argument("--pk-regression-tolerance", type=float, default=0.03, metavar="F",
                   help="Max allowed accuracy drop below pk-only baseline before revert (default 0.03).")

    g = p.add_argument_group("Shared scoring")
    g.add_argument("--reasoning-effort", default="low",
                   choices=["low", "medium", "high", "none"])
    g.add_argument("--cot-first",    action="store_true", default=True)
    g.add_argument("--no-cot-first", dest="cot_first", action="store_false")

    g = p.add_argument_group("Output")
    g.add_argument("--output-dir", default="runs/hybrid_run", metavar="DIR")
    g.add_argument("--resume",     action="store_true", default=False,
                   help="Load cheatsheet_current from --output-dir/phase2 if it exists.")

    return p


def main() -> None:
    args    = _build_parser().parse_args()
    api_key = get_api_key()

    model_score      = args.model_score      or args.model
    model_rule_patch = args.model_rule_patch or args.model
    model_casestudy  = args.model_casestudy  or args.model
    output_dir       = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reasoning_effort = None if args.reasoning_effort == "none" else args.reasoning_effort

    task_spec = _load_task(args.task)

    # ── Oracle ────────────────────────────────────────────────────────────────
    oracle = None
    if not args.no_oracle:
        if not args.oracle_csv:
            print(
                "[Error] --oracle-csv is required unless --no-oracle is set.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        oracle = load_oracle_csv(Path(args.oracle_csv))
        print(f"[oracle] {len(oracle)} traces loaded.", file=sys.stderr)
    else:
        print("[oracle] Disabled.", file=sys.stderr)

    # ── Dataset ───────────────────────────────────────────────────────────────
    all_items = load_jsonl(Path(args.dataset))
    if args.limit is not None:
        all_items = all_items[: args.limit]
    print(f"[data] {len(all_items)} items loaded.", file=sys.stderr)

    # ── Prescore ──────────────────────────────────────────────────────────────
    prescore_map = None
    if args.prescore_file:
        prescore_map = json.loads(Path(args.prescore_file).read_text(encoding="utf-8"))
        print(f"[prescore] {len(prescore_map)} pre-scored items.", file=sys.stderr)

    # ── Rule set (Phase 1) ────────────────────────────────────────────────────
    rule_set = None
    if args.rule_set:
        rule_set = _load_rule_set(args.rule_set)
        print(f"[rule-set] {rule_set.summary()}", file=sys.stderr)
    else:
        print("[rule-set] None provided — Phase 1 will be skipped.", file=sys.stderr)

    # ── Cheatsheet initialisation ─────────────────────────────────────────────
    initial_cheatsheet: Cheatsheet | None = None

    if args.resume:
        cp = output_dir / "phase2" / "cheatsheet_current.json"
        if cp.exists():
            initial_cheatsheet = Cheatsheet.load(output_dir / "phase2" / "cheatsheet_current")
            print(f"[resume] Loaded checkpoint: {initial_cheatsheet.summary()}", file=sys.stderr)
        else:
            print(f"[resume] No checkpoint at {cp} — starting fresh.", file=sys.stderr)

    if initial_cheatsheet is None:
        prior_knowledge = ""
        if args.prior_knowledge:
            pk_path = Path(args.prior_knowledge)
            if not pk_path.exists():
                raise SystemExit(f"Error: --prior-knowledge not found: {pk_path}")
            prior_knowledge = pk_path.read_text(encoding="utf-8").strip()

        if args.init_cheatsheet:
            initial_cheatsheet = Cheatsheet.load(Path(args.init_cheatsheet))
            if prior_knowledge:
                initial_cheatsheet.prior_knowledge = prior_knowledge
        else:
            initial_cheatsheet = Cheatsheet(roadmap="", prior_knowledge=prior_knowledge)

        if args.ablate_prior_segments and initial_cheatsheet:
            to_disable = [s.strip() for s in args.ablate_prior_segments.split(",") if s.strip()]
            for seg_id in to_disable:
                found = initial_cheatsheet.disable_prior_segment(seg_id)
                label = "disabled" if found else "not found"
                print(f"[ablate] prior segment '{seg_id}': {label}", file=sys.stderr)
            remaining = sum(1 for s in initial_cheatsheet.prior_knowledge_segments if s.get("enabled", True))
            print(f"[ablate] {remaining}/{len(initial_cheatsheet.prior_knowledge_segments)} segments active",
                  file=sys.stderr)

        if args.no_render_limit and initial_cheatsheet:
            initial_cheatsheet.no_limit = True

    # ── Header ────────────────────────────────────────────────────────────────
    print(
        f"\n{'='*65}\n"
        f"ICR_hybrid Pipeline\n"
        f"  task             : {args.task}\n"
        f"  dataset          : {Path(args.dataset).name}  ({len(all_items)} items)\n"
        f"  rule-set         : {Path(args.rule_set).name if args.rule_set else 'none (Phase 1 skipped)'}\n"
        f"  oracle           : {'yes' if oracle else 'none'}\n"
        f"  model-score      : {model_score}\n"
        f"  model-rule-patch : {model_rule_patch}\n"
        f"  model-casestudy  : {model_casestudy}\n"
        f"  Phase 1 iters    : {args.max_rule_iters}  goal={args.rule_acc_goal:.0%}\n"
        f"  Phase 2 iters    : {args.max_cs_iters}\n"
        f"  output-dir       : {output_dir}\n"
        f"{'='*65}",
        file=sys.stderr,
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    result = run_hybrid_loop(
        train_items=all_items,
        val_items=None,
        initial_rule_set=rule_set,
        initial_cheatsheet=initial_cheatsheet,
        model_score=model_score,
        model_rule_patch=model_rule_patch,
        model_casestudy=model_casestudy,
        api_key=api_key,
        oracle=oracle,
        oracle_min_similarity=args.oracle_min_similarity,
        max_rule_iters=args.max_rule_iters,
        rule_acc_goal=args.rule_acc_goal,
        rule_static_iters=args.rule_static_iters,
        rule_bin_threshold=args.rule_bin_threshold,
        rule_fix_rate_threshold=args.rule_fix_rate_threshold,
        rule_regress_threshold=args.rule_regress_threshold,
        rule_concurrency=args.rule_concurrency,
        rule_partition_concurrency=args.rule_partition_concurrency,
        run_initial_ablation=args.initial_ablation,
        run_midpoint_ablation=args.midpoint_ablation,
        max_cs_iters=args.max_cs_iters,
        cs_static_iters=args.cs_static_iters,
        cs_bin_threshold=args.cs_bin_threshold,
        cs_retirement_threshold=args.cs_retirement_threshold,
        cs_fix_rate_threshold=args.cs_fix_rate_threshold,
        cs_regress_threshold=args.cs_regress_threshold,
        cs_n_candidates=args.cs_n_candidates,
        cs_candidate_rounds=args.cs_candidate_rounds,
        cs_similarity_gate=not args.no_similarity_gate,
        cs_concurrency=args.cs_concurrency,
        cs_partition_concurrency=args.cs_partition_concurrency,
        cs_min_pool_for_regression=args.cs_min_pool_for_regression,
        reasoning_effort=reasoning_effort,
        cot_first=args.cot_first,
        task_spec=task_spec,
        prescore_map=prescore_map,
        auto_rule_init=args.auto_rule_init,
        n_bootstrap_failures=args.bootstrap_n,
        pk_regression_guard=args.pk_regression_guard,
        pk_regression_tolerance=args.pk_regression_tolerance,
        output_dir=output_dir,
        log=True,
    )

    # ── Report ────────────────────────────────────────────────────────────────
    print(
        f"\n{'='*65}\n"
        f"ICR_hybrid Results\n"
        f"{'='*65}\n"
        f"  Phase 1  rule patches   : {result.n_rule_patches}\n"
        f"  Phase 1  iters          : {result.n_outer_iters_rule}\n"
        f"  Phase 1  accuracy       : {result.accuracy_after_rules:.1%}\n"
        f"  Phase 2  case studies   : {result.n_case_studies_added}\n"
        f"  Phase 2  merges         : {result.n_merges}\n"
        f"  Phase 2  iters          : {result.n_outer_iters_cs}\n"
        f"  Final    accuracy       : {result.accuracy_final:.1%}\n"
        f"{'='*65}",
        file=sys.stderr,
    )

    # Save update log
    (output_dir / "hybrid_update_log.json").write_text(
        json.dumps(result.update_log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save final cheatsheet
    result.cheatsheet.save(output_dir / "cheatsheet_final")

    # Save partition summaries
    (output_dir / "partition_summary_rule.json").write_text(
        json.dumps(result.partition_summary_rule, indent=2), encoding="utf-8"
    )
    (output_dir / "partition_summary_cs.json").write_text(
        json.dumps(result.partition_summary_cs, indent=2), encoding="utf-8"
    )

    # Print the final cheatsheet to stdout (for piping / inspection)
    print(result.cheatsheet.render())


if __name__ == "__main__":
    main()
