"""
eval_cs_ablation.py — Isolate case study contribution across model families.

For each task, scores two variants side-by-side on every model:
  full     — v3 cheatsheet (prior_knowledge + case studies)
  pk_only  — same prior_knowledge, no case studies

Delta = full - pk_only per model tells us whether the ACTIVATE IF case
studies help/hurt on each model family vs. what the plain CS-ICL-style
prior_knowledge alone achieves.

Usage:
    python3 eval_cs_ablation.py \\
        --tasks formal_fallacies web_of_lies \\
        --concurrency 20 \\
        --out runs/cs_ablation_results.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / "ICR_partition" / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.data import load_jsonl
from utils.llm_client import get_api_key
from utils.scorer import score_batch

CS_ICL_FILES = {
    "boolean_expressions":     ("bbh_boolean_expressions",              "gen_gpt-4.1-2025-04-14_0_1000.txt"),
    "causal_judgement":        ("bbh_causal_judgement",                 "gen_gpt-4.1-2025-04-14_0_1000.txt"),
    "date_understanding":      ("bbh_date_understanding",               "gen_gpt-4.1_0_1000.txt"),
    "disambiguation_qa":       ("bbh_disambiguation_qa",                "gen_gpt-4.1-2025-04-14_0_1000.txt"),
    "formal_fallacies":        ("bbh_formal_fallacies",                 "gen_gpt-4.1_0_1000.txt"),
    "geometric_shapes":        ("bbh_geometric_shapes",                 "gen_gpt-4.1-2025-04-14_0_1000.txt"),
    "logical_deduction_three": ("bbh_logical_deduction_three_objects",  "gen_gpt-4.1_0_1000.txt"),
    "navigate":                ("bbh_navigate",                         "gen_gpt-4.1_0_1000.txt"),
    "snarks":                  ("bbh_snarks",                           "gen_gpt-4.1_0_1000.txt"),
    "sports_understanding":    ("bbh_sports_understanding",             "gen_gpt-4.1-2025-04-14_0_1000.txt"),
    "web_of_lies":             ("bbh_web_of_lies",                      "gen_gpt-4.1_0_1000.txt"),
}

TASK_CFG = {
    "formal_fallacies": {
        "module":    "tasks.formal_fallacies",
        "attr":      "FORMAL_FALLACIES_TASK",
        "test_file": "datasets/bbh/formal_fallacies_test.jsonl",
        "run_dir":   "runs/bbh_v3/formal_fallacies",
    },
    "web_of_lies": {
        "module":    "tasks.web_of_lies",
        "attr":      "WEB_OF_LIES_TASK",
        "test_file": "datasets/bbh/web_of_lies_test.jsonl",
        "run_dir":   "runs/bbh_v3/web_of_lies",
    },
    "causal_judgement": {
        "module":    "tasks.causal_judgement",
        "attr":      "CAUSAL_JUDGEMENT_TASK",
        "test_file": "datasets/bbh/causal_judgement_test.jsonl",
        "run_dir":   "runs/bbh_v3/causal_judgement",
    },
    "geometric_shapes": {
        "module":    "tasks.geometric_shapes",
        "attr":      "GEOMETRIC_TASK",
        "test_file": "datasets/bbh/geometric_shapes_test.jsonl",
        "run_dir":   "runs/bbh_v3/geometric_shapes",
    },
    "boolean_expressions": {
        "module":    "tasks.boolean_expressions",
        "attr":      "BBH_BOOLEAN_TASK",
        "test_file": "datasets/bbh/boolean_expressions_test.jsonl",
        "run_dir":   "runs/bbh_v3/boolean_expressions",
    },
    "disambiguation_qa": {
        "module":    "tasks.disambiguation_qa",
        "attr":      "DISAMBIGUATION_TASK",
        "test_file": "datasets/bbh/disambiguation_qa_test.jsonl",
        "run_dir":   "runs/bbh_v3/disambiguation_qa",
    },
    "logical_deduction_three": {
        "module":    "tasks.logical_deduction",
        "attr":      "LOGICAL_DEDUCTION_3_TASK",
        "test_file": "datasets/bbh/logical_deduction_three_objects_test.jsonl",
        "run_dir":   "runs/bbh_v3/logical_deduction_three",
    },
    "sports_understanding": {
        "module":    "tasks.sports_understanding",
        "attr":      "SPORTS_TASK",
        "test_file": "datasets/bbh/sports_understanding_test.jsonl",
        "run_dir":   "runs/bbh_v3/sports_understanding",
    },
    "navigate": {
        "module":    "tasks.navigate",
        "attr":      "NAVIGATE_TASK",
        "test_file": "datasets/bbh/navigate_test.jsonl",
        "run_dir":   "runs/bbh_v3/navigate",
    },
    "snarks": {
        "module":    "tasks.snarks",
        "attr":      "SNARKS_TASK",
        "test_file": "datasets/bbh/snarks_test.jsonl",
        "run_dir":   "runs/bbh_v3/snarks",
    },
    "date_understanding": {
        "module":    "tasks.date_understanding",
        "attr":      "DATE_UNDERSTANDING_TASK",
        "test_file": "datasets/bbh/date_understanding_test.jsonl",
        "run_dir":   "runs/bbh_v3/date_understanding",
    },
    "gpqa_diamond": {
        "module":    "tasks.gpqa_diamond",
        "attr":      "GPQA_DIAMOND_TASK",
        "test_file": "datasets/gpqa_diamond/gpqa_diamond_test.jsonl",
        "run_dir":   "runs/bbh_gpqa/gpqa_diamond",
    },
}

MODELS = [
    ("openai/gpt-4.1-mini",                 "gpt-4.1-mini [train model]"),
    ("openai/gpt-4.1",                      "gpt-4.1"),
    ("anthropic/claude-3-7-sonnet-20250219","claude-3.7-sonnet"),
    ("google/gemini-2.0-flash-001",         "gemini-2.0-flash"),
    ("meta-llama/llama-3.3-70b-instruct",   "llama-3.3-70b"),
]


def _score(items, cs_text, task_spec, model, api_key, concurrency, label, force_cot=False) -> float:
    eval_fn = getattr(task_spec, "build_eval_prompt", None)
    if eval_fn is not None and not force_cot:
        from utils.llm_client import call_llm_batch
        prompts = [eval_fn(cs_text, item) for item in items]
        responses = call_llm_batch(
            prompts, model=model, api_key=api_key,
            temperature=0.0, max_tokens=32,
            concurrency=concurrency,
            progress_label=label,
            reasoning_effort=None,
        )
        def _check(item, resp):
            if resp is None:
                return False
            predicted = task_spec.parse_verdict(resp.content)
            if predicted is None:
                predicted = resp.content.strip()
            return task_spec.is_correct(predicted, item)
        correct = sum(_check(item, resp) for item, resp in zip(items, responses))
    else:
        correct_items, _ = score_batch(
            items, cs_text, model, api_key,
            concurrency=concurrency,
            temperature=0.0,
            reasoning_effort=None,
            cot_first=True,
            task_spec=task_spec,
        )
        correct = len(correct_items)
    acc = correct / len(items) if items else 0.0
    print(f"  [{label}] {acc:.1%}", file=sys.stderr)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks",            nargs="+", default=list(TASK_CFG))
    ap.add_argument("--models",           nargs="+", default=None,
                    help="Restrict to these model IDs (exact match)")
    ap.add_argument("--concurrency",      type=int,  default=20)
    ap.add_argument("--out",              default="runs/cs_ablation_results.json")
    ap.add_argument("--transferability",  default="runs/transferability_results.json",
                    help="Path to transferability_results.json — reuse full scores (verdict mode)")
    ap.add_argument("--cot",             action="store_true",
                    help="Force reasoning scorer (score_batch cot_first=True) for all conditions")
    ap.add_argument("--cot-results",     default="runs/cot_mode_results.json",
                    help="Path to cot_mode_results.json — reuse ours_cot full scores for gpt-4.1-mini")
    ap.add_argument("--baseline",        action="store_true",
                    help="Score with empty cheatsheet — measures raw model capability, no CS")
    ap.add_argument("--gold-fewshot",   action="store_true",
                    help="Score cheatsheet_gold_fewshot.txt (PK + gold worked examples)")
    ap.add_argument("--cs-only",        action="store_true",
                    help="Score cheatsheet_cs_only.txt (case studies only, no PK)")
    ap.add_argument("--run-dir-overrides", nargs="+", default=[], metavar="TASK:PATH",
                    help="Override run_dir per task: 'causal_judgement:runs/bbh_cj_no_p2oracle'")
    ap.add_argument("--reasoning-first",  action="store_true",
                    help="Use reasoning-first prompt format (VERDICT on last line) for tasks that support it")
    ap.add_argument("--full-only",        action="store_true",
                    help="Skip pk_only scoring; only score the full cheatsheet")
    ap.add_argument("--cs-icl-dir",      default="../cheat-sheet-icl/data/cheat_prompt",
                    help="Root directory of CS-ICL cheatsheets; enables cs_icl scoring when set")
    ap.add_argument("--no-csicl",        action="store_true",
                    help="Disable CS-ICL comparison even if --cs-icl-dir is valid")
    args = ap.parse_args()

    # Build run_dir override map
    run_dir_overrides: dict[str, Path] = {}
    for spec in args.run_dir_overrides:
        task, _, path = spec.partition(":")
        run_dir_overrides[task.strip()] = Path(path.strip())

    api_key = get_api_key()

    # ── Run logger ────────────────────────────────────────────────────────────
    from utils.run_logger import RunLogger, make_run_id, set_logger
    _run_logger = RunLogger(
        log_base="runs/logs/eval",
        run_id=make_run_id("cs_ablation"),
        config=vars(args),
    )
    set_logger(_run_logger)
    print(f"[log] {_run_logger.log_dir}", file=sys.stderr)

    # Load pre-existing full scores so we don't re-score them.
    # Skip when --reasoning-first is set: old verdict-only cached scores are not
    # compatible with RF format and would silently mix scoring regimes.
    transfer_data: dict = {}
    if not args.cot and not args.reasoning_first:
        tp = Path(args.transferability)
        if tp.exists():
            transfer_data = json.loads(tp.read_text(encoding="utf-8"))
            print(f"Loaded verdict full scores from {tp}", file=sys.stderr)

    cot_data: dict = {}
    if args.cot:
        cp = Path(args.cot_results)
        if cp.exists():
            cot_data = json.loads(cp.read_text(encoding="utf-8"))
            print(f"Loaded CoT full scores (gpt-4.1-mini) from {cp}", file=sys.stderr)

    results: dict[str, dict] = {}

    for task_name in args.tasks:
        if task_name not in TASK_CFG:
            print(f"Unknown task: {task_name}", file=sys.stderr)
            continue

        cfg       = TASK_CFG[task_name]
        run_dir   = run_dir_overrides.get(task_name, Path(cfg["run_dir"]))
        task_spec = getattr(importlib.import_module(cfg["module"]), cfg["attr"])
        items     = load_jsonl(Path(cfg["test_file"]))

        if args.reasoning_first:
            mod = importlib.import_module(cfg["module"])
            rf_builder = getattr(mod, f"_{task_name}_scoring_prompt_rf", None)
            rf_parser  = getattr(mod, f"_parse_{task_name}_rf", None)
            if rf_builder is not None and rf_parser is not None:
                import copy
                task_spec = copy.copy(task_spec)
                task_spec.build_scoring_prompt = rf_builder
                task_spec.parse_verdict = rf_parser
                task_spec.build_eval_prompt = None  # force score_batch path so RF prompt is used
                print(f"  [reasoning-first] prompt patched for {task_name}", file=sys.stderr)

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Task: {task_name}  ({len(items)} test items)", file=sys.stderr)

        pkonly_txt = ""
        if not args.baseline and not args.cs_only and not args.gold_fewshot and not args.full_only:
            for _pk_name in ("cheatsheet_phase1_pk_final.txt", "cheatsheet_pkonly.txt"):
                _pk_path = run_dir / _pk_name
                if _pk_path.exists():
                    pkonly_txt = _pk_path.read_text(encoding="utf-8").strip()
                    print(f"  pk_only : {len(pkonly_txt):,} chars  (source: {_pk_name})", file=sys.stderr)
                    break
            else:
                print(f"  [warn] no pk_only cheatsheet found for {task_name} — skipping pk_only", file=sys.stderr)

        task_transfer = transfer_data.get(task_name, {})
        task_cot      = cot_data.get(task_name, {})
        task_results: dict[str, dict] = {}

        full_txt = (run_dir / "cheatsheet_final.txt").read_text(encoding="utf-8").strip()

        # Load CS-ICL cheatsheet if requested
        cs_icl_txt = None
        if not args.no_csicl and task_name in CS_ICL_FILES:
            cs_icl_dir, cs_icl_file = CS_ICL_FILES[task_name]
            cs_icl_path = Path(args.cs_icl_dir) / cs_icl_dir / cs_icl_file
            if cs_icl_path.exists():
                cs_icl_txt = cs_icl_path.read_text(encoding="utf-8").strip()
                print(f"  cs_icl  : {len(cs_icl_txt):,} chars", file=sys.stderr)
            else:
                print(f"  [warn] CS-ICL not found: {cs_icl_path}", file=sys.stderr)

        active_models = [
            (mid, lbl) for mid, lbl in MODELS
            if args.models is None or any(mid == f or mid.endswith(f) for f in args.models)
        ]
        for model_id, label in active_models:
            print(f"\n  Model: {label}", file=sys.stderr)

            if args.baseline:
                baseline_acc = _score(items, "", task_spec, model_id, api_key,
                                      args.concurrency, f"{label}/baseline", force_cot=True)
                task_results[model_id] = {
                    "label":    label,
                    "baseline": round(baseline_acc, 4),
                }
                continue

            if args.gold_fewshot:
                gf_path = run_dir / "cheatsheet_gold_fewshot.txt"
                if not gf_path.exists():
                    print(f"  [warn] gold_fewshot not found: {gf_path}", file=sys.stderr)
                    continue
                gf_txt = gf_path.read_text(encoding="utf-8").strip()
                gf_acc = _score(items, gf_txt, task_spec, model_id, api_key,
                                args.concurrency, f"{label}/gold_fewshot", force_cot=True)
                task_results[model_id] = {
                    "label":       label,
                    "gold_fewshot": round(gf_acc, 4),
                }
                continue

            if args.cs_only:
                cso_path = run_dir / "cheatsheet_cs_only.txt"
                if not cso_path.exists():
                    print(f"  [warn] cs_only not found: {cso_path}", file=sys.stderr)
                    continue
                cso_txt = cso_path.read_text(encoding="utf-8").strip()
                cso_acc = _score(items, cso_txt, task_spec, model_id, api_key,
                                 args.concurrency, f"{label}/cs_only", force_cot=True)
                task_results[model_id] = {
                    "label":   label,
                    "cs_only": round(cso_acc, 4),
                }
                continue

            if args.cot:
                if model_id == "openai/gpt-4.1-mini" and task_cot.get("ours_cot") is not None:
                    full_acc = task_cot["ours_cot"]
                    print(f"  [{label}/full] {full_acc:.1%}  (from cot_mode_results)", file=sys.stderr)
                else:
                    full_acc = _score(items, full_txt, task_spec, model_id, api_key,
                                      args.concurrency, f"{label}/full", force_cot=True)
            else:
                full_key = f"ours/{model_id}"
                if full_key in task_transfer:
                    full_acc = task_transfer[full_key]
                    print(f"  [{label}/full] {full_acc:.1%}  (from transferability run)", file=sys.stderr)
                else:
                    full_acc = _score(items, full_txt, task_spec, model_id, api_key,
                                      args.concurrency, f"{label}/full")

            if args.full_only or not pkonly_txt:
                task_results[model_id] = {
                    "label": label,
                    "full":  round(full_acc, 4),
                }
            else:
                pkonly_acc = _score(items, pkonly_txt, task_spec, model_id, api_key,
                                    args.concurrency, f"{label}/pk_only", force_cot=args.cot)
                delta_cs   = full_acc - pkonly_acc
                task_results[model_id] = {
                    "label":     label,
                    "full":      round(full_acc,   4),
                    "pk_only":   round(pkonly_acc, 4),
                    "delta_cs":  round(delta_cs,   4),
                }

            if cs_icl_txt is not None:
                cs_icl_acc = _score(items, cs_icl_txt, task_spec, model_id, api_key,
                                    args.concurrency, f"{label}/cs_icl", force_cot=args.cot)
                task_results[model_id]["cs_icl"] = round(cs_icl_acc, 4)

        results[task_name] = task_results
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n  Saved intermediate results → {args.out}", file=sys.stderr)

    # Summary table
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"CASE STUDY CONTRIBUTION (full − pk_only) per model family", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    for task_name, task_results in results.items():
        print(f"\n  {task_name}", file=sys.stderr)
        print(f"  {'Model':<28} {'full':>7} {'pk_only':>8} {'Δ_cs':>7}", file=sys.stderr)
        print(f"  {'-'*52}", file=sys.stderr)
        for model_id, r in task_results.items():
            marker = " ← train model" if "gpt-4.1-mini" in model_id else ""
            if "full" in r and "pk_only" in r:
                print(
                    f"  {r['label']:<28} {r['full']:>6.1%}  {r['pk_only']:>6.1%}  {r['delta_cs']:>+6.1%}{marker}",
                    file=sys.stderr,
                )
            elif "cs_only" in r:
                print(f"  {r['label']:<28} cs_only={r['cs_only']:>6.1%}{marker}", file=sys.stderr)
            elif "baseline" in r:
                print(f"  {r['label']:<28} baseline={r['baseline']:>6.1%}{marker}", file=sys.stderr)
            elif "gold_fewshot" in r:
                print(f"  {r['label']:<28} gold_fewshot={r['gold_fewshot']:>6.1%}{marker}", file=sys.stderr)

    print(f"\nFull results → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
