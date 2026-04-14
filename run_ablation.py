#!/usr/bin/env python3
"""
run_ablation.py — Case study generation ablation study.

Runs ICR_select once per prompt variant, evaluates the resulting cheatsheet
against a SAIR eval prescore file, and writes a summary table.

Each variant is a (activate_if × emphasis × length) combination:
  activate_if : strict | loose
  emphasis    : both | correct | wrong
  length      : 900 | 400

Usage:
    python run_ablation.py \\
        --dataset ../SAIR_eval_pipeline/datasets/hard1.jsonl \\
        --prior-knowledge ../SAIR_eval_pipeline/prompts/NeuriCo_cheatsheet.txt \\
        --oracle-csv gpt5.4_hard_correct.csv \\
        --prescore ../SAIR_eval_pipeline/results/refine_20260411_230141/iter_00/icr_select_iter_01/prescore.json \\
        --model-score deepseek-r1-32b \\
        --model-casestudy gpt-4o \\
        --output-dir runs/ablation

Evaluation: after each variant, scores the full dataset with the resulting
cheatsheet using the scoring model and records accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Case study generation ablation over prompt variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",         required=True, metavar="FILE")
    p.add_argument("--prior-knowledge", required=True, metavar="FILE")
    p.add_argument("--oracle-csv",      default=None,  metavar="FILE")
    p.add_argument("--prescore",        default=None,  metavar="FILE",
                   help="Pre-computed SAIR eval scores (skips initial scoring pass)")
    p.add_argument("--model-score",     default="deepseek-r1-32b", metavar="MODEL")
    p.add_argument("--model-casestudy", default="gpt-4o",          metavar="MODEL")
    p.add_argument("--bin-threshold",   type=int, default=5)
    p.add_argument("--batch-size",      type=int, default=13)
    p.add_argument("--concurrency",     type=int, default=25)
    p.add_argument("--n-candidates",    type=int, default=3)
    p.add_argument("--output-dir",      default="runs/ablation", metavar="DIR")
    p.add_argument("--variants",        default=None, metavar="A,B,...",
                   help="Comma-separated subset of variant names to run, e.g. "
                        "strict_both_900,loose_correct_400. Default: all 12.")
    p.add_argument("--resume",          action="store_true",
                   help="Skip variants whose output dir already has cheatsheet_final.txt")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    from utils.data import load_jsonl, is_true
    from utils.cheatsheet import Cheatsheet
    from utils.llm_client import get_api_key
    from utils.scorer import score_batch
    from ICR_reasoning.core.oracle import load_oracle_csv
    from ICR_select.prompts.ablation_templates import ABLATION_VARIANTS, VARIANT_NAMES
    from ICR_select.training.loop import run_training_loop

    api_key    = get_api_key()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset)
    all_items    = load_jsonl(dataset_path)
    prior_text   = Path(args.prior_knowledge).read_text(encoding="utf-8").strip()

    oracle = None
    if args.oracle_csv:
        oracle_path = Path(args.oracle_csv)
        if not oracle_path.is_absolute():
            oracle_path = Path(__file__).parent / oracle_path
        oracle = load_oracle_csv(oracle_path)

    prescore_map: dict = {}
    if args.prescore:
        raw = json.loads(Path(args.prescore).read_text(encoding="utf-8"))
        # Support both list-of-dicts and dict-keyed-by-id formats
        if isinstance(raw, list):
            prescore_map = {item["id"]: item for item in raw if "id" in item}
        else:
            prescore_map = raw
        print(f"[ablation] Loaded {len(prescore_map)} prescore entries.", file=sys.stderr)

    # Which variants to run
    all_keys  = list(ABLATION_VARIANTS.keys())
    run_keys  = all_keys
    if args.variants:
        name_to_key = {v: k for k, v in VARIANT_NAMES.items()}
        run_keys = [name_to_key[n.strip()] for n in args.variants.split(",")]

    results: list[dict] = []
    summary_path = output_dir / "summary.json"

    # Load existing summary if resuming
    if args.resume and summary_path.exists():
        results = json.loads(summary_path.read_text(encoding="utf-8"))
        done_names = {r["variant"] for r in results}
        run_keys = [k for k in run_keys if VARIANT_NAMES[k] not in done_names]
        print(f"[ablation] Resuming — {len(run_keys)} variants remaining.", file=sys.stderr)

    for key in run_keys:
        act, emp, lng = key
        name = VARIANT_NAMES[key]
        variant_dir = output_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"[ablation] Variant: {name}  (activate_if={act}, emphasis={emp}, length={lng})", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        # Patch the prompt for this variant
        import ICR_select.generators.case_study as _cs_gen_mod
        import ICR_reasoning.prompts.templates as _tmpl_mod
        prompt_variant = ABLATION_VARIANTS[key]
        original_prompt = _tmpl_mod.CASE_STUDY_WITH_REASONING_PROMPT
        _tmpl_mod.CASE_STUDY_WITH_REASONING_PROMPT = prompt_variant

        # Also patch FLUSH_MAX_TOKENS to match length variant
        import ICR_select.prompts.templates as _sel_tmpl
        original_flush = _sel_tmpl.FLUSH_MAX_TOKENS
        _sel_tmpl.FLUSH_MAX_TOKENS = int(lng / 4 * 1.5)  # ~tokens for char budget

        t0 = time.time()
        try:
            cheatsheet = Cheatsheet(roadmap=prior_text, prior_knowledge=prior_text)

            result = run_training_loop(
                cheatsheet=cheatsheet,
                train_items=all_items,
                val_items=None,
                model_score=args.model_score,
                model_casestudy=args.model_casestudy,
                api_key=api_key,
                bin_threshold=args.bin_threshold,
                batch_size=args.batch_size,
                concurrency=args.concurrency,
                flush_remainder=True,
                output_dir=variant_dir,
                log=True,
                reasoning_effort=None,
                prescore_map=prescore_map,
                oracle=oracle,
                n_candidates=args.n_candidates,
                flush_strategy="retry",
            )

            # Save cheatsheet
            result.cheatsheet.save(variant_dir / "cheatsheet_final")
            cs_text = result.cheatsheet.render()
            (variant_dir / "cheatsheet_final.txt").write_text(cs_text, encoding="utf-8")

            # Evaluate on full dataset
            print(f"\n  [ablation] Evaluating cheatsheet on {len(all_items)} items ...", file=sys.stderr)
            correct, wrong = score_batch(
                all_items, cs_text, args.model_score, api_key,
                concurrency=args.concurrency, reasoning_effort=None, cot_first=False,
                progress_label=f"eval-{name}",
            )
            accuracy = len(correct) / len(all_items) if all_items else 0.0
            elapsed  = time.time() - t0

            entry = {
                "variant":          name,
                "activate_if":      act,
                "emphasis":         emp,
                "length":           lng,
                "accuracy":         round(accuracy, 4),
                "n_case_studies":   len(result.cheatsheet.case_studies),
                "train_accuracy":   round(result.train_accuracy, 4),
                "elapsed_seconds":  round(elapsed),
            }
            results.append(entry)
            summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

            print(
                f"\n  [ablation] {name}: accuracy={accuracy:.1%}  "
                f"case_studies={len(result.cheatsheet.case_studies)}  "
                f"elapsed={elapsed:.0f}s",
                file=sys.stderr,
            )

        except Exception as exc:
            print(f"\n  [ablation] {name} FAILED: {exc}", file=sys.stderr)
            results.append({"variant": name, "error": str(exc)})
            summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        finally:
            # Restore original prompt
            _tmpl_mod.CASE_STUDY_WITH_REASONING_PROMPT = original_prompt
            _sel_tmpl.FLUSH_MAX_TOKENS = original_flush

    # Print final summary table
    print(f"\n{'='*60}", file=sys.stderr)
    print("ABLATION SUMMARY", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"{'Variant':<30} {'Accuracy':>8} {'#CS':>4} {'Train':>8}", file=sys.stderr)
    print("-" * 55, file=sys.stderr)
    for r in sorted(results, key=lambda x: x.get("accuracy", 0), reverse=True):
        if "error" in r:
            print(f"  {r['variant']:<28} ERROR: {r['error'][:30]}", file=sys.stderr)
        else:
            print(
                f"  {r['variant']:<28} {r['accuracy']:>8.1%} {r['n_case_studies']:>4} "
                f"{r['train_accuracy']:>8.1%}",
                file=sys.stderr,
            )
    print(f"\nFull results: {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
