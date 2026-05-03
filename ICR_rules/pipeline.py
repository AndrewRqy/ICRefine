"""
ICR_rules pipeline — Rule-patch variant of ICRefine.

Usage:
  python -m ICR_rules.pipeline \\
    --cheatsheet path/to/neurico_v2_slim_opt2_fr1fix.jinja2 \\
    --oracle gpt5.4_hard_correct.csv \\
    --dataset datasets/hard2.jsonl \\
    --failure-dir path/to/sair_results/gemma/hard2 \\
    --model-score google/gemma-4-31b-it \\
    --model-patch openai/gpt-5.4 \\
    --output-dir runs/icr_rules_001

--failure-dir (optional): path to a SAIR eval results folder containing per-item
  JSON files. If provided, failures are loaded from there and balanced with an
  equal number of correct items from --dataset. If omitted, the pipeline scores
  --dataset from scratch to find failures.
"""
from __future__ import annotations

import argparse
import builtins
import functools
import json
import os
import random
import sys
from pathlib import Path

# Force-flush all print output so progress is visible when piped to tee
builtins.print = functools.partial(builtins.print, flush=True)

# Ensure ICRefine root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from utils.data import load_jsonl
from utils.llm_client import get_api_key
from ICR_reasoning.core.oracle import load_oracle_csv

from .rules.parser import parse_cheatsheet_file
from .training.loop import RuleLoopConfig, run_rule_loop
from .training.scorer import score_batch_sair


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ICR_rules — rule-patch refinement pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cheatsheet", required=True, help="Path to neurico-style jinja2 cheatsheet")
    p.add_argument("--oracle", required=True, help="Path to GPT-5.4 oracle CSV")
    p.add_argument("--dataset", required=True, help="Path to .jsonl dataset (e.g. hard2.jsonl)")
    p.add_argument("--failure-dir", default=None,
                   help="SAIR eval results dir with per-item JSON files. "
                        "If given, failures come from here; otherwise scored from scratch.")
    p.add_argument("--model-score", default="google/gemma-4-31b-it",
                   help="Model to use for SAIR scoring (measures true deployment accuracy)")
    p.add_argument("--model-patch", default="openai/gpt-5.4",
                   help="Model to generate rule patches (oracle-grade)")
    p.add_argument("--output-dir", default="runs/icr_rules", help="Output directory")
    p.add_argument("--max-iters", type=int, default=6)
    p.add_argument("--bin-threshold", type=int, default=3)
    p.add_argument("--fix-rate", type=float, default=0.20)
    p.add_argument("--regress-threshold", type=float, default=0.20)
    p.add_argument("--concurrency", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-static-iters", type=int, default=2,
                   help="Stop after this many consecutive iterations with no patch accepted (default: 2)")
    p.add_argument("--no-ablation", action="store_true", help="Skip ablation pre-pass")
    p.add_argument("--ablation-only", action="store_true",
                   help="Run ablation on the full dataset then exit — skips test-set balancing and patch loop")
    p.add_argument("--api-key", default=None, help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    return p


# ---------------------------------------------------------------------------
# Test set construction
# ---------------------------------------------------------------------------

def build_test_set(
    all_items: list[dict],
    failure_dir: Path | None,
    rule_set,
    model_score: str,
    api_key: str,
    seed: int = 42,
    concurrency: int = 50,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Returns (test_items, failure_items, correct_items).
    If failure_dir is given, loads failures from pre-scored SAIR results.
    Otherwise, scores all_items from scratch.
    test_items = failures + equal number of correct items (balanced).
    """
    rng = random.Random(seed)

    if failure_dir is not None and failure_dir.exists():
        print(f"[pipeline] Loading pre-scored results from {failure_dir}")
        failures, correct_pool = _load_sair_results(failure_dir, all_items)
    else:
        print(f"[pipeline] Scoring {len(all_items)} items to find failures...")
        correct_pool, failures = score_batch_sair(
            all_items, rule_set, model_score, api_key, concurrency=concurrency
        )

    # Balance: equal failures and correct
    n = len(failures)
    if len(correct_pool) > n:
        sampled_correct = rng.sample(correct_pool, n)
    else:
        sampled_correct = correct_pool

    test_items = failures + sampled_correct
    rng.shuffle(test_items)

    print(f"[pipeline] Test set: {len(failures)} failures + {len(sampled_correct)} correct = {len(test_items)} items")
    return test_items, failures, sampled_correct


def _load_sair_results(
    results_dir: Path,
    all_items: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Load per-item JSON files from a SAIR eval results directory."""
    item_lookup = {item["id"]: item for item in all_items}
    failures: list[dict] = []
    correct: list[dict] = []

    for json_file in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text())
        except Exception:
            continue

        item_id = data.get("id")
        base_item = item_lookup.get(item_id, {})
        annotated = {
            **base_item,
            **{k: data[k] for k in ("equation1", "equation2", "answer") if k in data},
            "predicted": data.get("verdict"),
            "reasoning": data.get("reasoning", ""),
            "raw_response": data.get("raw_response", ""),
            "correct": data.get("correct", False),
            "id": item_id,
        }
        if data.get("correct"):
            correct.append(annotated)
        else:
            failures.append(annotated)

    print(f"[pipeline] Loaded {len(failures)} failures, {len(correct)} correct from {results_dir}")
    return failures, correct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cheatsheet → RuleSet
    print(f"[pipeline] Parsing cheatsheet: {args.cheatsheet}")
    rule_set = parse_cheatsheet_file(args.cheatsheet)
    print(f"[pipeline] {rule_set.summary()}")

    # Print rule inventory
    print("[pipeline] Rules found:")
    for s in rule_set.sections:
        if s.rules:
            ids = ", ".join(r.id for r in s.rules)
            print(f"  {s.name}: {ids}")

    # Load oracle
    print(f"[pipeline] Loading oracle: {args.oracle}")
    oracle = load_oracle_csv(args.oracle)
    print(f"[pipeline] Oracle: {len(oracle)} correct entries")

    # Load dataset
    print(f"[pipeline] Loading dataset: {args.dataset}")
    all_items = load_jsonl(args.dataset)
    print(f"[pipeline] Dataset: {len(all_items)} items")

    failure_dir = Path(args.failure_dir) if args.failure_dir else None

    if args.ablation_only:
        # Use full dataset directly — no balancing, no patch loop
        test_items = all_items
        print(f"[pipeline] --ablation-only: using all {len(test_items)} items")
    else:
        # Build balanced test set
        test_items, failure_items, correct_items = build_test_set(
            all_items=all_items,
            failure_dir=failure_dir,
            rule_set=rule_set,
            model_score=args.model_score,
            api_key=api_key,
            seed=args.seed,
            concurrency=args.concurrency,
        )

        # Save test set manifest
        manifest_path = output_dir / "test_set.json"
        manifest_path.write_text(json.dumps(
            {"n_failures": len(failure_items), "n_correct": len(correct_items),
             "n_total": len(test_items), "seed": args.seed,
             "failure_ids": [i["id"] for i in failure_items],
             "correct_ids": [i["id"] for i in correct_items]},
            indent=2
        ))

    # Config
    cfg = RuleLoopConfig(
        model_score=args.model_score,
        model_patch=args.model_patch,
        api_key=api_key,
        output_dir=output_dir,
        max_outer_iters=0 if args.ablation_only else args.max_iters,
        bin_threshold=args.bin_threshold,
        fix_rate_threshold=args.fix_rate,
        regress_threshold=args.regress_threshold,
        partition_concurrency=4,
        score_concurrency=args.concurrency,
        run_ablation_prepass=not args.no_ablation,
        max_static_iters=args.max_static_iters,
    )

    # Run
    result = run_rule_loop(
        initial_rule_set=rule_set,
        train_items=test_items,
        oracle=oracle,
        cfg=cfg,
    )

    # Final report
    print("\n" + "=" * 60)
    print(f"ICR_rules complete")
    print(f"  Patches applied : {result.n_patches_applied}")
    print(f"  Final accuracy  : {result.final_accuracy:.1%}")
    print(f"  Output dir      : {output_dir}")
    print("=" * 60)

    # Save final cheatsheet as jinja2
    final_path = output_dir / "cheatsheet_final.jinja2"
    final_path.write_text(result.rule_set.render(), encoding="utf-8")
    print(f"[pipeline] Final cheatsheet saved: {final_path} ({result.rule_set.byte_size()/1024:.1f} KB)")


if __name__ == "__main__":
    main()
