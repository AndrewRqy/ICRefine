"""
pipeline.py — End-to-end cheatsheet generation and refinement pipeline.

Stages
------
1. INIT   — Build the initial cheatsheet (decision tree + seed case studies).
            Three modes:
              a. generate   — LLM writes the decision tree from seed examples (default)
              b. --init-txt — load a plain .txt file as the decision tree; case studies
                              start empty and accumulate during training
              c. --init-cheatsheet — load a previously saved cheatsheet (.json sidecar)
2. TRAIN  — HypoGenic-style loop: score mini-batches, collect failures in a bin,
            flush when full → generate a new case study → append to cheatsheet.
3. REPORT — Print final accuracy, save all artifacts, optionally write final
            cheatsheet to --cheatsheet-out (overwrites).

Usage
-----
    python -m ICR_naive.pipeline \\
        --dataset path/to/dataset.jsonl \\
        --output-dir runs/run_001

    # Use an existing cheatsheet as starting point
    python -m ICR_naive.pipeline \\
        --dataset path/to/dataset.jsonl \\
        --init-txt path/to/prior_knowledge.txt \\
        --model-score openai/gpt-oss-120b \\
        --model-casestudy openai/gpt-4o \\
        --bin-threshold 3 --batch-size 10 \\
        --output-dir runs/naive_run \\
        --cheatsheet-out path/to/output_cheatsheet.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from .core.cheatsheet import Cheatsheet
from .core.data import load_jsonl, split_dataset
from .core.llm_client import get_api_key
from .generators.initial import DEFAULT_MODEL, generate_initial_cheatsheet
from .training.loop import run_training_loop

load_dotenv(Path(__file__).parent / ".env")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Iterative cheatsheet refinement pipeline (HypoGenic-style).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data ---
    g = p.add_argument_group("Data")
    g.add_argument("--dataset", required=True, metavar="FILE",
                   help="Training dataset .jsonl (equation1, equation2, answer)")
    g.add_argument("--val-dataset", default=None, metavar="FILE",
                   help="Separate validation .jsonl. If omitted, --val-split is used.")
    g.add_argument("--val-split", type=float, default=0.2, metavar="FRAC",
                   help="Fraction held out as validation (ignored if --val-dataset given)")
    g.add_argument("--seed", type=int, default=42, metavar="SEED")

    # --- Init stage ---
    g = p.add_argument_group("Init stage (mutually exclusive modes)")
    mx = g.add_mutually_exclusive_group()
    mx.add_argument("--init-txt", default=None, metavar="FILE",
                    help="Use this plain .txt file as the decision tree; "
                         "case studies start empty. e.g. NeuriCo_cheatsheet.txt")
    mx.add_argument("--init-cheatsheet", default=None, metavar="PATH",
                    help="Load a previously saved cheatsheet (base path, no extension; "
                         "reads <path>.json)")
    g.add_argument("--n-seed-examples", type=int, default=30, metavar="N",
                   help="Examples reserved for LLM decision-tree generation "
                        "(ignored if --init-txt or --init-cheatsheet given)")
    g.add_argument("--n-seed-studies", type=int, default=3, metavar="N",
                   help="Number of seed case studies generated at init")
    g.add_argument("--init-temperature", type=float, default=0.3, metavar="T")

    # --- Training loop ---
    g = p.add_argument_group("Training loop")
    g.add_argument("--bin-threshold", type=int, default=5, metavar="N",
                   help="Failures needed to trigger a case study generation")
    g.add_argument("--batch-size", type=int, default=20, metavar="N",
                   help="Items scored per mini-batch")
    g.add_argument("--concurrency", type=int, default=10, metavar="N",
                   help="Parallel API requests during scoring")
    g.add_argument("--casestudy-temperature", type=float, default=0.3, metavar="T")
    g.add_argument("--no-flush-remainder", action="store_true",
                   help="Do not flush leftover failures at the end of training")
    g.add_argument("--reasoning-effort", default="low",
                   choices=["low", "medium", "high", "none"],
                   help="Reasoning effort for scoring model (low/medium/high/none). "
                        "Use 'none' for non-reasoning models like gpt-4o-mini.")

    # --- Models ---
    g = p.add_argument_group("Models")
    g.add_argument("--model", default=DEFAULT_MODEL, metavar="MODEL_ID",
                   help="Default model for all stages")
    g.add_argument("--model-init",      default=None, metavar="MODEL_ID",
                   help="Override model for initial cheatsheet generation")
    g.add_argument("--model-score",     default=None, metavar="MODEL_ID",
                   help="Override model for scoring items during training")
    g.add_argument("--model-casestudy", default=None, metavar="MODEL_ID",
                   help="Override model for case study generation")

    # --- Output ---
    g = p.add_argument_group("Output")
    g.add_argument("--output-dir", default="runs/run", metavar="DIR",
                   help="Directory to save all run artifacts")
    g.add_argument("--cheatsheet-out", default=None, metavar="FILE",
                   help="Write the final cheatsheet to this file (overwrites)")

    return p


def main() -> None:
    args = _build_parser().parse_args()
    api_key = get_api_key()

    model_init      = args.model_init      or args.model
    model_score     = args.model_score     or args.model
    model_casestudy = args.model_casestudy or args.model
    output_dir      = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Error: dataset not found: {dataset_path}")
    all_items = load_jsonl(dataset_path)

    if args.val_dataset:
        val_path = Path(args.val_dataset)
        if not val_path.exists():
            raise SystemExit(f"Error: val-dataset not found: {val_path}")
        val_items = load_jsonl(val_path)
        seed_items, train_items, _ = split_dataset(
            all_items, val_fraction=0.0,
            seed_examples=args.n_seed_examples, seed=args.seed,
        )
    else:
        seed_items, train_items, val_items = split_dataset(
            all_items,
            val_fraction=args.val_split,
            seed_examples=args.n_seed_examples,
            seed=args.seed,
        )

    init_mode = (
        f"txt ({Path(args.init_txt).name})"          if args.init_txt else
        f"cheatsheet ({args.init_cheatsheet})"        if args.init_cheatsheet else
        "generate"
    )
    _log(
        f"\n{'='*60}\n"
        f"ICRefine Pipeline\n"
        f"  dataset        : {dataset_path.name}  ({len(all_items)} items)\n"
        f"  init mode      : {init_mode}\n"
        f"  seed / train / val : {len(seed_items)} / {len(train_items)} / {len(val_items)}\n"
        f"  model-init     : {model_init}\n"
        f"  model-score    : {model_score}\n"
        f"  model-casestudy: {model_casestudy}\n"
        f"  bin-threshold  : {args.bin_threshold}\n"
        f"  output-dir     : {output_dir}\n"
        f"{'='*60}"
    )

    # ------------------------------------------------------------------
    # Stage 1: Init
    # ------------------------------------------------------------------
    if args.init_txt:
        txt_path = Path(args.init_txt)
        if not txt_path.exists():
            raise SystemExit(f"Error: --init-txt file not found: {txt_path}")
        roadmap    = txt_path.read_text(encoding="utf-8").strip()
        cheatsheet = Cheatsheet(roadmap=roadmap)
        train_items = seed_items + train_items
        _log(
            f"\n[Stage 1] Loaded roadmap from {txt_path.name} "
            f"({len(roadmap):,} chars). Case studies start empty."
        )

    elif args.init_cheatsheet:
        _log(f"\n[Stage 1] Loading cheatsheet from {args.init_cheatsheet} ...")
        cheatsheet  = Cheatsheet.load(Path(args.init_cheatsheet))
        train_items = seed_items + train_items
        _log(f"  Loaded: {cheatsheet.summary()}")

    else:
        _log(f"\n[Stage 1] Generating initial cheatsheet from {len(seed_items)} seed examples ...")
        cheatsheet = generate_initial_cheatsheet(
            items=seed_items,
            model=model_init,
            api_key=api_key,
            n_seed_true=args.n_seed_examples // 2,
            n_seed_false=args.n_seed_examples // 2,
            n_studies=args.n_seed_studies,
            seed=args.seed,
            temperature=args.init_temperature,
        )
        cheatsheet.save(output_dir / "cheatsheet_init")
        _log(f"  {cheatsheet.summary()}")

    # ------------------------------------------------------------------
    # Stage 2: Training loop
    # ------------------------------------------------------------------
    _log(f"\n[Stage 2] Training loop over {len(train_items)} items ...")
    result = run_training_loop(
        cheatsheet=cheatsheet,
        train_items=train_items,
        val_items=val_items or None,
        model_score=model_score,
        model_casestudy=model_casestudy,
        api_key=api_key,
        bin_threshold=args.bin_threshold,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        casestudy_temperature=args.casestudy_temperature,
        flush_remainder=not args.no_flush_remainder,
        output_dir=output_dir,
        log=True,
        reasoning_effort=None if args.reasoning_effort == "none" else args.reasoning_effort,
    )

    # ------------------------------------------------------------------
    # Stage 3: Report & save
    # ------------------------------------------------------------------
    _log(f"\n[Stage 3] Results:")
    _log(f"  case_studies_added : {result.n_case_studies_added}")
    _log(f"  train_accuracy     : {result.train_accuracy:.1%}")
    if result.val_accuracy is not None:
        _log(f"  val_accuracy       : {result.val_accuracy:.1%}")
    _log(f"  {result.cheatsheet.summary()}")

    result.cheatsheet.save(output_dir / "cheatsheet_final")
    _log(f"\nFinal cheatsheet: {output_dir / 'cheatsheet_final.txt'}")

    if args.cheatsheet_out:
        out_path = Path(args.cheatsheet_out)
        out_path.write_text(result.cheatsheet.render(), encoding="utf-8")
        _log(f"Written to: {out_path}")

    print(result.cheatsheet.render())


if __name__ == "__main__":
    main()
