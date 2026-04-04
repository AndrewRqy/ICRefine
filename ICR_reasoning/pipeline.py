"""
ICR_reasoning/pipeline.py — Post-think aware cheatsheet refinement pipeline.

Identical flow to ICR_naive with three differences:
  1. Scoring captures both content (post-think) and thinking (full CoT) per item.
  2. Failures passed to the bin carry their post-think, which is included in the
     case study generation prompt so the LLM can see WHY the model failed.
  3. A reasoning analysis report is saved after training.

Theoretical basis: Heddaya et al. (ACL 2026) — post-think text preserves
deductive logical structure at 25× higher density than externally prompted
summaries, making it a better teaching signal for corrective case studies.

Usage
-----
    # Start from NeuriCo cheatsheet, train on hard1 with reasoning-aware case studies
    python -m ICR_reasoning.pipeline \\
        --dataset ../SAIR_evaluation_pipeline/datasets/hard1.jsonl \\
        --init-txt ../SAIR_evaluation_pipeline/prompts/NeuriCo_cheatsheet.txt \\
        --model-score openai/gpt-oss-120b \\
        --model-casestudy openai/gpt-4o \\
        --bin-threshold 3 --batch-size 10 \\
        --output-dir runs/reasoning_hard1 \\
        --cheatsheet-out ../SAIR_evaluation_pipeline/prompts/NeuriCo_cheatsheet_reasoning.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from ICR_naive.core.cheatsheet import Cheatsheet
from ICR_naive.core.data import _is_true, load_jsonl, split_dataset
from ICR_naive.generators.initial import DEFAULT_MODEL, generate_initial_cheatsheet
from .core.llm_client import get_api_key
from .training.loop import run_training_loop
from .training.scorer import score_batch
from .analysis.reasoning_analyzer import analyze_items, print_report, save_report

load_dotenv(Path(__file__).parent.parent / "SAIR_evaluation_pipeline" / ".env")
load_dotenv(Path(__file__).parent / ".env")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ICR_reasoning — post-think aware iterative cheatsheet refinement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Data")
    g.add_argument("--dataset",     required=True, metavar="FILE")
    g.add_argument("--val-dataset", default=None,  metavar="FILE")
    g.add_argument("--val-split",   type=float, default=0.2, metavar="FRAC")
    g.add_argument("--seed",        type=int,   default=42,  metavar="SEED")
    g.add_argument("--limit",       type=int,   default=None, metavar="N",
                   help="Cap training items to the first N (useful for quick tests).")

    g = p.add_argument_group("Init stage (mutually exclusive modes)")
    mx = g.add_mutually_exclusive_group()
    mx.add_argument("--init-txt", default=None, metavar="FILE",
                    help="Plain .txt file used as the decision tree; case studies start empty")
    mx.add_argument("--init-cheatsheet", default=None, metavar="PATH",
                    help="Previously saved cheatsheet (base path, no extension)")
    g.add_argument("--n-seed-examples",  type=int, default=30,  metavar="N")
    g.add_argument("--n-seed-studies",   type=int, default=3,   metavar="N")
    g.add_argument("--init-temperature", type=float, default=0.3, metavar="T")

    g = p.add_argument_group("Training loop")
    g.add_argument("--bin-threshold",          type=int,   default=5,   metavar="N")
    g.add_argument("--batch-size",             type=int,   default=20,  metavar="N")
    g.add_argument("--concurrency",            type=int,   default=10,  metavar="N")
    g.add_argument("--casestudy-temperature",  type=float, default=0.3, metavar="T")
    g.add_argument("--no-flush-remainder",     action="store_true")
    g.add_argument("--no-dt-patch",            action="store_true",
                   help="Skip applying decision tree patches (case studies only).")
    g.add_argument("--cot-first",              action="store_true", default=False,
                   help="Put REASONING before VERDICT in scoring prompt (default: off, matches SAIR eval).")
    g.add_argument("--no-analysis",            action="store_true",
                   help="Skip the final reasoning analysis stage.")
    g.add_argument("--reasoning-effort", default="low",
                   choices=["low", "medium", "high", "none"],
                   help="Reasoning effort for scoring model. Use 'none' for non-reasoning models.")

    g = p.add_argument_group("Models")
    g.add_argument("--model",           default=DEFAULT_MODEL, metavar="MODEL_ID")
    g.add_argument("--model-init",      default=None, metavar="MODEL_ID")
    g.add_argument("--model-score",     default=None, metavar="MODEL_ID")
    g.add_argument("--model-casestudy", default=None, metavar="MODEL_ID")

    g = p.add_argument_group("Output")
    g.add_argument("--output-dir",     default="runs/reasoning_run", metavar="DIR")
    g.add_argument("--cheatsheet-out", default=None, metavar="FILE",
                   help="Write final cheatsheet to this file (overwrites)")

    return p


def main() -> None:
    args    = _build_parser().parse_args()
    api_key = get_api_key()

    model_init      = args.model_init      or args.model
    model_score     = args.model_score     or args.model
    model_casestudy = args.model_casestudy or args.model
    output_dir      = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reasoning_effort = None if args.reasoning_effort == "none" else args.reasoning_effort

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
        f"txt ({Path(args.init_txt).name})"    if args.init_txt else
        f"cheatsheet ({args.init_cheatsheet})" if args.init_cheatsheet else
        "generate"
    )
    _log(
        f"\n{'='*60}\n"
        f"ICR_reasoning Pipeline\n"
        f"  dataset        : {dataset_path.name}  ({len(all_items)} items)\n"
        f"  init mode      : {init_mode}\n"
        f"  seed/train/val : {len(seed_items)}/{len(train_items)}/{len(val_items)}\n"
        f"  model-init     : {model_init}\n"
        f"  model-score    : {model_score}\n"
        f"  model-casestudy: {model_casestudy}\n"
        f"  reasoning-effort: {reasoning_effort}\n"
        f"  dt-patch       : {'enabled' if not args.no_dt_patch else 'disabled'}\n"
        f"  cot-first      : {'enabled' if args.cot_first else 'disabled'}\n"
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
            raise SystemExit(f"Error: --init-txt not found: {txt_path}")
        cheatsheet  = Cheatsheet(decision_tree=txt_path.read_text(encoding="utf-8").strip())
        train_items = seed_items + train_items
        _log(f"\n[Stage 1] Loaded decision tree from {txt_path.name}. Case studies start empty.")

    elif args.init_cheatsheet:
        cheatsheet  = Cheatsheet.load(Path(args.init_cheatsheet))
        train_items = seed_items + train_items
        _log(f"\n[Stage 1] Loaded cheatsheet: {cheatsheet.summary()}")

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
    if args.limit is not None:
        train_items = train_items[: args.limit]
        _log(f"\n[Stage 2] Limited to first {len(train_items)} training items (--limit {args.limit}).")
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
        apply_dt_patch=not args.no_dt_patch,
        cot_first=args.cot_first,
        output_dir=output_dir,
        log=True,
        reasoning_effort=reasoning_effort,
    )

    # ------------------------------------------------------------------
    # Stage 3: Reasoning analysis (skipped with --no-analysis)
    # ------------------------------------------------------------------
    if not args.no_analysis:
        _log("\n[Stage 3] Running reasoning analysis ...")
        correct_final, wrong_final = score_batch(
            train_items,
            cheatsheet_text=result.cheatsheet.render(),
            model=model_score,
            api_key=api_key,
            concurrency=args.concurrency,
            reasoning_effort=reasoning_effort,
            cot_first=args.cot_first,
            progress_label="analysis",
        )
        report = analyze_items(correct_final, wrong_final)
        print_report(report)
        save_report(report, output_dir / "reasoning_analysis.json")
        _log(f"  Analysis saved to {output_dir / 'reasoning_analysis.json'}")
    else:
        _log("\n[Stage 3] Skipped (--no-analysis).")

    # ------------------------------------------------------------------
    # Stage 4: Report & save
    # ------------------------------------------------------------------
    _log(f"\n[Stage 4] Results:")
    _log(f"  case_studies_added : {result.n_case_studies_added}")
    _log(f"  train_accuracy     : {result.train_accuracy:.1%}")
    if result.val_accuracy is not None:
        _log(f"  val_accuracy       : {result.val_accuracy:.1%}")
    _log(f"  {result.cheatsheet.summary()}")

    result.cheatsheet.save(output_dir / "cheatsheet_final")

    if args.cheatsheet_out:
        out_path = Path(args.cheatsheet_out)
        out_path.write_text(result.cheatsheet.render(), encoding="utf-8")
        _log(f"Written to: {out_path}")

    print(result.cheatsheet.render())


if __name__ == "__main__":
    main()
