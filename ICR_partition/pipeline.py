"""
ICR_partition/pipeline.py — Partition-parallel iterative cheatsheet refinement.

Key defaults (different from ICR_select)
-----------------------------------------
* --oracle-csv is required unless --no-oracle is passed (oracle ON by default).
* Roadmap starts empty by default; use --init-roadmap to provide one.
* Partition-parallel solving: each structural failure class is solved
  concurrently and independently.
* Per-partition regression: each case study is regression-checked against
  correct items from the same structural class, not a global reservoir.
* Bin retirement: partitions with few residual failures are retired after
  each outer iteration, focusing compute on the hard tail.

Usage
-----
    python -m ICR_partition.pipeline \\
        --dataset path/to/dataset.jsonl \\
        --oracle-csv path/to/gpt5.4_normal_default.csv \\
        --prior-knowledge path/to/NeuriCo_cheatsheet.txt \\
        --model-score deepseek-r1-32b \\
        --model-casestudy gpt-4o \\
        --output-dir runs/partition_run

    # Without oracle (falls back to both-wrong enrichment only):
    python -m ICR_partition.pipeline \\
        --dataset path/to/dataset.jsonl \\
        --no-oracle \\
        --model-score deepseek-r1-32b \\
        --model-casestudy gpt-4o \\
        --output-dir runs/partition_run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from utils.cheatsheet import Cheatsheet
from utils.data import load_jsonl
from utils.llm_client import get_api_key
from ICR_reasoning.core.oracle import load_oracle_csv
from .training.loop import run_partition_loop
from .training.partition import print_partition_table

load_dotenv(Path(__file__).parent / ".env")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Run lock — prevents two processes writing to the same output directory
# ---------------------------------------------------------------------------

def _acquire_lock(output_dir: Path) -> Path:
    """
    Write a .icr_lock file to output_dir.  Raises SystemExit if one already exists.
    Returns the lock path so the caller can release it on exit.
    """
    lock_path = output_dir / ".icr_lock"
    if lock_path.exists():
        info = {}
        try:
            info = json.loads(lock_path.read_text())
        except Exception:
            pass
        pid  = info.get("pid", "unknown")
        dir_ = info.get("output_dir", str(output_dir))
        _log(
            f"\n[Error] Output directory is locked by another process (PID {pid}).\n"
            f"  Lock file: {lock_path}\n"
            f"  If the previous run crashed, delete the lock file and retry:\n"
            f"    rm {lock_path}"
        )
        raise SystemExit(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "output_dir": str(output_dir)}),
        encoding="utf-8",
    )
    return lock_path


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ICR_partition — partition-parallel iterative cheatsheet refinement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Data")
    g.add_argument("--dataset",      required=True, metavar="FILE")
    g.add_argument("--limit",        type=int, default=None, metavar="N",
                   help="Cap training items to first N.")
    g.add_argument("--seed",         type=int, default=42)
    g.add_argument("--prescore-file", default=None, metavar="FILE",
                   help="JSON file of pre-computed scores (id → result). "
                        "Skips the initial scoring pass when provided.")

    g = p.add_argument_group("Oracle (on by default)")
    g.add_argument("--oracle-csv",  default=None, metavar="FILE",
                   help="Path to oracle CSV (e.g. gpt5.4_normal_default.csv). "
                        "Required unless --no-oracle is set.")
    g.add_argument("--no-oracle",   action="store_true",
                   help="Disable oracle enrichment. Failures will have no oracle "
                        "nearest-neighbour trace; case study quality may decrease.")
    g.add_argument("--oracle-min-similarity", type=float, default=0.25, metavar="F",
                   help="Jaccard threshold for nearest-oracle structural match.")

    g = p.add_argument_group("Cheatsheet initialisation (roadmap OFF by default)")
    mx = g.add_mutually_exclusive_group()
    mx.add_argument("--init-roadmap",    default=None, metavar="FILE",
                    help="Load a reasoning roadmap as the trainable roadmap. "
                         "By default the roadmap starts empty and only case studies are built.")
    mx.add_argument("--init-cheatsheet", default=None, metavar="PATH",
                    help="Load a full cheatsheet JSON as the starting point.")
    g.add_argument("--prior-knowledge",  default=None, metavar="FILE",
                   help="Frozen knowledge prefix (e.g. NeuriCo prompt). "
                        "Injected before the roadmap and case studies in every scoring call.")
    g.add_argument("--no-render-limit",  action="store_true", default=False,
                   help="Disable character caps on the rendered cheatsheet "
                        "(useful with large prior-knowledge prompts).")

    g = p.add_argument_group("Partition loop")
    g.add_argument("--bin-threshold",      type=int,   default=3,   metavar="N",
                   help="Minimum failures per partition to attempt case study generation.")
    g.add_argument("--retirement-threshold", type=int, default=2,   metavar="N",
                   help="Retire a partition when residual failures fall below this.")
    g.add_argument("--max-outer-iters",    type=int,   default=5,   metavar="N",
                   help="Maximum outer iterations (each re-scores active bins).")
    g.add_argument("--partition-concurrency", type=int, default=8,  metavar="N",
                   help="Max partitions solved concurrently.")
    g.add_argument("--concurrency",        type=int,   default=25,  metavar="N",
                   help="LLM API concurrency for score_batch calls.")

    g = p.add_argument_group("Candidate generation")
    g.add_argument("--n-candidates",    type=int,   default=3,    metavar="N")
    g.add_argument("--candidate-rounds", type=int,  default=3,    metavar="N",
                   help="Max retry rounds per bin when gates fail.")

    g = p.add_argument_group("Quality gates")
    g.add_argument("--fix-rate-threshold",   type=float, default=0.30, metavar="F",
                   help="Minimum fraction of failures a candidate must fix.")
    g.add_argument("--regress-threshold",    type=float, default=0.15, metavar="F",
                   help="Maximum regression rate on the partition's correct pool.")
    g.add_argument("--min-pool-for-regression", type=int, default=5,  metavar="N",
                   help="Skip regression gate when correct_pool has fewer than N items.")
    g.add_argument("--no-similarity-gate",   action="store_true",
                   help="Skip LLM similarity/dedup check.")

    g = p.add_argument_group("Scoring")
    g.add_argument("--reasoning-effort", default="low",
                   choices=["low", "medium", "high", "none"])
    g.add_argument("--cot-first",    action="store_true", default=True,
                   help="REASONING before VERDICT in scoring prompt (default: on).")
    g.add_argument("--no-cot-first", dest="cot_first", action="store_false")

    g = p.add_argument_group("Models")
    g.add_argument("--model",           default="deepseek-r1-32b", metavar="MODEL_ID")
    g.add_argument("--model-score",     default=None, metavar="MODEL_ID")
    g.add_argument("--model-score-2",   default=None, metavar="MODEL_ID",
                   help="Optional second scoring model for ensemble scoring. "
                        "When set, each item is scored by both models in parallel; "
                        "failures carry _wrong_weight proportional to how many models failed.")
    g.add_argument("--model-score-weights", default=None, metavar="W1,W2",
                   help="Comma-separated weights for --model-score and --model-score-2 "
                        "(e.g. '1.0,1.0'). Defaults to equal weights.")
    g.add_argument("--model-casestudy", default=None, metavar="MODEL_ID")

    g = p.add_argument_group("Output")
    g.add_argument("--output-dir",     default="runs/partition_run", metavar="DIR")
    g.add_argument("--cheatsheet-out", default=None, metavar="FILE",
                   help="Write final rendered cheatsheet to this path.")
    g.add_argument("--resume", action="store_true", default=False,
                   help="Load cheatsheet_current.json from --output-dir if it exists "
                        "and skip initialisation.")

    return p


def main() -> None:
    args    = _build_parser().parse_args()
    api_key = get_api_key()

    model_score     = args.model_score     or args.model
    model_score_2   = args.model_score_2   or None
    model_casestudy = args.model_casestudy or args.model
    model_score_weights: list[float] | None = None
    if args.model_score_weights:
        try:
            model_score_weights = [float(w) for w in args.model_score_weights.split(",")]
        except ValueError:
            raise SystemExit(f"Error: --model-score-weights must be comma-separated floats, "
                             f"e.g. '1.0,1.0'. Got: {args.model_score_weights}")
    output_dir      = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reasoning_effort = None if args.reasoning_effort == "none" else args.reasoning_effort

    # Acquire run lock — prevents two processes from writing to the same directory
    lock_path = _acquire_lock(output_dir)

    # ------------------------------------------------------------------
    # Oracle — ON by default
    # ------------------------------------------------------------------
    oracle = None
    if not args.no_oracle:
        if not args.oracle_csv:
            _log(
                "\n[Error] --oracle-csv is required unless --no-oracle is set.\n"
                "  Provide: --oracle-csv path/to/gpt5.4_normal_default.csv\n"
                "  Or disable: --no-oracle"
            )
            raise SystemExit(1)
        oracle = load_oracle_csv(Path(args.oracle_csv))
        _log(f"\n[Oracle] {len(oracle)} correct reasoning traces loaded.")
    else:
        _log("\n[Oracle] Disabled via --no-oracle.")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    all_items = load_jsonl(Path(args.dataset))
    if args.limit is not None:
        all_items = all_items[: args.limit]

    # ------------------------------------------------------------------
    # Prescore map
    # ------------------------------------------------------------------
    prescore_map: dict | None = None
    if args.prescore_file:
        prescore_map = json.loads(Path(args.prescore_file).read_text(encoding="utf-8"))
        _log(f"\n[Prescore] {len(prescore_map)} pre-scored items loaded.")

    # ------------------------------------------------------------------
    # Cheatsheet initialisation (roadmap empty by default)
    # ------------------------------------------------------------------
    _resumed = False
    cheatsheet: Cheatsheet

    if args.resume:
        cp = output_dir / "cheatsheet_current.json"
        if cp.exists():
            cheatsheet = Cheatsheet.load(str(output_dir / "cheatsheet_current"))
            if args.no_render_limit:
                cheatsheet.no_limit = True
            _log(f"\n[Resume] Loaded checkpoint: {cheatsheet.summary()}")
            _resumed = True
        else:
            _log(f"\n[Resume] No checkpoint found at {cp} — starting fresh.")

    if not _resumed:
        prior_knowledge = ""
        if args.prior_knowledge:
            pk_path = Path(args.prior_knowledge)
            if not pk_path.exists():
                raise SystemExit(f"Error: --prior-knowledge not found: {pk_path}")
            prior_knowledge = pk_path.read_text(encoding="utf-8").strip()
            _log(f"\n[Init] Prior knowledge: {pk_path.name} ({len(prior_knowledge)} chars)")

        if args.init_cheatsheet:
            cheatsheet = Cheatsheet.load(Path(args.init_cheatsheet))
            if prior_knowledge:
                cheatsheet.prior_knowledge = prior_knowledge
            _log(f"\n[Init] Loaded cheatsheet: {cheatsheet.summary()}")

        elif args.init_roadmap:
            roadmap_path = Path(args.init_roadmap)
            if not roadmap_path.exists():
                raise SystemExit(f"Error: --init-roadmap not found: {roadmap_path}")
            cheatsheet = Cheatsheet(
                roadmap=roadmap_path.read_text(encoding="utf-8").strip(),
                prior_knowledge=prior_knowledge,
            )
            _log(f"\n[Init] Loaded roadmap from {roadmap_path.name}. Case studies start empty.")

        else:
            # Default: empty roadmap, case studies built from scratch
            cheatsheet = Cheatsheet(roadmap="", prior_knowledge=prior_knowledge)
            _log(
                f"\n[Init] Empty roadmap (roadmap off by default). "
                f"Case studies will be built from failures.\n"
                + (f"  Prior knowledge: {len(prior_knowledge)} chars" if prior_knowledge else
                   "  No prior knowledge.")
            )

        if args.no_render_limit:
            cheatsheet.no_limit = True

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    _log(
        f"\n{'='*65}\n"
        f"ICR_partition Pipeline\n"
        f"  dataset          : {Path(args.dataset).name}  ({len(all_items)} items)\n"
        f"  oracle           : {'yes (' + Path(args.oracle_csv).name + ')' if oracle else 'none'}\n"
        f"  roadmap          : {'yes' if cheatsheet.roadmap else 'none (default)'}\n"
        f"  prior_knowledge  : {len(cheatsheet.prior_knowledge)} chars\n"
        f"  model-score      : {model_score}\n"
        f"  model-casestudy  : {model_casestudy}\n"
        f"  bin-threshold    : {args.bin_threshold}\n"
        f"  retirement-thresh: {args.retirement_threshold}\n"
        f"  max-outer-iters  : {args.max_outer_iters}\n"
        f"  partition-concur : {args.partition_concurrency}\n"
        f"  output-dir       : {output_dir}\n"
        f"{'='*65}"
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    try:
        result = run_partition_loop(
            cheatsheet=cheatsheet,
            train_items=all_items,
            val_items=None,
            model_score=model_score,
            model_casestudy=model_casestudy,
            api_key=api_key,
            model_score_2=model_score_2,
            model_score_weights=model_score_weights,
            oracle=oracle,
            oracle_min_similarity=args.oracle_min_similarity,
            bin_threshold=args.bin_threshold,
            retirement_threshold=args.retirement_threshold,
            max_outer_iters=args.max_outer_iters,
            partition_concurrency=args.partition_concurrency,
            concurrency=args.concurrency,
            n_candidates=args.n_candidates,
            candidate_rounds=args.candidate_rounds,
            fix_rate_threshold=args.fix_rate_threshold,
            regress_threshold=args.regress_threshold,
            min_pool_for_regression=args.min_pool_for_regression,
            similarity_gate=not args.no_similarity_gate,
            reasoning_effort=reasoning_effort,
            cot_first=args.cot_first,
            prescore_map=prescore_map,
            output_dir=output_dir,
            log=True,
        )
    finally:
        _release_lock(lock_path)

    # ------------------------------------------------------------------
    # Report & save
    # ------------------------------------------------------------------
    _log(f"\n{'='*65}")
    _log(f"ICR_partition Results")
    _log(f"{'='*65}")
    _log(f"  train_accuracy      : {result.train_accuracy:.1%}")
    _log(f"  case_studies_added  : {result.n_case_studies_added}")
    _log(f"  merges              : {result.n_merges}")
    _log(f"  bins_discarded      : {result.n_bins_discarded}")
    _log(f"  bins_skipped        : {result.n_skipped}")
    _log(f"  bins_solved         : {result.n_bins_solved}")
    _log(f"  outer_iters         : {result.n_outer_iters}")
    _log(f"  {result.cheatsheet.summary()}")

    result.cheatsheet.save(output_dir / "cheatsheet_final")

    # Write partition summary JSON
    (output_dir / "partition_summary.json").write_text(
        json.dumps(result.partition_summary, indent=2), encoding="utf-8"
    )

    if args.cheatsheet_out:
        out_path = Path(args.cheatsheet_out)
        out_path.write_text(result.cheatsheet.render(), encoding="utf-8")
        _log(f"  Written to: {out_path}")

    print(result.cheatsheet.render())


if __name__ == "__main__":
    main()
