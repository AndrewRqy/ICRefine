"""
ICR_select/pipeline.py — Selective iterative cheatsheet refinement pipeline.

Every case study addition must pass four quality gates before entering the
cheatsheet, and the cheatsheet is periodically pruned and condensed.

Usage
-----
    python -m ICR_select.pipeline \\
        --dataset path/to/dataset.jsonl \\
        --prior-knowledge path/to/prior_knowledge.txt \\
        --model-score openai/gpt-oss-120b \\
        --model-casestudy openai/gpt-4o \\
        --bin-threshold 3 --batch-size 5 \\
        --val-split 0.0 \\
        --output-dir runs/select_run \\
        --cheatsheet-out path/to/output_cheatsheet.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from utils.cheatsheet import Cheatsheet
from utils.data import load_jsonl, split_dataset
from ICR_naive.generators.initial import DEFAULT_MODEL, generate_initial_cheatsheet
from utils.llm_client import get_api_key
from ICR_reasoning.core.oracle import load_oracle_csv
from .training.loop import run_training_loop
from .training.utility_gate import UtilityConfig
from .prompts.templates import N_CANDIDATES

load_dotenv(Path(__file__).parent / ".env")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ICR_select — selective validated iterative cheatsheet refinement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Data")
    g.add_argument("--dataset",     required=True, metavar="FILE")
    g.add_argument("--val-dataset", default=None,  metavar="FILE")
    g.add_argument("--val-split",   type=float, default=0.0,  metavar="FRAC")
    g.add_argument("--seed",        type=int,   default=42,   metavar="SEED")
    g.add_argument("--limit",       type=int,   default=None, metavar="N",
                   help="Cap training items to first N.")
    g.add_argument("--oracle-csv",  default=None,  metavar="FILE",
                   help="Path to GPT-5.4 oracle CSV (gpt5.4_normal_default.csv). "
                        "Enables disagreement-bin mining: wrong items are paired with "
                        "the structurally nearest oracle entry for richer case study teaching.")
    g.add_argument("--oracle-min-similarity", type=float, default=0.25, metavar="F",
                   help="Jaccard threshold for nearest-oracle structural match. "
                        "Items below this threshold go to the both-wrong bin. (default: 0.25)")
    g.add_argument("--prescore-file", default=None, metavar="FILE",
                   help="JSON file of pre-computed SAIR eval scores (id → result). "
                        "When provided, the initial scoring pass is skipped — "
                        "items are split into correct/wrong from this file instead.")

    g = p.add_argument_group("Init (mutually exclusive modes)")
    mx = g.add_mutually_exclusive_group()
    mx.add_argument("--init-roadmap",     default=None, metavar="FILE",
                   help="Plain-text file loaded as the trainable roadmap / decision tree; "
                        "case studies start empty.")
    mx.add_argument("--init-cheatsheet", default=None, metavar="PATH")
    g.add_argument("--prior-knowledge",  default=None, metavar="FILE",
                   help="Plain-text file loaded into the frozen prior_knowledge field "
                        "(e.g. NeuriCo prompt). Use alone to start with an empty trainable "
                        "roadmap, or with --init-roadmap / --init-cheatsheet to layer "
                        "frozen knowledge on top of an existing starting point.")
    g.add_argument("--n-seed-examples",  type=int,   default=30,  metavar="N")
    g.add_argument("--n-seed-studies",   type=int,   default=3,   metavar="N")
    g.add_argument("--init-temperature", type=float, default=0.3, metavar="T")

    g = p.add_argument_group("Training loop")
    g.add_argument("--bin-threshold", type=int,   default=5,   metavar="N")
    g.add_argument("--batch-size",    type=int,   default=10,  metavar="N")
    g.add_argument("--concurrency",   type=int,   default=10,  metavar="N")
    g.add_argument("--no-flush-remainder", action="store_true")
    g.add_argument("--cot-first",     action="store_true", default=False,
                   help="REASONING before VERDICT in scoring prompt (default: off, matches SAIR eval).")
    g.add_argument("--no-cot-first",  dest="cot_first", action="store_false")
    g.add_argument("--reasoning-effort", default="low",
                   choices=["low", "medium", "high", "none"])

    g = p.add_argument_group("Quality gates")
    g.add_argument("--n-candidates",       type=int,   default=N_CANDIDATES, metavar="N",
                   help="Candidates generated per bin flush.")
    g.add_argument("--flush-strategy",    default="default", choices=["default", "retry"],
                   help="'default': discard bin on gate failure. "
                        "'retry': retry up to --candidate-rounds times, passing the "
                        "previous candidate's still-wrong items as context each round.")
    g.add_argument("--candidate-rounds",  type=int,   default=3, metavar="N",
                   help="Max retry rounds per bin flush when --flush-strategy retry (default: 3).")
    g.add_argument("--fix-rate-threshold", type=float, default=0.30, metavar="F",
                   help="Minimum fraction of failures a candidate must fix.")
    g.add_argument("--regress-threshold",  type=float, default=0.15, metavar="F",
                   help="Maximum fraction of correct-pool items a candidate may break.")
    g.add_argument("--min-pool-for-regression", type=int, default=10, metavar="N",
                   help="Skip regression gate when correct_pool has fewer than N items "
                        "(avoids false rejections when pool is too small to be statistically meaningful).")
    g.add_argument("--no-similarity-gate", action="store_true",
                   help="Skip LLM similarity/dedup check (faster, less selective).")
    g.add_argument("--validate-merge", action="store_true",
                   help="Before committing a merge, verify the merged entry fixes at least as "
                        "many failures as the existing one. If not, add as a new entry instead.")

    g = p.add_argument_group("Utility gate (replaces fix-rate + regression when slices are large enough)")
    g.add_argument("--utility-gate", action="store_true",
                   help="Enable continuous utility scoring gate U(c) = ΔVmatch + λΔVgap − μRegress − νlen.")
    g.add_argument("--utility-lambda",    type=float, default=0.5,  metavar="F",
                   help="λ: Vgap improvement weight (default: 0.5).")
    g.add_argument("--utility-mu",        type=float, default=1.0,  metavar="F",
                   help="μ: regression penalty weight (default: 1.0).")
    g.add_argument("--utility-nu",        type=float, default=0.1,  metavar="F",
                   help="ν: length penalty per 1 000 chars rendered (default: 0.1).")
    g.add_argument("--utility-threshold", type=float, default=0.0,  metavar="F",
                   help="Accept candidate when U > threshold (default: 0.0).")
    g.add_argument("--utility-min-slice", type=int,   default=5,    metavar="N",
                   help="Min items per Vmatch/Vgap slice; below this → fall back to classic gates (default: 5).")

    g = p.add_argument_group("Maintenance")
    g.add_argument("--ablation-every", type=int, default=5, metavar="N",
                   help="Run ablation pruning every N flushes.")
    g.add_argument("--condense-at",    type=int, default=6, metavar="N",
                   help="Run condensation when case_studies reaches this count.")

    g = p.add_argument_group("Models")
    g.add_argument("--model",           default=DEFAULT_MODEL, metavar="MODEL_ID")
    g.add_argument("--model-init",      default=None, metavar="MODEL_ID")
    g.add_argument("--model-score",     default=None, metavar="MODEL_ID")
    g.add_argument("--model-casestudy", default=None, metavar="MODEL_ID")

    g = p.add_argument_group("Output")
    g.add_argument("--output-dir",     default="runs/select_run", metavar="DIR")
    g.add_argument("--cheatsheet-out", default=None, metavar="FILE")
    g.add_argument("--no-render-limit", action="store_true", default=False,
                   help="Disable all character caps on the rendered cheatsheet (roadmap, "
                        "per-case-study, and total budget). Every case study is included in "
                        "full. Useful when using large prior-knowledge prompts (e.g. NeuriCo) "
                        "that would otherwise crowd out the ICR roadmap and case bank.")

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
    all_items = load_jsonl(Path(args.dataset))

    # ------------------------------------------------------------------
    # Load oracle (optional)
    # ------------------------------------------------------------------
    oracle = load_oracle_csv(Path(args.oracle_csv)) if args.oracle_csv else None
    if oracle:
        _log(f"\n[Oracle] {len(oracle)} correct reasoning traces loaded.")

    # ------------------------------------------------------------------
    # Load prescore map (optional) — avoids redundant initial scoring pass
    # ------------------------------------------------------------------
    import json as _json
    prescore_map: dict | None = None
    if args.prescore_file:
        prescore_map = _json.loads(Path(args.prescore_file).read_text(encoding="utf-8"))
        _log(f"\n[Prescore] {len(prescore_map)} pre-scored items loaded — skipping initial scoring pass.")

    # Apply --limit before splitting so val_items is proportional to the
    # limited dataset rather than the full 1000-item file.
    if args.limit is not None:
        all_items = all_items[: args.limit]

    if args.val_dataset:
        val_items = load_jsonl(Path(args.val_dataset))
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
        f"roadmap ({Path(args.init_roadmap).name})" if args.init_roadmap else
        f"cheatsheet ({args.init_cheatsheet})"       if args.init_cheatsheet else
        f"prior-knowledge only (empty DT)"           if args.prior_knowledge else
        "generate"
    )
    _log(
        f"\n{'='*60}\n"
        f"ICR_select Pipeline\n"
        f"  dataset        : {Path(args.dataset).name}  ({len(all_items)} items)\n"
        f"  init mode      : {init_mode}\n"
        f"  seed/train/val : {len(seed_items)}/{len(train_items)}/{len(val_items)}\n"
        f"  model-score    : {model_score}\n"
        f"  model-casestudy: {model_casestudy}\n"
        f"  reasoning-effort: {reasoning_effort}\n"
        f"  cot-first      : {args.cot_first}\n"
        f"  n-candidates   : {args.n_candidates}\n"
        f"  flush-strategy : {args.flush_strategy}"
        + (f" (rounds={args.candidate_rounds})" if args.flush_strategy == "retry" else "") + "\n"
        f"  fix-rate-gate  : ≥{args.fix_rate_threshold:.0%}\n"
        f"  regress-gate   : ≤{args.regress_threshold:.0%}  (min_pool={args.min_pool_for_regression})\n"
        f"  similarity-gate: {'off' if args.no_similarity_gate else 'on'}\n"
        f"  ablation-every : {args.ablation_every}\n"
        f"  condense-at    : {args.condense_at}\n"
        f"  output-dir     : {output_dir}\n"
        f"{'='*60}"
    )

    # ------------------------------------------------------------------
    # Stage 1: Init
    # ------------------------------------------------------------------

    # Load optional frozen prior knowledge (e.g. NeuriCo prompt).
    prior_knowledge = ""
    if args.prior_knowledge:
        pk_path = Path(args.prior_knowledge)
        if not pk_path.exists():
            raise SystemExit(f"Error: --prior-knowledge not found: {pk_path}")
        prior_knowledge = pk_path.read_text(encoding="utf-8").strip()
        _log(f"\n[Stage 1] Loaded prior knowledge from {pk_path.name} ({len(prior_knowledge)} chars).")

    if args.init_roadmap:
        txt_path = Path(args.init_roadmap)
        if not txt_path.exists():
            raise SystemExit(f"Error: --init-roadmap not found: {txt_path}")
        cheatsheet = Cheatsheet(
            roadmap=txt_path.read_text(encoding="utf-8").strip(),
            prior_knowledge=prior_knowledge,
        )
        _log(f"\n[Stage 1] Loaded roadmap from {txt_path.name}. Case studies start empty.")
        train_items = seed_items + train_items

    elif args.init_cheatsheet:
        cheatsheet = Cheatsheet.load(Path(args.init_cheatsheet))
        if prior_knowledge:
            cheatsheet.prior_knowledge = prior_knowledge
        train_items = seed_items + train_items
        _log(f"\n[Stage 1] Loaded cheatsheet: {cheatsheet.summary()}")

    elif prior_knowledge:
        # Prior knowledge provided alone — start with empty trainable roadmap
        cheatsheet = Cheatsheet(roadmap="", prior_knowledge=prior_knowledge)
        train_items = seed_items + train_items
        _log(f"\n[Stage 1] Prior knowledge loaded ({len(prior_knowledge)} chars). Roadmap starts empty.")

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

    # Apply no-render-limit flag — must be set before any .render() call
    if args.no_render_limit:
        cheatsheet.no_limit = True
        _log("\n[Stage 1] Render limits disabled — all character caps bypassed.")

    # ------------------------------------------------------------------
    # Stage 2: Training (inner loop only, or outer DT revision loop)
    # ------------------------------------------------------------------
    if args.limit is not None:
        _log(f"\n[Stage 2] Dataset limited to {len(all_items)} items (train={len(train_items)}, val={len(val_items)}).")

    # Build utility config if gate is enabled
    utility_config = UtilityConfig(
        lam=args.utility_lambda,
        mu=args.utility_mu,
        nu=args.utility_nu,
        threshold=args.utility_threshold,
        min_slice=args.utility_min_slice,
    ) if args.utility_gate else None

    # Shared kwargs passed to run_training_loop in every round
    inner_kwargs = dict(
        model_score=model_score,
        model_casestudy=model_casestudy,
        api_key=api_key,
        bin_threshold=args.bin_threshold,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        n_candidates=args.n_candidates,
        candidate_rounds=args.candidate_rounds,
        flush_strategy=args.flush_strategy,
        oracle=oracle,
        oracle_min_similarity=args.oracle_min_similarity,
        prescore_map=prescore_map,
        fix_rate_threshold=args.fix_rate_threshold,
        regress_threshold=args.regress_threshold,
        min_pool_for_regression=args.min_pool_for_regression,
        similarity_gate=not args.no_similarity_gate,
        validate_merge=args.validate_merge,
        utility_gate=args.utility_gate,
        utility_config=utility_config,
        ablation_every=args.ablation_every,
        condense_at=args.condense_at,
        flush_remainder=not args.no_flush_remainder,
        cot_first=args.cot_first,
        reasoning_effort=reasoning_effort,
        log=True,
    )

    _log(f"\n[Stage 2] Case study accumulation loop ...")
    result = run_training_loop(
        cheatsheet=cheatsheet,
        train_items=train_items,
        val_items=val_items or None,
        output_dir=output_dir,
        **inner_kwargs,
    )
    final_cheatsheet = result.cheatsheet
    final_train_acc  = result.train_accuracy
    final_val_acc    = result.val_accuracy
    extra_stats: dict = {
        "case_studies_added": result.n_case_studies_added,
        "merges": result.n_merges,
        "bins_discarded": result.n_bins_discarded,
        "bins_skipped": result.n_bins_skipped,
        "ablation_pruned": result.n_ablation_pruned,
        "condensations": result.n_condensations,
        "disagree_items": result.n_disagree,
        "both_wrong_items": result.n_both_wrong,
        "utility_accepted": result.n_utility_accepted,
        "utility_fallbacks": result.n_utility_fallbacks,
    }

    # ------------------------------------------------------------------
    # Stage 3: Report & save
    # ------------------------------------------------------------------
    _log(f"\n[Stage 3] Results:")
    _log(f"  train_accuracy : {final_train_acc:.1%}")
    if final_val_acc is not None:
        _log(f"  val_accuracy   : {final_val_acc:.1%}")
    for k, v in extra_stats.items():
        _log(f"  {k:<22}: {v}")
    _log(f"  {final_cheatsheet.summary()}")

    final_cheatsheet.save(output_dir / "cheatsheet_final")

    if args.cheatsheet_out:
        out_path = Path(args.cheatsheet_out)
        out_path.write_text(final_cheatsheet.render(), encoding="utf-8")
        _log(f"  Written to: {out_path}")

    print(final_cheatsheet.render())


if __name__ == "__main__":
    main()
