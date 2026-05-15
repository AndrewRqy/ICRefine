"""
ICR_holistic/pipeline.py — CLI entry point.

Algorithm:
  1. Generate initial cheatsheet from all training examples (CS-ICL style, model_gen).
     If cheatsheet_iter0.txt already exists in output_dir, reuse it (cache).
     Override with --init-cheatsheet to supply your own starting point.
  2. Run the holistic refinement loop (see training/loop.py).

Key design choices vs ICR_rules / refinement_pipeline:
  - Base cheatsheet is FREE-FORM PROSE, not a jinja2 rule template.
    CS-ICL warm-start means the initial cheatsheet already captures the task's
    hardest patterns in natural language.
  - Per-bin fix/regression scoring uses (current_cs + new content) temporarily;
    the holistic rewrite then integrates everything into a single coherent text.
  - Regression is RECORDED but never gates acceptance — the rewriter sees all
    regressions and is responsible for scoping new rules appropriately.
  - One model_gen call per iteration for the rewrite (not one per bin).

Usage:
    python -m ICR_holistic.pipeline \\
        --task         magma \\
        --dataset      datasets/hard3.jsonl \\
        --oracle       datasets/hard3_gpt54_oracle.csv \\
        --model-score  openai/gpt-oss-120b \\
        --model-gen    openai/gpt-4.1 \\
        --output-dir   runs/holistic_hard3 \\
        --max-iters    5

    # Start from an existing cheatsheet instead of generating CS-ICL:
    python -m ICR_holistic.pipeline \\
        --init-cheatsheet runs/holistic_hard3/cheatsheet_iter2.txt \\
        --task magma --dataset datasets/hard3.jsonl ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data import load_jsonl
from utils.llm_client import get_api_key, call_llm
from ICR_reasoning.core.oracle import load_oracle_csv
from tasks.registry import get_task, TASK_REGISTRY

from .prompts import CSICL_GENERATION_PROMPT
from .training.loop import HolisticLoopConfig, run_holistic_loop


def _load_task_spec(task_name: str):
    try:
        return get_task(task_name)
    except KeyError:
        print(f"[pipeline] WARNING: unknown task {task_name!r} — no task spec loaded\n"
              f"           Known tasks: {', '.join(sorted(TASK_REGISTRY))}",
              file=sys.stderr)
        return None
    except Exception as e:
        print(f"[pipeline] WARNING: could not load task spec for {task_name!r}: {e}",
              file=sys.stderr)
        return None


def _format_item_for_csicl(item: dict, oracle: dict) -> str:
    """Format one training item for the CSICL_GENERATION_PROMPT."""
    eq1 = item.get("equation1", "?")
    eq2 = item.get("equation2", "?")
    answer_bool = item.get("answer", False)
    ans = "True (implies)" if answer_bool else "False (does not imply)"
    question = f'Does "{eq1}" imply "{eq2}"?'
    reasoning = oracle.get(item.get("id", ""), "")
    if reasoning:
        return f"Question: {question}\nReasoning: {reasoning}\nAnswer: {ans}"
    return f"Question: {question}\nAnswer: {ans}"


def generate_initial_cheatsheet(
    train_items: list[dict],
    oracle: dict,
    model: str,
    api_key: str,
    max_tokens: int = 4000,
    task_spec=None,
) -> str:
    """Generate a CS-ICL warm-start cheatsheet from all training items."""
    fmt_fn = (
        task_spec.format_for_csicl
        if task_spec is not None and task_spec.format_for_csicl is not None
        else _format_item_for_csicl
    )
    formatted  = [fmt_fn(item, oracle) for item in train_items]
    dataset_str = "\n\n".join(formatted)
    prompt      = CSICL_GENERATION_PROMPT.format(dataset_str=dataset_str)
    print(f"[init] Generating CS-ICL cheatsheet from {len(train_items)} items "
          f"({len(prompt):,} chars prompt)...")
    resp = call_llm(
        prompt, model=model, api_key=api_key,
        max_tokens=max_tokens, temperature=0.0, reasoning_effort=None,
    )
    return resp.content.strip()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ICR_holistic — CS-ICL warm start + iterative holistic refinement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--task",           default="magma",
                    help="Task name: magma | formal_fallacies | causal_judgement | ...")
    ap.add_argument("--dataset",        required=True,
                    help="Path to .jsonl training dataset")
    ap.add_argument("--oracle",         default=None,
                    help="Path to oracle CSV (optional; enriches bin generation prompts)")
    ap.add_argument("--model-score",    default="openai/gpt-oss-120b",
                    help="Scoring model — measures real deployment accuracy each iteration")
    ap.add_argument("--model-gen",      default="openai/gpt-4.1",
                    help="Generation model — writes CS-ICL, bin rules, and holistic rewrites")
    ap.add_argument("--output-dir",     default="runs/holistic",
                    help="Directory for all outputs (cheatsheets, bin outputs, analysis)")
    ap.add_argument("--max-iters",      type=int,   default=5)
    ap.add_argument("--bin-threshold",  type=int,   default=3,
                    help="Min failures per bin to activate it")
    ap.add_argument("--fix-rate",       type=float, default=0.10,
                    help="Min fraction of bin failures that must be fixed to accept a rule")
    ap.add_argument("--regression-pool",type=int,   default=100,
                    help="Hard cap on correct items per bin for regression check")
    ap.add_argument("--regression-pool-fraction", type=float, default=0.10,
                    help="Regression pool = min(cap, fraction * n_train)")
    ap.add_argument("--concurrency",    type=int,   default=50,
                    help="Concurrency for score_batch calls")
    ap.add_argument("--bin-concurrency",type=int,   default=4,
                    help="Parallel workers for bin generation")
    ap.add_argument("--init-cheatsheet",default=None,
                    help="Path to an existing cheatsheet (overrides CS_ICL_Initial_Prompt lookup)")
    ap.add_argument("--init-max-tokens",type=int,   default=4000,
                    help="Max tokens for initial CS-ICL generation (fallback only)")
    ap.add_argument("--cs-icl-seed",    type=int,   default=0,
                    help="Seed index in CS_ICL_Initial_Prompt filenames (gen_model_<seed>_tokens.txt)")
    ap.add_argument("--cs-icl-tokens",  type=int,   default=None,
                    help="Max-token variant to use from CS_ICL_Initial_Prompt (default: highest available)")
    ap.add_argument("--bin-retry",      type=int,   default=2,
                    help="Max generation attempts per bin before discarding")
    ap.add_argument("--rollback",       action="store_true", default=False,
                    help="If set, roll back to best cheatsheet when training acc drops")
    ap.add_argument("--min-pool-for-net-gate", type=int, default=4,
                    help="Skip net-score gate for bins whose correct pool is smaller than this")
    ap.add_argument("--beam-size",          type=int,   default=2,
                    help="Holistic rewrite beam size: 1=A only, 2=A+A2(t=0.3), 3=A+A2+A3(t=0.5)…")
    ap.add_argument("--val-split",          type=float, default=0.0,
                    help="Fraction of training items held out for val acceptance gating (0=disabled)")
    ap.add_argument("--val-seed",           type=int,   default=42,
                    help="RNG seed for val/opt split")
    ap.add_argument("--val-gate-threshold", type=float, default=0.0,
                    help="Absolute val accuracy floor for acceptance (0 = relative comparison)")
    ap.add_argument("--fix-rate-escalation", type=float, default=0.0,
                    help="Amount added to fix-rate threshold each iteration (e.g. 0.02)")
    ap.add_argument("--bin-threshold-escalation", type=int, default=0,
                    help="Amount added to bin-threshold each iteration")
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help="Stop after N consecutive iterations with no new best (0 = disabled)")
    ap.add_argument("--bin-max-tokens",      type=int, default=900,
                    help="Max tokens for per-bin rule/example generation")
    ap.add_argument("--rewriter-max-tokens", type=int, default=3000,
                    help="Max tokens for holistic cheatsheet rewrite")
    ap.add_argument("--rewriter-cs-max-chars", type=int, default=4000,
                    help="Max chars of current cheatsheet fed into the rewriter prompt")
    ap.add_argument("--secondary-model",     default=None,
                    help="If set, apply paired delta gate using this model at bin evaluation")
    ap.add_argument("--secondary-tolerance", type=int, default=1,
                    help="Max allowed accuracy drop (items) on secondary model per bin (default 1)")
    ap.add_argument("--slowandsteady",        action="store_true", default=False,
                    help="Rewriter selects at most 3 candidates per iteration; rest deferred to next iter")
    ap.add_argument("--rewrite-min-fix",     type=int, default=3,
                    help="Min wrong items the rewrite must fix to pass gate; retries if below (0=disabled)")
    ap.add_argument("--rewrite-gate-retries",type=int, default=3,
                    help="Max rewrite attempts before accepting best seen (used when rewrite-min-fix > 0)")
    ap.add_argument("--rewrite-max-broken",   type=int, default=-1,
                    help="Max correct items rewrite may break; triggers retry + feeds broken cases as caution (-1=disabled)")
    ap.add_argument("--rewrite-min-net-gain", type=int, default=-999,
                    help="Min (n_fixed - n_broken) required to pass gate; triggers retry with caution (-999=disabled)")
    ap.add_argument("--no-oracle-injection", action="store_true", default=False,
                    help="Disable injection of correct reasoning into the bin generator context")
    args = ap.parse_args()

    api_key    = get_api_key()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_items = load_jsonl(args.dataset)
    print(f"[pipeline] Dataset : {len(train_items)} items from {args.dataset}")

    oracle: dict = {}
    if args.oracle:
        oracle = load_oracle_csv(args.oracle)
        print(f"[pipeline] Oracle  : {len(oracle)} entries from {args.oracle}")
    else:
        has_inline = sum(1 for it in train_items if it.get("reason") or it.get("oracle_reasoning"))
        if has_inline:
            print(f"[pipeline] Oracle  : inline ({has_inline}/{len(train_items)} items have 'reason' field)")
        else:
            print("[pipeline] Oracle  : none — no oracle CSV and no inline 'reason' field in dataset")

    task_spec = _load_task_spec(args.task)
    if task_spec:
        print(f"[pipeline] Task    : {args.task}")

    # ── Initial cheatsheet ────────────────────────────────────────────────────
    init_cs_path = output_dir / "cheatsheet_iter0.txt"

    if args.init_cheatsheet:
        initial_cs = Path(args.init_cheatsheet).read_text(encoding="utf-8").strip()
        print(f"[pipeline] Init CS : {args.init_cheatsheet} ({len(initial_cs):,} chars)")
        # Write to iter0 slot so the loop's save is consistent
        init_cs_path.write_text(initial_cs, encoding="utf-8")

    elif init_cs_path.exists():
        initial_cs = init_cs_path.read_text(encoding="utf-8").strip()
        print(f"[pipeline] Init CS : using cached {init_cs_path} ({len(initial_cs):,} chars)")

    else:
        # Try to load from the CS_ICL_Initial_Prompt folder before generating.
        # Folder candidates: CS_ICL_Initial_Prompt/{task} and CS_ICL_Initial_Prompt/bbh_{task}
        _cs_icl_root = Path(__file__).parent.parent / "CS_ICL_Initial_Prompt"
        _cs_icl_dir  = None
        for _candidate in (args.task, f"bbh_{args.task}"):
            _d = _cs_icl_root / _candidate
            if _d.is_dir():
                _cs_icl_dir = _d
                break

        if _cs_icl_dir is not None:
            # Filter by seed, then by token count
            _seed_str = str(args.cs_icl_seed)
            _cs_files = [
                p for p in _cs_icl_dir.glob("*.txt")
                if p.stem.split("_")[-2] == _seed_str        # gen_model_<seed>_tokens
            ]
            if not _cs_files:
                # Fall back: any file in the dir (seed may not match exactly)
                _cs_files = list(_cs_icl_dir.glob("*.txt"))
            if not _cs_files:
                print(f"[pipeline] ERROR: CS_ICL_Initial_Prompt/{_cs_icl_dir.name}/ exists but is empty.",
                      file=sys.stderr)
                sys.exit(1)
            if args.cs_icl_tokens is not None:
                _tok_str = str(args.cs_icl_tokens)
                _match = [p for p in _cs_files if p.stem.split("_")[-1] == _tok_str]
                if not _match:
                    print(f"[pipeline] WARNING: no file with tokens={args.cs_icl_tokens} "
                          f"in {_cs_icl_dir.name}/, using highest available.", file=sys.stderr)
                else:
                    _cs_files = _match
            # Among remaining candidates, pick the one with the highest token count
            _cs_files.sort(
                key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0,
                reverse=True,
            )
            _chosen = _cs_files[0]
            initial_cs = _chosen.read_text(encoding="utf-8").strip()
            init_cs_path.write_text(initial_cs, encoding="utf-8")
            print(f"[pipeline] Init CS : {_chosen} ({len(initial_cs):,} chars)")
        else:
            # No pre-built cheatsheet folder — fall back to LLM generation.
            initial_cs = generate_initial_cheatsheet(
                train_items, oracle, args.model_gen, api_key, args.init_max_tokens,
                task_spec=task_spec,
            )
            init_cs_path.write_text(initial_cs, encoding="utf-8")
            print(f"[pipeline] Init CS : generated ({len(initial_cs):,} chars) → {init_cs_path}")

    # ── Run loop ──────────────────────────────────────────────────────────────
    cfg = HolisticLoopConfig(
        model_score=args.model_score,
        model_gen=args.model_gen,
        api_key=api_key,
        output_dir=output_dir,
        task_spec=task_spec,
        oracle=oracle,
        max_iters=args.max_iters,
        bin_threshold=args.bin_threshold,
        fix_rate_threshold=args.fix_rate,
        regression_pool_size=args.regression_pool,
        regression_pool_fraction=args.regression_pool_fraction,
        score_concurrency=args.concurrency,
        bin_concurrency=args.bin_concurrency,
        rollback_to_best=args.rollback,
        bin_retry=args.bin_retry,
        min_pool_for_net_gate=args.min_pool_for_net_gate,
        beam_size=args.beam_size,
        fix_rate_escalation=args.fix_rate_escalation,
        bin_threshold_escalation=args.bin_threshold_escalation,
        early_stop_patience=args.early_stop_patience,
        val_split=args.val_split,
        val_seed=args.val_seed,
        val_gate_threshold=args.val_gate_threshold,
        bin_max_tokens=args.bin_max_tokens,
        rewriter_max_tokens=args.rewriter_max_tokens,
        rewriter_cs_max_chars=args.rewriter_cs_max_chars,
        no_oracle_injection=args.no_oracle_injection,
        secondary_model=args.secondary_model,
        secondary_tolerance=args.secondary_tolerance,
        slowandsteady=args.slowandsteady,
        rewrite_min_fix=args.rewrite_min_fix,
        rewrite_gate_retries=args.rewrite_gate_retries,
        rewrite_max_broken=args.rewrite_max_broken,
        rewrite_min_net_gain=args.rewrite_min_net_gain,
    )

    result = run_holistic_loop(
        initial_cheatsheet=initial_cs,
        train_items=train_items,
        cfg=cfg,
    )

    print(f"\n{'=' * 60}")
    print(f"ICR_holistic complete")
    print(f"  Iterations     : {result.n_iters}")
    print(f"  Final accuracy : {result.final_accuracy:.1%}  (on training set)")
    print(f"  Best accuracy  : {result.best_accuracy:.1%}  (iter {result.best_iter})"
          f"  → cheatsheet_best.txt")
    print(f"  Output dir     : {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
