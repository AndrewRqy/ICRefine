"""
icr_holistic_v7/pipeline.py — CLI for the original holistic v7 pipeline.

Usage:
    python -m icr_holistic_v7.pipeline \\
        --task causal_judgement \\
        --dataset datasets/bbh/causal_judgement_train_labeled.jsonl \\
        --init-cheatsheet CS_ICL_Initial_Prompt/bbh_causal_judgement/gen_gpt-4.1-mini_0_1000.txt \\
        --output-dir ICR_paper_ready/holistic_cj_v7_1000 \\
        --rollback --beam-size 2
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

from ICR_holistic.prompts import CSICL_GENERATION_PROMPT
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


def generate_initial_cheatsheet(
    train_items: list[dict],
    oracle: dict,
    model: str,
    api_key: str,
    max_tokens: int = 4000,
    task_spec=None,
) -> str:
    def _fmt(item, _oracle):
        eq1 = item.get("equation1", "?")
        eq2 = item.get("equation2", "?")
        answer = item.get("answer", False)
        ans = "True (implies)" if answer else "False (does not imply)"
        question = f'Does "{eq1}" imply "{eq2}"?'
        reasoning = _oracle.get(item.get("id", ""), "")
        if reasoning:
            return f"Question: {question}\nReasoning: {reasoning}\nAnswer: {ans}"
        return f"Question: {question}\nAnswer: {ans}"

    fmt_fn = (
        task_spec.format_for_csicl
        if task_spec is not None and task_spec.format_for_csicl is not None
        else _fmt
    )
    formatted = [fmt_fn(item, oracle) for item in train_items]
    dataset_str = "\n\n".join(formatted)
    prompt = CSICL_GENERATION_PROMPT.format(dataset_str=dataset_str)
    print(f"[init] Generating CS-ICL cheatsheet from {len(train_items)} items "
          f"({len(prompt):,} chars prompt)...")
    resp = call_llm(
        prompt, model=model, api_key=api_key,
        max_tokens=max_tokens, temperature=0.0, reasoning_effort=None,
    )
    return resp.content.strip()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="icr_holistic_v7 — original holistic v7 pipeline (dual beam + val gate)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--task",           default="magma")
    ap.add_argument("--dataset",        required=True)
    ap.add_argument("--oracle",         default=None)
    ap.add_argument("--model-score",    default="openai/gpt-4.1-mini")
    ap.add_argument("--model-gen",      default="openai/gpt-4.1-mini")
    ap.add_argument("--output-dir",     default="runs/holistic_v7")
    ap.add_argument("--max-iters",      type=int,   default=8)
    ap.add_argument("--bin-threshold",  type=int,   default=2)
    ap.add_argument("--fix-rate",       type=float, default=0.10)
    ap.add_argument("--regression-pool",type=int,   default=100)
    ap.add_argument("--regression-pool-fraction", type=float, default=0.10)
    ap.add_argument("--concurrency",    type=int,   default=50)
    ap.add_argument("--bin-concurrency",type=int,   default=4)
    ap.add_argument("--init-cheatsheet",default=None,
                    help="Path to initial cheatsheet (skips CS_ICL_Initial_Prompt lookup)")
    ap.add_argument("--init-max-tokens",type=int,   default=4000)
    ap.add_argument("--cs-icl-seed",    type=int,   default=0)
    ap.add_argument("--cs-icl-tokens",  type=int,   default=None)
    ap.add_argument("--bin-retry",      type=int,   default=2)
    ap.add_argument("--rollback",       action="store_true", default=False)
    ap.add_argument("--min-pool-for-net-gate", type=int, default=4)
    ap.add_argument("--early-stop-patience",   type=int, default=0)
    ap.add_argument("--beam-size",      type=int,   default=2,
                    help="1=A only (standard prompt), 2=A+B (standard + conservative)")
    ap.add_argument("--val-split",      type=float, default=0.0,
                    help="Fraction held out for val gate (0=disabled)")
    ap.add_argument("--val-seed",       type=int,   default=42)
    ap.add_argument("--val-gate-threshold", type=float, default=0.0,
                    help="Absolute val accuracy floor (0=relative: candidate > current)")
    ap.add_argument("--fix-rate-escalation",  type=float, default=0.0)
    ap.add_argument("--bin-threshold-escalation", type=int, default=0)
    ap.add_argument("--oracle-rewrite-injection", action="store_true", default=False,
                    help="Inject oracle reasoning + wrong CoT into rewriter regression block")
    args = ap.parse_args()

    api_key    = get_api_key()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_items = load_jsonl(args.dataset)
    print(f"[pipeline] Dataset : {len(train_items)} items from {args.dataset}")

    oracle: dict = {}
    if args.oracle:
        oracle = load_oracle_csv(args.oracle)
        print(f"[pipeline] Oracle  : {len(oracle)} entries from {args.oracle}")
    else:
        has_inline = sum(1 for it in train_items if it.get("reason") or it.get("oracle_reasoning"))
        if has_inline:
            print(f"[pipeline] Oracle  : inline ({has_inline}/{len(train_items)} items)")
        else:
            print("[pipeline] Oracle  : none")

    task_spec = _load_task_spec(args.task)
    if task_spec:
        print(f"[pipeline] Task    : {args.task}")

    # ── Initial cheatsheet ────────────────────────────────────────────────────
    init_cs_path = output_dir / "cheatsheet_iter0.txt"

    if args.init_cheatsheet:
        initial_cs = Path(args.init_cheatsheet).read_text(encoding="utf-8").strip()
        print(f"[pipeline] Init CS : {args.init_cheatsheet} ({len(initial_cs):,} chars)")
        init_cs_path.write_text(initial_cs, encoding="utf-8")

    elif init_cs_path.exists():
        initial_cs = init_cs_path.read_text(encoding="utf-8").strip()
        print(f"[pipeline] Init CS : cached {init_cs_path} ({len(initial_cs):,} chars)")

    else:
        _cs_icl_root = Path(__file__).parent.parent / "CS_ICL_Initial_Prompt"
        _cs_icl_dir  = None
        for _candidate in (args.task, f"bbh_{args.task}", f"agieval_{args.task}"):
            _d = _cs_icl_root / _candidate
            if _d.is_dir():
                _cs_icl_dir = _d
                break

        if _cs_icl_dir is not None:
            _seed_str = str(args.cs_icl_seed)
            _cs_files = [
                p for p in _cs_icl_dir.glob("*.txt")
                if p.stem.split("_")[-2] == _seed_str
            ]
            if not _cs_files:
                _cs_files = list(_cs_icl_dir.glob("*.txt"))
            if not _cs_files:
                print(f"[pipeline] ERROR: {_cs_icl_dir} exists but is empty.", file=sys.stderr)
                sys.exit(1)
            if args.cs_icl_tokens is not None:
                _tok_str = str(args.cs_icl_tokens)
                _match = [p for p in _cs_files if p.stem.split("_")[-1] == _tok_str]
                if _match:
                    _cs_files = _match
            _cs_files.sort(
                key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0,
                reverse=True,
            )
            _chosen = _cs_files[0]
            initial_cs = _chosen.read_text(encoding="utf-8").strip()
            init_cs_path.write_text(initial_cs, encoding="utf-8")
            print(f"[pipeline] Init CS : {_chosen} ({len(initial_cs):,} chars)")
        else:
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
        early_stop_patience=args.early_stop_patience,
        beam_size=args.beam_size,
        oracle_rewrite_injection=args.oracle_rewrite_injection,
        val_split=args.val_split,
        val_seed=args.val_seed,
        val_gate_threshold=args.val_gate_threshold,
        fix_rate_escalation=args.fix_rate_escalation,
        bin_threshold_escalation=args.bin_threshold_escalation,
    )

    result = run_holistic_loop(
        initial_cheatsheet=initial_cs,
        train_items=train_items,
        cfg=cfg,
    )

    print(f"\n{'=' * 60}")
    print(f"icr_holistic_v7 complete")
    print(f"  Iterations     : {result.n_iters}")
    print(f"  Final accuracy : {result.final_accuracy:.1%}")
    print(f"  Best accuracy  : {result.best_accuracy:.1%}  (iter {result.best_iter})"
          f"  → cheatsheet_best.txt")
    print(f"  Output dir     : {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
