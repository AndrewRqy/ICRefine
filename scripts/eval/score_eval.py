"""
score_eval.py — Evaluate a saved cheatsheet on a dataset, with resume support.

Results are streamed to a JSONL cache file as each item completes.
If interrupted and re-run with the same --output, already-scored items
are skipped and scoring resumes from where it left off.

Usage:
    python3 score_eval.py \
        --dataset path/to/dataset.jsonl \
        --cheatsheet runs/my_run/cheatsheet_current \
        --model deepseek-r1-32b \
        --concurrency 50 \
        --no-cot-first \
        --output runs/eval_hard_combined.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.cheatsheet import Cheatsheet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.data import load_jsonl, is_true
from utils.llm_client import get_api_key, call_llm
from utils.scorer import score_items_streaming
from utils.parser import parse_response as _parse, normalize as _normalize

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


def _default_output(dataset: str, cheatsheet: str) -> Path:
    ds   = Path(dataset).stem
    cs   = Path(cheatsheet).stem
    return Path("runs") / f"eval_{ds}_{cs}.jsonl"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate a cheatsheet on a dataset (resumable).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",     required=True, metavar="FILE")
    p.add_argument("--cheatsheet",  required=True, metavar="PATH",
                   help="Path without extension — loads <path>.json if present, else <path>.txt")
    p.add_argument("--output",      default=None,  metavar="FILE",
                   help="JSONL file to stream results into (enables resume). "
                        "Defaults to runs/eval_<dataset>_<cheatsheet>.jsonl")
    p.add_argument("--model",       default="deepseek-r1-32b", metavar="MODEL")
    p.add_argument("--concurrency", type=int, default=50, metavar="N")
    p.add_argument("--reasoning-effort", default="low",
                   choices=["low", "medium", "high", "none"])
    p.add_argument("--cot-first",    action="store_true", default=False)
    p.add_argument("--no-cot-first", dest="cot_first", action="store_false")
    p.add_argument("--no-render-limit", action="store_true", default=False)
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="Evaluate only the first N items.")
    args = p.parse_args()

    api_key          = get_api_key()
    from utils.run_logger import RunLogger, make_run_id, set_logger
    _run_logger = RunLogger(log_base="runs/logs/eval", run_id=make_run_id("score_eval"), config=vars(args))
    set_logger(_run_logger)
    print(f"[log] {_run_logger.log_dir}", file=sys.stderr)
    reasoning_effort = None if args.reasoning_effort == "none" else args.reasoning_effort
    output_path      = Path(args.output) if args.output else _default_output(args.dataset, args.cheatsheet)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load cheatsheet
    # ------------------------------------------------------------------
    cs_path = Path(args.cheatsheet)
    if cs_path.with_suffix(".json").exists():
        cheatsheet = Cheatsheet.load(cs_path)
    else:
        text = cs_path.with_suffix(".txt").read_text(encoding="utf-8")
        cheatsheet = Cheatsheet(roadmap="", prior_knowledge=text)
    if args.no_render_limit:
        cheatsheet.no_limit = True

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    all_items = load_jsonl(Path(args.dataset))
    if args.limit:
        all_items = all_items[: args.limit]

    # ------------------------------------------------------------------
    # Resume: load already-scored items from the output JSONL
    # ------------------------------------------------------------------
    already_scored: dict[str, dict] = {}   # id → scored item
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                already_scored[entry["id"]] = entry
            except Exception:
                pass

    remaining = [it for it in all_items if it["id"] not in already_scored]

    print(
        f"\nEval: {Path(args.dataset).name}  |  {cs_path.name}  |  {args.model}\n"
        f"  total={len(all_items)}  already_scored={len(already_scored)}  "
        f"remaining={len(remaining)}",
        file=sys.stderr,
    )

    if not remaining:
        print("  All items already scored — skipping API calls.", file=sys.stderr)
    else:
        # ------------------------------------------------------------------
        # Stream-score remaining items, writing each result immediately
        # ------------------------------------------------------------------
        cs_text = cheatsheet.render()
        correct_so_far = 0
        with output_path.open("a", encoding="utf-8") as fout:
            pbar = tqdm(total=len(remaining), unit="item", desc="scoring")
            for scored_item in score_items_streaming(
                remaining,
                get_cheatsheet=lambda: cs_text,
                model=args.model,
                api_key=api_key,
                concurrency=args.concurrency,
                reasoning_effort=reasoning_effort,
                cot_first=args.cot_first,
            ):
                fout.write(json.dumps(scored_item, ensure_ascii=False) + "\n")
                fout.flush()
                already_scored[scored_item["id"]] = scored_item
                if scored_item.get("predicted") == scored_item.get("expected"):
                    correct_so_far += 1
                done_so_far = len(already_scored)
                pbar.set_postfix(acc=f"{correct_so_far/done_so_far:.1%}" if done_so_far else "n/a")
                pbar.update(1)
            pbar.close()

    # ------------------------------------------------------------------
    # Final accuracy over all scored items
    # ------------------------------------------------------------------
    correct = [it for it in already_scored.values() if it.get("predicted") == it.get("expected")]
    wrong   = [it for it in already_scored.values() if it.get("predicted") != it.get("expected")]
    total   = len(already_scored)
    accuracy = len(correct) / total if total else 0.0
    parse_errors = sum(1 for it in already_scored.values() if it.get("predicted") is None)

    print(f"\n{'='*50}")
    print(f"  Dataset   : {Path(args.dataset).name}  ({total} items)")
    print(f"  Cheatsheet: {cs_path.name}")
    print(f"  Model     : {args.model}")
    print(f"  Accuracy  : {accuracy:.1%}  ({len(correct)}/{total})")
    print(f"  Wrong     : {len(wrong)}")
    if parse_errors:
        print(f"  Parse err : {parse_errors}")
    print(f"  Results   : {output_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
