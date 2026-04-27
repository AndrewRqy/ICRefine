"""
oracle_generate.py — Single-pass scoring of a dataset by a high-capability model
to produce an oracle CSV compatible with ICR_partition's --oracle-csv flag.

Only items where the model answers correctly are written to the output CSV.
Results are streamed and the file is appended incrementally, so the script
is resumable: re-running with the same --output skips already-scored items.

Usage:
    python3 oracle_generate.py \
        --dataset datasets/hard3.jsonl \
        --model openai/gpt-5.4 \
        --output gpt5.4_hard3_oracle.csv \
        --concurrency 50 \
        --limit 200
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from utils.data import load_jsonl, is_true
from utils.llm_client import get_api_key, call_llm
from utils.parser import parse_response as _parse
from utils.scorer import _build_scoring_prompt, _render_features_block
from ICR_naive.prompts.templates import SCORING_MAX_TOKENS

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

load_dotenv(Path(__file__).parent / ".env")

_CSV_FIELDS = [
    "problem_id", "equation1", "equation2", "answer",
    "model_id", "response", "correct",
    "prompt_tokens", "completion_tokens", "elapsed_seconds",
]


def _build_prompt(item: dict) -> str:
    return _build_scoring_prompt("(No cheatsheet provided — reason from first principles.)", item)


def _load_completed(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done: set[str] = set()
    with open(output_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("problem_id"):
                done.add(row["problem_id"])
    return done


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate an oracle CSV by scoring a dataset with a high-capability model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",     required=True,  metavar="FILE")
    p.add_argument("--model",       default="openai/gpt-5.4", metavar="MODEL")
    p.add_argument("--output",      default=None,   metavar="FILE",
                   help="Oracle CSV output path. Defaults to gpt5.4_<dataset>.csv")
    p.add_argument("--concurrency", type=int, default=50, metavar="N")
    p.add_argument("--limit",       type=int, default=None, metavar="N",
                   help="Cap number of items to score.")
    p.add_argument("--reasoning-effort", default="low",
                   choices=["low", "medium", "high", "none"])
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    api_key          = get_api_key()
    reasoning_effort = None if args.reasoning_effort == "none" else args.reasoning_effort

    ds_path  = Path(args.dataset)
    out_path = Path(args.output) if args.output else Path(f"gpt5.4_{ds_path.stem}_oracle.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_items = load_jsonl(ds_path)
    if args.limit:
        all_items = all_items[: args.limit]

    completed_ids = _load_completed(out_path)
    pending = [it for it in all_items if it["id"] not in completed_ids]

    total    = len(all_items)
    skipped  = len(completed_ids)
    print(f"Dataset : {ds_path.name}  ({total} items, {skipped} already done, {len(pending)} pending)")
    print(f"Model   : {args.model}")
    print(f"Output  : {out_path}")

    write_header = not out_path.exists() or skipped == 0
    fout = open(out_path, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(fout, fieldnames=_CSV_FIELDS)
    if write_header:
        writer.writeheader()
        fout.flush()

    items_iter = iter(pending)
    pending_futures: dict = {}
    correct_count = 0
    done_count    = skipped

    def _submit(pool: ThreadPoolExecutor) -> bool:
        try:
            item = next(items_iter)
        except StopIteration:
            return False
        prompt = _build_prompt(item)
        f = pool.submit(
            call_llm, prompt, args.model, api_key,
            args.temperature, SCORING_MAX_TOKENS, reasoning_effort, args.seed,
        )
        pending_futures[f] = item
        return True

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for _ in range(args.concurrency):
            if not _submit(pool):
                break

        while pending_futures:
            done_set, _ = wait(pending_futures, return_when=FIRST_COMPLETED)
            for f in done_set:
                item = pending_futures.pop(f)
                ground_truth = is_true(item["answer"])
                try:
                    resp   = f.result()
                    parsed = _parse(resp.content)
                    verdict = (parsed.get("verdict") or "").strip().upper()
                    correct = (verdict == "TRUE") == ground_truth
                    row = {
                        "problem_id":         item["id"],
                        "equation1":          item["equation1"],
                        "equation2":          item["equation2"],
                        "answer":             str(ground_truth),
                        "model_id":           args.model,
                        "response":           resp.content,
                        "correct":            str(correct),
                        "prompt_tokens":      0,
                        "completion_tokens":  0,
                        "elapsed_seconds":    0.0,
                    }
                    writer.writerow(row)
                    fout.flush()
                    if correct:
                        correct_count += 1
                except Exception as exc:
                    print(f"  [warn] {item['id']} failed: {exc}", file=sys.stderr)

                done_count += 1
                _submit(pool)

                if done_count % 20 == 0 or done_count == total:
                    print(f"  {done_count}/{total}  correct so far: {correct_count + skipped}", flush=True)

    fout.close()
    accuracy = (correct_count + skipped) / total if total else 0.0
    print(f"\nDone. Accuracy: {accuracy:.1%}  Oracle entries written: {correct_count}  (+ {skipped} resumed)")
    print(f"Oracle CSV: {out_path}")


if __name__ == "__main__":
    main()
