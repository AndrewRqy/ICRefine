"""
generate_reasons.py — Generate CS-ICL style reasoning for training items.

Mirrors the cheat-sheet-icl generate_reason_api.py format exactly:
  META_HEADER + metaprompt + SEP + "Question: {input}\nAnswer: {answer}\nExplanation:"

Writes generated reasons back into the training JSONL in-place.
Items that already have a non-empty reason are skipped unless --overwrite is set.

Usage:
    python3 generate_reasons.py --tasks web_of_lies snarks formal_fallacies disambiguation_qa
    python3 generate_reasons.py --tasks web_of_lies --overwrite --concurrency 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / "ICR_partition" / ".env")

from utils.llm_client import get_api_key, call_llm_batch

META_HEADER = (
    "Given a question and its answer, provide a concise explanation of how the answer was "
    "derived. Follow the examples below.\n\n"
)
SEP = "\n###\n"

METAPROMPT_DIR = Path(__file__).parent.parent / "cheat-sheet-icl" / "data" / "metaprompt"

TASK_CFG = {
    "web_of_lies": {
        "train_file": "datasets/bbh/web_of_lies_train.jsonl",
        "metaprompt": "web_of_lies.txt",
    },
    "snarks": {
        "train_file": "datasets/bbh/snarks_train.jsonl",
        "metaprompt": "snarks.txt",
    },
    "formal_fallacies": {
        "train_file": "datasets/bbh/formal_fallacies_train.jsonl",
        "metaprompt": "formal_fallacies.txt",
    },
    "disambiguation_qa": {
        "train_file": "datasets/bbh/disambiguation_qa_train.jsonl",
        "metaprompt": "disambiguation_qa.txt",
    },
    "date_understanding": {
        "train_file": "datasets/bbh/date_understanding_train.jsonl",
        "metaprompt": "date_understanding.txt",
    },
}

DEFAULT_MODEL = "openai/gpt-4.1"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_jsonl(path: Path, items: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(it) for it in items), encoding="utf-8")


def build_prompt(metaprompt: str, item: dict) -> str:
    # Use input as-is: WOL inputs already start with "Question:", others don't.
    # Metaprompt examples are written to match each task's actual input format.
    q = item["input"].strip()
    a = item["answer"].strip()
    example = f"{q}\nAnswer: {a}\nExplanation:"
    return META_HEADER + metaprompt + SEP + example


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks",       nargs="+", default=list(TASK_CFG))
    ap.add_argument("--model",       default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=30)
    ap.add_argument("--max-tokens",  type=int, default=512)
    ap.add_argument("--overwrite",   action="store_true",
                    help="Re-generate even for items that already have a reason")
    args = ap.parse_args()

    api_key = get_api_key()

    for task_name in args.tasks:
        if task_name not in TASK_CFG:
            print(f"Unknown task: {task_name}", file=sys.stderr)
            continue

        cfg = TASK_CFG[task_name]
        train_path = Path(cfg["train_file"])
        metaprompt_path = METAPROMPT_DIR / cfg["metaprompt"]

        metaprompt = metaprompt_path.read_text(encoding="utf-8").strip()
        items = load_jsonl(train_path)

        needs_reason = [
            i for i, it in enumerate(items)
            if args.overwrite or not it.get("reason", "").strip()
        ]

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Task: {task_name}  ({len(needs_reason)}/{len(items)} items need reasons)", file=sys.stderr)

        if not needs_reason:
            print("  Nothing to do.", file=sys.stderr)
            continue

        prompts = [build_prompt(metaprompt, items[i]) for i in needs_reason]
        responses = call_llm_batch(
            prompts,
            model=args.model,
            api_key=api_key,
            temperature=0.0,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            progress_label=task_name,
            reasoning_effort=None,
        )

        n_ok = 0
        for idx, resp in zip(needs_reason, responses):
            if resp is None:
                print(f"  [warn] item {idx} failed — reason left empty", file=sys.stderr)
                continue
            reason = resp.content.strip()
            if reason.startswith("Explanation:"):
                reason = reason[len("Explanation:"):].strip()
            items[idx]["reason"] = reason
            n_ok += 1

        save_jsonl(train_path, items)
        print(f"  Generated {n_ok}/{len(needs_reason)} reasons → {train_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
