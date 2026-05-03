#!/usr/bin/env python3
"""
gen_csicl_ext.py — Generate CS-ICL cheat sheets for the 6 extended BBH tasks
using our standard OpenAI API client (not Azure).

Replicates the exact CS-ICL pipeline:
  1. Format training items as  "Question: ...\nReasoning: ...\nAnswer: ..."
  2. Wrap in the CS-ICL cheat-sheet generation prompt
  3. Call the model, write output to cheat-sheet-icl/data/cheat_prompt/{task}/

Output files (one per seed) match the CS-ICL naming convention:
  gen_{model}_{shot}_{seed}.txt  e.g. gen_gpt-4.1_0_1000.txt

Usage:
    python3 gen_csicl_ext.py                        # all 6 tasks, gpt-4.1, seeds 1000/2000/3000
    python3 gen_csicl_ext.py --seeds 1000           # single seed only
    python3 gen_csicl_ext.py --tasks web_of_lies    # single task
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / "ICR_partition" / ".env")

from utils.llm_client import call_llm, get_api_key  # noqa: E402

# ---------------------------------------------------------------------------
# Task config
# ---------------------------------------------------------------------------

TASKS = [
    {
        "name":        "formal_fallacies",
        "cs_icl_name": "bbh_formal_fallacies",
        "train_jsonl": "datasets/bbh/formal_fallacies_train.jsonl",
        "train_n":     150,
    },
    {
        "name":        "logical_deduction_three",
        "cs_icl_name": "bbh_logical_deduction_three_objects",
        "train_jsonl": "datasets/bbh/logical_deduction_three_objects_train.jsonl",
        "train_n":     150,
    },
    {
        "name":        "web_of_lies",
        "cs_icl_name": "bbh_web_of_lies",
        "train_jsonl": "datasets/bbh/web_of_lies_train.jsonl",
        "train_n":     150,
    },
    {
        "name":        "date_understanding",
        "cs_icl_name": "bbh_date_understanding",
        "train_jsonl": "datasets/bbh/date_understanding_train.jsonl",
        "train_n":     150,
    },
    {
        "name":        "navigate",
        "cs_icl_name": "bbh_navigate",
        "train_jsonl": "datasets/bbh/navigate_train.jsonl",
        "train_n":     150,
    },
    {
        "name":        "snarks",
        "cs_icl_name": "bbh_snarks",
        "train_jsonl": "datasets/bbh/snarks_train.jsonl",
        "train_n":     100,   # CS-ICL convention for snarks
    },
]

CSICL_DIR = Path("../cheat-sheet-icl/data/cheat_prompt")

_GEN_PROMPT = (
    "Create a cheat sheet based on the examples below. "
    "You will be asked to answer questions similar to these examples during the test, "
    "without being allowed to refer to the examples at that time. "
    "Your task here is to make a cheat sheet that will help you answer such problems correctly. "
    "First, carefully read the examples below and identify which ones you find most difficult to answer.\n\n"
    "{dataset_str}\n\n"
    "Now, create a cheat sheet to help you solve the difficult examples. "
    "Exclude any content that is easy for you, and only include specific, detailed points "
    "to address the challenging ones.\n\n"
)


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def format_item(item: dict) -> str:
    reason = item.get("reason", "").strip()
    if reason:
        return f"Question: {item['input']}\nReasoning: {reason}\nAnswer: {item['answer']}"
    return f"Question: {item['input']}\nAnswer: {item['answer']}"


_SEP = "\n###\n"


def gen_cheatsheet(task: dict, model: str, api_key: str, max_tokens: int, seed: int) -> str:
    items = load_jsonl(task["train_jsonl"])[: task["train_n"]]
    random.seed(seed)
    random.shuffle(items)
    formatted = [format_item(it) for it in items]
    dataset_str = "\n\n".join(formatted)
    prompt = _GEN_PROMPT.format(dataset_str=dataset_str)

    print(f"  seed={seed}  ({len(items)} train items, prompt ~{len(prompt)} chars) ...", flush=True)
    resp = call_llm(prompt, model=model, api_key=api_key, max_tokens=max_tokens, temperature=0.0)
    cheatsheet = resp.content.strip()

    # Replicate the CS-ICL paper's output format: append two training examples so the
    # target model knows the expected Q/Reasoning/Answer format at inference time.
    show_examples = _SEP.join(formatted[:2])
    return f"{cheatsheet}\n\nFollow the format of the examples below in your response.\n\n{show_examples}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="openai/gpt-4.1")
    p.add_argument("--max-tokens", type=int, default=4000)
    p.add_argument("--seeds",      nargs="+", type=int, default=[1000, 2000, 3000],
                   help="Random seeds to generate (default: 1000 2000 3000)")
    p.add_argument("--tasks",      nargs="+", default=None,
                   help="Subset of task names to generate (default: all 6)")
    args = p.parse_args()

    api_key = get_api_key()
    model_short = args.model.split("/")[-1]

    task_filter = set(args.tasks) if args.tasks else None
    tasks = [t for t in TASKS if task_filter is None or t["name"] in task_filter]

    for task in tasks:
        print(f"\n[{task['name']}]")
        out_dir = CSICL_DIR / task["cs_icl_name"]
        out_dir.mkdir(parents=True, exist_ok=True)

        for seed in args.seeds:
            cs_text = gen_cheatsheet(task, args.model, api_key, args.max_tokens, seed)
            outname = f"gen_{model_short}_0_{seed}.txt"
            out_path = out_dir / outname
            out_path.write_text(cs_text, encoding="utf-8")
            print(f"  Saved → {out_path}  ({len(cs_text)} chars)")

    print("\nDone.")


if __name__ == "__main__":
    main()
