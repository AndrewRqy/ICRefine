#!/usr/bin/env python3
"""
download_bbh_tasks.py — Download additional BBH task data and format it to
match our existing datasets/bbh/ convention.

Source:
  Examples  : https://github.com/suzgunmirac/BIG-Bench-Hard (bbh/*.json)
  CoT chains: https://github.com/suzgunmirac/BIG-Bench-Hard (cot-prompts/*.txt)

Output format per item:
  { "input": str, "answer": str, "reason": str, "id": str }

Split: first 150 items → _train.jsonl, remaining up to 100 → _test.jsonl.
For tasks with fewer than 250 examples all items are used (train takes 60%).
"""

from __future__ import annotations

import json
import math
import re
import sys
import urllib.request
from pathlib import Path

OUT_DIR = Path("datasets/bbh")

TASKS = [
    "formal_fallacies",
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "web_of_lies",
    "date_understanding",
    "navigate",
    "snarks",
    "tracking_shuffled_objects_three_objects",
    "penguins_in_a_table",
    "object_counting",
    "reasoning_about_colored_objects",
    "temporal_sequences",
]

BASE_EXAMPLES = (
    "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard"
    "/main/bbh/{task}.json"
)
BASE_COT = (
    "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard"
    "/main/cot-prompts/{task}.txt"
)


def fetch(url: str) -> str:
    print(f"  GET {url}", flush=True)
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8")


def parse_cot_file(text: str) -> dict[str, str]:
    """
    Parse the few-shot CoT prompt file to extract per-question reasoning.
    Format: Q: <question>\nA: Let's think step by step. <chain>\n\n
    Returns {question_snippet: reasoning}.
    """
    chains: dict[str, str] = {}
    blocks = re.split(r"\n\nQ:", text)
    for block in blocks[1:]:       # first block is the task preamble
        lines = block.strip().split("\n")
        q_line = lines[0].strip()
        a_lines = []
        for line in lines[1:]:
            if line.startswith("A:"):
                a_lines.append(line[2:].strip())
            elif a_lines:
                a_lines.append(line)
        reason = " ".join(a_lines).strip()
        chains[q_line[:80]] = reason   # key = first 80 chars of question
    return chains


def download_task(task: str) -> None:
    out_train = OUT_DIR / f"{task}_train.jsonl"
    out_test  = OUT_DIR / f"{task}_test.jsonl"

    if out_train.exists() and out_test.exists():
        print(f"  [{task}] already exists — skipping.")
        return

    # ── Fetch examples ────────────────────────────────────────────────────────
    try:
        raw = fetch(BASE_EXAMPLES.format(task=task))
    except Exception as exc:
        print(f"  [{task}] ERROR fetching examples: {exc}")
        return

    data = json.loads(raw)
    task_prefix = data.get("task_prefix", "").strip()
    examples    = data["examples"]

    # ── Fetch CoT chains (best-effort) ────────────────────────────────────────
    cot_map: dict[str, str] = {}
    try:
        cot_text = fetch(BASE_COT.format(task=task))
        cot_map  = parse_cot_file(cot_text)
    except Exception:
        print(f"  [{task}] CoT file not found — reason field will be empty.")

    # ── Build items ───────────────────────────────────────────────────────────
    items = []
    for i, ex in enumerate(examples):
        raw_input = ex["input"].strip()
        answer    = ex.get("target", ex.get("answer", "")).strip()

        # Prepend task prefix if not already present in input
        if task_prefix and task_prefix[:30] not in raw_input:
            full_input = task_prefix + "\n\n" + raw_input
        else:
            full_input = raw_input

        # Look up CoT chain by question snippet
        snippet = raw_input[:80]
        reason  = cot_map.get(snippet, "")

        items.append({
            "input":  full_input,
            "answer": answer,
            "reason": reason,
            "id":     f"{task}_{i:04d}",
        })

    # ── Split ─────────────────────────────────────────────────────────────────
    n = len(items)
    if n >= 250:
        n_train, n_test = 150, 100
    else:
        n_train = math.ceil(n * 0.6)
        n_test  = n - n_train

    train_items = items[:n_train]
    test_items  = items[n_train: n_train + n_test]

    # ── Write ─────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(out_train, "w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(out_test, "w", encoding="utf-8") as f:
        for item in test_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f"  [{task}] {n} examples → "
        f"train={len(train_items)}  test={len(test_items)}  "
        f"cot_coverage={len(cot_map)} chains"
    )


def main() -> None:
    tasks = sys.argv[1:] if len(sys.argv) > 1 else TASKS
    print(f"Downloading {len(tasks)} BBH tasks to {OUT_DIR}/\n")
    for task in tasks:
        print(f"\n[{task}]")
        download_task(task)
    print("\nDone.")


if __name__ == "__main__":
    main()
