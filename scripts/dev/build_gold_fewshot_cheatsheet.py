"""
build_gold_fewshot_cheatsheet.py

For each oracle task, constructs a "PK + gold few-shot" cheatsheet by:
  1. Taking cheatsheet_phase1_pk_final.txt (Phase 1 PK text, no case studies)
  2. Appending N gold worked examples from the training set using item["reason"]

This creates a direct comparison against the ACTIVATE IF case studies without
running any new LLM generation.

Usage:
    python3 build_gold_fewshot_cheatsheet.py --n-examples 5
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.data import load_jsonl

ORACLE_TASKS = {
    "boolean_expressions": {
        "train": "datasets/bbh/boolean_expressions_train.jsonl",
        "run_dir": "runs/bbh_v3/boolean_expressions",
        "answer_label": "Answer",
    },
    "causal_judgement": {
        "train": "datasets/bbh/causal_judgement_train.jsonl",
        "run_dir": "runs/bbh_v3/causal_judgement",
        "answer_label": "Answer",
    },
    "disambiguation_qa": {
        "train": "datasets/bbh/disambiguation_qa_train.jsonl",
        "run_dir": "runs/bbh_v3/disambiguation_qa",
        "answer_label": "Answer",
    },
    "geometric_shapes": {
        "train": "datasets/bbh/geometric_shapes_train.jsonl",
        "run_dir": "runs/bbh_v3/geometric_shapes",
        "answer_label": "Answer",
    },
    "logical_deduction_three": {
        "train": "datasets/bbh/logical_deduction_three_objects_train.jsonl",
        "run_dir": "runs/bbh_v3/logical_deduction_three",
        "answer_label": "Answer",
    },
    "sports_understanding": {
        "train": "datasets/bbh/sports_understanding_train.jsonl",
        "run_dir": "runs/bbh_v3/sports_understanding",
        "answer_label": "Answer",
    },
    "web_of_lies": {
        "train": "datasets/bbh/web_of_lies_train.jsonl",
        "run_dir": "runs/bbh_v3/web_of_lies",
        "answer_label": "Answer",
    },
    "formal_fallacies": {
        "train": "datasets/bbh/formal_fallacies_train.jsonl",
        "run_dir": "runs/bbh_v3/formal_fallacies",
        "answer_label": "Answer",
    },
    "snarks": {
        "train": "datasets/bbh/snarks_train.jsonl",
        "run_dir": "runs/bbh_v3/snarks",
        "answer_label": "Answer",
    },
    "date_understanding": {
        "train": "datasets/bbh/date_understanding_train.jsonl",
        "run_dir": "runs/bbh_v3/date_understanding",
        "answer_label": "Answer",
    },
}


def build_gold_fewshot(pk_text: str, examples: list[dict], n: int) -> str:
    oracle_items = [x for x in examples if x.get("reason")]
    random.shuffle(oracle_items)
    selected = oracle_items[:n]

    lines = [pk_text.strip(), "", "=== WORKED EXAMPLES ===", ""]
    for i, item in enumerate(selected, 1):
        lines.append(f"--- Example {i} ---")
        lines.append(f"Question: {item['input'].strip()}")
        lines.append(f"Reasoning: {item['reason'].strip()}")
        lines.append(f"Answer: {item['answer'].strip()}")
        lines.append("")

    return "\n".join(lines).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks",      nargs="+", default=list(ORACLE_TASKS))
    ap.add_argument("--n-examples", type=int,  default=5)
    ap.add_argument("--seed",       type=int,  default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    for task_name in args.tasks:
        if task_name not in ORACLE_TASKS:
            print(f"Unknown task: {task_name}")
            continue

        cfg     = ORACLE_TASKS[task_name]
        run_dir = Path(cfg["run_dir"])
        pk_path = run_dir / "cheatsheet_phase1_pk_final.txt"

        if not pk_path.exists():
            print(f"[{task_name}] pk_final not found, skipping")
            continue

        pk_text = pk_path.read_text(encoding="utf-8")
        items   = load_jsonl(Path(cfg["train"]))
        cs_text = build_gold_fewshot(pk_text, items, args.n_examples)

        out_path = run_dir / "cheatsheet_gold_fewshot.txt"
        out_path.write_text(cs_text, encoding="utf-8")

        n_oracle = sum(1 for x in items if x.get("reason"))
        print(f"[{task_name}] {n_oracle} oracle items → {args.n_examples} selected  "
              f"({len(cs_text):,} chars) → {out_path}")


if __name__ == "__main__":
    main()
