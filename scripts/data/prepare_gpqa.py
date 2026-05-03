"""prepare_gpqa.py — Convert GPQA Diamond HF dataset to BBH-compatible jsonl.

Shuffles the four answer choices into (A)–(D) positions with a fixed seed,
tracks the correct letter, and writes train (100 items) / test (98 items).

Usage:
    HF_TOKEN=hf_... python3 scripts/data/prepare_gpqa.py \
        --out-dir datasets/gpqa_diamond
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build_item(row: dict, idx: int, rng: random.Random) -> dict:
    question = row["Question"].strip()
    correct  = row["Correct Answer"].strip()
    wrongs   = [
        row["Incorrect Answer 1"].strip(),
        row["Incorrect Answer 2"].strip(),
        row["Incorrect Answer 3"].strip(),
    ]
    choices = [correct] + wrongs
    rng.shuffle(choices)

    letters = ["(A)", "(B)", "(C)", "(D)"]
    correct_letter = letters[choices.index(correct)]

    options_block = "\n".join(
        f"{letters[i]} {choices[i]}" for i in range(4)
    )
    input_text = f"{question}\n\nOptions:\n{options_block}"

    return {
        "input":        input_text,
        "answer":       correct_letter,
        "id":           f"gpqa_diamond_{idx:04d}",
        "subdomain":    row.get("Subdomain", ""),
        "domain":       row.get("High-level domain", ""),
        "reason":       row.get("Explanation", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="datasets/gpqa_diamond")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--n-train", type=int, default=100)
    args = ap.parse_args()

    from huggingface_hub import login
    import os
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)

    from datasets import load_dataset
    rows = list(load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"])

    rng_shuffle = random.Random(args.seed)
    rng_shuffle.shuffle(rows)

    rng_choices = random.Random(args.seed + 1)
    items = [build_item(r, i, rng_choices) for i, r in enumerate(rows)]

    train_items = items[:args.n_train]
    test_items  = items[args.n_train:]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, data: list[dict]) -> None:
        path.write_text(
            "\n".join(json.dumps(d) for d in data) + "\n",
            encoding="utf-8",
        )

    write_jsonl(out / "gpqa_diamond_train.jsonl", train_items)
    write_jsonl(out / "gpqa_diamond_test.jsonl",  test_items)

    print(f"Wrote {len(train_items)} train / {len(test_items)} test → {out}/")
    print(f"Domains: { {r['domain'] for r in items} }")

    # Sanity check: answer distribution
    from collections import Counter
    print("Answer dist (train):", Counter(r["answer"] for r in train_items))


if __name__ == "__main__":
    main()
