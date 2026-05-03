#!/usr/bin/env python3
"""
gen_csicl_magma.py — Prepare a balanced train/test split from hard2.jsonl,
generate LLM reasoning for each training item, then create a CS-ICL cheat
sheet using the same prompt as the CS-ICL paper.

Steps:
  1. Split hard2.jsonl → datasets/magma_train.jsonl + datasets/magma_test.jsonl
  2. For each train item, call the LLM for a reasoning chain
  3. Build the CS-ICL cheat-sheet generation prompt from the reasoned examples
  4. Call the LLM once more to produce the cheat sheet
  5. Save to ../cheat-sheet-icl/data/cheat_prompt/magma/gen_<model>_0_1000.txt

The cheat sheet can then be scored by eval_magma_comparison.py.

Usage:
    python3 gen_csicl_magma.py [--model openai/gpt-4.1-mini]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / "ICR_partition" / ".env")

from utils.llm_client import call_llm, get_api_key  # noqa: E402

TRAIN_PATH    = Path("datasets/magma_train.jsonl")
TEST_PATH     = Path("datasets/magma_test.jsonl")
ORACLE_CSV    = Path("datasets/hard3_gpt54_oracle.csv")   # GPT-5.4, correct only
CSICL_OUT_DIR = Path("../cheat-sheet-icl/data/cheat_prompt/magma")

_CHEAT_PROMPT = (
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


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def make_split() -> tuple[list[dict], list[dict]]:
    items = load_jsonl(HARD2_PATH)
    trues  = [it for it in items if it["answer"] is True]
    falses = [it for it in items if it["answer"] is False]
    train = sorted(trues[:75] + falses[:75],  key=lambda x: x["index"])
    test  = sorted(trues[75:] + falses[75:],  key=lambda x: x["index"])
    write_jsonl(train, TRAIN_PATH)
    write_jsonl(test,  TEST_PATH)
    print(f"Split: train={len(train)} (75T/75F)  test={len(test)} (25T/25F)")
    print(f"  → {TRAIN_PATH}  {TEST_PATH}")
    return train, test


def load_oracle_reasons(train: list[dict]) -> list[str]:
    """Load GPT-5.4 oracle reasoning for each train item (all items guaranteed correct)."""
    import csv as _csv
    with open(ORACLE_CSV, encoding="utf-8") as f:
        oracle_rows = list(_csv.DictReader(f))
    oracle_map = {r["problem_id"]: r for r in oracle_rows}
    reasons = []
    used = 0
    for it in train:
        row = oracle_map.get(it["id"])
        if row:
            reasons.append(row.get("response", "").strip())
            used += 1
        else:
            reasons.append("")
    print(f"Oracle reasoning loaded: {used}/{len(train)} items (GPT-5.4, all correct)")
    return reasons


def format_item(item: dict, reason: str) -> str:
    ans_str = "True (implies)" if item["answer"] else "False (does not imply)"
    q = f'Does "{item["equation1"]}" imply "{item["equation2"]}"?'
    if reason.strip():
        return f"Question: {q}\nReasoning: {reason.strip()}\nAnswer: {ans_str}"
    return f"Question: {q}\nAnswer: {ans_str}"


def generate_cheatsheet(
    train: list[dict],
    reasons: list[str],
    model: str,
    api_key: str,
    max_tokens: int,
) -> str:
    dataset_str = "\n\n".join(format_item(it, r) for it, r in zip(train, reasons))
    prompt = _CHEAT_PROMPT.format(dataset_str=dataset_str)
    print(f"\nGenerating cheat sheet (prompt ~{len(prompt)} chars) ...")
    resp = call_llm(prompt, model=model, api_key=api_key, max_tokens=max_tokens, temperature=0.0)
    return resp.content


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="openai/gpt-4.1-mini")
    p.add_argument("--max-tokens",  type=int, default=4000)
    p.add_argument("--skip-split",  action="store_true",
                   help="Skip writing split files (use existing)")
    args = p.parse_args()

    api_key = get_api_key()

    # Split is pre-built (hard3 GPT-5.4 correct subset); just load it
    train = load_jsonl(TRAIN_PATH)
    print(f"Loaded train split: {len(train)} items")

    # Use oracle reasoning chains (from c21c__gemma-4-31b-it, 92.5% accurate on hard2)
    reasons   = load_oracle_reasons(train)
    cs_text   = generate_cheatsheet(train, reasons, args.model, api_key, args.max_tokens)

    model_short = args.model.split("/")[-1]
    outname     = f"gen_{model_short}_0_1000.txt"
    CSICL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CSICL_OUT_DIR / outname
    out_path.write_text(cs_text, encoding="utf-8")
    print(f"\nCheat sheet saved → {out_path}  ({len(cs_text)} chars)")
    print("\nDone.")


if __name__ == "__main__":
    main()
