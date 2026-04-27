#!/usr/bin/env python3
"""
gen_icr_bootstrap.py — Generate a CS-ICL style bootstrap cheatsheet for ICR.

For each task, format the first --bootstrap-n training items as Q/A pairs,
generate a CS-ICL style cheat sheet, and save it as a Cheatsheet JSON that
can be passed to the pipeline via --init-cheatsheet.

The resulting JSON seeds Phase 2 with a general-knowledge roadmap derived
from examples, so ICR adds targeted case studies on top of an informed baseline
rather than starting from a blank roadmap.

Usage:
    python3 gen_icr_bootstrap.py [--model openai/gpt-4.1-mini] [--bootstrap-n 75]
    python3 gen_icr_bootstrap.py --tasks formal_fallacies snarks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "ICR_partition" / ".env")

from utils.llm_client import call_llm, get_api_key  # noqa: E402
from utils.cheatsheet import Cheatsheet             # noqa: E402

# ---------------------------------------------------------------------------
# Task config
# ---------------------------------------------------------------------------

TASKS = [
    {
        "name":        "formal_fallacies",
        "train_jsonl": "datasets/bbh/formal_fallacies_train.jsonl",
        "bootstrap_n": 75,
    },
    {
        "name":        "logical_deduction_three",
        "train_jsonl": "datasets/bbh/logical_deduction_three_objects_train.jsonl",
        "bootstrap_n": 75,
    },
    {
        "name":        "web_of_lies",
        "train_jsonl": "datasets/bbh/web_of_lies_train.jsonl",
        "bootstrap_n": 75,
    },
    {
        "name":        "date_understanding",
        "train_jsonl": "datasets/bbh/date_understanding_train.jsonl",
        "bootstrap_n": 75,
    },
    {
        "name":        "navigate",
        "train_jsonl": "datasets/bbh/navigate_train.jsonl",
        "bootstrap_n": 75,
    },
    {
        "name":        "snarks",
        "train_jsonl": "datasets/bbh/snarks_train.jsonl",
        "bootstrap_n": 50,
    },
]

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


def _parse_segments(text: str) -> list[dict]:
    """
    Split a CS-ICL cheat sheet into ablatable segments by the `---` divider.

    Each segment becomes a dict with keys:
      id      — "seg_0", "seg_1", ...
      title   — first ### / ## / #### heading found in the block, or "(preamble)"
      content — full block text (heading included)
      enabled — True (all segments on by default)
    """
    import re as _re
    raw_blocks = _re.split(r"\n---\n", text)
    segments: list[dict] = []
    for i, block in enumerate(raw_blocks):
        block = block.strip()
        if not block:
            continue
        m = _re.search(r"(#{2,4}[^\n]+)", block)
        title = m.group(1).strip() if m else "(preamble)"
        segments.append({
            "id":      f"seg_{i}",
            "title":   title,
            "content": block,
            "enabled": True,
        })
    return segments


def gen_bootstrap(task: dict, model: str, api_key: str, max_tokens: int,
                  bootstrap_n_override: int | None, out_base: Path) -> Path:
    n = bootstrap_n_override if bootstrap_n_override is not None else task["bootstrap_n"]
    items = load_jsonl(task["train_jsonl"])[:n]
    dataset_str = "\n\n".join(format_item(it) for it in items)
    prompt = _GEN_PROMPT.format(dataset_str=dataset_str)

    print(f"  Generating bootstrap cheatsheet ({len(items)} items, prompt ~{len(prompt)} chars) ...",
          flush=True)
    resp = call_llm(prompt, model=model, api_key=api_key, max_tokens=max_tokens, temperature=0.0)
    cs_text = resp.content

    segments = _parse_segments(cs_text)
    print(f"  Generated {len(cs_text)} chars → {len(segments)} segments.")
    for seg in segments:
        print(f"    [{seg['id']}] {len(seg['content']):4d} chars  {seg['title'][:60]}")

    out_dir = out_base / task["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bootstrap_cs"

    # Store CS-ICL text in prior_knowledge (no render cap, frozen by ICR) so
    # the roadmap budget stays free for Phase 1/2 gap-filling patches.
    # Segments are stored for selective ablation via --ablate-prior-segments.
    cs = Cheatsheet(
        roadmap="",
        case_studies=[],
        prior_knowledge=cs_text,
        prior_knowledge_segments=segments,
    )
    cs.save(out_path)
    print(f"  Saved → {out_path}.json")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="openai/gpt-4.1-mini")
    p.add_argument("--max-tokens",   type=int, default=4000)
    p.add_argument("--bootstrap-n",  type=int, default=None,
                   help="Override per-task bootstrap_n for all tasks.")
    p.add_argument("--out-base",     default="runs/bbh_bootstrap",
                   help="Base directory for output (default: runs/bbh_bootstrap)")
    p.add_argument("--tasks",        nargs="+", default=None,
                   help="Subset of task names to generate (default: all 6)")
    args = p.parse_args()

    api_key  = get_api_key()
    out_base = Path(args.out_base)

    task_filter = set(args.tasks) if args.tasks else None
    tasks = [t for t in TASKS if task_filter is None or t["name"] in task_filter]

    for task in tasks:
        print(f"\n[{task['name']}]")
        gen_bootstrap(task, args.model, api_key, args.max_tokens,
                      args.bootstrap_n, out_base)

    print("\nDone.")


if __name__ == "__main__":
    main()
