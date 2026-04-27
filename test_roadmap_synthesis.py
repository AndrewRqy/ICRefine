"""
test_roadmap_synthesis.py — Test roadmap synthesis from rules + case studies.

After Phase 1 generates rules and Phase 2 generates case studies, we ask
the model to derive a step-by-step reasoning procedure from them and
populate the Reasoning Roadmap section. This gives the model a HOW-to-reason
scaffold rather than just WHAT-to-conclude rules.

Usage:
    python3 test_roadmap_synthesis.py
"""
from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "ICR_partition" / ".env")

from utils.data import load_jsonl
from utils.llm_client import call_llm, get_api_key
from utils.scorer import score_batch

MODEL       = "openai/gpt-4.1-2025-04-14"
CONCURRENCY = 16
TEST_JSONL  = "datasets/bbh/causal_judgement_test.jsonl"
CHEATSHEET  = "runs/bbh_rerun/causal_judgement/cheatsheet_final.txt"
TASK_MODULE = "tasks.bbh_tasks"
TASK_ATTR   = "CAUSAL_JUDGEMENT_TASK"

# ---------------------------------------------------------------------------
# Roadmap synthesis prompt
# ---------------------------------------------------------------------------

ROADMAP_PROMPT = """\
You are improving a cheat sheet used by a language model to answer \
causal judgment questions.

The cheat sheet already has a set of rules (each encoding what conclusion \
to draw given a specific causal structure) and possibly some case studies. \
The problem is that models struggle to identify WHICH rule applies to a \
novel scenario — they have the rules but not a procedure for reasoning \
through a question to decide which rule fits.

=== EXISTING RULES AND CASE STUDIES ===
{rules_and_cases}

=== YOUR TASK ===
Write a Reasoning Roadmap: a short step-by-step procedure (4-6 steps) that \
a model should follow for ANY causal judgment question before applying the \
rules above. The roadmap should guide the model through the causal structure \
of the scenario so it can reliably identify the right rule. Each step should \
be a concrete action (e.g., "identify the agent whose causation is in \
question," "ask whether the outcome would have occurred without that agent's \
action"), not a restatement of the rules themselves.

The roadmap must:
- Be general enough to apply to any causal scenario
- Lead naturally into the rules (each step should narrow down which rule applies)
- Use plain language a typical person would follow
- Be concise (each step 1-2 sentences max)

Output ONLY the roadmap steps, formatted as a numbered list. \
No preamble, no explanation after.
"""

# ---------------------------------------------------------------------------

def _extract_rules_and_cases(cheatsheet: str) -> str:
    """Extract the rules + case studies sections, excluding the empty roadmap."""
    # Take everything between PRIOR KNOWLEDGE and REASONING ROADMAP
    m = re.search(r"=== PRIOR KNOWLEDGE ===(.*?)=== REASONING ROADMAP ===",
                  cheatsheet, re.DOTALL)
    if m:
        return m.group(1).strip()
    return cheatsheet.strip()


def _inject_roadmap(cheatsheet: str, roadmap: str) -> str:
    """Replace the empty REASONING ROADMAP section with the generated roadmap."""
    marker = "=== REASONING ROADMAP ==="
    if marker in cheatsheet:
        before = cheatsheet[:cheatsheet.index(marker) + len(marker)]
        after  = cheatsheet[cheatsheet.index(marker) + len(marker):]
        # Drop whatever (empty) content was after the marker
        after_stripped = re.sub(r"^\s*\n", "", after)
        return before + "\n\n" + roadmap + "\n" + after_stripped
    return cheatsheet + "\n\n" + marker + "\n\n" + roadmap


# ---------------------------------------------------------------------------

def main():
    api_key   = get_api_key()
    task_spec = getattr(importlib.import_module(TASK_MODULE), TASK_ATTR)
    cheatsheet_base = Path(CHEATSHEET).read_text(encoding="utf-8").strip()
    test_items      = load_jsonl(Path(TEST_JSONL))

    # Step 1 — baseline
    print("Step 1 — scoring test set with base cheatsheet ...")
    correct_base, _ = score_batch(
        test_items, cheatsheet_base, MODEL, api_key,
        concurrency=CONCURRENCY, reasoning_effort=None,
        cot_first=True, task_spec=task_spec,
    )
    acc_base = len(correct_base) / len(test_items)
    print(f"  base accuracy = {acc_base:.1%}")

    # Step 2 — generate roadmap from rules + case studies
    print("\nStep 2 — synthesising reasoning roadmap from rules ...")
    rules_and_cases = _extract_rules_and_cases(cheatsheet_base)
    prompt = ROADMAP_PROMPT.format(rules_and_cases=rules_and_cases)
    response = call_llm(prompt, model=MODEL, api_key=api_key,
                        max_tokens=600, temperature=0.3)
    roadmap = response.content.strip()
    print("\n--- GENERATED ROADMAP ---")
    print(roadmap)
    print("--- END ---\n")

    # Step 3 — inject roadmap and score
    cheatsheet_aug = _inject_roadmap(cheatsheet_base, roadmap)
    print("Step 3 — scoring test set with roadmap-augmented cheatsheet ...")
    correct_aug, _ = score_batch(
        test_items, cheatsheet_aug, MODEL, api_key,
        concurrency=CONCURRENCY, reasoning_effort=None,
        cot_first=True, task_spec=task_spec,
    )
    acc_aug = len(correct_aug) / len(test_items)

    print(f"\n{'='*55}")
    print(f"  Base (rules only)          :  {acc_base:.1%}  ({len(correct_base)}/{len(test_items)})")
    print(f"  + roadmap from rules       :  {acc_aug:.1%}  ({len(correct_aug)}/{len(test_items)})")
    print(f"  Delta                      :  {acc_aug - acc_base:+.1%}")
    print(f"  CS-ICL reference           :  70.1%")
    print(f"{'='*55}")

    out = {
        "base_acc":     acc_base,
        "roadmap_acc":  acc_aug,
        "delta":        acc_aug - acc_base,
        "roadmap":      roadmap,
        "cheatsheet_augmented": cheatsheet_aug,
    }
    out_path = Path("runs/bbh_rerun/causal_judgement/roadmap_synthesis_test.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nResult saved to {out_path}")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    main()
