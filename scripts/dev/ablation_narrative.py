#!/usr/bin/env python3
"""
ablation_narrative.py — Test Option 1: CS-ICL narrative format for Phase 2.

The causal cheatsheet already has CS-ICL-style named narratives in prior_knowledge
(from bootstrap_cheatsheet_fn). Phase 2 added structured ACTIVATE IF case studies.

This script tests whether generating *more narratives* in Phase 2 (instead of
structured case studies) beats the current approach.

Three configs scored on the causal test set:
  A — full cheatsheet (prior_knowledge + 2 structured case studies)   [current]
  B — prior_knowledge only (just the 7 bootstrap narratives, no case studies)
  D — prior_knowledge + LLM-generated additional narrative scenarios   [option 1]

Usage:
    python3 ablation_narrative.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_ROOT / "ICR_partition" / ".env")
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT))

from utils.data import load_jsonl
from utils.llm_client import get_api_key, call_llm
from utils.cheatsheet import Cheatsheet
from utils.scorer import score_batch
from tasks.bbh_tasks import CAUSAL_JUDGEMENT_TASK

MODEL       = "openai/gpt-4.1-mini"
CONCURRENCY = 32
DATA_DIR    = Path("datasets/bbh")
CS_PATH     = Path("runs/bbh_concrete/causal_judgement/cheatsheet_final")

_NARRATIVE_GEN_PROMPT = """\
You are an expert in causal reasoning. A weaker model keeps failing on these questions.

The model must answer YES/NO: "Did X cause Y?" from the perspective of a typical person.

Here are the training examples it got wrong:

{failure_lines}

Your job: write 2-3 NEW named scenario analogies (like the examples below) that a \
typical person can use to reason about these failures. Each scenario should illustrate \
a specific causal pattern (overdetermination, preemption, double prevention, joint \
sufficiency, proximate/distal, counterfactual dependence, or accidental combination).

Format each scenario EXACTLY like this:
=== [Causal Pattern]: [Memorable Title] ===
Scenario: [2-3 sentences. Concrete. Named actors. Clear causal chain.]
Causal structure: [1 sentence naming the pattern.]
Verdict: YES or NO (explain which actor if multiple)
Why a typical person says this: [1 sentence on the intuitive reasoning.]
Apply when: [1 sentence on when to use this analogy.]

Rules:
- Do NOT repeat the patterns already covered (listed below)
- Each scenario must target a different gap shown in the failures
- Keep each scenario under 120 words
- No bullet points, no headers beyond the === line

Already covered patterns (do not repeat):
{covered_patterns}
"""


def _format_failures(items: list[dict], max_items: int = 20) -> str:
    lines = []
    for i, it in enumerate(items[:max_items], 1):
        lines.append(
            f"[{i}] {it.get('input', '')[:250].strip()}\n"
            f"    Expected: {it.get('answer','?').strip()}  "
            f"Got: {it.get('predicted','?')}"
        )
    return "\n\n".join(lines)


def _extract_covered_patterns(prior_knowledge: str) -> str:
    """Pull === title === lines from existing prior_knowledge."""
    patterns = []
    for line in prior_knowledge.splitlines():
        line = line.strip()
        if line.startswith("===") and line.endswith("===") and len(line) > 6:
            patterns.append(line.strip("= ").strip())
    return "\n".join(f"  - {p}" for p in patterns) if patterns else "  (none)"


def score_config(label: str, cs: Cheatsheet, test_items: list[dict], api_key: str) -> float:
    rendered = cs.render()
    correct, _ = score_batch(
        test_items, rendered, MODEL, api_key,
        concurrency=CONCURRENCY, reasoning_effort=None, cot_first=True,
        progress_label=label, task_spec=CAUSAL_JUDGEMENT_TASK,
    )
    acc = len(correct) / len(test_items)
    print(f"  [{label}]  acc={acc:.1%}  ({len(correct)}/{len(test_items)})")
    return acc


def find_failures(cs: Cheatsheet, train_items: list[dict], api_key: str) -> list[dict]:
    rendered = cs.render()
    _, wrong = score_batch(
        train_items, rendered, MODEL, api_key,
        concurrency=CONCURRENCY, reasoning_effort=None, cot_first=True,
        progress_label="find-failures", task_spec=CAUSAL_JUDGEMENT_TASK,
    )
    return wrong


def main():
    api_key = get_api_key()

    train_items = load_jsonl(DATA_DIR / "causal_judgement_train.jsonl")
    test_items  = load_jsonl(DATA_DIR / "causal_judgement_test.jsonl")
    print(f"train={len(train_items)}  test={len(test_items)}")

    # Load final cheatsheet (prior_knowledge + 2 structured case studies)
    full_cs = Cheatsheet.load(CS_PATH)
    pk_text = full_cs.prior_knowledge

    # ── Config A: full cheatsheet (current) ───────────────────────────────────
    print("\n--- Config A: full cheatsheet (prior_knowledge + structured case studies) ---")
    acc_a = score_config("A-full", full_cs, test_items, api_key)

    # ── Config B: prior_knowledge only (no structured case studies) ───────────
    print("\n--- Config B: prior_knowledge only (CS-ICL narratives, no case studies) ---")
    cs_b = Cheatsheet(roadmap="", case_studies=[], prior_knowledge=pk_text)
    acc_b = score_config("B-narratives-only", cs_b, test_items, api_key)

    # ── Config D: prior_knowledge + LLM-generated additional narratives ───────
    print("\n--- Config D: finding training failures with prior_knowledge only ---")
    failures = find_failures(cs_b, train_items, api_key)
    print(f"  {len(failures)} training failures found")

    if not failures:
        print("  No failures — skipping narrative generation")
        acc_d = acc_b
    else:
        covered = _extract_covered_patterns(pk_text)
        prompt  = _NARRATIVE_GEN_PROMPT.format(
            failure_lines=_format_failures(failures),
            covered_patterns=covered,
        )
        print("\n--- Generating additional narrative scenarios ---")
        response = call_llm(prompt, model=MODEL, api_key=api_key,
                            max_tokens=1200, temperature=0.4)
        new_narratives = response.content.strip()
        print(f"  Generated {len(new_narratives)} chars of new narratives")
        print("\n--- New narratives ---")
        print(new_narratives)

        extended_pk = pk_text.rstrip() + "\n\n" + new_narratives
        cs_d = Cheatsheet(roadmap="", case_studies=[], prior_knowledge=extended_pk)

        print("\n--- Config D: prior_knowledge + new narrative scenarios ---")
        acc_d = score_config("D-extended-narratives", cs_d, test_items, api_key)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("CAUSAL NARRATIVE ABLATION SUMMARY")
    print("="*55)
    print(f"  CS-ICL baseline (reference):        71.3%")
    print(f"  A — full cheatsheet (current):      {acc_a:.1%}")
    print(f"  B — narratives only (no CS):        {acc_b:.1%}")
    print(f"  D — narratives + extra narratives:  {acc_d:.1%}")
    print("="*55)


if __name__ == "__main__":
    main()
