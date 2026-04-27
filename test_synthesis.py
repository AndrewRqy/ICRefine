"""
test_synthesis.py — Test cross-cluster abstraction on causal_judgement.

Instead of generating case studies per failure cluster, we collect all
failures from the training set, show them together to the LLM, and ask
it to write a single abstract case study that captures the general
underlying reasoning principle.

Usage:
    python3 test_synthesis.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "ICR_partition" / ".env")

from utils.data import load_jsonl
from utils.llm_client import call_llm, get_api_key
from utils.scorer import score_batch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL       = "openai/gpt-4.1-2025-04-14"
CONCURRENCY = 16
TRAIN_JSONL = "datasets/bbh/causal_judgement_train.jsonl"
TEST_JSONL  = "datasets/bbh/causal_judgement_test.jsonl"
CHEATSHEET  = "runs/bbh_rerun/causal_judgement/cheatsheet_final.txt"
TASK_MODULE = "tasks.bbh_tasks"
TASK_ATTR   = "CAUSAL_JUDGEMENT_TASK"
MAX_FAILURES_SHOWN = 12   # how many diverse failures to show the synthesizer
SEED = 42

# ---------------------------------------------------------------------------
# Synthesis prompt
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = """\
You are an expert in causal reasoning improving a cheat sheet used by a \
language model to answer causal judgment questions.

Below are {n} questions the model answered WRONG, each paired with the \
CORRECT answer and the gold reasoning trace that explains why.

=== FAILED CASES ===
{failure_lines}

=== EXISTING CHEAT SHEET (for context — do not rewrite it) ===
{cheatsheet}

=== YOUR TASK ===
The failures above span different causal structures but likely share a \
common reasoning gap. Identify what that gap is and write ONE abstract \
case study that teaches the general principle needed to avoid it.

The case study must:
1. Identify the reasoning pattern by name (e.g., "Counterfactual Test for Proximate Causation")
2. Explain the general principle in 2-3 sentences
3. Give ONE concrete worked example (different from the failures above) \
   that illustrates how to apply the principle step by step
4. Be broad enough to apply to novel scenarios, not just the specific \
   failure cases shown

Format:
--- Case Study: <name> ---
PRINCIPLE: <general reasoning principle>
EXAMPLE:
  Scenario: <concrete scenario>
  Correct reasoning: <step-by-step>
  Verdict: YES/NO
WHY THIS GENERALIZES: <one sentence on why this covers the pattern>
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_failures(failures: list[dict]) -> str:
    lines = []
    for i, it in enumerate(failures, 1):
        lines.append(f"[{i}] {it['input'].strip()[:300]}")
        lines.append(f"    Correct answer: {it['answer'].strip()}")
        lines.append(f"    Model predicted: {it.get('predicted', '?').strip()}")
        reason = it.get("reason", "")
        if reason:
            lines.append(f"    Gold reasoning: {reason.strip()[:300]}")
        lines.append("")
    return "\n".join(lines)


def _sample_diverse(failures: list[dict], n: int, seed: int) -> list[dict]:
    """Sample failures balanced by correct answer (YES/NO)."""
    rng = random.Random(seed)
    yes = [f for f in failures if f.get("answer", "").strip().lower() == "yes"]
    no  = [f for f in failures if f.get("answer", "").strip().lower() == "no"]
    rng.shuffle(yes); rng.shuffle(no)
    half = n // 2
    sampled = yes[:half] + no[:n - half]
    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import importlib
    api_key   = get_api_key()
    task_spec = getattr(importlib.import_module(TASK_MODULE), TASK_ATTR)
    cheatsheet_base = Path(CHEATSHEET).read_text(encoding="utf-8").strip()

    print("Step 1 — scoring train set to collect failures ...")
    train_items = load_jsonl(Path(TRAIN_JSONL))
    correct, wrong = score_batch(
        train_items, cheatsheet_base, MODEL, api_key,
        concurrency=CONCURRENCY, reasoning_effort=None,
        cot_first=True, task_spec=task_spec,
    )
    print(f"  train accuracy = {len(correct)/len(train_items):.1%}  "
          f"({len(correct)} correct, {len(wrong)} wrong)")

    if not wrong:
        print("No failures found — nothing to synthesize.")
        return

    # Sample diverse failures balanced by answer label
    sample = _sample_diverse(wrong, min(MAX_FAILURES_SHOWN, len(wrong)), SEED)
    print(f"  sampled {len(sample)} diverse failures for synthesis")

    print("\nStep 2 — running synthesis to generate abstract case study ...")
    failure_lines = _format_failures(sample)
    prompt = SYNTHESIS_PROMPT.format(
        n=len(sample),
        failure_lines=failure_lines,
        cheatsheet=cheatsheet_base,
    )
    response = call_llm(prompt, model=MODEL, api_key=api_key,
                        max_tokens=800, temperature=0.3)
    synthesis = response.content.strip()
    print("\n--- SYNTHESIZED CASE STUDY ---")
    print(synthesis)
    print("--- END ---\n")

    # Append synthesized case study to cheatsheet
    cheatsheet_augmented = cheatsheet_base + "\n\n" + synthesis

    print("Step 3 — scoring test set with augmented cheatsheet ...")
    test_items = load_jsonl(Path(TEST_JSONL))
    correct_aug, _ = score_batch(
        test_items, cheatsheet_augmented, MODEL, api_key,
        concurrency=CONCURRENCY, reasoning_effort=None,
        cot_first=True, task_spec=task_spec,
    )
    acc_aug = len(correct_aug) / len(test_items)

    print("\nStep 4 — scoring test set with base cheatsheet (baseline) ...")
    correct_base, _ = score_batch(
        test_items, cheatsheet_base, MODEL, api_key,
        concurrency=CONCURRENCY, reasoning_effort=None,
        cot_first=True, task_spec=task_spec,
    )
    acc_base = len(correct_base) / len(test_items)

    print(f"\n{'='*50}")
    print(f"  Base cheatsheet accuracy :  {acc_base:.1%}  ({len(correct_base)}/{len(test_items)})")
    print(f"  + synthesis case study   :  {acc_aug:.1%}  ({len(correct_aug)}/{len(test_items)})")
    print(f"  Delta                    :  {acc_aug - acc_base:+.1%}")
    print(f"  CS-ICL reference         :  70.1%")
    print(f"{'='*50}")

    # Save result
    out = {
        "base_acc": acc_base,
        "augmented_acc": acc_aug,
        "delta": acc_aug - acc_base,
        "n_failures_shown": len(sample),
        "synthesis": synthesis,
    }
    Path("runs/bbh_rerun/causal_judgement/synthesis_test.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("Result saved to runs/bbh_rerun/causal_judgement/synthesis_test.json")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    main()
