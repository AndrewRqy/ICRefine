#!/usr/bin/env python3
"""
ablation_prompts.py — Prompt-style ablation for regressed BBH tasks.

For each task we load the post-Phase-1 cheatsheet (so Phase 1 rules are held
constant) and run Phase 2 with three different generation prompt styles:

  A — current custom prompt (baseline for this run)
  B — simplified / direct style
  C — example-anchored / few-shot style

Scores the held-out test split after each variant and prints a comparison table.

Usage:
    python3 ablation_prompts.py [--tasks sports geo causal] [--iters 3]

Tasks:
    sports  → sports_understanding
    geo     → geometric_shapes
    causal  → causal_judgement
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / "ICR_partition" / ".env")

from utils.data import load_jsonl
from utils.llm_client import get_api_key
from utils.cheatsheet import Cheatsheet
from utils.scorer import score_batch
from utils.task_spec import TaskSpec
from ICR_partition.training.loop import run_partition_loop
from tasks.bbh_tasks import (
    SPORTS_TASK, GEOMETRIC_TASK, CAUSAL_JUDGEMENT_TASK,
    _SPORTS_GEN_PROMPT, _GEO_GEN_PROMPT, _CAUSAL_GEN_PROMPT,
)

MODEL       = "openai/gpt-4.1-mini"
CONCURRENCY = 32
DATA_DIR    = Path("datasets/bbh")
RUNS_DIR    = Path("runs/bbh_concrete")

# ---------------------------------------------------------------------------
# Prompt variant definitions
# ---------------------------------------------------------------------------

# ── SPORTS ──────────────────────────────────────────────────────────────────

_SPORTS_B = (
    "You are an expert in sports rules. A model keeps failing on sports plausibility questions.\n"
    "Task: decide if a sentence about an athlete's action is plausible in their sport.\n\n"
    "=== REASONING ROADMAP ===\n{roadmap}\n=== END ROADMAP ===\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "Do three things:\n"
    "  1. Name the sport in the failure sentences.\n"
    "  2. State the exact sport rule that determines plausibility.\n"
    "  3. Write a case study that teaches that rule.\n\n"
    "Keep your diagnosis short. Go straight to the rule and the case study.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [sport + rule name] ===\n"
    "FAILURE_TYPE: A (wrong sport rule) or B (right rule, wrong sport)\n"
    "ACTIVATE IF:\n"
    "  - [sport name and action type]\n"
    "DO NOT ACTIVATE IF: [case where model is correct]\n"
    "COMMON WRONG MOVE: [specific wrong assumption model makes]\n"
    "NEXT CHECK: [exact sport rule to verify → YES or NO]\n"
    "WHY THIS WORKS: [1-2 sentences]\n"
    "SUPPORT:\n"
    "  • [example sentence]  |  Answer: YES/NO  — [rule note]\n"
    "TARGET_STEP: [roadmap step]\n\n"
    "OUTPUT 2 — ROADMAP PATCH\n"
    "=== ROADMAP PATCH ===\n"
    "[one-line rule, or '(none)']\n"
    "=== END ROADMAP PATCH ===\n"
    "{retry_context}"
)

_SPORTS_EXAMPLE_CS = """\
=== CASE STUDY: Hockey Puck vs Ball ===
FAILURE_TYPE: B
ACTIVATE IF:
  - sport: hockey
  - action_type: equipment
DO NOT ACTIVATE IF: sentence mentions ice or skates (model usually handles those)
COMMON WRONG MOVE: model assumes hockey uses a ball like soccer or basketball
NEXT CHECK: hockey uses a puck, not a ball → any sentence saying a hockey player
  "kicked the ball" or "dribbled the ball" is NOT plausible → NO
WHY THIS WORKS: the model conflates field hockey and ice hockey equipment rules.
SUPPORT:
  • "Wayne Gretzky dribbled the ball past the defender."  |  Answer: NO  — hockey uses a puck
TARGET_STEP: Step 2 — verify the equipment specific to this sport
=== ROADMAP PATCH ===
Hockey: puck not ball; goals scored by shooting, not kicking.
=== END ROADMAP PATCH ==="""

_SPORTS_C = (
    "You are an expert in sports rules. A model keeps failing on sports plausibility questions.\n"
    "Task: decide if a sentence about an athlete's action is plausible in their sport.\n\n"
    "=== REASONING ROADMAP ===\n{roadmap}\n=== END ROADMAP ===\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "Here is an example of a GOOD case study for this task:\n\n"
    f"{_SPORTS_EXAMPLE_CS}\n\n"
    "Now write a case study in exactly that style for the failures above.\n"
    "Focus on one specific sport + action combination. Be concrete.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [sport + action name] ===\n"
    "FAILURE_TYPE: A or B\n"
    "ACTIVATE IF:\n"
    "  - [sport and action]\n"
    "DO NOT ACTIVATE IF: [counter-case]\n"
    "COMMON WRONG MOVE: [model's specific error]\n"
    "NEXT CHECK: [exact rule check → YES or NO]\n"
    "WHY THIS WORKS: [1-2 sentences]\n"
    "SUPPORT:\n"
    "  • [example]  |  Answer: YES/NO  — [note]\n"
    "TARGET_STEP: [roadmap step]\n\n"
    "OUTPUT 2 — ROADMAP PATCH\n"
    "=== ROADMAP PATCH ===\n"
    "[one-line rule, or '(none)']\n"
    "=== END ROADMAP PATCH ===\n"
    "{retry_context}"
)

# ── GEOMETRIC ────────────────────────────────────────────────────────────────

_GEO_B = (
    "You are an expert in SVG geometry. A model fails at identifying shapes from SVG paths.\n"
    "The model can usually count vertices correctly but often maps the count to the wrong shape name.\n\n"
    "=== REASONING ROADMAP ===\n{roadmap}\n=== END ROADMAP ===\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "Focus on the NAME MAPPING step. The most common errors are:\n"
    "  4 vertices → model says 'square' but should check: is it rectangle? kite? rhombus?\n"
    "  6 vertices → model says 'hexagon' — correct, but sometimes confused with heptagon\n"
    "  7 vertices → model says 'hexagon' — off by one; should be heptagon\n"
    "  8 vertices → model says 'heptagon' — off by one; should be octagon\n"
    "  arc in path → model ignores arc and counts lines; should recognise circle or sector\n\n"
    "Write a case study targeting the specific confusion shown in the failures.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [vertex count + confused shape names] ===\n"
    "FAILURE_TYPE: A (miscounted) or B (correct count, wrong name)\n"
    "ACTIVATE IF:\n"
    "  - [n_vertices and which names are confused]\n"
    "DO NOT ACTIVATE IF: [case where model names correctly]\n"
    "COMMON WRONG MOVE: [specific wrong name and why model picks it]\n"
    "NEXT CHECK: [count → correct shape name → answer (A)–(J)]\n"
    "WHY THIS WORKS: [1-2 sentences on the name confusion]\n"
    "SUPPORT:\n"
    "  • [SVG command summary, e.g. '3 L + 1 M = 4 vertices = rectangle']  |  Answer: (X)  — [note]\n"
    "TARGET_STEP: [roadmap step]\n\n"
    "OUTPUT 2 — ROADMAP PATCH\n"
    "=== ROADMAP PATCH ===\n"
    "[one-line name-mapping rule, or '(none)']\n"
    "=== END ROADMAP PATCH ===\n"
    "{retry_context}"
)

_GEO_EXAMPLE_CS = """\
=== CASE STUDY: 7-Vertex Heptagon Mistaken for Hexagon ===
FAILURE_TYPE: B
ACTIVATE IF:
  - n_vertices: 7
  - error: wrong_shape_name
DO NOT ACTIVATE IF: path has arc commands (different rule applies)
COMMON WRONG MOVE: model counts 7 vertices correctly but writes 'hexagon' (6-sided) instead of 'heptagon'
NEXT CHECK: count L commands + 1 = total vertices. 7 vertices → heptagon (7 sides), NOT hexagon (6) → answer (G)
WHY THIS WORKS: hexagon (6) and heptagon (7) are adjacent and rarely seen, so the model defaults to hexagon.
SUPPORT:
  • M + 6×L + Z = 7 vertices  |  Answer: (G)  — heptagon, not hexagon
TARGET_STEP: Step 3 — map vertex count to shape name
=== ROADMAP PATCH ===
7 vertices → heptagon (not hexagon); 8 vertices → octagon (not heptagon).
=== END ROADMAP PATCH ==="""

_GEO_C = (
    "You are an expert in SVG geometry. A model fails at identifying shapes from SVG paths.\n\n"
    "=== REASONING ROADMAP ===\n{roadmap}\n=== END ROADMAP ===\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "Here is an example of a GOOD case study for this task:\n\n"
    f"{_GEO_EXAMPLE_CS}\n\n"
    "Write a case study in exactly that style for the failures above.\n"
    "Include a concrete SVG command summary in the SUPPORT line.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [vertex count + shape confusion] ===\n"
    "FAILURE_TYPE: A or B\n"
    "ACTIVATE IF:\n"
    "  - [n_vertices and error type]\n"
    "DO NOT ACTIVATE IF: [counter-case]\n"
    "COMMON WRONG MOVE: [specific wrong name]\n"
    "NEXT CHECK: [count rule → shape name → answer (A)–(J)]\n"
    "WHY THIS WORKS: [1-2 sentences]\n"
    "SUPPORT:\n"
    "  • [SVG command summary]  |  Answer: (X)  — [note]\n"
    "TARGET_STEP: [roadmap step]\n\n"
    "OUTPUT 2 — ROADMAP PATCH\n"
    "=== ROADMAP PATCH ===\n"
    "[one-line rule, or '(none)']\n"
    "=== END ROADMAP PATCH ===\n"
    "{retry_context}"
)

# ── CAUSAL ───────────────────────────────────────────────────────────────────

_CAUSAL_B = (
    "You are an expert in causal reasoning. A model fails on causal judgment questions.\n"
    "The model answers from the perspective of a typical person.\n\n"
    "=== REASONING ROADMAP ===\n{roadmap}\n=== END ROADMAP ===\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "SINGLE DIAGNOSTIC: Apply the counterfactual test first.\n"
    "  Ask: would the outcome have occurred WITHOUT this actor's action?\n"
    "  YES → the actor did NOT cause it (answer is likely NO)\n"
    "  NO  → the actor DID cause it (answer is likely YES)\n\n"
    "Secondary checks (only if counterfactual is inconclusive):\n"
    "  - Is this actor the proximate (immediate) cause or just background?\n"
    "  - Are two actors jointly required? (neither alone causes it)\n\n"
    "Write a case study teaching whichever check the failures are missing.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [check name, e.g. 'Counterfactual Test Fails for Background Cause'] ===\n"
    "FAILURE_TYPE: A (model skips counterfactual) or B (right test, wrong conclusion)\n"
    "ACTIVATE IF:\n"
    "  - [structural condition]\n"
    "DO NOT ACTIVATE IF: [case where model reasons correctly]\n"
    "COMMON WRONG MOVE: [what model concludes and why it's wrong]\n"
    "NEXT CHECK: [counterfactual or structural test → YES or NO]\n"
    "WHY THIS WORKS: [1-2 sentences]\n"
    "SUPPORT:\n"
    "  • [concrete scenario]  |  Answer: YES/NO  — [causal note]\n"
    "TARGET_STEP: [roadmap step]\n\n"
    "OUTPUT 2 — ROADMAP PATCH\n"
    "=== ROADMAP PATCH ===\n"
    "[one-line addition, or '(none)']\n"
    "=== END ROADMAP PATCH ===\n"
    "{retry_context}"
)

_CAUSAL_C = (
    "You are an expert in causal reasoning. A model fails on causal judgment questions.\n"
    "The model answers from the perspective of a typical person.\n\n"
    "=== REASONING ROADMAP ===\n{roadmap}\n=== END ROADMAP ===\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "Write a case study as a NAMED CONCRETE SCENARIO (like 'Two Wires', 'Bridge Collapse').\n"
    "Do NOT use abstract causal structure labels (no 'overdetermination', 'preemption', etc.).\n"
    "Instead: describe a specific everyday situation, explain why a typical person would say\n"
    "YES or NO, and give the reasoning in plain language.\n"
    "The scenario should pattern-match to the failures above so a model can recognise it.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [concrete scenario name, e.g. 'Two Gardeners One Hose'] ===\n"
    "FAILURE_TYPE: A (wrong intuition) or B (right intuition, wrong actor)\n"
    "ACTIVATE IF:\n"
    "  - [structural cue in plain language, e.g. 'two people each had to act for outcome']\n"
    "DO NOT ACTIVATE IF: [counter-case in plain language]\n"
    "COMMON WRONG MOVE: [plain-language description of error]\n"
    "NEXT CHECK: [plain-language question a typical person would ask → YES or NO]\n"
    "WHY THIS WORKS: [1-2 sentences on the everyday intuition]\n"
    "SUPPORT:\n"
    "  • [concrete 1-sentence scenario]  |  Answer: YES/NO  — [plain reasoning]\n"
    "TARGET_STEP: [roadmap step]\n\n"
    "OUTPUT 2 — ROADMAP PATCH\n"
    "=== ROADMAP PATCH ===\n"
    "[one-line plain-language addition, or '(none)']\n"
    "=== END ROADMAP PATCH ===\n"
    "{retry_context}"
)

# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

VARIANTS: dict[str, list[tuple[str, str]]] = {
    "sports": [
        ("A-current",      _SPORTS_GEN_PROMPT),
        ("B-rule-direct",  _SPORTS_B),
        ("C-few-shot",     _SPORTS_C),
    ],
    "geo": [
        ("A-current",      _GEO_GEN_PROMPT),
        ("B-name-confusion", _GEO_B),
        ("C-few-shot",     _GEO_C),
    ],
    "causal": [
        ("A-current",      _CAUSAL_GEN_PROMPT),
        ("B-counterfactual-first", _CAUSAL_B),
        ("C-scenario-analogy",     _CAUSAL_C),
    ],
}

TASK_CFG: dict[str, dict] = {
    "sports": {
        "task_spec":   SPORTS_TASK,
        "train_jsonl": DATA_DIR / "sports_understanding_train.jsonl",
        "test_jsonl":  DATA_DIR / "sports_understanding_test.jsonl",
        "phase1_cs":   RUNS_DIR / "sports_understanding" / "cheatsheet_phase1_final.json",
    },
    "geo": {
        "task_spec":   GEOMETRIC_TASK,
        "train_jsonl": DATA_DIR / "geometric_shapes_train.jsonl",
        "test_jsonl":  DATA_DIR / "geometric_shapes_test.jsonl",
        "phase1_cs":   RUNS_DIR / "geometric_shapes" / "cheatsheet_phase1_final.json",
    },
    "causal": {
        "task_spec":   CAUSAL_JUDGEMENT_TASK,
        "train_jsonl": DATA_DIR / "causal_judgement_train.jsonl",
        "test_jsonl":  DATA_DIR / "causal_judgement_test.jsonl",
        # causal has no phase1 (skipped); use concrete bootstrap as prior_knowledge
        "phase1_cs":   RUNS_DIR / "causal_judgement" / "ruleset_bootstrap.txt",
    },
}


def _load_starting_cheatsheet(task_key: str) -> Cheatsheet:
    cfg = TASK_CFG[task_key]
    path = cfg["phase1_cs"]
    if not path.exists():
        print(f"  [warn] {path} not found — starting from empty cheatsheet")
        return Cheatsheet(roadmap="", case_studies=[])
    if path.suffix == ".json":
        return Cheatsheet.load(path.with_suffix(""))  # load strips extension
    # .txt → treat as raw prior_knowledge text (causal bootstrap)
    text = path.read_text(encoding="utf-8").strip()
    return Cheatsheet(roadmap="", case_studies=[], prior_knowledge=text)


def run_variant(
    task_spec: TaskSpec,
    prompt_template: str,
    train_items: list[dict],
    test_items: list[dict],
    starting_cs: Cheatsheet,
    api_key: str,
    max_cs_iters: int,
    label: str,
) -> dict:
    variant_spec = dataclasses.replace(task_spec, generation_prompt_template=prompt_template)

    # Deep copy starting cheatsheet so variants don't share state
    cs = Cheatsheet(
        roadmap=starting_cs.roadmap,
        case_studies=list(starting_cs.case_studies),
        prior_knowledge=starting_cs.prior_knowledge,
        no_limit=starting_cs.no_limit,
    )

    result = run_partition_loop(
        cheatsheet=cs,
        train_items=train_items,
        val_items=None,
        model_score=MODEL,
        model_casestudy=MODEL,
        api_key=api_key,
        oracle=None,
        bin_threshold=3,
        retirement_threshold=2,
        max_outer_iters=max_cs_iters,
        partition_concurrency=8,
        concurrency=CONCURRENCY,
        n_candidates=3,
        candidate_rounds=2,
        fix_rate_threshold=0.30,
        regress_threshold=0.20,
        min_pool_for_regression=5,
        similarity_gate=True,
        task_spec=variant_spec,
        output_dir=None,
        log=True,
        cot_first=True,
        reasoning_effort=None,
        pk_regression_guard=True,
        pk_regression_tolerance=0.03,
    )

    correct, _ = score_batch(
        test_items, result.cheatsheet.render(), MODEL, api_key,
        concurrency=CONCURRENCY, reasoning_effort=None, cot_first=True,
        progress_label=f"{label}-eval", task_spec=variant_spec,
    )
    acc = len(correct) / len(test_items)
    return {
        "variant":        label,
        "test_acc":       acc,
        "n_cs_added":     result.n_case_studies_added,
        "n_cs_total":     len(result.cheatsheet.case_studies),
        "train_acc":      result.train_accuracy,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks",    nargs="+", default=["sports", "geo", "causal"],
                        choices=["sports", "geo", "causal"])
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Variant labels to run, e.g. A-current B-counterfactual-first. "
                             "Defaults to all variants for the task.")
    parser.add_argument("--iters",  type=int, default=3,
                        help="max_cs_iters per variant (default 3)")
    args = parser.parse_args()

    os_chdir = Path(__file__).parent
    import os; os.chdir(os_chdir)

    api_key = get_api_key()

    all_results: dict[str, list[dict]] = {}

    for task_key in args.tasks:
        cfg        = TASK_CFG[task_key]
        task_spec  = cfg["task_spec"]
        train_items = load_jsonl(cfg["train_jsonl"])
        test_items  = load_jsonl(cfg["test_jsonl"])
        starting_cs = _load_starting_cheatsheet(task_key)

        print(f"\n{'='*65}")
        print(f"TASK: {task_key}  train={len(train_items)}  test={len(test_items)}")
        print(f"  starting cheatsheet: {len(starting_cs.case_studies)} case studies, "
              f"{len(starting_cs.prior_knowledge)} chars prior_knowledge")
        print(f"{'='*65}")

        task_results = []
        variants_to_run = [
            (label, tpl) for label, tpl in VARIANTS[task_key]
            if args.variants is None or label in args.variants
        ]
        for variant_label, prompt_template in variants_to_run:
            print(f"\n--- Variant {variant_label} ---")
            r = run_variant(
                task_spec=task_spec,
                prompt_template=prompt_template,
                train_items=train_items,
                test_items=test_items,
                starting_cs=starting_cs,
                api_key=api_key,
                max_cs_iters=args.iters,
                label=f"{task_key}/{variant_label}",
            )
            task_results.append(r)
            print(f"  → test_acc={r['test_acc']:.1%}  cs_added={r['n_cs_added']}")

        all_results[task_key] = task_results

    # ── Results table ──────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("ABLATION RESULTS")
    print(f"{'='*65}")
    for task_key, rows in all_results.items():
        print(f"\n  {task_key}")
        print(f"  {'Variant':<28} {'Test Acc':>9}  {'CS Added':>9}  {'Train Acc':>10}")
        print(f"  {'-'*60}")
        best_acc = max(r["test_acc"] for r in rows)
        for r in rows:
            marker = " ◀" if r["test_acc"] == best_acc else ""
            print(f"  {r['variant']:<28} {r['test_acc']:>8.1%}  "
                  f"{r['n_cs_added']:>9}  {r['train_acc']:>9.1%}{marker}")

    # Save results
    out_path = Path("runs/ablation_prompts.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
