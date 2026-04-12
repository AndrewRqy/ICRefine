"""
smoke_roadmap.py — Quick smoke test for roadmap synthesis.

Loads a real cheatsheet from a previous run, calls run_roadmap_synthesis
with no train_seen (skips scoring), and prints the result.

Usage:
    python smoke_roadmap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Resolve paths ─────────────────────────────────────────────────────────────
ICR_DIR  = Path(__file__).resolve().parent
PREV_CS  = ICR_DIR.parent / (
    "SAIR_eval_pipeline/results/refine_20260410_073754"
    "/iter_00/icr_select_iter_01_r1/cheatsheet_final.json"
)

if str(ICR_DIR) not in sys.path:
    sys.path.insert(0, str(ICR_DIR))

from utils.cheatsheet import Cheatsheet
from utils.case_study  import CaseStudy
from utils.llm_client  import get_api_key
from ICR_select.training.roadmap_synthesizer import run_roadmap_synthesis

# ── Load cheatsheet ────────────────────────────────────────────────────────────
data = json.loads(PREV_CS.read_text(encoding="utf-8"))
cs   = Cheatsheet(
    roadmap         = data.get("roadmap", ""),
    case_studies    = [CaseStudy.from_dict(c) for c in data.get("case_studies", [])],
    prior_knowledge = data.get("prior_knowledge", ""),
)

print(f"Loaded cheatsheet: {len(cs.case_studies)} case study/studies, "
      f"{len(cs.prior_knowledge)} chars of prior knowledge.")

# ── Minimal failure context (one hand-crafted item) ────────────────────────────
failures = [
    {
        "equation1":  "x = ((y * (x * y)) * z) * w",
        "equation2":  "x = (y * (x * z)) * (y * w)",
        "answer":     True,
        "predicted":  "FALSE",
        "post_think": "Tried to build a counterexample but didn't check symmetry.",
    }
]

# ── Run synthesis (no train_seen → skip scoring) ───────────────────────────────
api_key    = get_api_key()
model_cs   = "openai/gpt-4o"

print(f"\nCalling run_roadmap_synthesis (model={model_cs}) ...")
result = run_roadmap_synthesis(
    cheatsheet      = cs,
    failures        = failures,
    model_casestudy = model_cs,
    model_score     = "openai/gpt-4o",   # unused when train_seen=None
    api_key         = api_key,
    train_seen      = None,              # skip accuracy validation
    log             = True,
)

# ── Report ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"accepted          : {result.accepted}")
print(f"n_case_studies    : {result.n_case_studies_used}")
print(f"roadmap_chars     : {len(result.roadmap)}")
if result.roadmap:
    print("\n--- ROADMAP ---")
    print(result.roadmap)
print("=" * 60)
print("\nSmoke test PASSED." if result.accepted else "\nSmoke test: synthesis rejected (format issue?).")
