"""
ICR_select/prompts/ablation_templates.py — Prompt variants for case study ablation study.

Axes:
  activate_if : "strict"  — conditions must be mechanical syntax checks only
                "loose"   — natural language description of the pattern is fine
  emphasis    : "both"    — show wrong move AND correct move
                "correct" — correct move only (what to do)
                "wrong"   — wrong move only (what to avoid)
  length      : 900       — current budget
                400       — compressed, forces distillation to one key insight

12 variants total: 2 × 3 × 2.

Each variant is a complete CASE_STUDY_WITH_REASONING_PROMPT replacement with
identical structure ({roadmap}, {case_studies}, {already_covered}, {failure_lines})
so generate_candidates() can swap them in without any other code changes.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared preamble and structure — identical across all variants
# ---------------------------------------------------------------------------

_PREAMBLE = """\
You are an expert in universal algebra working on equational theories of magmas.
A magma is a set with a single binary operation * and no other axioms.

You are writing a TEACHING NOTE for a weaker reasoning model that keeps making \
the same mistake.

The cheatsheet the model uses has two parts:

=== REASONING ROADMAP ===
{roadmap}
=== END REASONING ROADMAP ===

=== EXISTING CASE STUDIES ===
{case_studies}
=== END CASE STUDIES ===

=== PATTERNS ALREADY COVERED — YOUR CASE STUDY MUST NOT RESTATE THESE ===
{already_covered}
Your new case study MUST address a gap NOT covered above.
=== END ALREADY COVERED ===

The following examples were ALL predicted INCORRECTLY by the weaker model.

=== FAILURES WITH INCORRECT MODEL REASONING ===

{failure_lines}

=== YOUR TASK ===
"""

_SUFFIX = """\

OUTPUT 2 — ROADMAP PATCH (max 400 chars)
Targeted corrections to the roadmap that would have prevented these failures.
Write only aspects that need to change.

=== ROADMAP PATCH ===
[changes]
=== END PATCH ===

Output ONLY these two sections — no preamble, no sign-off.\
"""

# ---------------------------------------------------------------------------
# ACTIVATE IF instruction — strict vs loose
# ---------------------------------------------------------------------------

_ACTIVATE_IF_STRICT = """\
ACTIVATE IF:
  - [condition 1 — MUST be a mechanical syntax check: e.g. "E1 is absorbing", \
"E2 has more distinct variables than E1", "left side of E1 has exactly N * operations"]
  - [condition 2 — another mechanical check; omit if only one needed]
  (Every condition must be answerable by counting or inspecting variables/operators \
— no algebraic reasoning, no vague qualifiers like "complex" or "similar".)\
"""

_ACTIVATE_IF_LOOSE = """\
ACTIVATE IF:
  - [condition 1 — describe the structural pattern of E1/E2 that identifies this case]
  - [condition 2 — add if needed]
  (Describe the pattern clearly enough that a model can recognise it.)\
"""

# ---------------------------------------------------------------------------
# Step instructions — emphasis variants
# ---------------------------------------------------------------------------

_STEPS_BOTH = """\
Step 1 — DIAGNOSE: What did the weaker model do wrong? Quote or paraphrase the \
exact wrong move from its reasoning trace. Be specific.

Step 2 — CORRECT MOVE: What should the model do instead? State the one mechanical \
check that produces the right answer — answerable by direct inspection of the equations.

Step 3 — TRIGGER: The precise structural conditions that identify this failure pattern. \
Be narrow — a trigger that fires on 2–3 cases correctly is better than one that \
fires on 10 and gets half wrong.

Step 4 — ANTI-TRIGGER: 1–2 similar cases where the weaker model's approach is fine.\
"""

_STEPS_CORRECT = """\
Step 1 — CORRECT MOVE: What is the one mechanical check that produces the right \
answer for these failures? It must be answerable by direct inspection of the \
equations — no proof, no judgment.

Step 2 — TRIGGER: The precise structural conditions that identify when this check \
is needed. Be narrow — prefer a trigger that fires on 2–3 cases correctly over \
one that fires broadly.

Step 3 — ANTI-TRIGGER: 1–2 similar cases where this check is NOT needed.\
"""

_STEPS_WRONG = """\
Step 1 — WRONG MOVE: What does the weaker model consistently do wrong here? \
Quote or paraphrase the exact mistake from the reasoning traces. Be specific — \
name the heuristic, shortcut, or missing check.

Step 2 — TRIGGER: The precise structural conditions that identify when this wrong \
move occurs. Be narrow — prefer a trigger that fires on 2–3 cases correctly.

Step 3 — ANTI-TRIGGER: 1–2 similar cases where the weaker model's approach is \
actually fine.\
"""

# ---------------------------------------------------------------------------
# Output format — 900 vs 400 char budget
# ---------------------------------------------------------------------------

def _output_format_900(activate_if: str, emphasis: str) -> str:
    common_wrong = (
        'COMMON WRONG MOVE: [1 sentence — what the weaker model does wrong here]\n'
        if emphasis in ("both", "wrong") else ""
    )
    next_check = (
        'NEXT CHECK: [the one mechanical thing to do instead — end with "If yes → TRUE/FALSE."]\n'
        if emphasis in ("both", "correct") else ""
    )
    return f"""\
OUTPUT 1 — CASE STUDY (max 900 chars)

=== CASE STUDY: [short title] ===
FAILURE_TYPE: A or B
{activate_if}
DO NOT ACTIVATE IF: [closest case where this should NOT fire]
{common_wrong}{next_check}WHY THIS WORKS: [1–2 sentence justification]
SUPPORT:
  • E1 = ...  |  E2 = ...  |  Answer: TRUE/FALSE  — [brief note]
  • E1 = ...  |  E2 = ...  |  Answer: TRUE/FALSE  — [brief note]
TARGET_STEP: [roadmap aspect this corrects]\
"""


def _output_format_400(activate_if: str, emphasis: str) -> str:
    common_wrong = (
        'COMMON WRONG MOVE: [one clause — the exact mistake]\n'
        if emphasis in ("both", "wrong") else ""
    )
    next_check = (
        'NEXT CHECK: [one mechanical check — end with "If yes → TRUE/FALSE."]\n'
        if emphasis in ("both", "correct") else ""
    )
    return f"""\
OUTPUT 1 — CASE STUDY (max 400 chars — be ruthlessly concise, one key insight only)

=== CASE STUDY: [title] ===
FAILURE_TYPE: A or B
{activate_if}
{common_wrong}{next_check}WHY THIS WORKS: [1 sentence only]
SUPPORT:
  • E1 = ...  |  E2 = ...  |  Answer: TRUE/FALSE  — [brief note]
  • E1 = ...  |  E2 = ...  |  Answer: TRUE/FALSE  — [brief note]
TARGET_STEP: [aspect]\
"""


# ---------------------------------------------------------------------------
# Variant builder
# ---------------------------------------------------------------------------

def build_prompt(
    activate_if: str,   # "strict" | "loose"
    emphasis:    str,   # "both" | "correct" | "wrong"
    length:      int,   # 900 | 400
) -> str:
    """Build a complete CASE_STUDY_WITH_REASONING_PROMPT for the given variant."""
    act_if_block = _ACTIVATE_IF_STRICT if activate_if == "strict" else _ACTIVATE_IF_LOOSE
    steps_block  = {
        "both":    _STEPS_BOTH,
        "correct": _STEPS_CORRECT,
        "wrong":   _STEPS_WRONG,
    }[emphasis]
    output_block = (
        _output_format_900(act_if_block, emphasis)
        if length == 900
        else _output_format_400(act_if_block, emphasis)
    )
    return _PREAMBLE + steps_block + "\n\n" + output_block + _SUFFIX


# ---------------------------------------------------------------------------
# All 12 variants — keyed by (activate_if, emphasis, length)
# ---------------------------------------------------------------------------

ABLATION_VARIANTS: dict[tuple[str, str, int], str] = {
    (act, emp, lng): build_prompt(act, emp, lng)
    for act in ("strict", "loose")
    for emp in ("both", "correct", "wrong")
    for lng in (900, 400)
}

VARIANT_NAMES = {
    (act, emp, lng): f"{act}_{emp}_{lng}"
    for act in ("strict", "loose")
    for emp in ("both", "correct", "wrong")
    for lng in (900, 400)
}
