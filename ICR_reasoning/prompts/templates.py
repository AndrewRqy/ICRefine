"""
ICR_reasoning/prompts/templates.py — Prompt templates for the reasoning-aware pipeline.

Shared templates (ROADMAP_PROMPT, CASE_STUDIES_PROMPT, SCORING_PROMPT)
are imported directly from ICR_naive — they are identical.

The key addition is CASE_STUDY_WITH_REASONING_PROMPT, which extends the naive
case study prompt by including each failure's post-think text alongside the
equation pair. Per Heddaya et al. (ACL 2026), post-think preserves logical
scaffolding at 25× higher deductive marker density than external summaries,
making it a better teaching signal for identifying what went wrong.
"""

from __future__ import annotations

# Re-use all shared templates from ICR_naive unchanged
from ICR_naive.prompts.templates import (
    ROADMAP_PROMPT,
    DECISION_TREE_PROMPT,  # backward-compat alias
    CASE_STUDIES_PROMPT,
    SCORING_PROMPT,
    SCORING_PROMPT_COT_FIRST,
    DT_MAX_TOKENS,
    CS_MAX_TOKENS,
    SCORING_MAX_TOKENS,
)

__all__ = [
    "ROADMAP_PROMPT",
    "DECISION_TREE_PROMPT",
    "CASE_STUDIES_PROMPT",
    "SCORING_PROMPT",
    "SCORING_PROMPT_COT_FIRST",
    "DT_MAX_TOKENS",
    "CS_MAX_TOKENS",
    "SCORING_MAX_TOKENS",
    "CASE_STUDY_WITH_REASONING_PROMPT",
    "FLUSH_MAX_TOKENS",
]

# Token budget for the combined case study + DT patch response.
# Case study (all fields): ~900 chars = ~225 tokens
# DT patch: ~800 chars = ~200 tokens
# 1.3× headroom → ~550 tokens; round up to give the model room.
FLUSH_MAX_TOKENS = 900


# ---------------------------------------------------------------------------
# Reasoning-aware case study generation
# ---------------------------------------------------------------------------

CASE_STUDY_WITH_REASONING_PROMPT = """\
You are an expert in universal algebra working on equational theories of magmas.
A magma is a set with a single binary operation * and no other axioms.

You are writing a TEACHING NOTE for a weaker reasoning model that keeps making \
the same mistake. Your job is to diagnose WHY it fails and teach the exact fix — \
either a key piece of missing algebraic knowledge, or a wrong/missing reasoning \
pattern compared to how a stronger model handles the same case.

The cheatsheet the model uses has two parts:

=== REASONING ROADMAP ===
{roadmap}
=== END REASONING ROADMAP ===

=== EXISTING CASE STUDIES ===
{case_studies}
=== END CASE STUDIES ===

The following examples were ALL predicted INCORRECTLY by the weaker model.
The ground-truth verdict and the weaker model's wrong reasoning are shown.
Where available, a CORRECT oracle reasoning trace from a stronger model is shown \
for contrast — this is your primary signal for what the weaker model is missing.

=== FAILURES WITH INCORRECT MODEL REASONING ===

{failure_lines}

=== YOUR TASK ===

Step 0 — DIAGNOSE the failure type. Choose exactly one:
  TYPE A — MISSING KNOWLEDGE: The weaker model's reasoning strategy was reasonable
    but it lacks a key algebraic fact (a lemma, identity, or structural property).
    Signal: the oracle trace invokes a fact or consequence that the weaker model
    never considers, even though the weaker model followed a plausible path.
    Example: model doesn't know "absorbing E1 forces all elements equal → TRUE always".
  TYPE B — WRONG/MISSING REASONING PATTERN: The weaker model has the relevant tools
    but applies the wrong one, stops too early, skips a necessary check, or follows
    a bad heuristic that a stronger model avoids.
    Signal: the oracle trace and the weaker model both start similarly, but the
    oracle takes a different branch or performs an extra check the weaker model skips.

Step 1 — For TYPE A: State the missing lemma or algebraic fact precisely in one
  sentence. Explain what structural condition triggers it and what it implies.
  For TYPE B: Quote or paraphrase the exact wrong move from the weaker model's trace.
  Name the correct move the stronger model takes instead, at the same decision point.

Step 2 — Find the CORRECT MOVE: the specific mechanical check that produces the
  right answer. It must be something the model can execute by direct inspection
  of the equation syntax — no proof, no judgment, just structural reading.
  For TYPE A this is often: "check whether [lemma condition] holds → if yes, verdict
  follows immediately." For TYPE B this is the alternative branch/check to take.

Step 3 — Find the TRIGGER: the precise structural conditions that distinguish these
  equations from superficially similar ones where the same mistake would not occur.
  Be narrow. A trigger that fires on too many cases causes regressions and is worse
  than no case study at all. Prefer a trigger that fires on 2–3 cases correctly
  over one that fires on 10 cases and gets half wrong.

Step 4 — Find the ANTI-TRIGGER: 1–2 structurally similar cases where this teaching
  note should NOT fire (the shortcut or the weaker model's approach is actually fine).

Now produce TWO outputs:

OUTPUT 1 — CASE STUDY (max 900 chars)
Write the teaching note in EXACTLY this format, with these exact field names:

=== CASE STUDY: [short title — name the missing lemma (TYPE A) or the mistaken shortcut/structural trap (TYPE B)] ===
FAILURE_TYPE: A or B
ACTIVATE IF:
  - [condition 1 — one structural fact about E1 or E2 that must be true]
  - [condition 2 — ...]
  (All conditions must hold. If any is false, do not use this note.)
DO NOT ACTIVATE IF: [1 sentence — the closest structural case where the shortcut is actually correct]
COMMON WRONG MOVE: [1 sentence — TYPE A: "Does not apply [missing lemma/fact]..."; TYPE B: start with a verb: "Applies...", "Stops at...", "Treats...", "Ignores..."]
NEXT CHECK: [the one mechanical thing to verify instead — must be answerable by direct inspection; end with what it means: "If yes → TRUE. If no → FALSE." or "If yes → proceed to STEP N."]
WHY THIS WORKS: [TYPE A: state the missing lemma explicitly and why it resolves the case. TYPE B: explain why the stronger model's reasoning branch is correct and the weaker model's is not. 1–2 sentences.]
SUPPORT:
  • E1 = ...  |  E2 = ...  |  Answer: TRUE/FALSE  — [one phrase: what structural fact the trigger catches]
  • E1 = ...  |  E2 = ...  |  Answer: TRUE/FALSE  — [one phrase]
TARGET_STEP: [roadmap aspect this corrects, e.g. "ASPECT 2" or "ASPECT 4"]

OUTPUT 2 — ROADMAP PATCH (max 800 chars)
One or more targeted corrections to the reasoning roadmap that would have prevented
these failures. Write only aspects that need to be ADDED or MODIFIED — do not
rewrite the whole roadmap. Name the aspect being refined.

=== ROADMAP PATCH ===
[ASPECT X EXCEPTION / INSERT AFTER ASPECT Y / NEW ASPECT Z]
[corrected or new rule text]
...
=== END PATCH ===

Output ONLY these two sections — no preamble, no sign-off.\
"""
