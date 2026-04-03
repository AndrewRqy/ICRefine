"""
ICR_reasoning/prompts/templates.py — Prompt templates for the reasoning-aware pipeline.

Shared templates (DECISION_TREE_PROMPT, CASE_STUDIES_PROMPT, SCORING_PROMPT)
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
    DECISION_TREE_PROMPT,
    CASE_STUDIES_PROMPT,
    SCORING_PROMPT,
    SCORING_PROMPT_COT_FIRST,
    DT_MAX_TOKENS,
    CS_MAX_TOKENS,
    SCORING_MAX_TOKENS,
)

__all__ = [
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
# Case study: ~600 chars = ~150 tokens
# DT patch:   ~800 chars = ~200 tokens
# 1.3× headroom → 455 tokens; round up.
FLUSH_MAX_TOKENS = 600


# ---------------------------------------------------------------------------
# Reasoning-aware case study generation
# ---------------------------------------------------------------------------

CASE_STUDY_WITH_REASONING_PROMPT = """\
You are an expert in universal algebra working on equational theories of magmas.
A magma is a set with a single binary operation * and no other axioms.

You are improving a cheatsheet that an LLM uses to decide whether E1 implies E2.
The cheatsheet has two parts:

=== DECISION TREE ===
{decision_tree}
=== END DECISION TREE ===

=== EXISTING CASE STUDIES ===
{case_studies}
=== END CASE STUDIES ===

The following examples were ALL predicted INCORRECTLY by the model.
The ground-truth verdict is shown. The model's post-think reasoning is also shown —
this reasoning IS WRONG and led to the wrong verdict.

=== FAILURES WITH INCORRECT MODEL REASONING ===

{failure_lines}

=== YOUR TASK ===

For each failure, the model's reasoning IS WRONG — treat it as a flawed argument.
Identify the EXACT step or claim where it breaks down:
  - Which decision tree step did it apply incorrectly?
  - Did it misclassify the equation form, miscount variables, or miss a structural feature?
  - What should the correct reasoning have been?

Produce TWO outputs:

OUTPUT 1 — CASE STUDY (max 600 chars)
A new case study that names the specific flaw, gives the correct rule, and cites
2-4 failures with a one-line note on where their reasoning broke.

OUTPUT 2 — DECISION TREE PATCH (max 800 chars)
One or more targeted corrections to the decision tree that would have prevented
these failures. Write only the steps that need to be ADDED or MODIFIED — do not
rewrite the whole tree. Each patch item should name the original step it refines
(e.g. "STEP 4 EXCEPTION" or "INSERT AFTER STEP 3") and state the corrected rule.

Format your response EXACTLY as:

=== CASE STUDY: [short descriptive title] ===
PATTERN: [the exact decision tree step or rule the model misapplied]
RULE: IF [condition] THEN [TRUE / FALSE / lean TRUE / lean FALSE]
WHY: [1-2 sentences showing why the model's conclusion is wrong]
EXAMPLES:
  • E1 = ...  |  E2 = ...  |  Answer: TRUE/FALSE  — [where the reasoning broke]
  • ...
EXCEPTIONS: [one sentence, or "None"]

=== DECISION TREE PATCH ===
[STEP X EXCEPTION / INSERT AFTER STEP Y / NEW STEP Z]
[corrected or new rule text]
...
=== END PATCH ===

Output ONLY these two sections — no preamble, no sign-off.\
"""
