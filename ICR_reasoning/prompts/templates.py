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
# Core fields (IDENTIFY/ACTION/WHY/EXAMPLES/DOES NOT APPLY TO): ~600 chars = ~150 tokens
# New structured fields (FEATURE_SIGNATURE/COMMON_WRONG_MOVE/TARGET_STEP/NEXT_CHECK): ~200 chars
# DT patch: ~800 chars = ~200 tokens
# 1.3× headroom → ~715 tokens; round up.
FLUSH_MAX_TOKENS = 900


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

Where available, a CORRECT reasoning trace from a high-accuracy oracle model is also
shown for contrast. Use this to understand not just what the model did wrong, but what
the right approach looks like for this structural case.

=== FAILURES WITH INCORRECT MODEL REASONING ===

{failure_lines}

=== YOUR TASK ===

For each failure, the WRONG reasoning is a flawed argument — do not follow it.
Where a CORRECT reasoning (oracle) is shown, use it to understand the right approach.
Identify the EXACT structural feature shared by these specific failures that the
decision tree does not currently handle correctly. Be as specific as possible —
do NOT write a general rule about broad categories like "GENERAL implies GENERAL".
Instead, describe the precise structural fingerprint that these equations share:
  - What is the exact form of E1 (nesting depth, variable pattern, left/right structure)?
  - What is the exact form of E2?
  - What specific feature of this pair causes the model to go wrong?
  - What concrete check would have given the right answer for these cases?

The IDENTIFY section is the most important part. A case study that fires on too
many cases is worse than useless — it will cause regressions. It is better to
write a very narrow rule that only fires on 2-3 cases correctly than a broad
rule that fires on 10 cases but gets half of them wrong.

Produce TWO outputs:

OUTPUT 1 — CASE STUDY (max 800 chars)
Format your response EXACTLY as:

=== CASE STUDY: [short specific title — name the structural feature, not just the error type] ===
IDENTIFY: [precise checklist of conditions that must ALL be true before this fires —
           be specific about equation structure, variable counts, nesting, form types.
           Use one bullet per condition. If any condition is not met, this does NOT apply.]
ACTION: [exactly what to conclude when IDENTIFY conditions are met]
WHY: [1-2 sentences — the mathematical reason this specific structure leads to this verdict]
EXAMPLES:
  • E1 = ...  |  E2 = ...  |  Answer: TRUE/FALSE  — [what structural feature matches]
  • ...
DOES NOT APPLY TO: [1-2 sentence description of similar-looking cases where this rule
                    should NOT fire — the boundary condition]
FEATURE_SIGNATURE: [one compact tag summarising the structural pattern, e.g. "absorbing→general_L0" or "standard_3var_implies_standard_1var"]
COMMON_WRONG_MOVE: [one sentence — what the model typically does wrong in these cases]
TARGET_STEP: [the decision tree step or rule this case study corrects, e.g. "STEP 4" or "RULE 5"]
NEXT_CHECK: [what to do after this fires — either "DONE: TRUE", "DONE: FALSE", or "PROCEED TO: STEP N"]

OUTPUT 2 — DECISION TREE PATCH (max 800 chars)
One or more targeted corrections to the decision tree that would have prevented
these failures. Write only the steps that need to be ADDED or MODIFIED — do not
rewrite the whole tree. Each patch item should name the original step it refines
(e.g. "STEP 4 EXCEPTION" or "INSERT AFTER STEP 3") and state the corrected rule.

=== DECISION TREE PATCH ===
[STEP X EXCEPTION / INSERT AFTER STEP Y / NEW STEP Z]
[corrected or new rule text]
...
=== END PATCH ===

Output ONLY these two sections — no preamble, no sign-off.\
"""
