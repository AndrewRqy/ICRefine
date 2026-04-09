"""
prompts/templates.py — All LLM prompt templates and their token budgets.

Every string that gets sent to an LLM lives here. To change what the model
is asked to do, edit the relevant template. Token budgets are derived from
the cheatsheet size constants so they stay in sync automatically.

Templates
---------
ROADMAP_PROMPT         — initial generation: ask LLM to write a reasoning roadmap
CASE_STUDIES_PROMPT    — initial generation: ask LLM to write seed case studies
CASE_STUDY_PROMPT      — training loop: ask LLM to write a case study from failures
SCORING_PROMPT         — training/eval: ask LLM to predict TRUE/FALSE for one pair

Token budgets
-------------
DT_MAX_TOKENS          — max tokens for a roadmap response
CS_MAX_TOKENS          — max tokens for a single case study response
SCORING_MAX_TOKENS     — max tokens for a scoring response (short)
"""

from __future__ import annotations

from ..core.cheatsheet import CASE_STUDY_MAX_CHARS, ROADMAP_MAX_CHARS

# ~4 chars/token; 1.2× headroom so the LLM can overshoot slightly —
# the hard char cap in cheatsheet.py will truncate the rest.
DT_MAX_TOKENS      = int(ROADMAP_MAX_CHARS    / 4 * 1.2)
CS_MAX_TOKENS      = int(CASE_STUDY_MAX_CHARS / 4 * 1.2)
SCORING_MAX_TOKENS = 8_000   # 8K is enough for DeepSeek-R1-14B think + structured output; 16K lets it exhaust the budget mid-think


# ---------------------------------------------------------------------------
# Initial cheatsheet generation — reasoning roadmap
# ---------------------------------------------------------------------------

ROADMAP_PROMPT = """\
You are an expert in universal algebra, specifically in equational theories of magmas.
A magma is a set with a single binary operation * and no other axioms.
"E1 implies E2" means every magma satisfying E1 also satisfies E2.

Below are labeled examples showing whether E1 implies E2.

=== LABELED EXAMPLES ===

{example_lines}

=== YOUR TASK ===

Design a REASONING ROADMAP — a structured guide that tells the model HOW to think
about whether E1 implies E2, not just which bucket to classify into.

Requirements:
- Each aspect must have a clear structural CHECK (something you can compute by \
  inspecting the equations, no proof required) with a definitive IF YES / IF NO outcome.
- Aspects should be ordered from most reliable / highest signal to least.
- Cover at least: trivial/singleton/absorbing/standard/general form detection, \
  variable count comparison, left-side operation count comparison, \
  substitution instance check, and a default fallback.
- Ground every aspect in evidence from the examples above.
- Write for an LLM reader: be explicit, use examples inline, avoid vague language.

LENGTH CONSTRAINT: The entire roadmap must fit in 2,500 characters.
Be dense and precise — one or two lines per aspect, inline examples in brackets.
Do not pad with filler; every sentence must be actionable.

Output ONLY the roadmap text — no preamble.
Format each aspect as:
  ASPECT N: [short name]
  CHECK: [the specific mechanical question]
  IF YES: [what to conclude or do next]
  IF NO: [what to conclude or do next]\
"""

# Backward-compat alias
DECISION_TREE_PROMPT = ROADMAP_PROMPT


# ---------------------------------------------------------------------------
# Initial cheatsheet generation — seed case studies
# ---------------------------------------------------------------------------

CASE_STUDIES_PROMPT = """\
You are an expert in universal algebra, specifically in equational theories of magmas.

Here is a reasoning roadmap for determining whether E1 implies E2 over all magmas:

{roadmap}

Below are additional labeled examples that illustrate specific patterns:

=== LABELED EXAMPLES ===

{example_lines}

=== YOUR TASK ===

Write {n_studies} CASE STUDIES drawn from the examples above.
Each case study should:
  - Identify a specific structural pattern present in several examples
  - Explain why that pattern leads to TRUE or FALSE
  - State a concrete rule: IF [condition] THEN [verdict]
  - Give 2-3 inline examples from the dataset as evidence

LENGTH CONSTRAINT: Each case study must fit in 600 characters.
Keep PATTERN to one sentence, WHY to two sentences max, each EXAMPLE to one line.
Omit anything that does not directly support the rule.

Format each case study as:
=== CASE STUDY: [short descriptive title] ===
PATTERN: [one sentence describing the structural feature]
RULE: IF [condition] THEN [verdict]
WHY: [1-2 sentence mathematical explanation]
EXAMPLES:
  • E1 = ...  |  E2 = ...  |  Answer: TRUE/FALSE  — [brief reason]
  • ...

Output ONLY the case studies, separated by a blank line. No preamble.\
"""


# ---------------------------------------------------------------------------
# Training loop — case study from failure bin
# ---------------------------------------------------------------------------

CASE_STUDY_PROMPT = """\
You are an expert in universal algebra working on equational theories of magmas.
A magma is a set with a single binary operation * and no other axioms.

You are improving a decision cheatsheet that an LLM uses to determine whether
Equation 1 (E1) implies Equation 2 (E2) over all magmas.

=== CURRENT CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

The cheatsheet made WRONG predictions on the following examples:

{failure_lines}

=== YOUR TASK ===

These failures share a structural pattern the current cheatsheet does not handle.
Write ONE new case study that:
  1. Identifies the specific structural pattern common to (most of) these failures.
  2. Explains mathematically why that pattern leads to the correct verdict.
  3. States a concrete, actionable rule: IF [structural condition] THEN [verdict].
  4. Uses 2-4 of the failure examples as inline evidence.
  5. Notes any exceptions or sub-cases within this failure batch.

LENGTH CONSTRAINT: The entire case study must fit in 600 characters.
Keep PATTERN to one sentence, WHY to two sentences max, each EXAMPLE to one line.
EXCEPTIONS in one sentence or "None". Cut anything that doesn't support the rule.

Format:
=== CASE STUDY: [short descriptive title] ===
PATTERN: [one sentence — the structural feature that identifies this class]
RULE: IF [condition on E1 and/or E2] THEN [TRUE / FALSE / lean TRUE / lean FALSE]
WHY: [1-2 sentence mathematical justification]
EXAMPLES:
  • E1 = ...  |  E2 = ...  |  Answer: TRUE/FALSE  — [brief reason]
  • ...
EXCEPTIONS: [one sentence, or "None"]

Output ONLY the case study section — no preamble, no sign-off.\
"""


# ---------------------------------------------------------------------------
# Scoring — predict TRUE/FALSE for one pair given the cheatsheet
# ---------------------------------------------------------------------------

SCORING_PROMPT = """\
You are a mathematician specializing in equational theories of magmas.
Your task is to determine whether Equation 1 ({equation1}) implies Equation 2 \
({equation2}) over all magmas.

Use the following decision guide to inform your determination:

{cheatsheet}

Output format (use exact headers without any additional text or formatting):
VERDICT: must be exactly TRUE or FALSE (in the same line).
REASONING: must be non-empty. If you apply ASPECT sections from the decision guide, \
open each clause with its checkpoint tag — [CK:A1] for ASPECT 1, [CK:A2] for ASPECT 2, \
and so on — so that the aspect you are applying is unambiguous.
PROOF: if VERDICT is TRUE, provide a proof; otherwise leave empty.
COUNTEREXAMPLE: if VERDICT is FALSE, provide a counterexample magma; otherwise leave empty.\
"""

# Variant: reasoning written BEFORE the verdict so the model cannot anchor on a
# verdict first and reverse-engineer justification. This forces a genuine reasoning
# trace that is available as post_think for the case study generator.
SCORING_PROMPT_COT_FIRST = """\
You are a mathematician specializing in equational theories of magmas.
Your task is to determine whether Equation 1 ({equation1}) implies Equation 2 \
({equation2}) over all magmas.

Use the following decision guide to inform your determination:

{cheatsheet}

Work through your reasoning step by step BEFORE stating your verdict.
Output format (use exact headers in this exact order, no extra text):
REASONING: apply the decision guide aspect by aspect. For each ASPECT you consult, \
begin that clause with its checkpoint tag: [CK:A1] for ASPECT 1, [CK:A2] for ASPECT 2, \
etc. Then explain which check fires and why. This tagging is required — it is used to \
track which aspects are applied correctly vs. incorrectly.
VERDICT: must be exactly TRUE or FALSE (in the same line, after REASONING).
PROOF: if VERDICT is TRUE, provide a proof; otherwise leave empty.
COUNTEREXAMPLE: if VERDICT is FALSE, provide a counterexample magma; otherwise leave empty.\
"""
