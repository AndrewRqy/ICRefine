"""
prompts/templates.py — Core LLM prompt templates for the ICR pipeline.

All strings sent to the LLM live here. Token budgets are derived from the
cheatsheet size constants in utils/cheatsheet.py so they stay in sync.

Templates
---------
ROADMAP_PROMPT                   — initial generation: reasoning roadmap
CASE_STUDIES_PROMPT              — initial generation: seed case studies
CASE_STUDY_PROMPT                — training loop: case study from failure bin
CASE_STUDY_WITH_REASONING_PROMPT — reasoning-aware variant (includes post-think)
SCORING_PROMPT                   — scoring: predict TRUE/FALSE given cheatsheet
SCORING_PROMPT_COT_FIRST         — scoring with checkpoint-tagged reasoning trace

Token budgets
-------------
DT_MAX_TOKENS      — max tokens for a roadmap response
CS_MAX_TOKENS      — max tokens for a single case study response
SCORING_MAX_TOKENS — max tokens for a scoring response
FLUSH_MAX_TOKENS   — max tokens for a combined case study + roadmap patch response
"""

from __future__ import annotations

from utils.cheatsheet import CASE_STUDY_MAX_CHARS, ROADMAP_MAX_CHARS

# ~4 chars/token; 1.2× headroom so the LLM can overshoot slightly —
# the hard char cap in cheatsheet.py will truncate the rest.
DT_MAX_TOKENS      = int(ROADMAP_MAX_CHARS    / 4 * 1.2)
CS_MAX_TOKENS      = int(CASE_STUDY_MAX_CHARS / 4 * 1.2)
SCORING_MAX_TOKENS = 8_192

# Case study (~900 chars) + roadmap patch (~800 chars) with 1.3× headroom
FLUSH_MAX_TOKENS = 900


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

{features_block}\
Use the following decision guide to inform your determination:

{cheatsheet}

CRITICAL INSTRUCTION: The VERY FIRST LINE of your response must be either:
  VERDICT: TRUE
  VERDICT: FALSE
Do NOT write anything before this line. Not a single word. Start with VERDICT immediately.
After the verdict line you may provide reasoning, proof, or counterexample.
Even if you are uncertain, you MUST commit to a verdict — write VERDICT: TRUE or VERDICT: FALSE \
based on your best assessment or lean. Never leave the verdict blank or say "I don't know".

Output format:
VERDICT: TRUE or FALSE  ← THIS MUST BE YOUR FIRST LINE, NO EXCEPTIONS.
REASONING: apply the decision guide step by step.
PROOF: if VERDICT is TRUE, provide a proof; otherwise leave empty.
COUNTEREXAMPLE: if VERDICT is FALSE, provide a counterexample magma; otherwise leave empty.\
"""

# Variant: reasoning written BEFORE the verdict so the model cannot anchor on a
# verdict first and reverse-engineer justification.
SCORING_PROMPT_COT_FIRST = """\
You are a mathematician specializing in equational theories of magmas.
Your task is to determine whether Equation 1 ({equation1}) implies Equation 2 \
({equation2}) over all magmas.

{features_block}\
Use the following decision guide to inform your determination:

{cheatsheet}

CRITICAL INSTRUCTION: The VERY FIRST LINE of your response must be either:
  VERDICT: TRUE
  VERDICT: FALSE
Do NOT write anything before this line. Not a single word. Start with VERDICT immediately.
After the verdict line you may provide reasoning, proof, or counterexample.
Even if you are uncertain, you MUST commit to a verdict — write VERDICT: TRUE or VERDICT: FALSE \
based on your best assessment or lean. Never leave the verdict blank or say "I don't know".

Output format:
VERDICT: TRUE or FALSE  ← THIS MUST BE YOUR FIRST LINE, NO EXCEPTIONS.
REASONING: apply the decision guide aspect by aspect. For each ASPECT you consult, \
begin that clause with its checkpoint tag: [CK:A1] for ASPECT 1, [CK:A2] for ASPECT 2, \
etc.
PROOF: if VERDICT is TRUE, provide a proof; otherwise leave empty.
COUNTEREXAMPLE: if VERDICT is FALSE, provide a counterexample magma; otherwise leave empty.\
"""


# ---------------------------------------------------------------------------
# Reasoning-aware case study generation (ICR_reasoning / ICR_select)
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

=== PATTERNS ALREADY COVERED — YOUR CASE STUDY MUST NOT RESTATE THESE ===
{already_covered}
Your new case study MUST address a gap NOT covered above. If your ACTIVATE IF
conditions would fire on the same equation pairs as any pattern listed above,
you are restating existing knowledge — discard that idea and find a genuinely
new pattern in the failure examples.
=== END ALREADY COVERED ===

The following examples were ALL predicted INCORRECTLY by the weaker model.
The ground-truth verdict and the weaker model's wrong reasoning are shown.
Where available, a CORRECT oracle reasoning trace from a stronger model is shown \
for contrast — this is your primary signal for what the weaker model is missing.

=== FAILURES WITH INCORRECT MODEL REASONING ===

{failure_lines}

=== YOUR TASK ===

{polarity_instruction}

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

  FEATURE VOCABULARY — use these exact terms in ACTIVATE IF conditions wherever
  applicable (the scoring model can read them from the PRECOMPUTED FEATURES block):
    bare(E1), vars(E1), size(E1), imb(E1), LP(E1), RP(E1), XOR(E1), AB(E1)
    topShape(E1): "v-m" (var*product), "m-v" (product*var), "m-m" (product*product)
    xTop(E1): "left" | "right" | "both"  (where bare var x sits in top split)
    Lx(E1): TRUE if leftmost RHS variable is x  |  Rx(E1): TRUE if rightmost is x
    square(E1): TRUE if RHS contains u*u subterm
    rhsVars(E1): distinct variable count on RHS (product side)

  For TRUE-polarity bins with bare E1 and proj_class=nested, express the trigger
  using topShape/xTop/Lx/Rx/square/rhsVars — these map directly to the STEP 0B
  contradiction motifs (C1–C13) in the decision protocol. Name the motif if it
  applies (e.g. "matches C4: rhsVars=3, Lx=FALSE, Rx=FALSE, xTop=right, topShape=v-m").

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
