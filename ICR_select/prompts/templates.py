"""
ICR_select/prompts/templates.py — All prompt templates for the selective ICR mode.

Imports shared templates from ICR_naive/ICR_reasoning and adds the five new
mechanism prompts:
  SIMILARITY_CHECK_PROMPT  — is this candidate a duplicate of an existing entry?
  MERGE_PROMPT             — merge two similar case studies into one
  CONDENSATION_PROMPT      — rewrite N case studies as M denser ones
"""

from __future__ import annotations

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
from ICR_reasoning.prompts.templates import (
    CASE_STUDY_WITH_REASONING_PROMPT,
    FLUSH_MAX_TOKENS,
)

# ---------------------------------------------------------------------------
# Candidate generation config
# ---------------------------------------------------------------------------

N_CANDIDATES        = 3       # candidates generated per bin flush
CANDIDATE_TEMPS     = [0.3, 0.6, 0.9]   # one per candidate — diversity via temperature
CORRECT_POOL_MAX    = 40      # max items kept in correct pool for regression check

# ---------------------------------------------------------------------------
# Retry context — appended to generation prompt when flush_strategy="retry"
# ---------------------------------------------------------------------------

RETRY_CONTEXT_TEMPLATE = """\
=== PREVIOUS ATTEMPT (REJECTED: {reason_desc}) ===
The following case study was generated but rejected because {reason_desc}:

{prev_candidate}

Items STILL WRONG after applying it ({n_still_wrong} items):
{still_wrong_lines}

Your new case study MUST:
- Use DIFFERENT IDENTIFY conditions than the rejected attempt above.
- Directly address what the previous attempt missed — focus on the patterns \
shared by the still-wrong items.
- NOT copy the ACTION or WHY from the rejected attempt verbatim.\
"""

# Note: {n_still_wrong} is filled by generate_candidates before formatting
# ---------------------------------------------------------------------------
# Similarity gate — deduplicate before adding
# ---------------------------------------------------------------------------

SIMILARITY_CHECK_PROMPT = """\
You are reviewing a NEW case study before adding it to a cheatsheet for magma \
equation implication decisions.

=== EXISTING CASE STUDIES ===
{existing}

=== NEW CANDIDATE ===
{candidate}

Compare the IDENTIFY conditions of the new candidate against each existing entry.
Two case studies are duplicates if their IDENTIFY conditions would fire on the
same set of equation pairs — even if they are worded differently or have different
titles. Focus on the structural conditions (equation forms, variable counts,
nesting patterns), not the surface text.

Reply with EXACTLY one of the following — no explanation, no extra text:
  ADD       — the IDENTIFY conditions are genuinely distinct from all existing entries
  SKIP      — an existing entry has IDENTIFY conditions that cover the same cases
  MERGE:N   — the new candidate's IDENTIFY conditions overlap with entry N and \
should be merged into it (replace N with the integer number shown above)\
"""

SIMILARITY_MAX_TOKENS = 12   # only needs "ADD", "SKIP", or "MERGE:3"

# ---------------------------------------------------------------------------
# Merge — combine two similar case studies
# ---------------------------------------------------------------------------

MERGE_PROMPT = """\
Merge these two case studies about magma equation implication into ONE combined entry.

Rules:
- Combine the IDENTIFY conditions from both: the merged entry should fire when
  either set of conditions is met. If the conditions overlap, write the tightest
  combined checklist that covers both without becoming too broad.
- Keep the most informative example from each.
- The merged result must fit within 600 characters total.
- The DOES NOT APPLY TO section must cover the boundary cases from both originals.

=== CASE STUDY A ===
{cs_a}

=== CASE STUDY B ===
{cs_b}

Output ONLY the merged case study using this exact format:
=== CASE STUDY: [short specific title] ===
IDENTIFY: [combined checklist of conditions — all must be true for this to fire]
ACTION: [what to conclude]
WHY: [1-2 sentence mathematical justification]
EXAMPLES:
  • E1 = ...  |  E2 = ...  |  Answer: TRUE/FALSE  — [brief reason]
DOES NOT APPLY TO: [boundary cases where this should not fire]\
"""

MERGE_MAX_TOKENS = CS_MAX_TOKENS   # same budget as a regular case study

# ---------------------------------------------------------------------------
# Condensation — restructure a growing list into a denser one
# ---------------------------------------------------------------------------

CONDENSATION_PROMPT = """\
You are condensing a growing list of case studies in a cheatsheet for magma \
equation implication. The cheatsheet is getting large; your job is to rewrite \
the case studies as fewer, denser entries that preserve all useful information.

=== REASONING ROADMAP (for context) ===
{roadmap}

=== CURRENT CASE STUDIES ({n_current} entries) ===
{case_studies}

Rewrite these as exactly {n_target} condensed case studies that:
1. Cover all distinct structural patterns from the originals.
2. Eliminate redundant examples — keep the most informative one per pattern.
3. Each entry fits within 600 characters.
4. Each entry references the specific roadmap aspect it corrects.
5. Are ordered from most-common failure pattern to least-common.

Output ONLY the case studies, each starting with:
=== CASE STUDY: [short title] ===
...followed by PATTERN / RULE / WHY / EXAMPLES / EXCEPTIONS fields.\
"""

CONDENSATION_MAX_TOKENS = int(CS_MAX_TOKENS * 4)  # up to 4 condensed entries

# ---------------------------------------------------------------------------
# DT revision — analyse which steps are broken, then rewrite them
# ---------------------------------------------------------------------------

DT_STEP_ANALYSIS_PROMPT = """\
You are analysing why a reasoning model keeps getting wrong answers when \
using a decision tree for magma equation implication.

Below are failures with the model's actual reasoning trace (what it wrote \
step-by-step before giving a wrong verdict).

=== REASONING ROADMAP (what the model was following) ===
{roadmap}

=== FAILURES WITH REASONING TRACES ({n_failures} items) ===
{failure_lines}

For each roadmap aspect or rule, identify:
  1. Was it misapplied in any failure? (wrong classification, wrong rule fired, etc.)
  2. How many failures involved this step?
  3. What exactly went wrong — quote the model's own words.

Output a ranked list of BROKEN STEPS from most to least frequent, like:
BROKEN STEP: [step name, e.g. "STEP 1" or "RULE 4"]
FREQUENCY: [N/total failures]
PROBLEM: [one sentence — what the model consistently gets wrong here]
EVIDENCE: [1-2 direct quotes from the reasoning traces above]
---
(repeat for each broken step, most frequent first)
DONE\
"""

DT_STEP_ANALYSIS_MAX_TOKENS = 1_200

DT_REVISION_PROMPT = """\
You are rewriting specific broken aspects in a reasoning roadmap for magma equation \
implication.  The model has been systematically misapplying the aspects listed \
below.  Rewrite ONLY those aspects to fix the identified problems.  Leave all \
other aspects exactly as written.

=== STEP ANALYSIS (what is broken and why) ===
{step_analysis}

=== CURRENT REASONING ROADMAP ===
{roadmap}

Requirements for the revised tree:
- Fix only the broken steps — do not restructure or rename working steps.
- Make the fix concrete and mechanical: add explicit checks, worked examples \
  inline, or clarifying sub-rules so the same mistake cannot recur.
- The entire revised roadmap must fit within 2,500 characters.
- Preserve the existing aspect numbering and headers.

Output ONLY the complete revised reasoning roadmap — no preamble, no commentary.\
"""

DT_REVISION_MAX_TOKENS = DT_MAX_TOKENS

# ---------------------------------------------------------------------------
# Roadmap synthesis — build a routing controller over the case bank
# ---------------------------------------------------------------------------

ROADMAP_SYNTHESIS_PROMPT = """\
You are writing a REASONING ROADMAP for the task: does "E1 imply E2 over all \
magmas"? (Every magma satisfying E1 also satisfies E2.)

The model has two sources of guidance that work together:
  1. PRIOR KNOWLEDGE — fixed general rules (do NOT restate or restructure these).
  2. CASE BANK — a set of case studies indexed by structural feature. When the
     model encounters an equation pair, the most relevant case studies are surfaced
     automatically based on structural matching. The case studies carry the detailed
     fine-grained reasoning; the roadmap does not need to repeat it.

Your roadmap is the CONTROLLER that bridges these two sources: it tells the model
which structural dimension to probe first and when to consult the case bank.

=== PRIOR KNOWLEDGE (fixed — do not repeat) ===
{prior_knowledge}

=== CASE STUDIES IN THE BANK ===
{case_studies}

=== SAMPLE FAILURES WITH REASONING TRACES ===
{failure_lines}

DESIGN PRINCIPLES — read carefully before writing:

1. The roadmap is NAVIGATION, not knowledge.
   Each aspect identifies a structural dimension and routes the model to the right
   cases or prior-knowledge rules. Do NOT encode verdicts or detailed reasoning that
   belongs in the case studies — that would make this prompt monolithic and would
   override fine-grained case-level guidance on every query.

2. IF YES / IF NO branches name the signal and route — they do NOT give a verdict.
   Correct: "E1 has a fresh rhs variable — case bank entries for absorbing patterns apply."
   Wrong:   "E1 is absorbing — therefore E1 implies E2."
   The case bank, not the roadmap, closes the argument.

3. Every CHECK must be purely mechanical — answerable by counting variables,
   checking set membership, or inspecting syntax. No semantic inference.

4. Keep it short. The roadmap must fit alongside case bank entries in a single prompt.
   Verbosity belongs in the case studies, not here.

Format each aspect as:

ASPECT N: [what structural dimension this probes]
  CHECK: [the specific mechanical question — binary, answerable by syntax inspection]
  IF YES: [one-line structural signal] — consult CASE BANK for [brief pattern description]
  IF NO:  [one-line structural signal] — consult CASE BANK for [brief pattern description] \
or proceed to ASPECT N+1
  WATCH OUT: [one misclassification trap the case bank commonly catches at this checkpoint]

Rules:
- 3 to 5 aspects. More = confusion.
- Order from highest-signal / most reliable check to lowest.
- Aspects are INDEPENDENT — do not chain on each other's output.
- Do not restate anything already clearly handled by the prior knowledge.
- End with a SYNTHESIS note (1–2 sentences): how to combine aspect signals and
  case bank guidance into a final verdict. Do not enumerate structural cases here —
  that is the case bank's job.

Output ONLY the roadmap starting with ASPECT 1 — no preamble.\
"""

ROADMAP_SYNTHESIS_MAX_TOKENS = 1_600
