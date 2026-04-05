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
    DECISION_TREE_PROMPT,
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

=== DECISION TREE (for context) ===
{decision_tree}

=== CURRENT CASE STUDIES ({n_current} entries) ===
{case_studies}

Rewrite these as exactly {n_target} condensed case studies that:
1. Cover all distinct structural patterns from the originals.
2. Eliminate redundant examples — keep the most informative one per pattern.
3. Each entry fits within 600 characters.
4. Each entry references the specific decision tree step it corrects.
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

=== DECISION TREE (what the model was following) ===
{decision_tree}

=== FAILURES WITH REASONING TRACES ({n_failures} items) ===
{failure_lines}

For each decision tree step or rule, identify:
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
You are rewriting specific broken steps in a decision tree for magma equation \
implication.  The model has been systematically misapplying the steps listed \
below.  Rewrite ONLY those steps to fix the identified problems.  Leave all \
other steps exactly as written.

=== STEP ANALYSIS (what is broken and why) ===
{step_analysis}

=== CURRENT DECISION TREE ===
{decision_tree}

Requirements for the revised tree:
- Fix only the broken steps — do not restructure or rename working steps.
- Make the fix concrete and mechanical: add explicit checks, worked examples \
  inline, or clarifying sub-rules so the same mistake cannot recur.
- The entire revised tree must fit within 2,500 characters.
- Preserve the existing step numbering and headers.

Output ONLY the complete revised decision tree — no preamble, no commentary.\
"""

DT_REVISION_MAX_TOKENS = DT_MAX_TOKENS
