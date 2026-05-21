"""ICR_sfcr/prompts.py — Prompt templates for SFCR rule generation."""

# ---------------------------------------------------------------------------
# Flat pool
# ---------------------------------------------------------------------------

RULE_GENERATION_PROMPT = """\
Cheatsheet:
{anchor_cheatsheet}

SHARED FAILURES — fix these:
{v_shared_block}

BOUNDARY CASES — must NOT trigger:
{v_private_block}

Write one rule addressing the shared failures. Do not duplicate cheatsheet content.

RULE:
USE WHEN:"""

# ---------------------------------------------------------------------------
# Subtype-targeted
# ---------------------------------------------------------------------------

RULE_GENERATION_PROMPT_SUBTYPE = """\
Cheatsheet:
{anchor_cheatsheet}

TARGET SUBTYPE — {subtype_description}:
{v_subtype_block}

OTHER FAILURES (context only):
{v_other_block}

BOUNDARY CASES — must NOT trigger:
{v_private_block}

Write one rule for the TARGET SUBTYPE only. Do not duplicate cheatsheet content.

RULE:
USE WHEN:"""

# ---------------------------------------------------------------------------
# Subtype clustering
# ---------------------------------------------------------------------------

SUBTYPE_CLUSTER_PROMPT = """\
Group these failing examples into 2-4 subtypes by shared failure pattern.

{v_shared_block}

Output valid JSON only:
{{"subtypes": [{{"label": "short label", "description": "one sentence", "indices": [0, 3, 7]}}]}}

Indices are 0-based. Each example must appear in exactly one subtype."""

# ---------------------------------------------------------------------------
# Repair loop
# ---------------------------------------------------------------------------

RULE_REPAIR_PROMPT = """\
This rule was rejected: {reject_reason}

RULE: {rule}
USE WHEN: {use_when}

{mis_triggered_section}
{no_gain_section}

Rewrite USE WHEN to fix the rejection. Keep the core rule insight.

RULE:
USE WHEN:"""

# ---------------------------------------------------------------------------
# Compressed rationale
# ---------------------------------------------------------------------------

COMPRESSED_RATIONALE_PROMPT = """\
Summarize the key insight from this reasoning in one sentence.

{reasoning}

One-sentence insight:"""
