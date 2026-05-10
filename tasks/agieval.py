"""tasks/agieval.py — TaskSpec for AGIEval English subsets.

Tasks:
  agieval_lsat_ar   — LSAT Analytical Reasoning (5-option A–E)
  agieval_lsat_lr   — LSAT Logical Reasoning    (5-option A–E)
  agieval_logiqa_en — LogiQA English            (4-option A–D)
"""
from __future__ import annotations

from utils.task_spec import TaskSpec
from tasks.utils import (
    _make_eval_prompt,
    _parse_mc,
    _mc_correct,
    _mc_label,
    _extract_reasoning,
    _format_failure,
    _rule_score_prompt,
    _generic_polarity,
)


_AGIEVAL_SCORING = """\
You are answering a {task_label} question.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

VERDICT: {verdict_fmt}  ← FIRST LINE.
REASONING: Apply the cheatsheet strategies step by step. \
Name the specific rule or strategy you used. \
If no cheatsheet rule applied, say so and reason from first principles."""

_AGIEVAL_SCORING_COT = """\
You are answering a {task_label} question.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Think step by step, applying the cheatsheet strategies. Then give your verdict.

VERDICT: {verdict_fmt}  ← FIRST LINE.
REASONING: Apply the cheatsheet strategies step by step. \
Name the specific rule or strategy you used."""


_AGIEVAL_SCORING_RF = """\
You are answering a {task_label} question.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Work through the problem step by step using the cheatsheet, then state your answer.
Your response MUST end with this exact line (no other text after it):
VERDICT: (A)   ← replace (A) with your chosen option"""


def _make_agieval_scoring_prompt(task_label: str, verdict_fmt: str):
    def _score(cs: str, item: dict, cot: bool = False) -> str:
        t = _AGIEVAL_SCORING_COT if cot else _AGIEVAL_SCORING
        return t.format(
            task_label=task_label,
            cheatsheet=cs,
            question=item["input"],
            verdict_fmt=verdict_fmt,
        )
    return _score


def _make_agieval_scoring_prompt_rf(task_label: str):
    def _score(cs: str, item: dict, cot: bool = True) -> str:
        return _AGIEVAL_SCORING_RF.format(
            task_label=task_label,
            cheatsheet=cs,
            question=item["input"],
        )
    return _score


def _parse_agieval_rf(text: str) -> str | None:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip()
            import re
            m = re.search(r"\(([A-E])\)", val, re.IGNORECASE)
            if m:
                return f"({m.group(1).upper()})"
    return None


def _agieval_partition_key(item: dict) -> tuple:
    label = item.get("semantic_label")
    if label and label != "(unknown)":
        return (label,)
    return (True,)


def _agieval_key_to_conds(key: tuple) -> list[str]:
    if len(key) == 1 and key[0] is not True:
        return [f"question type: {key[0]}"]
    return ["reasoning question from this task domain"]


import re as _re

def _lsat_ar_could_be_true_subtype(item: dict) -> str:
    """Split could_be_true into list-type vs single-element based on question stem."""
    # Extract the final question sentence (after the last blank line or at end of input)
    text = item.get("input", "")
    # The question stem is typically the last paragraph before the answer choices
    parts = _re.split(r'\n\s*\n', text.strip())
    stem = parts[-1] if parts else text
    # "list" questions ask for a complete or partial arrangement
    if _re.search(r'\b(list|order|schedule|arrangement|sequence|assign)\b', stem, _re.IGNORECASE):
        return "could_be_true_list"
    return "could_be_true_single"


def _lsat_ar_cross_partition_key(item: dict) -> tuple:
    """Cross-partition by game type × question type for lsat_ar.
    could_be_true is further split into list vs single-element sub-types.
    """
    game  = item.get("semantic_label", "")
    qtype = item.get("question_type_label", "")
    if not game or game == "(unknown)":
        return (True,)
    if not qtype or qtype == "(unknown)":
        return (game,)
    if qtype == "could_be_true":
        qtype = _lsat_ar_could_be_true_subtype(item)
    return (game, qtype)


def _lsat_ar_cross_key_to_conds(key: tuple) -> list[str]:
    if len(key) == 2:
        return [f"game type: {key[0]}", f"question type: {key[1]}"]
    if len(key) == 1 and key[0] is not True:
        return [f"game type: {key[0]}"]
    return ["LSAT Analytical Reasoning question"]


# ── Per-task label prompts ────────────────────────────────────────────────────

_LSAT_AR_LABEL_PROMPT = """\
Analyze this LSAT Analytical Reasoning (logic game) question and identify the game type.

QUESTION:
{question}

Identify the PRIMARY logic game structure:
- ordering: arrange entities in a linear sequence (first through last, earliest to latest)
- grouping: divide entities into two or more groups or categories
- assignment: assign attributes or roles to each entity (one property per entity)
- mapping: place entities in a spatial or positional grid (rows/columns, offices/floors)
- hybrid: combines two structures (e.g., ordering + grouping, assignment + ordering)
- other: does not fit the above categories

Output EXACTLY two lines:
GAME_TYPE: <type>
REASON: <one sentence explanation>"""


def _build_lsat_ar_label_prompt(item: dict) -> str:
    return _LSAT_AR_LABEL_PROMPT.format(question=item.get("input", "").strip())


_LSAT_AR_TYPES = {"ordering", "grouping", "assignment", "mapping", "hybrid", "other"}


def _parse_lsat_ar_label(text: str) -> str:
    for line in text.strip().splitlines():
        if line.upper().startswith("GAME_TYPE:"):
            label = line.split(":", 1)[1].strip().lower()
            if label in _LSAT_AR_TYPES:
                return label
            for t in _LSAT_AR_TYPES:
                if t in label:
                    return t
    return "other"


_LSAT_AR_QT_LABEL_PROMPT = """\
Analyze this LSAT Analytical Reasoning (logic game) question and identify the QUESTION TYPE \
based on what is being asked — not the game structure.

QUESTION:
{question}

Identify the PRIMARY question type from what the question stem asks you to determine:
- must_be_true: which answer MUST be true / is necessarily true given the constraints
- cannot_be_true: which answer CANNOT be true / is impossible given the constraints
- could_be_true: which answer COULD be true / is possible (may be true, could be accurate, etc.)
- complete_arrangement: give a complete and accurate list or ordering of all entities
- minimum_maximum: find the minimum or maximum number of a specific entity placement
- if_then: hypothetically, IF a given condition holds, what must/could/cannot be true
- other: does not clearly fit the above types

Output EXACTLY two lines:
QUESTION_TYPE: <type>
REASON: <one sentence explanation>"""


def _build_lsat_ar_qt_label_prompt(item: dict) -> str:
    return _LSAT_AR_QT_LABEL_PROMPT.format(question=item.get("input", "").strip())


_LSAT_AR_QT_TYPES = {
    "must_be_true", "cannot_be_true", "could_be_true",
    "complete_arrangement", "minimum_maximum", "if_then", "other",
}


def _parse_lsat_ar_qt_label(text: str) -> str:
    for line in text.strip().splitlines():
        if line.upper().startswith("QUESTION_TYPE:"):
            label = line.split(":", 1)[1].strip().lower().replace(" ", "_")
            if label in _LSAT_AR_QT_TYPES:
                return label
            for t in _LSAT_AR_QT_TYPES:
                if t in label or label in t:
                    return t
    return "other"


_LSAT_LR_LABEL_PROMPT = """\
Analyze this LSAT Logical Reasoning question and identify the question type.

QUESTION:
{question}

Identify the PRIMARY question type from the stem (the final question being asked):
- strengthen: find evidence or reasoning that supports the argument's conclusion
- weaken: find evidence or reasoning that undermines the argument's conclusion
- assumption: identify an unstated premise the argument requires or takes for granted
- inference: identify what must be true / can be concluded from the given information
- flaw: identify the logical error or reasoning mistake in the argument
- parallel: find an argument with the same logical structure as the given argument
- explain: resolve an apparent paradox or reconcile two seemingly contradictory facts
- evaluate: identify information that would most help assess whether the argument is correct
- other: question type does not fit the above categories

Output EXACTLY two lines:
QUESTION_TYPE: <type>
REASON: <one sentence explanation>"""


def _build_lsat_lr_label_prompt(item: dict) -> str:
    return _LSAT_LR_LABEL_PROMPT.format(question=item.get("input", "").strip())


_LSAT_LR_TYPES = {
    "strengthen", "weaken", "assumption", "inference",
    "flaw", "parallel", "explain", "evaluate", "other",
}


def _parse_lsat_lr_label(text: str) -> str:
    for line in text.strip().splitlines():
        if line.upper().startswith("QUESTION_TYPE:"):
            label = line.split(":", 1)[1].strip().lower()
            if label in _LSAT_LR_TYPES:
                return label
            for t in _LSAT_LR_TYPES:
                if t in label:
                    return t
    return "other"


_LOGIQA_LABEL_PROMPT = """\
Analyze this LogiQA logical reading comprehension question and identify the reasoning type.

QUESTION:
{question}

Identify the PRIMARY reasoning pattern required to answer the question:
- deductive: apply a general rule or principle to reach a specific conclusion (top-down)
- causal: identify cause-and-effect relationships or explain why something happened
- conditional: reason about if-then relationships, necessary/sufficient conditions
- analogy: apply reasoning by structural similarity between two situations
- inductive: generalize from specific examples to a broader pattern or conclusion
- other: does not fit the above categories

Output EXACTLY two lines:
REASONING_TYPE: <type>
REASON: <one sentence explanation>"""


def _build_logiqa_label_prompt(item: dict) -> str:
    return _LOGIQA_LABEL_PROMPT.format(question=item.get("input", "").strip())


_LOGIQA_TYPES = {"deductive", "causal", "conditional", "analogy", "inductive", "other"}


def _parse_logiqa_label(text: str) -> str:
    for line in text.strip().splitlines():
        if line.upper().startswith("REASONING_TYPE:"):
            label = line.split(":", 1)[1].strip().lower()
            if label in _LOGIQA_TYPES:
                return label
            for t in _LOGIQA_TYPES:
                if t in label:
                    return t
    return "other"


_AGIEVAL_GEN_PROMPT_TMPL = (
    "You are an expert in {task_label} helping a model that keeps failing on these questions.\n\n"
    "=== EXISTING CASE STUDIES ===\n{{case_studies}}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{{already_covered}}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{{failure_lines}}\n\n"
    "=== YOUR TASK ===\n{{polarity_instruction}}\n\n"
    "DIAGNOSE:\n"
    "  1. What reasoning step is the model missing in these failures?\n"
    "  2. What is the key distinction between the correct answer and the distractors?\n"
    "  3. What general pattern do these failures share?\n\n"
    "TRANSFERABILITY REQUIREMENT:\n"
    "Your case study must encode an ABSTRACT REASONING PATTERN — not memorize specific questions.\n"
    "  • DO NOT copy specific text, names, or scenarios from the failures above.\n"
    "  • SUPPORT examples must be freshly invented scenarios illustrating the pattern.\n"
    "  • The teaching note should apply to any similar question in this domain.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [{task_label}]: [Memorable Pattern Name] ===\n"
    "FAILURE_TYPE: A (wrong reasoning strategy) or B (right strategy, wrong application)\n"
    "ACTIVATE IF:\n"
    "  - question involves: [describe the reasoning structure that triggers this case study]\n"
    "  - the key step is: [what the model needs to do to distinguish the correct answer]\n"
    "DO NOT ACTIVATE IF: [superficially similar question where the standard approach works]\n"
    "COMMON WRONG MOVE: [what the model incorrectly selects and why it is fooled]\n"
    "NEXT CHECK: [a plain question to resolve the ambiguity → points to the correct option]\n"
    "WHY THIS WORKS: [1-2 sentences on the underlying reasoning principle]\n"
    "SUPPORT:\n"
    "  • [concrete example scenario]  |  Answer: (X)  — [brief note]\n"
    "{{retry_context}}"
)


def _make_agieval_gen_prompt(task_label: str) -> str:
    return _AGIEVAL_GEN_PROMPT_TMPL.format(task_label=task_label)


def _format_agieval_for_csicl(item: dict, oracle: dict) -> str:
    """Serialize one agieval item for CS-ICL warm-start generation."""
    question = item.get("input", "").strip()
    answer   = item.get("answer", "?")
    reasoning = item.get("reason", "") or oracle.get(item.get("id", ""), "")
    if reasoning:
        return f"Question:\n{question}\n\nReasoning: {reasoning}\n\nAnswer: {answer}"
    return f"Question:\n{question}\n\nAnswer: {answer}"


def _make_agieval_task(
    task_name: str,
    task_label: str,
    verdict_fmt: str,
    patch_domain: str,
    build_label_prompt=None,
    parse_label=None,
    label_field: str = "semantic_label",
    partition_key=None,
    partition_key_to_conditions=None,
) -> TaskSpec:
    return TaskSpec(
        build_scoring_prompt=_make_agieval_scoring_prompt(task_label, verdict_fmt),
        is_correct=_mc_correct,
        answer_label=_mc_label,
        parse_verdict=_parse_mc,
        extract_post_think=_extract_reasoning,
        partition_key=partition_key or _agieval_partition_key,
        partition_key_to_conditions=partition_key_to_conditions or _agieval_key_to_conds,
        format_failure=_format_failure,
        generation_prompt_template=_make_agieval_gen_prompt(task_label),
        build_polarity_instruction=_generic_polarity,
        task_name=task_name,
        build_scoring_prompt_rf=_make_agieval_scoring_prompt_rf(task_label),
        build_rule_scoring_prompt=_rule_score_prompt,
        build_eval_prompt=_make_eval_prompt(verdict_fmt),
        patch_domain=patch_domain,
        build_label_prompt=build_label_prompt,
        parse_label=parse_label,
        label_field=label_field,
        format_for_csicl=_format_agieval_for_csicl,
    )


AGIEVAL_LSAT_AR_TASK = _make_agieval_task(
    "agieval_lsat_ar",
    "LSAT Analytical Reasoning",
    "(A), (B), (C), (D), or (E)",
    "LSAT analytical reasoning: logic games, constraint satisfaction, scheduling, and ordering",
    build_label_prompt=_build_lsat_ar_qt_label_prompt,
    parse_label=_parse_lsat_ar_qt_label,
    label_field="question_type_label",
    partition_key=_lsat_ar_cross_partition_key,
    partition_key_to_conditions=_lsat_ar_cross_key_to_conds,
)

AGIEVAL_LSAT_LR_TASK = _make_agieval_task(
    "agieval_lsat_lr",
    "LSAT Logical Reasoning",
    "(A), (B), (C), (D), or (E)",
    "LSAT logical reasoning: argument analysis, assumptions, strengthening/weakening, and logical flaws",
    build_label_prompt=_build_lsat_lr_label_prompt,
    parse_label=_parse_lsat_lr_label,
)

AGIEVAL_LOGIQA_EN_TASK = _make_agieval_task(
    "agieval_logiqa_en",
    "LogiQA Logical Reading Comprehension",
    "(A), (B), (C), or (D)",
    "logical reading comprehension: deductive reasoning, categorical logic, and argument evaluation",
    build_label_prompt=_build_logiqa_label_prompt,
    parse_label=_parse_logiqa_label,
)

# ── RF scoring functions (for RULE augmentation pipeline) ────────────────────
_agieval_lsat_ar_scoring_prompt_rf   = _make_agieval_scoring_prompt_rf("LSAT Analytical Reasoning")
_agieval_lsat_lr_scoring_prompt_rf   = _make_agieval_scoring_prompt_rf("LSAT Logical Reasoning")
_agieval_logiqa_en_scoring_prompt_rf = _make_agieval_scoring_prompt_rf("LogiQA Logical Reading Comprehension")

# Per-task parse aliases so eval_cs_ablation.py can find _parse_{task_name}_rf
_parse_agieval_lsat_ar_rf   = _parse_agieval_rf
_parse_agieval_lsat_lr_rf   = _parse_agieval_rf
_parse_agieval_logiqa_en_rf = _parse_agieval_rf
