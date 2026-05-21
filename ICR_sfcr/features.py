"""ICR_sfcr/features.py — Task-aware feature extraction for feature-tag routing.

Replaces raw keyword matching over USE WHEN with task-specific structural tags.
The router matches rule positive_tags against item feature tags, enabling
precise activation even when surface vocabulary differs.

Public API
----------
extract_item_features(task, item) -> set[str]
    Extract all applicable tags for an item given its task.

extract_rule_tags(rule) -> tuple[set[str], set[str]]
    Return (positive_tags, negative_tags) from a rule dict.

route_by_features(rule, item, task, min_tag_matches=1) -> bool
    Return True when the rule should activate on the item under feature routing.
    A rule with no positive_tags falls back to returning False (caller should
    use lexical routing instead).
"""
from __future__ import annotations

import re
from typing import Callable

# ---------------------------------------------------------------------------
# Tag vocabularies (one set per task)
# ---------------------------------------------------------------------------

# Causal Judgement — causal structure subtypes
_CJ_TAGS: frozenset[str] = frozenset({
    "joint_and",
    "necessary_not_sufficient",
    "simultaneous_conditions",
    "independently_sufficient",
    "overdetermination",
    "preemption",
    "omission_non_action",
    "norm_violation",
    "intentional_action",
    "background_condition",
    "temporal_order",
})

# Geometric Shapes — SVG/path structural features
_GS_TAGS: frozenset[str] = frozenset({
    "has_arc",
    "arc_equal_radii",
    "arc_nonzero_rotation",
    "multi_subpath",
    "closed_path",
    "open_path",
    "line_only",
    "n_vertices_3",
    "n_vertices_4",
    "n_vertices_5",
    "n_vertices_6",
    "n_vertices_7",
    "n_vertices_8",
    "quadrilateral_candidate",
    "triangle_candidate",
    "circle_or_arc_candidate",
    "ellipse_or_arc_candidate",
})

# LogiQA / LSAT-LR — logical reasoning question subtypes
_LOGIQA_TAGS: frozenset[str] = frozenset({
    "comparative_constraint",
    "ranking",
    "conditional_chain",
    "necessary_assumption",
    "must_be_true",
    "weaken",
    "strengthen",
    "flaw",
    "parallel_reasoning",
    "principle",
    "resolve_paradox",
    "quantifier_scope",
    "causal_explanation",
    "definition_question",
    "analogy",
    "general_principle",
})

# Disambiguation QA — pronoun resolution features
_DQ_TAGS: frozenset[str] = frozenset({
    "pronoun_present",
    "ambiguous_pronoun",
    "two_candidate_antecedents",
    "number_agreement",
    "gender_agreement",
    "nearest_candidate",
    "syntactic_subject_candidate",
    "object_candidate",
    "recency_conflict",
    "world_knowledge_required",
    "single_candidate",
})

TASK_TAG_VOCAB: dict[str, frozenset[str]] = {
    "causal_judgement":       _CJ_TAGS,
    "geometric_shapes":       _GS_TAGS,
    "agieval_logiqa_en":      _LOGIQA_TAGS,
    "agieval_lsat_lr":        _LOGIQA_TAGS,
    "disambiguation_qa":      _DQ_TAGS,
}

# ---------------------------------------------------------------------------
# Causal Judgement feature extractor
# ---------------------------------------------------------------------------

_CJ_PATTERNS: list[tuple[str, list[str]]] = [
    ("joint_and", [
        r"\bboth\b", r"\bonly if both\b", r"\btogether\b",
        r"\bat the same time\b", r"\bnot enough by itself\b",
        r"\bjointly\b", r"\ball (of|at once)\b",
    ]),
    ("independently_sufficient", [
        r"\beither\b.{0,40}\bwould\b", r"\beach would\b",
        r"\bwould have happened anyway\b", r"\bindependently\b",
        r"\bsufficient (on its own|alone|by itself)\b",
    ]),
    ("preemption", [
        r"\bbefore .{0,30} could\b", r"\bprevented\b", r"\bstopped\b",
        r"\bpre.?empt\b", r"\bbeaten to\b",
    ]),
    ("norm_violation", [
        r"\bsupposed to\b", r"\bnot supposed to\b",
        r"\bagainst the rule\b", r"\bignored (the )?signal\b",
        r"\bviolat\b", r"\bforbidden\b", r"\bprohibited\b",
    ]),
    ("omission_non_action", [
        r"\bfailed to\b", r"\bdid not\b", r"\bdidn.t\b",
        r"\bneglected\b", r"\brefused\b", r"\bomission\b",
        r"\binaction\b",
    ]),
    ("intentional_action", [
        r"\bintentionally\b", r"\bdeliberately\b", r"\bon purpose\b",
        r"\bmeant to\b", r"\bwanted to\b",
    ]),
    ("overdetermination", [
        r"\boverdet\b", r"\bboth (causes|would have)\b",
        r"\bwould have occurred (anyway|regardless)\b",
        r"\bduplicate (cause|factor)\b",
    ]),
    ("background_condition", [
        r"\benabling condition\b", r"\bbackground\b",
        r"\bprecondition\b", r"\bnecessary background\b",
    ]),
    ("temporal_order", [
        r"\bbefore\b", r"\bafter\b", r"\bfirst\b", r"\bthen\b",
        r"\bsequence\b", r"\border of\b",
    ]),
    ("necessary_not_sufficient", [
        r"\bnecessary but not sufficient\b", r"\bnecessary condition\b",
        r"\brequired but\b",
    ]),
    ("simultaneous_conditions", [
        r"\bat the same time\b", r"\bsimultaneously\b",
        r"\bconcurrently\b", r"\bco.?occur\b",
    ]),
]


def _extract_cj_features(item: dict) -> set[str]:
    text = _item_text(item).lower()
    tags: set[str] = set()
    for tag, patterns in _CJ_PATTERNS:
        for pat in patterns:
            if re.search(pat, text):
                tags.add(tag)
                break
    return tags


# ---------------------------------------------------------------------------
# Geometric Shapes feature extractor
# ---------------------------------------------------------------------------

_GS_ARC_RE = re.compile(r"\bA\s+[\d.]+\s+[\d.]+\s+([\d.]+)", re.IGNORECASE)
_GS_M_RE   = re.compile(r"\bM\b",  re.IGNORECASE)
_GS_L_RE   = re.compile(r"\bL\b",  re.IGNORECASE)
_GS_Z_RE   = re.compile(r"\bZ\b",  re.IGNORECASE)


def _extract_gs_features(item: dict) -> set[str]:
    text = _item_text(item)
    tags: set[str] = set()

    arcs = _GS_ARC_RE.findall(text)
    if arcs:
        tags.add("has_arc")
        radii_vals = re.findall(r"\bA\s+([\d.]+)\s+([\d.]+)", text, re.IGNORECASE)
        if any(r1 == r2 for r1, r2 in radii_vals):
            tags.add("arc_equal_radii")
            tags.add("circle_or_arc_candidate")
        else:
            tags.add("ellipse_or_arc_candidate")
        rotations = [float(r) for r in arcs]
        if any(r != 0.0 for r in rotations):
            tags.add("arc_nonzero_rotation")

    m_count = len(_GS_M_RE.findall(text))
    if m_count > 1:
        tags.add("multi_subpath")

    l_count = len(_GS_L_RE.findall(text))
    if l_count > 0 and not arcs:
        tags.add("line_only")

    if _GS_Z_RE.search(text):
        tags.add("closed_path")
    else:
        tags.add("open_path")

    # Vertex count heuristic: count L commands + 1 for start M
    n_vertices = l_count + 1 if l_count > 0 else 0
    tag = f"n_vertices_{n_vertices}"
    if tag in _GS_TAGS:
        tags.add(tag)
    if n_vertices == 4:
        tags.add("quadrilateral_candidate")
    elif n_vertices == 3:
        tags.add("triangle_candidate")

    return tags


# ---------------------------------------------------------------------------
# LogiQA / LSAT-LR feature extractor
# ---------------------------------------------------------------------------

_LOGIQA_PATTERNS: list[tuple[str, list[str]]] = [
    ("comparative_constraint", [
        r"\bmore (than|likely)\b", r"\bless (than|likely)\b",
        r"\bgreater\b", r"\bsmaller\b", r"\bhigher\b", r"\blower\b",
        r"\btaller\b", r"\bolder\b", r"\byounger\b", r"\bheavier\b",
    ]),
    ("ranking", [
        r"\brank\b", r"\border\b", r"\bfirst\b.*\bsecond\b",
        r"\bhighest\b", r"\blowest\b", r"\bbest\b", r"\bworst\b",
        r"\bposition\b", r"\bshortest\b", r"\btallest\b",
        r"\boldest\b", r"\byoungest\b", r"\bheaviest\b", r"\blightest\b",
    ]),
    ("conditional_chain", [
        r"\bif .{0,40} then\b", r"\bonly if\b", r"\bunless\b",
        r"\bwhenever\b", r"\bprovided that\b",
    ]),
    ("necessary_assumption", [
        r"\bnecessary assumption\b", r"\bthe argument (assumes|presupposes)\b",
        r"\bmust be true for\b", r"\brequires that\b",
    ]),
    ("must_be_true", [
        r"\bmust be true\b", r"\bcannot be false\b",
        r"\bnecessarily (true|follows)\b", r"\bwhich of the following must\b",
    ]),
    ("weaken", [
        r"\bweaken\b", r"\bundermine\b", r"\bcast doubt\b",
        r"\bchallenges\b", r"\bcounts against\b",
    ]),
    ("strengthen", [
        r"\bstrengthen\b", r"\bsupport\b", r"\bbolster\b",
        r"\bcounts in favor\b", r"\bmost helps\b",
    ]),
    ("flaw", [
        r"\bflaw\b", r"\berror in reasoning\b", r"\bfallacy\b",
        r"\bvulnerable to\b", r"\bweakness\b",
    ]),
    ("parallel_reasoning", [
        r"\bparallel\b", r"\bmost similar\b", r"\bsame pattern\b",
        r"\banalogous reasoning\b",
    ]),
    ("resolve_paradox", [
        r"\bresolve\b", r"\bexplain (the|this) (paradox|discrepancy|apparent)\b",
        r"\breconcile\b", r"\bapparently contradictory\b",
    ]),
    ("causal_explanation", [
        r"\bcause\b", r"\bexplain why\b", r"\blead to\b",
        r"\bresult in\b", r"\bdue to\b",
    ]),
    ("definition_question", [
        r"\bdefine\b", r"\bmeaning of\b", r"\bwhat (is|does) .{0,20} mean\b",
        r"\baccording to the passage\b",
    ]),
    ("quantifier_scope", [
        r"\ball\b.{0,30}\bsome\b", r"\bsome\b.{0,30}\bnone\b",
        r"\beveryone\b", r"\bno one\b", r"\bat least one\b", r"\bevery\b",
    ]),
    ("principle", [
        r"\bprinciple\b", r"\bgeneral rule\b", r"\bpolicy\b",
        r"\bstandard\b", r"\bguideline\b",
    ]),
    ("analogy", [
        r"\banalogy\b", r"\bjust as\b", r"\bsimilar to\b",
        r"\blike .{0,20} so\b",
    ]),
    ("general_principle", [
        r"\bin general\b", r"\bbroadly\b", r"\btypically\b",
        r"\busually\b", r"\boften\b",
    ]),
]


def _extract_logiqa_features(item: dict) -> set[str]:
    text = _item_text(item).lower()
    tags: set[str] = set()
    for tag, patterns in _LOGIQA_PATTERNS:
        for pat in patterns:
            if re.search(pat, text):
                tags.add(tag)
                break
    return tags


# ---------------------------------------------------------------------------
# Disambiguation QA feature extractor
# ---------------------------------------------------------------------------

_DQ_PRONOUN_RE = re.compile(
    r"\b(he|she|they|him|her|them|his|hers|their|it|its)\b", re.IGNORECASE
)
_DQ_AMBIG_PRONOUNS = {"he", "she", "they", "him", "her", "them", "his", "hers", "their"}
_DQ_NAME_RE = re.compile(r"\b[A-Z][a-z]+\b")


def _extract_dq_features(item: dict) -> set[str]:
    text = _item_text(item)
    tags: set[str] = set()
    lower = text.lower()

    pronouns = _DQ_PRONOUN_RE.findall(text)
    if pronouns:
        tags.add("pronoun_present")
        ambig = [p for p in pronouns if p.lower() in _DQ_AMBIG_PRONOUNS]
        if ambig:
            tags.add("ambiguous_pronoun")

    names = _DQ_NAME_RE.findall(text)
    unique_names = len(set(names))
    if unique_names >= 2:
        tags.add("two_candidate_antecedents")
    elif unique_names == 1:
        tags.add("single_candidate")

    # Gender agreement heuristics
    has_she = bool(re.search(r"\b(she|her|hers)\b", lower))
    has_he  = bool(re.search(r"\b(he|him|his)\b", lower))
    if has_she or has_he:
        tags.add("gender_agreement")

    # Number agreement heuristics
    has_they = bool(re.search(r"\b(they|them|their)\b", lower))
    if has_they and unique_names >= 2:
        tags.add("number_agreement")

    if "world knowledge" in lower or "generally known" in lower:
        tags.add("world_knowledge_required")

    return tags


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_EXTRACTORS: dict[str, Callable[[dict], set[str]]] = {
    "causal_judgement":  _extract_cj_features,
    "geometric_shapes":  _extract_gs_features,
    "agieval_logiqa_en": _extract_logiqa_features,
    "agieval_lsat_lr":   _extract_logiqa_features,
    "disambiguation_qa": _extract_dq_features,
}


def _item_text(item: dict) -> str:
    """Return the primary text of an item for feature extraction."""
    if item.get("input"):
        return str(item["input"])
    parts = []
    for key in ("question", "prompt", "text", "sentence"):
        if item.get(key):
            parts.append(str(item[key]))
    return "\n".join(parts) if parts else str(item)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_item_features(task: str, item: dict) -> set[str]:
    """Extract all applicable feature tags for an item given its task.

    Returns an empty set for unknown tasks (falls back to lexical routing).
    """
    extractor = _EXTRACTORS.get(task)
    if extractor is None:
        return set()
    return extractor(item)


def extract_rule_tags(rule: dict) -> tuple[set[str], set[str]]:
    """Return (positive_tags, negative_tags) from a rule dict.

    Tags may be stored as a list (YAML/JSON) or a comma-separated string.
    """
    def _to_set(val) -> set[str]:
        if not val:
            return set()
        if isinstance(val, (list, tuple)):
            return {str(t).strip() for t in val if str(t).strip()}
        return {t.strip() for t in str(val).split(",") if t.strip()}

    pos = _to_set(rule.get("positive_tags"))
    neg = _to_set(rule.get("negative_tags"))
    return pos, neg


def route_by_features(
    rule: dict,
    item: dict,
    task: str,
    *,
    min_tag_matches: int = 1,
) -> bool:
    """Return True when rule should activate on item under feature-tag routing.

    Logic:
    1. Extract positive and negative tags from the rule.
    2. If no positive_tags defined → return False (use lexical fallback).
    3. Extract item features for the task.
    4. Veto: if any negative_tag is in item features → False.
    5. Activate: if |positive_tags ∩ item_features| >= min_tag_matches → True.
    """
    pos_tags, neg_tags = extract_rule_tags(rule)
    if not pos_tags:
        return False

    item_features = extract_item_features(task, item)

    # Veto check — any single negative tag match blocks the rule
    if neg_tags & item_features:
        return False

    # Activation check
    return len(pos_tags & item_features) >= min_tag_matches
