"""ICR_sfcr/activation.py - USE WHEN routing (lexical + feature-tag).

SF-CR treats a memory atom as:
    (rule_text, activation_predicate, boundary_predicate)

Two routing strategies are supported:

  keyword (default)
      Deterministic lexical matching over USE WHEN / DO NOT USE WHEN text.
      At least ``min_matches`` content terms must appear in the item text.

  feature
      Task-aware tag-based routing via ICR_sfcr.features.
      Requires the rule to carry ``positive_tags`` / ``negative_tags`` fields.
      Falls back to keyword routing when tags are absent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# Minimal English stop-word list plus generic instruction words that make
# routing too broad for SF-CR safety validation.
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "by", "for", "with", "from", "as", "into", "through", "during",
    "and", "or", "but", "not", "no", "nor", "so", "yet", "both", "either",
    "this", "that", "these", "those", "it", "its", "if", "then", "than",
    "when", "where", "which", "who", "what", "how", "there", "their",
    "they", "we", "you", "he", "she", "i", "my", "your", "our", "his",
    "her", "about", "between", "each", "more", "most", "other", "some",
    "such", "only", "same", "also", "any", "all",
    # Generic reasoning/prompting terms.
    "verify", "consider", "logic", "reason", "reasoning", "explicitly",
    "check", "ask", "determine", "question", "answer", "rule", "use",
    "using", "applies", "apply", "input", "contains", "contains",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_QUOTED_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')


RouterType = Literal["keyword", "feature"]


@dataclass(frozen=True)
class ActivationConfig:
    min_matches: int = 2
    allow_empty_use_when_global: bool = False


@dataclass
class ActivationDetails:
    active: bool
    use_when_terms: list[str] = field(default_factory=list)
    do_not_use_when_terms: list[str] = field(default_factory=list)
    use_when_phrases: list[str] = field(default_factory=list)
    do_not_use_when_phrases: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    matched_phrases: list[str] = field(default_factory=list)
    boundary_matched_terms: list[str] = field(default_factory=list)
    boundary_matched_phrases: list[str] = field(default_factory=list)
    vetoed_by_boundary: bool = False
    empty_use_when: bool = False
    # Feature routing extras
    router_type: str = "keyword"
    matched_pos_tags: list[str] = field(default_factory=list)
    matched_neg_tags: list[str] = field(default_factory=list)
    item_features: list[str] = field(default_factory=list)


def item_text(item: dict) -> str:
    """Return a stable text representation for activation checks."""
    if item.get("input"):
        return str(item["input"])
    fields = []
    for key in ("question", "prompt", "text", "sentence", "equation1", "equation2"):
        if key in item and item[key] is not None:
            fields.append(str(item[key]))
    if fields:
        return "\n".join(fields)
    return str(item)


def _terms(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokens:
        if tok in _STOP_WORDS or len(tok) <= 2:
            continue
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def _normalise_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _phrases(text: str) -> list[str]:
    phrases: list[str] = []

    for match in _QUOTED_RE.finditer(text):
        raw = match.group(1) or match.group(2) or ""
        phrase = _normalise_phrase(raw)
        if len(_terms(phrase)) >= 2:
            phrases.append(phrase)

    # Comma/newline/semicolon separated clauses are useful as lightweight
    # phrase predicates when they are specific enough.
    for raw in re.split(r"[,;\n]+", text):
        phrase = _normalise_phrase(raw)
        if len(_terms(phrase)) >= 3:
            phrases.append(phrase)

    seen: set[str] = set()
    out: list[str] = []
    for phrase in phrases:
        if phrase not in seen:
            seen.add(phrase)
            out.append(phrase)
    return out


def parse_activation_text(text: str) -> tuple[list[str], list[str]]:
    """Return (content_terms, content_phrases) for a predicate string."""
    return _terms(text), _phrases(text)


def _match_terms(terms: list[str], question_terms: set[str]) -> list[str]:
    return [t for t in terms if t in question_terms]


def _match_phrases(phrases: list[str], question_text: str) -> list[str]:
    q = _normalise_phrase(question_text)
    return [p for p in phrases if p and p in q]


def activation_details(
    rule: dict,
    item: dict,
    *,
    min_matches: int = 2,
    min_veto_matches: int = 1,
    allow_empty_use_when_global: bool = False,
) -> ActivationDetails:
    """Evaluate USE WHEN / DO NOT USE WHEN for one rule-item pair.

    min_veto_matches: how many DO NOT USE WHEN content-terms (or phrases) must
    match before the veto fires.  Default 1 preserves the original behaviour.
    Set to 2 to require at least two discriminating terms before blocking a rule,
    which prevents broad domain vocabulary from vetoing every item in a task.
    """
    use_when = str(rule.get("use_when", "") or "")
    do_not = str(rule.get("do_not_use_when", "") or "")
    q_text = item_text(item)
    q_terms = set(_terms(q_text))

    use_terms, use_phrases = parse_activation_text(use_when)
    veto_terms, veto_phrases = parse_activation_text(do_not)
    matched_terms = _match_terms(use_terms, q_terms)
    matched_phrases = _match_phrases(use_phrases, q_text)
    boundary_terms = _match_terms(veto_terms, q_terms)
    boundary_phrases = _match_phrases(veto_phrases, q_text)

    empty_use_when = not use_when.strip()
    if empty_use_when:
        active = allow_empty_use_when_global
    elif matched_phrases:
        active = True
    else:
        active = len(matched_terms) >= max(1, min_matches)

    vetoed = (len(boundary_terms) + len(boundary_phrases)) >= max(1, min_veto_matches)
    if vetoed:
        active = False

    return ActivationDetails(
        active=active,
        use_when_terms=use_terms,
        do_not_use_when_terms=veto_terms,
        use_when_phrases=use_phrases,
        do_not_use_when_phrases=veto_phrases,
        matched_terms=matched_terms,
        matched_phrases=matched_phrases,
        boundary_matched_terms=boundary_terms,
        boundary_matched_phrases=boundary_phrases,
        vetoed_by_boundary=vetoed,
        empty_use_when=empty_use_when,
    )


def activation_details_feature(
    rule: dict,
    item: dict,
    task: str,
    *,
    min_tag_matches: int = 1,
    min_veto_matches: int = 1,
    allow_empty_use_when_global: bool = False,
) -> ActivationDetails:
    """Feature-tag routing variant of activation_details.

    Uses positive_tags / negative_tags from the rule dict when available.
    Falls back to keyword routing when the rule has no positive_tags.
    """
    from .features import extract_item_features, extract_rule_tags, route_by_features

    pos_tags, neg_tags = extract_rule_tags(rule)

    if not pos_tags:
        # No feature tags defined — fall back to keyword routing
        det = activation_details(
            rule, item,
            min_matches=2,
            min_veto_matches=min_veto_matches,
            allow_empty_use_when_global=allow_empty_use_when_global,
        )
        det.router_type = "keyword_fallback"
        return det

    item_features = extract_item_features(task, item)
    vetoed_tags   = list(neg_tags & item_features)
    matched_pos   = list(pos_tags & item_features)
    vetoed        = len(vetoed_tags) >= min_veto_matches if neg_tags else bool(vetoed_tags)
    active        = (not vetoed) and (len(matched_pos) >= min_tag_matches)

    return ActivationDetails(
        active=active,
        router_type="feature",
        matched_pos_tags=matched_pos,
        matched_neg_tags=vetoed_tags,
        item_features=sorted(item_features),
        vetoed_by_boundary=bool(vetoed_tags),
        empty_use_when=not bool(pos_tags),
    )


def matches_use_when(
    use_when: str,
    question_text: str,
    *,
    min_matches: int = 2,
    allow_empty_use_when_global: bool = False,
) -> bool:
    """Backward-compatible USE WHEN matcher."""
    return activation_details(
        {"use_when": use_when, "do_not_use_when": ""},
        {"input": question_text},
        min_matches=min_matches,
        allow_empty_use_when_global=allow_empty_use_when_global,
    ).active


def route_rule(
    rule: dict,
    item: dict,
    *,
    min_matches: int = 2,
    min_veto_matches: int = 1,
    allow_empty_use_when_global: bool = False,
    router_type: RouterType = "keyword",
    task: str = "",
) -> bool:
    """Return True when rule should be exposed to item under routed mode."""
    if router_type == "feature" and task:
        return activation_details_feature(
            rule, item, task,
            min_tag_matches=1,
            min_veto_matches=min_veto_matches,
            allow_empty_use_when_global=allow_empty_use_when_global,
        ).active
    return activation_details(
        rule,
        item,
        min_matches=min_matches,
        min_veto_matches=min_veto_matches,
        allow_empty_use_when_global=allow_empty_use_when_global,
    ).active


def _rule_block(rule: dict, *, memory_format: str = "rule") -> str:
    """Render a rule dict as a formatted text block.

    memory_format:
      "rule"              — RULE + USE WHEN + DO NOT USE WHEN only  (B1)
      "rule_check"        — + CHECK field                            (B2)
      "rule_check_example"— + CHECK + MICRO-EXAMPLE                  (B3)
    """
    lines = [f"RULE: {rule['rule']}"]
    if rule.get("use_when"):
        lines.append(f"USE WHEN: {rule['use_when']}")
    if rule.get("do_not_use_when"):
        lines.append(f"DO NOT USE WHEN: {rule['do_not_use_when']}")
    if memory_format in ("rule_check", "rule_check_example") and rule.get("check"):
        lines.append(f"CHECK: {rule['check']}")
    if memory_format == "rule_check_example" and rule.get("micro_example"):
        lines.append(f"MICRO-EXAMPLE: {rule['micro_example']}")
    return "\n".join(lines)


def build_cheatsheet(
    anchor: str,
    accepted_rules: list[dict],
    mode: str = "global",
    question_text: str = "",
    *,
    router_min_matches: int = 2,
    router_min_veto_matches: int = 1,
    allow_empty_use_when_global: bool = False,
    router_type: RouterType = "keyword",
    task: str = "",
    memory_format: str = "rule",
) -> str:
    """
    Construct the final cheatsheet for a question.

    mode="global" appends all rules unconditionally.
    mode="routed" appends only rules whose activation predicate matches.
    memory_format controls how each rule block is rendered.
    """
    if not accepted_rules:
        return anchor

    if mode == "global":
        active = accepted_rules
    elif mode == "routed":
        item = {"input": question_text}
        active = [
            r for r in accepted_rules
            if route_rule(
                r,
                item,
                min_matches=router_min_matches,
                min_veto_matches=router_min_veto_matches,
                allow_empty_use_when_global=allow_empty_use_when_global,
                router_type=router_type,
                task=task,
            )
        ]
    else:
        raise ValueError(f"Unknown routing mode: {mode!r}. Use 'global' or 'routed'.")

    if not active:
        return anchor

    rule_section = "\n\n--- ADDITIONAL RULES ---\n" + "\n\n".join(
        _rule_block(r, memory_format=memory_format) for r in active
    )
    return anchor.rstrip() + rule_section


def activation_summary(
    accepted_rules: list[dict],
    items: list[dict],
    *,
    router_min_matches: int = 2,
    router_min_veto_matches: int = 1,
    allow_empty_use_when_global: bool = False,
) -> dict:
    """Return per-rule activation statistics over a set of items."""
    summary = {}
    for i, rule in enumerate(accepted_rules):
        details = [
            activation_details(
                rule,
                it,
                min_matches=router_min_matches,
                min_veto_matches=router_min_veto_matches,
                allow_empty_use_when_global=allow_empty_use_when_global,
            )
            for it in items
        ]
        hits = sum(1 for d in details if d.active)
        vetoes = sum(1 for d in details if d.vetoed_by_boundary)
        terms, veto_terms = parse_activation_text(rule.get("use_when", ""))[0], parse_activation_text(rule.get("do_not_use_when", ""))[0]
        summary[f"rule_{i}"] = {
            "rule_prefix": rule["rule"][:60],
            "use_when": rule.get("use_when", "")[:120],
            "use_when_terms": terms,
            "do_not_use_when_terms": veto_terms,
            "activation_n": hits,
            "activation_pct": hits / len(items) if items else 0.0,
            "vetoed_by_boundary_count": vetoes,
        }
    return summary
