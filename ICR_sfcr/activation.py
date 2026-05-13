"""ICR_sfcr/activation.py — USE WHEN routing and cheatsheet construction.

Two inference modes:
  global  — anchor + ALL accepted rules prepended unconditionally
  routed  — anchor + only rules whose USE WHEN trigger matches the input question

V1 routing uses simple keyword overlap: tokenise the USE WHEN clause,
strip English stop words, and check whether any content term appears in
the lowercased question text.  The plan explicitly calls for "a simple
keyword or symbolic trigger" — this is not meant to be a retrieval system.
"""
from __future__ import annotations

import re

# Minimal English stop-word list — enough to filter article/preposition noise
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "by", "for", "with", "from", "as", "into", "through", "during",
    "and", "or", "but", "not", "no", "nor", "so", "yet", "both", "either",
    "this", "that", "these", "those", "it", "its", "if", "then", "than",
    "when", "where", "which", "who", "what", "how", "there", "their",
    "they", "we", "you", "he", "she", "i", "my", "your", "our", "his",
    "her", "its", "about", "between", "each", "more", "most", "other",
    "some", "such", "only", "same", "also", "any", "all",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _keywords(text: str) -> set[str]:
    """Extract lowercased non-stop-word tokens from text."""
    tokens = _TOKEN_RE.findall(text.lower())
    return {t for t in tokens if t not in _STOP_WORDS and len(t) > 2}


def matches_use_when(use_when: str, question_text: str) -> bool:
    """
    Return True if the USE WHEN clause has any keyword overlap with question_text.
    An empty USE WHEN clause always matches (unconditional rule).
    """
    if not use_when.strip():
        return True
    trigger_kws = _keywords(use_when)
    if not trigger_kws:
        return True
    question_kws = _keywords(question_text)
    return bool(trigger_kws & question_kws)


def _rule_block(rule: dict) -> str:
    lines = [f"RULE: {rule['rule']}"]
    if rule.get("use_when"):
        lines.append(f"USE WHEN: {rule['use_when']}")
    if rule.get("do_not_use_when"):
        lines.append(f"DO NOT USE WHEN: {rule['do_not_use_when']}")
    if rule.get("check"):
        lines.append(f"CHECK: {rule['check']}")
    return "\n".join(lines)


def build_cheatsheet(
    anchor: str,
    accepted_rules: list[dict],
    mode: str = "global",
    question_text: str = "",
) -> str:
    """
    Construct the final cheatsheet for a given question.

    mode="global"  — append all accepted rules unconditionally
    mode="routed"  — append only rules whose USE WHEN matches question_text
    """
    if not accepted_rules:
        return anchor

    if mode == "global":
        active = accepted_rules
    elif mode == "routed":
        active = [
            r for r in accepted_rules
            if matches_use_when(r.get("use_when", ""), question_text)
        ]
    else:
        raise ValueError(f"Unknown routing mode: {mode!r}. Use 'global' or 'routed'.")

    if not active:
        return anchor

    rule_section = "\n\n--- ADDITIONAL RULES ---\n" + "\n\n".join(
        _rule_block(r) for r in active
    )
    return anchor.rstrip() + rule_section


def activation_summary(
    accepted_rules: list[dict],
    items: list[dict],
) -> dict:
    """
    Return per-rule activation statistics over a set of items.
    Used for logging and reporting.
    """
    summary = {}
    for i, rule in enumerate(accepted_rules):
        use_when = rule.get("use_when", "")
        hits = sum(
            1 for it in items
            if matches_use_when(use_when, it.get("input", ""))
        )
        summary[f"rule_{i}"] = {
            "rule_prefix":   rule["rule"][:60],
            "use_when":      use_when[:80],
            "activation_n":  hits,
            "activation_pct": hits / len(items) if items else 0.0,
        }
    return summary
