"""
parser.py — Parse a neurico-style cheatsheet (jinja2 or plain text) into a RuleSet.

Sections are delimited by lines of 44+ dashes. Within each section, rules are
identified by known ID prefixes. Multi-line rules (e.g. TR-4 with VERIFY annotation)
are captured until the next rule starts.

Sections modelled:
  STEP 0  — PROJ-SIMPLE, PROJ-COMPLEX, CPLEMMA
  STEP 0B — M1 … M10
  STEP 1  — feature definitions (no mutable rules, stored as preamble only)
  STEP 2  — SEP-LP, SEP-RP, SEP-SET, SEP-XOR, SEP-AB
  STEP 3  — FG-1 … FG-N
  STEP 4  — TR-N / FR-N (in order of appearance)
  STEP 5  — ER-1 … ER-N
  RESPONSE FORMAT, CASE PATTERNS → footer
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

from .rule import Rule, Section, RuleSet, DIVIDER

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIVIDER_RE = re.compile(r"^-{44,}\s*$", re.MULTILINE)

# Map section title prefix → internal name
SECTION_MAP = {
    "STEP 0B": "STEP 0B",
    "STEP 0":  "STEP 0",
    "STEP 1":  "STEP 1",
    "STEP 2":  "STEP 2",
    "STEP 3":  "STEP 3",
    "STEP 4":  "STEP 4",
    "STEP 5":  "STEP 5",
    "RESPONSE FORMAT": "RESPONSE FORMAT",
    "CASE PATTERNS":   "CASE PATTERNS",
}

# Sections that are purely fixed text (no mutable rules extracted)
FIXED_SECTIONS = {"STEP 1", "RESPONSE FORMAT", "CASE PATTERNS"}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_cheatsheet_file(path: str | Path) -> RuleSet:
    text = Path(path).read_text(encoding="utf-8")
    return parse_cheatsheet_text(text, source_path=str(path))


def parse_cheatsheet_text(text: str, source_path: str = "") -> RuleSet:
    chunks = DIVIDER_RE.split(text)

    # chunks[0] = intro (before first divider)
    # chunks[1] = section title, chunks[2] = section content, ...
    intro = chunks[0]

    sections: List[Section] = []
    footer_parts: List[str] = []
    in_footer = False

    i = 1
    while i < len(chunks) - 1:
        title_chunk = chunks[i].strip()
        content_chunk = chunks[i + 1] if i + 1 < len(chunks) else ""
        i += 2

        section_name = _resolve_section_name(title_chunk)

        if section_name in ("RESPONSE FORMAT", "CASE PATTERNS") or in_footer:
            in_footer = True
            footer_parts.append(f"\n{DIVIDER}\n{title_chunk}\n{DIVIDER}")
            footer_parts.append(content_chunk)
            continue

        if section_name is None:
            # Unknown title — treat as part of current footer if already started
            footer_parts.append(f"\n{DIVIDER}\n{title_chunk}\n{DIVIDER}")
            footer_parts.append(content_chunk)
            continue

        if section_name in FIXED_SECTIONS:
            # No mutable rules; store as a section with only preamble
            sections.append(Section(
                name=section_name,
                title=title_chunk,
                preamble=content_chunk,
                rules=[],
                postamble="",
            ))
            continue

        preamble, rules, postamble = _extract_rules(content_chunk, section_name)
        sections.append(Section(
            name=section_name,
            title=title_chunk,
            preamble=preamble,
            rules=rules,
            postamble=postamble,
        ))

    footer = "".join(footer_parts)
    return RuleSet(intro=intro, sections=sections, footer=footer, source_path=source_path)


# ---------------------------------------------------------------------------
# Rule identification in a reasoning trace
# ---------------------------------------------------------------------------

def identify_triggered_rule(reasoning: str) -> Optional[str]:
    """
    Scan a model's reasoning trace and return the rule ID most likely
    responsible for the (wrong) verdict.

    Returns None if no specific rule can be identified.
    """
    r = reasoning.lower()

    # Check for CASE PATTERNS override first (most common Gemma FP source)
    if any(kw in r for kw in ("case pattern", "leftmost=rightmost", "rhs shape y*(y", "forcing sentence")):
        return "CASE-PATTERNS"

    # Check for CPLEMMA (non-bare constant-product lemma)
    if any(kw in r for kw in ("constant-product", "cplemma", "non-bare constant")):
        return "CPLEMMA"

    # Rule IDs to check, ordered so that more specific patterns come first
    ordered_ids = [
        "TR-1", "TR-2", "TR-3", "TR-4", "TR-5",
        "FR-1", "FR-2",
        "FG-1", "FG-2", "FG-3", "FG-4",
        "ER-1", "ER-2", "ER-3", "ER-4",
        "M10", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
        "SEP-LP", "SEP-RP", "SEP-SET", "SEP-XOR", "SEP-AB",
    ]
    trigger_patterns = re.compile(
        r"(fires|match(?:es)?|trigger(?:s)?|→|->)\s*(?:TRUE|FALSE)", re.IGNORECASE
    )
    for rule_id in ordered_ids:
        rid = rule_id.lower().replace("-", r"[\-\s]?")
        pattern = rf"\b{rid}\b"
        hit = re.search(pattern, r)
        if hit:
            # Confirm it's near a trigger word
            context_start = max(0, hit.start() - 40)
            context_end = min(len(r), hit.end() + 80)
            context = r[context_start:context_end]
            if trigger_patterns.search(context) or "fires" in context or "match" in context:
                return rule_id

    # Fallback: just find any mentioned rule ID
    for rule_id in ordered_ids:
        rid = rule_id.lower().replace("-", r"[\-\s]?")
        if re.search(rf"\b{rid}\b", r):
            return rule_id

    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_section_name(title: str) -> Optional[str]:
    for prefix, name in SECTION_MAP.items():
        if title.strip().startswith(prefix):
            return name
    return None


def _extract_rules(content: str, section_name: str) -> Tuple[str, List[Rule], str]:
    """
    Split section content into (preamble, list[Rule], postamble).
    Each rule is identified by a known ID pattern for the given section.
    """
    lines = content.split("\n")
    rule_starts: List[Tuple[int, str]] = []  # (line_index, rule_id)

    for i, line in enumerate(lines):
        rule_id = _detect_rule_id(line, section_name)
        if rule_id is not None:
            rule_starts.append((i, rule_id))

    if not rule_starts:
        return content, [], ""

    first_rule_line = rule_starts[0][0]
    preamble = "\n".join(lines[:first_rule_line])

    rules: List[Rule] = []
    for idx, (start_line, rule_id) in enumerate(rule_starts):
        # Rule text ends just before the next rule starts
        if idx + 1 < len(rule_starts):
            end_line = rule_starts[idx + 1][0]
        else:
            end_line = len(lines)

        # Trim trailing blank lines from rule text
        while end_line > start_line and not lines[end_line - 1].strip():
            end_line -= 1

        rule_text = "\n".join(lines[start_line:end_line])
        verdict = _infer_verdict(rule_text)
        section_key = _section_key(section_name)
        rules.append(Rule(id=rule_id, section=section_key, text=rule_text, verdict=verdict))

    # Postamble: text after the last rule ends (not after it starts)
    last_rule_start = rule_starts[-1][0]
    last_rule_end = len(lines)
    while last_rule_end > last_rule_start and not lines[last_rule_end - 1].strip():
        last_rule_end -= 1
    postamble = "\n".join(lines[last_rule_end:])

    return preamble, rules, postamble


def _detect_rule_id(line: str, section_name: str) -> Optional[str]:
    """Return the rule ID if this line starts a rule, else None."""

    if section_name == "STEP 0":
        stripped = line.strip()
        if re.match(r"^SIMPLE PROJECTIONS", stripped):
            return "PROJ-SIMPLE"
        if re.match(r"^COMPLEX PROJECTIONS", stripped):
            return "PROJ-COMPLEX"
        if re.match(r"^NON-BARE CONSTANT-PRODUCT", stripped):
            return "CPLEMMA"
        return None

    if section_name == "STEP 0B":
        m = re.match(r"^\s{2}(M\d+):", line)
        return m.group(1) if m else None

    if section_name == "STEP 2":
        sep_map = [
            ("SEP-LP",  r"^\s{2}LP\(A\)=T"),
            ("SEP-RP",  r"^\s{2}RP\(A\)=T"),
            ("SEP-SET", r"^\s{2}SET\(A\)=T"),
            ("SEP-XOR", r"^\s{2}XOR\(A\)=T"),
            ("SEP-AB",  r"^\s{2}AB\(A\)=T"),
        ]
        for sep_id, pattern in sep_map:
            if re.match(pattern, line):
                return sep_id
        return None

    if section_name == "STEP 3":
        m = re.match(r"^\s{2}(FG-\d+):", line)
        return m.group(1) if m else None

    if section_name == "STEP 4":
        m = re.match(r"^\s{2}(TR-\d+|FR-\d+):", line)
        return m.group(1) if m else None

    if section_name == "STEP 5":
        m = re.match(r"^\s{2}(ER-\d+):", line)
        return m.group(1) if m else None

    return None


def _infer_verdict(rule_text: str) -> Optional[str]:
    if "→  TRUE" in rule_text or "->  TRUE" in rule_text:
        return "TRUE"
    if "→  FALSE" in rule_text or "->  FALSE" in rule_text:
        return "FALSE"
    return None


def _section_key(section_name: str) -> str:
    return section_name.lower().replace(" ", "")
