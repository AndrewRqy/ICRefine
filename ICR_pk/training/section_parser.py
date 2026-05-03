"""
section_parser.py — Parse prior knowledge text into addressable sections.

Sections are delimited by '=== TITLE ===' headers.  The text before the
first header becomes a PREAMBLE section.  Each section is assigned a
stable integer index used for ablation, insertion, and pruning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class PKSection:
    index: int
    title: str
    content: str          # full text including the === HEADER === line
    pruned: bool = False  # marked True when flagged for removal


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_sections(pk_text: str) -> list[PKSection]:
    """
    Split PK text into sections on '=== ... ===' boundaries.
    Returns sections in document order with contiguous integer indices.
    """
    header_re = re.compile(r"(===\s*[^=\n]+?\s*===)", re.MULTILINE)
    parts = header_re.split(pk_text.strip())

    sections: list[PKSection] = []
    idx = 0

    # Text before the first header — treat as a preamble section
    if parts and parts[0].strip():
        sections.append(PKSection(index=idx, title="PREAMBLE", content=parts[0].strip()))
        idx += 1

    i = 1
    while i < len(parts):
        header = parts[i].strip()
        body   = parts[i + 1].strip() if i + 1 < len(parts) else ""
        title  = re.sub(r"^===\s*|\s*===$", "", header).strip()
        content = f"{header}\n\n{body}" if body else header
        sections.append(PKSection(index=idx, title=title, content=content))
        idx += 1
        i += 2

    return sections


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_pk(sections: list[PKSection], skip_index: int | None = None) -> str:
    """
    Render all non-pruned sections into a single PK string.
    Optionally skip one section by index (used for ablation).
    """
    parts = [
        s.content for s in sections
        if not s.pruned and s.index != skip_index
    ]
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Mutation helpers
# ---------------------------------------------------------------------------

def append_section(sections: list[PKSection], title: str, content: str) -> list[PKSection]:
    """Return a new list with a new section appended at the end."""
    new_idx = max((s.index for s in sections), default=-1) + 1
    return sections + [PKSection(index=new_idx, title=title, content=content)]


def reindex(sections: list[PKSection]) -> list[PKSection]:
    """Re-assign contiguous indices (use after pruning to keep indices dense)."""
    return [
        PKSection(index=i, title=s.title, content=s.content, pruned=s.pruned)
        for i, s in enumerate(sections)
    ]
