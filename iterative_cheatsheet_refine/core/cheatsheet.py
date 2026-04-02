"""
cheatsheet.py — Cheatsheet data structure.

A Cheatsheet has two parts:
  decision_tree  — a structured step-by-step decision procedure (generated once at init)
  case_studies   — a growing list of case study strings (added during training)

render() produces the full text injected into the evaluation prompt, enforcing
size limits so the total stays within ~10 kb:
  - Decision tree is hard-capped at DECISION_TREE_MAX_CHARS.
  - Each case study is hard-capped at CASE_STUDY_MAX_CHARS.
  - Case studies are included newest-first (most recent failures are most relevant)
    until TOTAL_RENDER_MAX_CHARS would be exceeded.

The JSON sidecar always stores the full untruncated content so nothing is lost.

Persistence: save() writes <path>.txt (rendered) and <path>.json (full structured).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Size budget  (all in characters)
# ---------------------------------------------------------------------------

DECISION_TREE_MAX_CHARS = 2_500   # ~600 tokens — covers 8-10 detailed steps
CASE_STUDY_MAX_CHARS    =   600   # ~150 tokens — one focused rule + 2-3 examples
TOTAL_RENDER_MAX_CHARS  = 9_500   # leaves headroom under 10 kb after headers

DECISION_TREE_HEADER = "=== DECISION TREE ==="
CASE_STUDIES_HEADER  = "=== CASE STUDIES ==="
CASE_STUDY_DIVIDER   = "--- Case Study {n}: {title} ---"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Cheatsheet:
    decision_tree: str
    case_studies: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> str:
        """
        Produce the full cheatsheet text ready for injection into a prompt.

        Applies hard size limits:
          - Decision tree capped at DECISION_TREE_MAX_CHARS
          - Each case study capped at CASE_STUDY_MAX_CHARS
          - Total output capped at TOTAL_RENDER_MAX_CHARS
            (case studies are included newest-first; older ones are dropped
            if they would exceed the budget)

        Format:
            === DECISION TREE ===
            <decision tree>

            === CASE STUDIES ===
            --- Case Study N: <title> ---
            <case study body>
            ...
        """
        dt = _truncate(self.decision_tree.strip(), DECISION_TREE_MAX_CHARS)
        parts = [DECISION_TREE_HEADER, "", dt]
        budget = TOTAL_RENDER_MAX_CHARS - len("\n".join(parts))

        if self.case_studies:
            header_block = ["", CASE_STUDIES_HEADER]
            budget -= len("\n".join(header_block))
            cs_blocks: list[list[str]] = []

            # Newest-first so the most recent failures are always included
            for i, cs in enumerate(reversed(self.case_studies)):
                display_n = len(self.case_studies) - i   # keep original numbering
                title = _extract_title(cs)
                body  = _truncate(cs.strip(), CASE_STUDY_MAX_CHARS)
                block = [
                    "",
                    CASE_STUDY_DIVIDER.format(n=display_n, title=title),
                    body,
                ]
                block_size = len("\n".join(block))
                if block_size > budget:
                    break
                cs_blocks.append(block)
                budget -= block_size

            if cs_blocks:
                # Reverse back to chronological order for display
                cs_blocks.reverse()
                parts += header_block
                for block in cs_blocks:
                    parts += block

        return "\n".join(parts)

    def render_size(self) -> int:
        """Return the character count of the rendered cheatsheet."""
        return len(self.render())

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_case_study(self, text: str) -> None:
        """Append a new case study. Called when the failure bin flushes."""
        self.case_studies.append(text.strip())
        rendered_size = self.render_size()
        if rendered_size > TOTAL_RENDER_MAX_CHARS:
            print(
                f"  [cheatsheet] rendered size {rendered_size:,} chars — oldest case "
                f"studies will be excluded from render but remain in JSON.",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Save the cheatsheet.
        <path>.txt  — rendered output (size-limited, what goes into prompts)
        <path>.json — full structured content (no truncation)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        path.with_suffix(".txt").write_text(self.render(), encoding="utf-8")
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "decision_tree": self.decision_tree,
                    "case_studies": self.case_studies,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Cheatsheet":
        """Load from the .json sidecar written by save()."""
        json_path = Path(path).with_suffix(".json")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return cls(
            decision_tree=data["decision_tree"],
            case_studies=data.get("case_studies", []),
        )

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def summary(self) -> str:
        rendered = self.render_size()
        dt_shown = min(len(self.decision_tree), DECISION_TREE_MAX_CHARS)
        return (
            f"case_studies={len(self.case_studies)}  "
            f"rendered={rendered:,} chars  "
            f"dt_chars={dt_shown}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int) -> str:
    """
    Truncate *text* to at most *max_chars* characters, breaking at a line
    boundary where possible so no line is cut mid-sentence.
    Appends '[truncated]' if content was removed.
    """
    if len(text) <= max_chars:
        return text

    # Try to break at the last newline before the limit
    cut = text.rfind("\n", 0, max_chars)
    if cut == -1 or cut < max_chars // 2:
        # No good line break — cut at word boundary
        cut = text.rfind(" ", 0, max_chars)
    if cut == -1:
        cut = max_chars

    return text[:cut].rstrip() + "\n[truncated]"


def _extract_title(case_study_text: str) -> str:
    """Pull a short title from the first meaningful line of a case study."""
    for line in case_study_text.splitlines():
        line = line.strip().lstrip("=- ").rstrip("=- ").strip()
        if line:
            return line[:72] + ("..." if len(line) > 72 else "")
    return "untitled"
