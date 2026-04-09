"""
cheatsheet.py — Cheatsheet data structure.

A Cheatsheet has two parts:
  decision_tree  — a structured step-by-step decision procedure (generated once at init)
  case_studies   — a growing list of CaseStudy records (added during training)

render() produces the full text injected into the evaluation prompt, enforcing
size limits so the total stays within ~10 kb:
  - Decision tree is hard-capped at DECISION_TREE_MAX_CHARS.
  - Each case study is hard-capped at CASE_STUDY_MAX_CHARS (applied to render()).
  - Case studies are included newest-first (most recent failures are most relevant)
    until TOTAL_RENDER_MAX_CHARS would be exceeded.

The JSON sidecar always stores the full untruncated structured content so nothing
is lost.  Each case study is stored as a rich structured dict (see CaseStudy.to_dict()).

Persistence: save() writes <path>.txt (rendered) and <path>.json (full structured).
Backward compatibility: load() accepts the old format (case_studies as list[str])
and transparently wraps each string with CaseStudy.from_text().
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from utils.case_study import CaseStudy


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

PRIOR_KNOWLEDGE_HEADER = "=== PRIOR KNOWLEDGE ==="
ROADMAP_HEADER         = "=== REASONING ROADMAP ==="


@dataclass
class Cheatsheet:
    decision_tree: str
    case_studies: list[CaseStudy] = field(default_factory=list)
    # Optional frozen prior knowledge (e.g. NeuriCo prompt).
    # Rendered before the decision tree / roadmap and never modified by ICR.
    prior_knowledge: str = ""

    def __post_init__(self) -> None:
        # Accept list[str] for backward compatibility — wrap each string.
        normalized: list[CaseStudy] = []
        for cs in self.case_studies:
            if isinstance(cs, str):
                normalized.append(CaseStudy.from_text(cs))
            else:
                normalized.append(cs)
        self.case_studies = normalized

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> str:
        """
        Produce the full cheatsheet text ready for injection into a prompt.

        Applies hard size limits:
          - Decision tree capped at DECISION_TREE_MAX_CHARS
          - Each case study render() capped at CASE_STUDY_MAX_CHARS
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
        parts = []
        if self.prior_knowledge.strip():
            parts += [PRIOR_KNOWLEDGE_HEADER, "", self.prior_knowledge.strip(), ""]

        header = ROADMAP_HEADER if self.decision_tree.lstrip().startswith("ASPECT") else DECISION_TREE_HEADER
        dt = _truncate(self.decision_tree.strip(), DECISION_TREE_MAX_CHARS)
        parts += [header, "", dt]
        budget = TOTAL_RENDER_MAX_CHARS - len("\n".join(parts))

        if self.case_studies:
            header_block = ["", CASE_STUDIES_HEADER]
            budget -= len("\n".join(header_block))
            cs_blocks: list[list[str]] = []

            # Newest-first so the most recent failures are always included
            for i, cs in enumerate(reversed(self.case_studies)):
                display_n = len(self.case_studies) - i   # keep original numbering
                body  = _truncate(cs.render(), CASE_STUDY_MAX_CHARS)
                block = [
                    "",
                    CASE_STUDY_DIVIDER.format(n=display_n, title=cs.title),
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

    def add_case_study(self, cs: "CaseStudy | str") -> None:
        """Append a new case study. Accepts CaseStudy or plain text string."""
        if isinstance(cs, str):
            cs = CaseStudy.from_text(cs)
        self.case_studies.append(cs)
        rendered_size = self.render_size()
        if rendered_size > TOTAL_RENDER_MAX_CHARS:
            print(
                f"  [cheatsheet] rendered size {rendered_size:,} chars — oldest case "
                f"studies will be excluded from render but remain in JSON.",
                file=sys.stderr,
            )

    def patch_decision_tree(self, patch_text: str) -> None:
        """
        Append a DT patch block produced by the reasoning-aware case study generator.

        Patches are appended after the existing decision tree text.  If the combined
        length exceeds DECISION_TREE_MAX_CHARS, render() will truncate the oldest
        content (from the beginning of the DT), so newly added rules survive.
        """
        patch_text = patch_text.strip()
        if not patch_text:
            return
        self.decision_tree = self.decision_tree.rstrip() + "\n\n" + patch_text
        dt_len = len(self.decision_tree)
        if dt_len > DECISION_TREE_MAX_CHARS:
            print(
                f"  [cheatsheet] decision_tree is {dt_len:,} chars after patch — "
                f"oldest DT content will be truncated in render (cap={DECISION_TREE_MAX_CHARS}).",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Save the cheatsheet.
        <path>.txt  — rendered output (size-limited, what goes into prompts)
        <path>.json — full structured content (no truncation; case studies as rich dicts)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        path.with_suffix(".txt").write_text(self.render(), encoding="utf-8")
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "decision_tree":   self.decision_tree,
                    "case_studies":    [cs.to_dict() for cs in self.case_studies],
                    "prior_knowledge": self.prior_knowledge,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Cheatsheet":
        """
        Load from the .json sidecar written by save().

        Backward compatible: if case_studies entries are plain strings (old format),
        each is parsed with CaseStudy.from_text().
        """
        json_path = Path(path).with_suffix(".json")
        data = json.loads(json_path.read_text(encoding="utf-8"))

        raw_studies = data.get("case_studies", [])
        case_studies: list[CaseStudy] = []
        for entry in raw_studies:
            if isinstance(entry, str):
                case_studies.append(CaseStudy.from_text(entry))
            elif isinstance(entry, dict):
                case_studies.append(CaseStudy.from_dict(entry))
            # else: skip malformed entries

        return cls(
            decision_tree=data["decision_tree"],
            case_studies=case_studies,
            prior_knowledge=data.get("prior_knowledge", ""),
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
