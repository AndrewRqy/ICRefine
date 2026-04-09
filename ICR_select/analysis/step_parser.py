"""
ICR_select/analysis/step_parser.py — Extract which roadmap checkpoints appear in COT traces.

Given a set of failing items (each with a post_think / thinking field) and
the current roadmap text, this module:

  1. Parses the canonical ASPECT N headers from the roadmap into checkpoint IDs
     ("A1", "A2", …).
  2. Scans each failure's reasoning trace for exact checkpoint tags ([CK:A1],
     [CK:A2], …) emitted by the scoring prompt.
  3. Returns a ranked misapplication profile: which aspects caused the most
     failures and example quotes that show exactly what went wrong.

Checkpoint ID format
--------------------
The scoring prompt instructs the model to write [CK:AN] (where N is the aspect
number) each time it applies that aspect of the roadmap.  The parser does exact
tag matching — no heuristic regex on free-form text — so paraphrased references
are neither falsely matched nor silently missed.

Backward compatibility
----------------------
Roadmaps that still use STEP N / RULE N headers (old format) return an empty
checkpoint list and therefore an empty profile.  That is the correct result:
old-format roadmaps are not compatible with checkpoint tagging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Checkpoint ID extraction from roadmap text
# ---------------------------------------------------------------------------

# Matches "ASPECT 1:", "ASPECT 2:", etc. at the start of a line.
_ASPECT_RE = re.compile(r"^ASPECT\s+(\d+)\b", re.IGNORECASE | re.MULTILINE)

# Matches emitted checkpoint tags in reasoning traces: [CK:A1], [CK:A12], …
_CK_TAG_RE = re.compile(r"\[CK:A(\d+)\]")


def extract_checkpoint_ids(roadmap_text: str) -> list[str]:
    """
    Return checkpoint IDs for every ASPECT found in the roadmap, in order.

    e.g. a roadmap with ASPECT 1, ASPECT 2, ASPECT 3 → ["A1", "A2", "A3"]
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for m in _ASPECT_RE.finditer(roadmap_text):
        cid = f"A{m.group(1)}"
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    return ordered


def extract_step_names(dt_text: str) -> list[str]:
    """Alias for extract_checkpoint_ids — kept for call-site compatibility."""
    return extract_checkpoint_ids(dt_text)


# ---------------------------------------------------------------------------
# Per-item checkpoint mention detection
# ---------------------------------------------------------------------------

def mentions_in_trace(trace: str, checkpoint_ids: list[str]) -> list[str]:
    """
    Return the subset of checkpoint_ids whose [CK:AN] tag appears in *trace*.

    Exact tag matching only — no regex guessing on free-form text.
    """
    found_nums = {m.group(1) for m in _CK_TAG_RE.finditer(trace)}
    return [cid for cid in checkpoint_ids if cid[1:] in found_nums]


# ---------------------------------------------------------------------------
# Misapplication profile
# ---------------------------------------------------------------------------

@dataclass
class StepMisapplication:
    step_name: str          # checkpoint ID, e.g. "A2"
    count: int              # how many failures mention this checkpoint
    fraction: float         # count / total failures
    evidence: list[str] = field(default_factory=list)   # short quotes from traces


@dataclass
class MisapplicationProfile:
    steps: list[StepMisapplication]   # ranked most → least frequent
    n_failures: int
    n_no_trace: int                   # failures with empty/missing trace


def _best_quote(trace: str, checkpoint_id: str, max_chars: int = 200) -> str:
    """Extract the clause around the [CK:AN] tag in *trace*."""
    tag = f"[CK:{checkpoint_id}]"
    idx = trace.find(tag)
    if idx == -1:
        return ""
    start   = max(0, idx - 40)
    end     = min(len(trace), idx + len(tag) + 160)
    snippet = trace[start:end].replace("\n", " ").strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(trace):
        snippet = snippet + "..."
    return snippet[:max_chars]


def build_profile(
    failures: list[dict],
    dt_text: str,
    max_evidence: int = 3,
) -> MisapplicationProfile:
    """
    Build a ranked misapplication profile from a list of failing items.

    Each item should have at least one of:
      post_think  — REASONING section text (preferred, denser signal)
      thinking    — full internal CoT trace (fallback)

    Items are expected to have been scored with a checkpoint-tagging prompt
    (SCORING_PROMPT or SCORING_PROMPT_COT_FIRST from ICR_naive/prompts/templates.py).
    Traces that contain no [CK:AN] tags contribute to n_no_trace.
    """
    checkpoint_ids = extract_checkpoint_ids(dt_text)
    counts:   dict[str, int]         = {cid: 0 for cid in checkpoint_ids}
    evidence: dict[str, list[str]]   = {cid: [] for cid in checkpoint_ids}
    n_no_trace = 0

    for item in failures:
        trace = (item.get("post_think") or "").strip()
        if not trace or len(trace) < 10:
            trace = (item.get("thinking") or "").strip()
        if not trace:
            n_no_trace += 1
            continue

        matched = mentions_in_trace(trace, checkpoint_ids)
        if not matched:
            n_no_trace += 1
            continue

        for cid in matched:
            counts[cid] += 1
            if len(evidence[cid]) < max_evidence:
                q = _best_quote(trace, cid)
                if q:
                    evidence[cid].append(q)

    steps = [
        StepMisapplication(
            step_name=cid,
            count=counts[cid],
            fraction=counts[cid] / len(failures) if failures else 0.0,
            evidence=evidence[cid],
        )
        for cid in checkpoint_ids
        if counts[cid] > 0
    ]
    steps.sort(key=lambda x: x.count, reverse=True)

    return MisapplicationProfile(
        steps=steps,
        n_failures=len(failures),
        n_no_trace=n_no_trace,
    )


def format_profile(profile: MisapplicationProfile) -> str:
    """Render the profile as a human/LLM-readable string for the revision prompt."""
    if not profile.steps:
        return (
            f"(No checkpoint tags found in {profile.n_failures} failure traces. "
            "Ensure the scoring prompt emits [CK:AN] tags and --cot-first is active.)"
        )

    lines = [
        f"Analysed {profile.n_failures} failures "
        f"({profile.n_no_trace} had no usable checkpoint tags).\n"
    ]
    for rank, step in enumerate(profile.steps, 1):
        aspect_label = f"ASPECT {step.step_name[1:]}"   # "A2" → "ASPECT 2"
        lines.append(
            f"BROKEN CHECKPOINT: {aspect_label} [{step.step_name}]\n"
            f"FREQUENCY: {step.count}/{profile.n_failures} failures "
            f"({step.fraction:.0%})\n"
        )
        for i, q in enumerate(step.evidence, 1):
            lines.append(f'  Evidence {i}: "{q}"')
        lines.append("---")
    return "\n".join(lines)
