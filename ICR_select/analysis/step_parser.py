"""
ICR_select/analysis/step_parser.py — Extract which DT steps appear in COT traces.

Given a set of failing items (each with a post_think / thinking field) and
the current decision tree text, this module:

  1. Parses the canonical step and rule names from the DT.
  2. Scans each failure's reasoning trace to find which steps were referenced.
  3. Returns a ranked misapplication profile: which steps caused the most failures
     and example quotes that show exactly what went wrong.

This profile is the input to the DT reviser — it tells the LLM precisely
which steps to fix and what the model does wrong when it applies them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Step extraction from DT text
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(
    r"(?:===\s*)?(STEP\s+\d+(?:\s*[:\-]\s*\S.*?)?|RULE\s+\d+)[:\s]",
    re.IGNORECASE,
)


def extract_step_names(dt_text: str) -> list[str]:
    """
    Pull the canonical step/rule identifiers from the DT text.
    Returns e.g. ["STEP 1", "STEP 2", "RULE 1", "RULE 2", ...] in order.
    """
    raw = _STEP_RE.findall(dt_text)
    seen, ordered = set(), []
    for r in raw:
        # Normalise to just "STEP N" or "RULE N"
        m = re.match(r"(STEP|RULE)\s*(\d+)", r, re.IGNORECASE)
        if m:
            key = f"{m.group(1).upper()} {m.group(2)}"
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


# ---------------------------------------------------------------------------
# Per-item step mention counting
# ---------------------------------------------------------------------------

def _make_step_pattern(step_name: str) -> re.Pattern:
    """Build a regex that finds a step reference in free-form reasoning text."""
    kind, num = step_name.split()   # e.g. "STEP", "1"
    return re.compile(
        rf"(?:{kind}[\s\-]*{num}|{kind.lower()}[\s\-]*{num})",
        re.IGNORECASE,
    )


def mentions_in_trace(trace: str, step_names: list[str]) -> list[str]:
    """Return the subset of step_names that appear in *trace*."""
    return [s for s in step_names if _make_step_pattern(s).search(trace)]


# ---------------------------------------------------------------------------
# Misapplication profile
# ---------------------------------------------------------------------------

@dataclass
class StepMisapplication:
    step_name: str
    count: int                                # how many failures mention this step
    fraction: float                           # count / total failures
    evidence: list[str] = field(default_factory=list)   # short quotes from traces


@dataclass
class MisapplicationProfile:
    steps: list[StepMisapplication]           # ranked most → least frequent
    n_failures: int
    n_no_trace: int                           # failures with empty/missing trace


def _best_quote(trace: str, step_name: str, max_chars: int = 200) -> str:
    """Extract the sentence or clause around the step mention."""
    pat = _make_step_pattern(step_name)
    m = pat.search(trace)
    if not m:
        return ""
    # Take up to max_chars centred on the match
    start = max(0, m.start() - 60)
    end   = min(len(trace), m.end() + 140)
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
    """
    step_names = extract_step_names(dt_text)
    counts: dict[str, int]        = {s: 0 for s in step_names}
    evidence: dict[str, list[str]] = {s: [] for s in step_names}
    n_no_trace = 0

    for item in failures:
        trace = (item.get("post_think") or "").strip()
        if not trace or len(trace) < 30:
            trace = (item.get("thinking") or "").strip()
        if not trace:
            n_no_trace += 1
            continue

        for step in mentions_in_trace(trace, step_names):
            counts[step] += 1
            if len(evidence[step]) < max_evidence:
                q = _best_quote(trace, step)
                if q:
                    evidence[step].append(q)

    steps = [
        StepMisapplication(
            step_name=s,
            count=counts[s],
            fraction=counts[s] / len(failures) if failures else 0.0,
            evidence=evidence[s],
        )
        for s in step_names
        if counts[s] > 0
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
        return "(No step references found in reasoning traces.)"

    lines = [
        f"Analysed {profile.n_failures} failures "
        f"({profile.n_no_trace} had no usable reasoning trace).\n"
    ]
    for rank, step in enumerate(profile.steps, 1):
        lines.append(
            f"BROKEN STEP: {step.step_name}\n"
            f"FREQUENCY: {step.count}/{profile.n_failures} failures "
            f"({step.fraction:.0%})\n"
        )
        for i, q in enumerate(step.evidence, 1):
            lines.append(f'  Evidence {i}: "{q}"')
        lines.append("---")
    return "\n".join(lines)
