"""
build_cs_only_cheatsheet.py

Extracts the case-study section from cheatsheet_final.txt, stripping the Phase 1
PK (prior knowledge) block. The resulting cheatsheet_cs_only.txt contains only
the ACTIVATE IF case studies — no abstract rules or principles.

Purpose: Ablation experiment — does the ACTIVATE IF structure alone (without
the PK foundation) provide any benefit? Comparing baseline → cs_only → pk_only
→ full isolates the contribution of each component.

Detection heuristic:
  - Standard tasks: split at "=== CASE STUDIES ===" header
  - CJ-style tasks:  split at the first "=== " line that is NOT
                     "=== PRIOR KNOWLEDGE ===" and NOT "=== CASE STUDIES ==="
  - Tasks with no case studies: skipped with a warning

Usage:
    python3 build_cs_only_cheatsheet.py
    python3 build_cs_only_cheatsheet.py --tasks web_of_lies causal_judgement
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

TASKS = {
    "web_of_lies":             "runs/bbh_v3/web_of_lies",
    "causal_judgement":        "runs/bbh_v3/causal_judgement",
    "geometric_shapes":        "runs/bbh_v3/geometric_shapes",
    "boolean_expressions":     "runs/bbh_v3/boolean_expressions",
    "formal_fallacies":        "runs/bbh_v3/formal_fallacies",
    "logical_deduction_three": "runs/bbh_v3/logical_deduction_three",
    "sports_understanding":    "runs/bbh_v3/sports_understanding",
    "snarks":                  "runs/bbh_v3/snarks",
    # navigate / disambiguation_qa have no case studies — skipped
}


def _find_cs_start(lines: list[str]) -> int | None:
    """Return the line index where the case study section begins, or None."""
    # Primary: explicit CASE STUDIES header
    for i, line in enumerate(lines):
        if line.strip() == "=== CASE STUDIES ===":
            return i

    # Fallback: first "=== ... ===" header that isn't PRIOR KNOWLEDGE or CASE STUDIES
    for i, line in enumerate(lines):
        if i == 0:
            continue
        if re.match(r"^===\s+.+\s+===$", line.strip()):
            tag = line.strip()
            if tag not in ("=== PRIOR KNOWLEDGE ===", "=== CASE STUDIES ==="):
                return i

    return None


def build_cs_only(final_txt: str) -> str | None:
    lines = final_txt.splitlines(keepends=True)
    idx = _find_cs_start(lines)
    if idx is None:
        return None
    return "".join(lines[idx:]).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    args = ap.parse_args()

    for task in args.tasks:
        if task not in TASKS:
            print(f"[{task}] unknown task — skipping")
            continue

        run_dir   = Path(TASKS[task])
        final_path = run_dir / "cheatsheet_final.txt"

        if not final_path.exists():
            print(f"[{task}] cheatsheet_final.txt not found — skipping")
            continue

        final_txt = final_path.read_text(encoding="utf-8")
        cs_only   = build_cs_only(final_txt)

        if cs_only is None:
            print(f"[{task}] no case study section found — skipping")
            continue

        out_path = run_dir / "cheatsheet_cs_only.txt"
        out_path.write_text(cs_only, encoding="utf-8")

        n_cs = cs_only.count("=== CASE STUDY:")
        if n_cs == 0:
            # CJ-style: count "===" blocks instead
            n_cs = len(re.findall(r"^===\s+.+\s+===$", cs_only, re.MULTILINE))

        print(f"[{task}] {n_cs} case study block(s)  ({len(cs_only):,} chars) → {out_path}")


if __name__ == "__main__":
    main()
