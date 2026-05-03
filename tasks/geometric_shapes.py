"""tasks/geometric_shapes.py — TaskSpec for the geometric_shapes BBH task."""

from __future__ import annotations

import os as _os
import re

from utils.task_spec import TaskSpec
from tasks.utils import (
    _make_eval_prompt,
    _parse_mc,
    _mc_correct,
    _mc_label,
    _extract_reasoning,
    _format_failure,
    _rule_score_prompt,
    _bootstrap_ruleset,
)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring prompts
# ─────────────────────────────────────────────────────────────────────────────

_GEO_SCORING = """\
You are identifying geometric shapes from SVG path descriptions.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

VERDICT: (A)–(J)  ← FIRST LINE.\
REASONING: Think step by step. Count M (move) and L (line) commands to determine the number of vertices, \
then name the resulting shape.
"""

_GEO_SCORING_COT = """\
You are identifying geometric shapes from SVG path descriptions.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Think step by step: count M (move) and L (line) commands to determine the number of vertices.
  3 vertices → triangle, 4 → rectangle/kite, 5 → pentagon, 6 → hexagon, 7 → heptagon, 8 → octagon.
  If the path uses A (arc) commands → circle or sector.

VERDICT: (A)–(J)  ← FIRST LINE.\
REASONING: Think step by step. Count M and L commands, then name the resulting shape.
"""


def _geo_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _GEO_SCORING_COT if cot else _GEO_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


_GEO_RF = """\
You are identifying geometric shapes from SVG path descriptions.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Output format - use plain text only, no markdown bold or headers:
REASONING: count M and L commands to determine vertices, apply the cheatsheet rules to identify the shape, then select the matching option.
VERDICT: (A)

(Replace (A) above with the correct option letter (A)-(J). VERDICT must be the last line, must be followed immediately by the option on the same line, and must appear exactly once.)"""


def _geometric_shapes_scoring_prompt_rf(cs: str, item: dict, cot: bool = True) -> str:
    return _GEO_RF.format(cheatsheet=cs, question=item["input"])


def _parse_geometric_shapes_rf(text: str):
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip().upper()
            m = re.search(r"\(([A-J])\)", val)
            if m:
                return f"({m.group(1)})"
    return None


def _geo_partition_key(item: dict) -> tuple:
    path = item["input"]
    n_l = len(re.findall(r"\bL\b", path))
    n_m = len(re.findall(r"\bM\b", path))
    has_arc = "A " in path or " A" in path
    n_vertices = n_l + (1 if n_m > 0 else 0)
    bucket = (min(n_vertices, 10), has_arc)
    return bucket


def _geo_key_to_conds(key: tuple) -> list[str]:
    n_verts, has_arc = key
    conds = [f"path has approximately {n_verts} vertices"]
    if has_arc:
        conds.append("path contains arc (A) commands")
    return conds


def _geo_polarity(polarity: str, failure_type: str, divergence_step: str) -> str:
    base = (
        f"POLARITY — model chose wrong shape option (majority correct answer is {polarity}):\n"
        "TYPE A: model doesn't know how to count SVG path vertices correctly (M vs L commands).\n"
        "TYPE B: model counts vertices correctly but maps the count to the wrong shape name."
    )
    if failure_type == "ABANDONMENT":
        base = "STRATEGY — ABANDONMENT: model gave up. Show the next step.\n\n" + base
    return base


def _geo_identify_rule(reasoning: str) -> str | None:
    m = re.search(r"RULE CITED:\s*(GS-\w+)", reasoning)
    if m:
        return m.group(1)
    m = re.search(r"\b(GS-\w+)", reasoning)
    return m.group(1) if m else None


_GEO_INTRO = ("You are identifying geometric shapes from SVG path descriptions.\n"
              "Apply these rules in order (stop at the first match):")
_GEO_FOOTER = (
    "\nIf no rule applies: count L (line) commands + 1 for the initial M to get vertices. "
    "Map: 3→triangle, 4→rectangle/kite, 5→pentagon, 6→hexagon, 7→heptagon, 8→octagon. "
    "A (arc) commands → circle or sector.\n\n"
    "VERDICT: (A)–(J)\n"
    "REASONING: Begin with the principle applied or 'No rule matched. Counted N vertices → <shape>'."
)


def _geo_bootstrap(failures, model, api_key):
    return _bootstrap_ruleset(
        failures, model, api_key,
        task_desc="SVG path geometric shape identification",
        rule_prefix="GS",
        concepts="SVG M (moveto) and L (lineto) commands, vertex counting, "
                 "shape names: triangle(3), rectangle(4), pentagon(5), hexagon(6), "
                 "heptagon(7), octagon(8), arc commands for circles/sectors",
        verdict_fmt="(A) through (J)",
        ruleset_intro=_GEO_INTRO,
        ruleset_footer=_GEO_FOOTER,
        section_title="GEOMETRIC SHAPE RULES",
    )


_GEO_V3 = (
    "You are an expert in SVG path geometry helping a model that fails on geometric shape identification.\n"
    "The task: given an SVG path string, identify which named shape (triangle, hexagon, etc.) it describes.\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES WITH INCORRECT MODEL REASONING ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "DIAGNOSE the failures by answering:\n"
    "  1. List the SVG commands in the path: count M (moveto), L (lineto), A (arc), Z (closepath).\n"
    "  2. Compute correct vertex count: number of L commands + 1 for the starting M.\n"
    "     Special cases: multiple M commands = multiple sub-paths; A (arc) commands = circle or sector.\n"
    "  3. Map vertex count to shape name:\n"
    "     3 → triangle  |  4 → rectangle or kite  |  5 → pentagon  |  6 → hexagon\n"
    "     7 → heptagon  |  8 → octagon  |  arc commands → circle or sector\n"
    "  4. Did the model fail at counting commands (TYPE A) or map the correct count to the wrong shape name (TYPE B)?\n\n"
    "STRUCTURAL VOCABULARY — use these exact terms in ACTIVATE IF:\n"
    "    n_vertices: exact vertex count from L+1 rule\n"
    "    has_arc: path contains A (arc) commands → circle or sector\n"
    "    has_multi_subpath: path contains multiple M commands → compound shape\n"
    "    error: miscounted_vertices (TYPE A) / wrong_shape_name (TYPE B)\n"
    "    answer_is_A through answer_is_J\n\n"
    "TRANSFERABILITY REQUIREMENT:\n"
    "Your case study must encode an ABSTRACT REASONING PATTERN — not memorize the specific failures above.\n"
    "  • DO NOT copy specific SVG path strings from the failures above.\n"
    "  • SUPPORT examples must be freshly constructed minimal SVG paths that illustrate the counting rule.\n"
    "  • The teaching note should apply to any SVG path with this vertex count or arc structure.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [short title, e.g. '7-Vertex Heptagon vs Hexagon Confusion'] ===\n"
    "FAILURE_TYPE: A (miscounted M/L commands) or B (correct count, wrong shape name)\n"
    "ACTIVATE IF:\n"
    "  - [n_vertices and error type from vocabulary above]\n"
    "DO NOT ACTIVATE IF: [case where vertex count is unambiguous and model is correct]\n"
    "COMMON WRONG MOVE: [what count or shape name the model produces incorrectly]\n"
    "NEXT CHECK: [count L commands + 1 → map to shape name → answer is (A)–(J)]\n"
    "WHY THIS WORKS: [1-2 sentences on the counting rule or name mapping]\n"
    "SUPPORT:\n"
    "  • [example: 'M 0 0 L 1 0 L 1 1 L 0 1 Z' = 3 L + 1 M = 4 vertices = rectangle]  |  Answer: (X)  — [note]\n"
    "{retry_context}"
)

_GEO_V4 = (
    "You are an expert in SVG path geometry helping a model that fails on geometric shape identification.\n"
    "The task: given an SVG path string, identify which named shape (triangle, hexagon, etc.) it describes.\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES WITH INCORRECT MODEL REASONING ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "DIAGNOSE the failures by answering:\n"
    "  1. List the SVG commands in the path: count M (moveto), L (lineto), A (arc), Z (closepath).\n"
    "  2. Compute correct vertex count: number of L commands + 1 for the starting M.\n"
    "     Special cases: multiple M commands = multiple sub-paths; A (arc) commands = circle or sector.\n"
    "  3. Map vertex count to shape name:\n"
    "     3 → triangle  |  4 → rectangle or kite  |  5 → pentagon  |  6 → hexagon\n"
    "     7 → heptagon  |  8 → octagon  |  arc commands → circle or sector\n"
    "  4. Did the model fail at counting commands (TYPE A) or map the correct count to the wrong shape name (TYPE B)?\n\n"
    "STRUCTURAL VOCABULARY — use these exact terms in ACTIVATE IF:\n"
    "    n_vertices: exact vertex count from L+1 rule (e.g. n_vertices=7)\n"
    "    has_arc: path contains A (arc) commands, implying circle or sector\n"
    "    has_multi_subpath: path contains multiple M commands, implying compound shape\n"
    "    error: miscounted_vertices (TYPE A) / wrong_shape_name (TYPE B)\n\n"
    "ACTIVATE IF — REQUIRED CONSTRAINTS:\n"
    "  Conditions must be derivable from SVG path structure alone, without reference to the answer.\n"
    "  • DO NOT include the correct answer option (A)–(J) as a condition. The answer is unknown at\n"
    "    inference time; encoding it is circular and causes the case study to overfit to the training set.\n"
    "  • DO NOT write conditions specific to a single training path instance. Use structural vocabulary\n"
    "    (n_vertices, has_arc, has_multi_subpath, error type) that generalises to unseen SVG paths.\n"
    "  • A valid ACTIVATE IF condition must apply to any SVG path sharing the same structural property,\n"
    "    not only to the particular coordinate values or sequences present in the training failures above.\n\n"
    "TRANSFERABILITY REQUIREMENT:\n"
    "Your case study must encode an ABSTRACT REASONING PATTERN — not memorize the specific failures above.\n"
    "  • DO NOT copy specific SVG path strings from the failures above.\n"
    "  • SUPPORT examples must be freshly constructed minimal SVG paths that illustrate the counting rule.\n"
    "  • The teaching note should apply to any SVG path with this vertex count or arc structure.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [short title, e.g. '7-Vertex Heptagon vs Hexagon Confusion'] ===\n"
    "FAILURE_TYPE: A (miscounted M/L commands) or B (correct count, wrong shape name)\n"
    "ACTIVATE IF:\n"
    "  - [structural property from vocabulary — e.g. 'n_vertices=7, error=wrong_shape_name']\n"
    "DO NOT ACTIVATE IF: [case where vertex count is unambiguous and model is correct]\n"
    "COMMON WRONG MOVE: [what count or shape name the model produces incorrectly]\n"
    "NEXT CHECK: [count L commands + 1 → map to shape name → answer is (A)–(J)]\n"
    "WHY THIS WORKS: [1-2 sentences on the counting rule or name mapping]\n"
    "SUPPORT:\n"
    "  • [example: 'M 0 0 L 1 0 L 1 1 L 0 1 Z' = 3 L + 1 M = 4 vertices = rectangle]  |  Answer: (X)  — [note]\n"
    "{retry_context}"
)

GEO_PROMPTS: dict[str, str] = {"v3": _GEO_V3, "v4": _GEO_V4}
_GEO_LATEST = "v3"
_geo_env = _os.environ.get("ICR_GEN_PROMPT_VERSION", "").strip()
_GEO_GEN_PROMPT = GEO_PROMPTS[_geo_env if _geo_env in GEO_PROMPTS else _GEO_LATEST]

GEOMETRIC_TASK = TaskSpec(
    build_scoring_prompt=_geo_scoring_prompt,
    is_correct=_mc_correct,
    answer_label=_mc_label,
    parse_verdict=_parse_mc,
    extract_post_think=_extract_reasoning,
    partition_key=_geo_partition_key,
    partition_key_to_conditions=_geo_key_to_conds,
    format_failure=_format_failure,
    generation_prompt_template=_GEO_GEN_PROMPT,
    build_polarity_instruction=_geo_polarity,
    task_name="geometric_shapes",
    build_rule_scoring_prompt=_rule_score_prompt,
    identify_triggered_rule=_geo_identify_rule,
    rule_id_regex=r"(GS-\w+)",
    bootstrap_ruleset=_geo_bootstrap,
    build_eval_prompt=_make_eval_prompt("(A) through (J)"),
)
