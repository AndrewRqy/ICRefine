"""tasks/movie_recommendation.py — TaskSpec for the movie_recommendation BBH task."""

from __future__ import annotations

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
)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring prompts
# ─────────────────────────────────────────────────────────────────────────────

_MOVIE_SCORING = """\
You are recommending movies based on similarity to a set of seed films.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

VERDICT: (A), (B), (C), or (D)  ← FIRST LINE.\
REASONING: Think step by step. Identify shared genre, era, tone, and audience of the seed films.
"""

_MOVIE_SCORING_COT = """\
You are recommending movies based on similarity to a set of seed films.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Think through: what genre, era, tone, and audience do the seed films share? Which option best matches?

VERDICT: (A), (B), (C), or (D)  ← FIRST LINE.\
REASONING: Think step by step. Identify shared genre, era, tone, and audience of the seed films.
"""


def _movie_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _MOVIE_SCORING_COT if cot else _MOVIE_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


def _movie_partition_key(item: dict) -> tuple:
    t = item["input"].lower()
    is_action = any(w in t for w in ["action", "thriller", "adventure", "superhero"])
    is_comedy = any(w in t for w in ["comedy", "romantic", "funny", "humor"])
    is_drama = any(w in t for w in ["drama", "serious", "emotional"])
    is_scifi = any(w in t for w in ["sci-fi", "science fiction", "fantasy", "animated"])
    return (is_action, is_comedy, is_scifi)


def _movie_key_to_conds(key: tuple) -> list[str]:
    is_action, is_comedy, is_scifi = key
    conds = []
    if is_action:
        conds.append("seed films include action / thriller / adventure genre")
    if is_comedy:
        conds.append("seed films include comedy / romantic-comedy genre")
    if is_scifi:
        conds.append("seed films include sci-fi / fantasy / animation genre")
    if not conds:
        conds.append("seed films are drama or mixed genre")
    return conds


def _movie_polarity(polarity: str, failure_type: str, divergence_step: str) -> str:
    base = (
        f"POLARITY — model chose wrong option (majority correct answer is {polarity}):\n"
        "TYPE A: model lacks knowledge of the seed films' shared genre, era, or tone.\n"
        "TYPE B: model knows the films but applies the wrong similarity criterion."
    )
    if failure_type == "ABANDONMENT":
        base = "STRATEGY — ABANDONMENT: model gave up. Show the next step.\n\n" + base
    return base


def _movie_identify_rule(reasoning: str) -> str | None:
    m = re.search(r"RULE CITED:\s*(MR-\w+)", reasoning)
    if m:
        return m.group(1)
    m = re.search(r"\b(MR-\w+)", reasoning)
    return m.group(1) if m else None


_MOVIE_GEN_PROMPT = (
    "You are an expert in film genre and tone helping a model that fails on movie recommendation questions.\n"
    "The task: given a set of seed films, pick which option film is most similar to them.\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES WITH INCORRECT MODEL REASONING ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "DIAGNOSE the failures by answering:\n"
    "  1. What do the seed films share? List genre, era, tone, audience, and director if relevant.\n"
    "  2. Which similarity axis is decisive for this question?\n"
    "     (genre / tone / era / director_style / franchise / audience_age / subject_matter)\n"
    "  3. Which distractor option shares a surface feature but fails on the decisive axis?\n"
    "  4. Did the model fail because it doesn't know the seed films (TYPE A),\n"
    "     or because it used the wrong similarity axis (TYPE B)?\n\n"
    "STRUCTURAL VOCABULARY — use these exact terms in ACTIVATE IF:\n"
    "    seed_genre: action / comedy / drama / scifi / horror / romance / animation / thriller\n"
    "    seed_era: 80s / 90s / 00s / 10s / modern\n"
    "    decisive_axis: genre / tone / director / franchise / audience_age / era\n"
    "    has_plausible_distractor: a wrong option shares a surface feature with seeds\n"
    "    answer_is_A / answer_is_B / answer_is_C / answer_is_D\n\n"
    "TRANSFERABILITY REQUIREMENT:\n"
    "Your case study must encode an ABSTRACT REASONING PATTERN — not memorize the specific failures above.\n"
    "  • DO NOT copy film titles from the failures above.\n"
    "  • SUPPORT examples must use different seed/option films that illustrate the axis pattern.\n"
    "  • The teaching note should apply to any movie recommendation question with this axis structure.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [short title naming the decisive axis, e.g. 'Tone Over Genre for 90s Action'] ===\n"
    "FAILURE_TYPE: A (model doesn't know the seed films) or B (wrong similarity axis)\n"
    "ACTIVATE IF:\n"
    "  - [seed_genre and decisive_axis from vocabulary above]\n"
    "DO NOT ACTIVATE IF: [case where the genre match is unambiguous and model is correct]\n"
    "COMMON WRONG MOVE: [which option the model picks and which surface feature it incorrectly matched]\n"
    "NEXT CHECK: [identify the decisive axis, compare each option → answer is (A), (B), (C), or (D)]\n"
    "WHY THIS WORKS: [1-2 sentences on the similarity axis]\n"
    "SUPPORT:\n"
    "  • [seed films + option films]  |  Answer: (A)/(B)/(C)/(D)  — [axis note]\n"
    "{retry_context}"
)

MOVIE_TASK = TaskSpec(
    build_scoring_prompt=_movie_scoring_prompt,
    is_correct=_mc_correct,
    answer_label=_mc_label,
    parse_verdict=_parse_mc,
    extract_post_think=_extract_reasoning,
    partition_key=_movie_partition_key,
    partition_key_to_conditions=_movie_key_to_conds,
    format_failure=_format_failure,
    generation_prompt_template=_MOVIE_GEN_PROMPT,
    build_polarity_instruction=_movie_polarity,
    task_name="movie_recommendation",
    build_rule_scoring_prompt=_rule_score_prompt,
    identify_triggered_rule=_movie_identify_rule,
    rule_id_regex=r"(MR-\w+)",
    bootstrap_ruleset=None,
    build_eval_prompt=_make_eval_prompt("(A), (B), (C), or (D)"),
)
