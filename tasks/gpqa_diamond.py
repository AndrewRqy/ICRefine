"""tasks/gpqa_diamond.py — TaskSpec for the GPQA Diamond benchmark.

GPQA Diamond: 198 graduate-level multiple-choice questions across
Physics, Chemistry, and Biology. Each item has 4 answer choices (A–D).
Expert validators answer correctly ~74% of the time; non-experts ~34%.
"""
from __future__ import annotations

import re

from utils.task_spec import TaskSpec
from tasks.utils import (
    _make_eval_prompt,
    _extract_reasoning,
    _format_failure,
)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring prompts
# ─────────────────────────────────────────────────────────────────────────────

_GPQA_RF = """\
You are solving a graduate-level multiple-choice question in science (physics, chemistry, or biology).

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Output format - use plain text only, no markdown bold or headers:
REASONING: apply any relevant cheatsheet rules, work through the problem step by step using domain knowledge, eliminate incorrect options, and identify the correct answer.
VERDICT: (A)

(Replace (A) above with the correct option letter (A)-(D). VERDICT must be the last line, must be followed immediately by the option on the same line, and must appear exactly once.)"""


def _gpqa_scoring_prompt_rf(cs: str, item: dict, cot: bool = True) -> str:
    return _GPQA_RF.format(cheatsheet=cs, question=item["input"])

# Alias expected by eval_cs_ablation.py --reasoning-first lookup convention
_gpqa_diamond_scoring_prompt_rf = _gpqa_scoring_prompt_rf


def _gpqa_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    return _GPQA_RF.format(cheatsheet=cs, question=item["input"])


# ─────────────────────────────────────────────────────────────────────────────
# Verdict parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_gpqa_verdict(text: str) -> str | None:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip().upper()
            m = re.search(r"\(([A-D])\)", val)
            if m:
                return f"({m.group(1)})"
    return None

# Alias expected by eval_cs_ablation.py --reasoning-first lookup convention
_parse_gpqa_diamond_rf = _parse_gpqa_verdict


def _gpqa_correct(predicted: str | None, item: dict) -> bool:
    if predicted is None:
        return False
    return predicted.strip().upper() == item["answer"].strip().upper()


def _gpqa_label(item: dict) -> str:
    return item["answer"]


# ─────────────────────────────────────────────────────────────────────────────
# Partitioning — by high-level domain (Physics / Chemistry / Biology)
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_MAP = {
    "physics":   "physics",
    "chemistry": "chemistry",
    "biology":   "biology",
}


def _gpqa_partition_key(item: dict) -> tuple:
    domain = item.get("domain", "").lower()
    for k, v in _DOMAIN_MAP.items():
        if k in domain:
            return (v,)
    return ("other",)


def _gpqa_key_to_conds(key: tuple) -> list[str]:
    return [f"domain = {key[0]}"]


# ─────────────────────────────────────────────────────────────────────────────
# Failure formatting
# ─────────────────────────────────────────────────────────────────────────────

def _gpqa_format_failure(item: dict) -> str:
    predicted  = item.get("predicted", "(none)")
    post_think = (item.get("post_think") or item.get("reasoning") or "").strip()
    lines = [
        f"QUESTION: {item['input']}",
        f"CORRECT: {item['answer']}",
        f"PREDICTED: {predicted or '(none)'}",
    ]
    if post_think:
        lines.append(f"MODEL REASONING (trimmed):\n{post_think[:600]}")
    if item.get("subdomain"):
        lines.append(f"SUBDOMAIN: {item['subdomain']}")
    exact = item.get("_oracle_exact", "")
    if exact:
        lines.append(f"CORRECT REASONING:\n{exact[:600]}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Polarity instruction
# ─────────────────────────────────────────────────────────────────────────────

def _gpqa_polarity(polarity: str, failure_type: str, divergence_step: str) -> str:
    return (
        f"POLARITY — model chose wrong option (majority correct answer is {polarity}).\n"
        "TYPE A: model lacks domain knowledge needed to answer (factual gap).\n"
        "TYPE B: model has the knowledge but applies it incorrectly (reasoning error).\n"
        "Focus on which conceptual step failed, not on the specific numeric values."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Case study generation prompt
# ─────────────────────────────────────────────────────────────────────────────

_GPQA_GEN_PROMPT = (
    "You are an expert scientist helping a model that fails on graduate-level "
    "multiple-choice questions in physics, chemistry, or biology.\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES WITH INCORRECT MODEL REASONING ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "DIAGNOSE the failures:\n"
    "  1. What conceptual principle or domain fact is required?\n"
    "  2. What does the model do wrong? (wrong equation, wrong approximation, "
    "wrong conceptual link, wrong elimination of distractors)\n"
    "  3. Is this a TYPE A failure (missing knowledge) or TYPE B (wrong application)?\n\n"
    "CRITICAL CONSTRAINTS on ACTIVATE IF conditions:\n"
    "  • Conditions must be derivable from the question structure alone, "
    "without reference to the answer.\n"
    "  • DO NOT encode the correct answer letter or value as a condition.\n"
    "  • Use abstract domain vocabulary (e.g. 'involves_uncertainty_principle', "
    "'requires_equilibrium_constant', 'involves_mendelian_genetics') that generalises "
    "across questions, not just the specific numeric values in the failures above.\n"
    "  • A valid ACTIVATE IF must apply to any question sharing the same "
    "conceptual structure — not only the specific items shown.\n\n"
    "TRANSFERABILITY REQUIREMENT:\n"
    "Encode an ABSTRACT REASONING PATTERN, not a memorised answer.\n"
    "  • DO NOT copy specific numbers or answer choices from the failures.\n"
    "  • SUPPORT examples must be freshly constructed minimal problems that "
    "illustrate the principle.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [short title, e.g. 'Heisenberg Uncertainty: Energy-Time Form'] ===\n"
    "FAILURE_TYPE: A (missing concept) or B (wrong reasoning step)\n"
    "DOMAIN: [Physics | Chemistry | Biology]\n"
    "ACTIVATE IF:\n"
    "  - [abstract condition — e.g. 'question_type=energy_level_resolution, "
    "involves=uncertainty_principle']\n"
    "DO NOT ACTIVATE IF: [questions where this principle does NOT apply]\n"
    "COMMON WRONG MOVE: [what the model typically does wrong]\n"
    "NEXT CHECK: [the correct reasoning chain to apply]\n"
    "WHY THIS WORKS: [1-2 sentences on the underlying principle]\n"
    "SUPPORT:\n"
    "  • [minimal example question]  |  Answer: (X)  — [note on why]\n"
    "{retry_context}"
)


# ─────────────────────────────────────────────────────────────────────────────
# Eval prompt (lightweight verdict-only)
# ─────────────────────────────────────────────────────────────────────────────

_GPQA_EVAL = """\
You are solving a graduate-level multiple-choice science question.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Reply with exactly one line: VERDICT: (X) where X is A, B, C, or D."""


def _gpqa_eval_prompt(cs: str, item: dict) -> str:
    return _GPQA_EVAL.format(cheatsheet=cs, question=item["input"])


# ─────────────────────────────────────────────────────────────────────────────
# TaskSpec
# ─────────────────────────────────────────────────────────────────────────────

GPQA_DIAMOND_TASK = TaskSpec(
    build_scoring_prompt=_gpqa_scoring_prompt,
    is_correct=_gpqa_correct,
    answer_label=_gpqa_label,
    parse_verdict=_parse_gpqa_verdict,
    extract_post_think=_extract_reasoning,
    partition_key=_gpqa_partition_key,
    partition_key_to_conditions=_gpqa_key_to_conds,
    format_failure=_gpqa_format_failure,
    generation_prompt_template=_GPQA_GEN_PROMPT,
    build_polarity_instruction=_gpqa_polarity,
    build_eval_prompt=_gpqa_eval_prompt,
    task_name="gpqa_diamond",
)
