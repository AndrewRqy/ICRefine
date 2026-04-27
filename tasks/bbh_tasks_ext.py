"""tasks/bbh_tasks_ext.py — TaskSpec implementations for 6 additional BBH tasks.

Tasks
-----
  FORMAL_FALLACIES_TASK           — formal_fallacies        (valid / invalid)
  LOGICAL_DEDUCTION_3_TASK        — logical_deduction_three_objects  ((A)/(B)/(C))
  WEB_OF_LIES_TASK                — web_of_lies             (Yes / No)
  DATE_UNDERSTANDING_TASK         — date_understanding      ((A)–(F))
  NAVIGATE_TASK                   — navigate                (Yes / No)
  SNARKS_TASK                     — snarks                  ((A) / (B))
"""

from __future__ import annotations

import re
from utils.task_spec import TaskSpec
from tasks.bbh_tasks import (
    _parse_mc, _parse_yesno, _mc_correct, _mc_label,
    _extract_reasoning, _format_failure, _rule_score_prompt,
    _bootstrap_ruleset,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_valid_invalid(content: str) -> str | None:
    m = re.search(r"VERDICT:\s*(valid|invalid)", content, re.IGNORECASE)
    return m.group(1).lower() if m else None


def _valid_invalid_correct(predicted: str | None, item: dict) -> bool:
    return predicted is not None and predicted.lower() == item["answer"].strip().lower()


def _valid_invalid_label(item: dict) -> str:
    return item["answer"].strip().lower()


def _yesno_correct(predicted: str | None, item: dict) -> bool:
    return predicted is not None and predicted.upper() == item["answer"].strip().upper()


def _yesno_label(item: dict) -> str:
    return item["answer"].strip()


def _trivial_key(item: dict) -> tuple:
    return ("all",)


def _trivial_conds(key: tuple) -> list[str]:
    return []


def _generic_polarity(polarity: str, failure_type: str, divergence_step: str) -> str:
    base = {
        "YES":   "POLARITY — FALSE NEGATIVE: model said NO/invalid/wrong option but correct answer is YES/valid/correct option.",
        "NO":    "POLARITY — FALSE POSITIVE: model said YES/valid/correct option but the answer is NO/invalid/other option.",
        "TRUE":  "POLARITY — FALSE NEGATIVE: model output the wrong answer.",
        "FALSE": "POLARITY — FALSE POSITIVE: model output the wrong answer.",
    }.get(polarity.upper(), "Diagnose whether the model applied the wrong rule or misread the structure.")
    if failure_type == "ABANDONMENT":
        base = "STRATEGY — ABANDONMENT: model gave up or produced no verdict. Show the next step.\n\n" + base
    return base


# ─────────────────────────────────────────────────────────────────────────────
# 1. FORMAL FALLACIES
# ─────────────────────────────────────────────────────────────────────────────

_FF_SCORING = """\
You are evaluating whether a deductive argument is valid or invalid.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

CRITICAL: Your VERY FIRST LINE must be:
  VERDICT: valid
  VERDICT: invalid

VERDICT: valid or invalid  ← FIRST LINE.
RULE CITED: <rule ID, e.g. FF-2> or NONE
REASONING: State the logical form (e.g. modus ponens, affirming the consequent) \
then explain why it is valid or invalid.\
"""

_FF_SCORING_COT = """\
You are evaluating whether a deductive argument is valid or invalid.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Think step by step: identify the logical form of the argument, then decide if \
the conclusion follows necessarily from the premises.

CRITICAL: Your VERY FIRST LINE must be:
  VERDICT: valid
  VERDICT: invalid

VERDICT: valid or invalid  ← FIRST LINE.
RULE CITED: <rule ID, e.g. FF-2> or NONE
REASONING: State the logical form then explain.\
"""


def _ff_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _FF_SCORING_COT if cot else _FF_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


def _ff_partition_key(item: dict) -> tuple:
    t = item["input"].lower()
    has_not_x_is_y = bool(re.search(r"whoever is not .+ is ", t))
    has_whoever    = "whoever" in t
    has_something  = "something" in t or "someone" in t
    return (has_not_x_is_y, has_whoever or has_something)


def _ff_key_to_conds(key: tuple) -> list[str]:
    has_not_x, has_whoever = key
    conds = []
    if has_not_x:
        conds.append("argument uses 'whoever is not X is Y' structure (common invalid converse)")
    if has_whoever:
        conds.append("argument uses universal quantifier (whoever / something / someone)")
    return conds or ["argument is a syllogism with stated premises and a conclusion"]


_FF_GEN_PROMPT = (
    "You are an expert in formal logic helping a model that keeps making errors on deductive validity.\n\n"
    "=== REASONING ROADMAP ===\n{roadmap}\n=== END ROADMAP ===\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "DIAGNOSE:\n"
    "  1. What named logical fallacy (or valid form) is this?\n"
    "     Common invalids: affirming the consequent, denying the antecedent, illicit conversion\n"
    "     Common valids:   modus ponens, modus tollens, hypothetical syllogism\n"
    "  2. What is the model confusing? (e.g. thinks a converse is equivalent to original)\n"
    "  3. What one-sentence rule would have prevented the error?\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [Fallacy Name or Valid Form] ===\n"
    "FAILURE_TYPE: A (wrong logical form identified) or B (right form, wrong validity verdict)\n"
    "ACTIVATE IF:\n"
    "  - argument structure looks like: [describe the surface form — e.g. 'if A then B; therefore if not-A then not-B']\n"
    "  - the conclusion attempts to: [what the argument is trying to derive]\n"
    "DO NOT ACTIVATE IF: [the structurally similar case that IS valid]\n"
    "COMMON WRONG MOVE: [what the model wrongly concludes and why]\n"
    "NEXT CHECK: [the one logical test to apply → valid or invalid]\n"
    "WHY THIS WORKS: [1-2 sentences on why the logical form (in)validates the conclusion]\n"
    "SUPPORT:\n"
    "  • [concrete everyday syllogism example]  |  Answer: valid/invalid  — [brief note]\n"
    "TARGET_STEP: [roadmap step this corrects]\n\n"
    "OUTPUT 2 — ROADMAP PATCH\n"
    "=== ROADMAP PATCH ===\n"
    "[one-line rule addition, or '(none)']\n"
    "=== END ROADMAP PATCH ===\n"
    "{retry_context}"
)


def _ff_bootstrap(failures, model, api_key):
    return _bootstrap_ruleset(
        failures, model, api_key,
        task_desc="formal deductive logic (valid vs invalid arguments)",
        rule_prefix="FF",
        concepts="modus ponens, modus tollens, affirming the consequent, denying the antecedent, "
                 "illicit conversion, hypothetical syllogism",
        verdict_fmt="valid or invalid",
        ruleset_intro="Evaluate the argument's logical form. Rules (apply first match):",
        ruleset_footer="\nIf no rule applies, check whether the conclusion follows necessarily "
                       "from the premises by logical form alone.\n\nVERDICT: valid or invalid\n"
                       "RULE CITED: <FF-N> or NONE\nREASONING: name the form, then explain.",
        section_title="FORMAL FALLACY RULES",
    )


FORMAL_FALLACIES_TASK = TaskSpec(
    build_scoring_prompt=_ff_scoring_prompt,
    is_correct=_valid_invalid_correct,
    answer_label=_valid_invalid_label,
    parse_verdict=_parse_valid_invalid,
    extract_post_think=_extract_reasoning,
    partition_key=_ff_partition_key,
    partition_key_to_conditions=_ff_key_to_conds,
    format_failure=_format_failure,
    generation_prompt_template=_FF_GEN_PROMPT,
    build_polarity_instruction=_generic_polarity,
    task_name="formal_fallacies",
    build_rule_scoring_prompt=_rule_score_prompt,
    bootstrap_ruleset=_ff_bootstrap,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOGICAL DEDUCTION (THREE OBJECTS)
# ─────────────────────────────────────────────────────────────────────────────

_LD3_SCORING = """\
You are solving a logical ordering puzzle with three objects.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

CRITICAL: Your VERY FIRST LINE must be:
  VERDICT: (A)
  VERDICT: (B)
  VERDICT: (C)

VERDICT: (A), (B), or (C)  ← FIRST LINE.
RULE CITED: <rule ID, e.g. LD-2> or NONE
REASONING: Show the ordering chain step by step, then pick the answer.\
"""

_LD3_SCORING_COT = """\
You are solving a logical ordering puzzle with three objects.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Build the full left-to-right (or ordered) sequence from the constraints, then answer.

CRITICAL: Your VERY FIRST LINE must be:
  VERDICT: (A)
  VERDICT: (B)
  VERDICT: (C)

VERDICT: (A), (B), or (C)  ← FIRST LINE.
RULE CITED: <rule ID, e.g. LD-2> or NONE
REASONING: Write out the full ordering, then pick.\
"""


def _ld3_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _LD3_SCORING_COT if cot else _LD3_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


_LD3_RELATIONS = {
    "left":    "positional (left/right)",
    "right":   "positional (left/right)",
    "older":   "temporal/comparative (older/newer)",
    "newer":   "temporal/comparative (older/newer)",
    "heavier": "weight comparison (heavier/lighter)",
    "lighter": "weight comparison (heavier/lighter)",
    "taller":  "size comparison (taller/shorter)",
    "shorter": "size comparison (taller/shorter)",
    "larger":  "size comparison (larger/smaller)",
    "smaller": "size comparison (larger/smaller)",
    "faster":  "speed comparison (faster/slower)",
    "cheaper": "cost comparison (cheaper/more expensive)",
}


def _ld3_partition_key(item: dict) -> tuple:
    t = item["input"].lower()
    for kw, rel_type in _LD3_RELATIONS.items():
        if kw in t:
            return (rel_type,)
    return ("general ordering",)


def _ld3_key_to_conds(key: tuple) -> list[str]:
    (rel,) = key
    return [f"ordering relation is {rel}"]


_LD3_GEN_PROMPT = (
    "You are an expert in logical deduction helping a model that fails at ordering puzzles with three objects.\n\n"
    "=== REASONING ROADMAP ===\n{roadmap}\n=== END ROADMAP ===\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "DIAGNOSE:\n"
    "  1. What step in building the ordering chain did the model get wrong?\n"
    "     (e.g. misread 'A is to the right of B' as 'B is to the right of A', "
    "or stopped after processing only 2 of 3 constraints)\n"
    "  2. What is the minimal ordering chain the model should have built?\n"
    "  3. What one check would have caught the error?\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [Ordering Error Type] ===\n"
    "FAILURE_TYPE: A (misread a constraint direction) or B (correct chain, wrong position read-off)\n"
    "ACTIVATE IF:\n"
    "  - the puzzle involves: [describe the relation type and what the model typically does wrong]\n"
    "  - the error pattern is: [what specific mistake the model makes in these failures]\n"
    "DO NOT ACTIVATE IF: [the case where the model handles this relation correctly]\n"
    "COMMON WRONG MOVE: [exactly how the model misreads or misapplies the constraint]\n"
    "NEXT CHECK: [the specific constraint to re-read carefully → which object is leftmost/smallest/etc.]\n"
    "WHY THIS WORKS: [1-2 sentences on the correct chain-building approach]\n"
    "SUPPORT:\n"
    "  • [mini 3-object ordering example]  |  Answer: (X)  — [brief note]\n"
    "TARGET_STEP: [roadmap step this corrects]\n\n"
    "OUTPUT 2 — ROADMAP PATCH\n"
    "=== ROADMAP PATCH ===\n"
    "[one-line rule addition, or '(none)']\n"
    "=== END ROADMAP PATCH ===\n"
    "{retry_context}"
)


def _ld3_bootstrap(failures, model, api_key):
    return _bootstrap_ruleset(
        failures, model, api_key,
        task_desc="logical deduction — ordering three objects from pairwise constraints",
        rule_prefix="LD",
        concepts="constraint chaining, transitive ordering, reading direction of comparisons, "
                 "positional vs comparative relations",
        verdict_fmt="(A), (B), or (C)",
        ruleset_intro="Build the full ordering from all constraints, then answer. Rules:",
        ruleset_footer="\nIf no rule applies, write out all constraints as a chain "
                       "and read off the answer.\n\nVERDICT: (A), (B), or (C)\n"
                       "RULE CITED: <LD-N> or NONE\nREASONING: show the chain.",
        section_title="LOGICAL DEDUCTION RULES",
    )


LOGICAL_DEDUCTION_3_TASK = TaskSpec(
    build_scoring_prompt=_ld3_scoring_prompt,
    is_correct=_mc_correct,
    answer_label=_mc_label,
    parse_verdict=_parse_mc,
    extract_post_think=_extract_reasoning,
    partition_key=_ld3_partition_key,
    partition_key_to_conditions=_ld3_key_to_conds,
    format_failure=_format_failure,
    generation_prompt_template=_LD3_GEN_PROMPT,
    build_polarity_instruction=_generic_polarity,
    task_name="logical_deduction_three_objects",
    build_rule_scoring_prompt=_rule_score_prompt,
    bootstrap_ruleset=_ld3_bootstrap,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. WEB OF LIES
# ─────────────────────────────────────────────────────────────────────────────

_WOL_SCORING = """\
You are tracing a chain of truth-tellers and liars.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Rules: A truth-teller's claim about someone is accurate. A liar's claim is the opposite.
Track each person's actual truth-value through the chain.

CRITICAL: Your VERY FIRST LINE must be:
  VERDICT: Yes
  VERDICT: No

VERDICT: Yes or No  ← FIRST LINE.
RULE CITED: <rule ID, e.g. WL-2> or NONE
REASONING: Trace each step of the chain showing how the truth-value flips or stays.\
"""

_WOL_SCORING_COT = """\
You are tracing a chain of truth-tellers and liars.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Think step by step: start from the first given fact and propagate through each claim.

CRITICAL: Your VERY FIRST LINE must be:
  VERDICT: Yes
  VERDICT: No

VERDICT: Yes or No  ← FIRST LINE.
RULE CITED: <rule ID, e.g. WL-2> or NONE
REASONING: Trace each step (person → truth-value) then give verdict.\
"""


def _wol_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _WOL_SCORING_COT if cot else _WOL_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


def _wol_partition_key(item: dict) -> tuple:
    # Count number of people in the chain (count "says" occurrences)
    n_says = item["input"].lower().count(" says ")
    chain_len = "short" if n_says <= 2 else "medium" if n_says <= 4 else "long"
    # Does the chain start with a liar?
    starts_with_lie = "lies." in item["input"].split(".")[0].lower() or \
                      "lies\n" in item["input"].lower()[:100]
    return (chain_len, starts_with_lie)


def _wol_key_to_conds(key: tuple) -> list[str]:
    chain_len, starts_lie = key
    conds = [f"chain length is {chain_len} ({{'short': '≤2', 'medium': '3–4', 'long': '5+'}}[chain_len] + ' claims')"]
    if starts_lie:
        conds.append("chain starts with someone who lies")
    else:
        conds.append("chain starts with someone who tells the truth")
    return conds


_WOL_GEN_PROMPT = (
    "You are an expert in propositional logic helping a model that fails at truth-chain problems.\n\n"
    "=== REASONING ROADMAP ===\n{roadmap}\n=== END ROADMAP ===\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "KEY INSIGHT: Each 'lies' in the chain flips the truth-value; each 'tells the truth' keeps it.\n"
    "Count the number of 'lies' in the chain. Even number of lies → same as the initial truth-value. "
    "Odd number → flipped.\n\n"
    "DIAGNOSE:\n"
    "  1. Where in the chain did the model lose track of the truth-value?\n"
    "  2. Did it miscount 'lies'? Misread 'tells the truth'? Stop too early?\n"
    "  3. What rule would prevent this?\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [Chain Error Type] ===\n"
    "FAILURE_TYPE: A (lost track mid-chain) or B (correct tracking, wrong final verdict)\n"
    "ACTIVATE IF:\n"
    "  - chain looks like: [describe the pattern — e.g. 'starts with liar, 3+ says claims']\n"
    "  - the model's error is: [what specific step it got wrong]\n"
    "DO NOT ACTIVATE IF: [the simpler case where the model traces correctly]\n"
    "COMMON WRONG MOVE: [exactly where the model drops the flip or counts wrong]\n"
    "NEXT CHECK: [count the 'lies' claims; even=same, odd=flipped → Yes or No]\n"
    "WHY THIS WORKS: [1-2 sentences on parity tracking]\n"
    "SUPPORT:\n"
    "  • [short chain example: A lies, B says A tells truth → B lies → No]  |  Answer: No  — [note]\n"
    "TARGET_STEP: [roadmap step this corrects]\n\n"
    "OUTPUT 2 — ROADMAP PATCH\n"
    "=== ROADMAP PATCH ===\n"
    "[one-line rule addition, or '(none)']\n"
    "=== END ROADMAP PATCH ===\n"
    "{retry_context}"
)


def _wol_bootstrap(failures, model, api_key):
    return _bootstrap_ruleset(
        failures, model, api_key,
        task_desc="web of lies — tracing truth-value through chains of truth-tellers and liars",
        rule_prefix="WL",
        concepts="truth propagation, lie flipping, parity of lies in chain, "
                 "chain length, starting truth-value",
        verdict_fmt="Yes or No",
        ruleset_intro="Trace the chain: each 'lies' flips the truth-value, each 'tells the truth' keeps it. Rules:",
        ruleset_footer="\nIf no rule applies, trace each person in order and track truth-value flips.\n\n"
                       "VERDICT: Yes or No\nRULE CITED: <WL-N> or NONE\nREASONING: trace the chain.",
        section_title="WEB OF LIES RULES",
    )


WEB_OF_LIES_TASK = TaskSpec(
    build_scoring_prompt=_wol_scoring_prompt,
    is_correct=_yesno_correct,
    answer_label=_yesno_label,
    parse_verdict=_parse_yesno,
    extract_post_think=_extract_reasoning,
    partition_key=_wol_partition_key,
    partition_key_to_conditions=_wol_key_to_conds,
    format_failure=_format_failure,
    generation_prompt_template=_WOL_GEN_PROMPT,
    build_polarity_instruction=_generic_polarity,
    task_name="web_of_lies",
    build_rule_scoring_prompt=_rule_score_prompt,
    bootstrap_ruleset=_wol_bootstrap,
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATE UNDERSTANDING
# ─────────────────────────────────────────────────────────────────────────────

_DU_SCORING = """\
You are solving a date arithmetic or calendar reasoning question.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

CRITICAL: Your VERY FIRST LINE must be:
  VERDICT: (A)
  VERDICT: (B)
  ... (through (F))

VERDICT: (A)–(F)  ← FIRST LINE.
RULE CITED: <rule ID, e.g. DU-2> or NONE
REASONING: State the starting date, apply the operation step by step, then pick the matching option.\
"""

_DU_SCORING_COT = """\
You are solving a date arithmetic or calendar reasoning question.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Think step by step: determine the starting date, apply the operation (add/subtract days or months, \
find day of week, convert format), and match to the options.

CRITICAL: Your VERY FIRST LINE must be:
  VERDICT: (A)
  VERDICT: (B)
  ... (through (F))

VERDICT: (A)–(F)  ← FIRST LINE.
RULE CITED: <rule ID, e.g. DU-2> or NONE
REASONING: Show each arithmetic step then pick.\
"""


def _du_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _DU_SCORING_COT if cot else _DU_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


def _du_partition_key(item: dict) -> tuple:
    t = item["input"].lower()
    if "day of the week" in t or "what day" in t:
        op = "weekday"
    elif "month" in t and ("ago" in t or "later" in t or "from now" in t):
        op = "month_arithmetic"
    elif "ago" in t or "later" in t or "from now" in t:
        op = "day_arithmetic"
    elif "mm/dd/yyyy" in t or "format" in t:
        op = "format_conversion"
    elif "yesterday" in t or "tomorrow" in t:
        op = "relative_day"
    else:
        op = "general"
    return (op,)


def _du_key_to_conds(key: tuple) -> list[str]:
    (op,) = key
    labels = {
        "weekday":         "question asks for the day of the week",
        "month_arithmetic":"question involves adding or subtracting months",
        "day_arithmetic":  "question involves adding or subtracting days",
        "format_conversion":"question involves converting between date formats",
        "relative_day":    "question uses relative terms like yesterday or tomorrow",
        "general":         "general date reasoning question",
    }
    return [labels.get(op, f"date operation: {op}")]


_DU_GEN_PROMPT = (
    "You are an expert in calendar arithmetic helping a model that fails on date reasoning questions.\n\n"
    "=== REASONING ROADMAP ===\n{roadmap}\n=== END ROADMAP ===\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "DIAGNOSE:\n"
    "  1. What date operation is being performed? (add days, add months, find weekday, convert format)\n"
    "  2. What is the starting date (explicit or implied)?\n"
    "  3. What specific arithmetic error did the model make?\n"
    "     (e.g. off by one in month count, wrong year for leap year, confused UK/US format)\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [Date Operation Type: Error Description] ===\n"
    "FAILURE_TYPE: A (wrong starting date) or B (right start, wrong arithmetic)\n"
    "ACTIVATE IF:\n"
    "  - question involves: [describe the operation type and the typical error pattern]\n"
    "  - the error looks like: [what the model computes wrong]\n"
    "DO NOT ACTIVATE IF: [the simpler date question this model handles correctly]\n"
    "COMMON WRONG MOVE: [the specific arithmetic mistake with example]\n"
    "NEXT CHECK: [the step-by-step arithmetic to verify → which option matches]\n"
    "WHY THIS WORKS: [1-2 sentences on the correct procedure]\n"
    "SUPPORT:\n"
    "  • [concrete date example]  |  Answer: (X)  — [brief note]\n"
    "TARGET_STEP: [roadmap step this corrects]\n\n"
    "OUTPUT 2 — ROADMAP PATCH\n"
    "=== ROADMAP PATCH ===\n"
    "[one-line rule addition, or '(none)']\n"
    "=== END ROADMAP PATCH ===\n"
    "{retry_context}"
)


def _du_bootstrap(failures, model, api_key):
    return _bootstrap_ruleset(
        failures, model, api_key,
        task_desc="date understanding — calendar arithmetic and date format reasoning",
        rule_prefix="DU",
        concepts="adding/subtracting days and months, day-of-week calculation, "
                 "UK vs US date formats, month lengths, leap years",
        verdict_fmt="(A) through (F)",
        ruleset_intro="Identify the starting date and operation, then apply arithmetic. Rules:",
        ruleset_footer="\nIf no rule applies, determine the starting date from context, "
                       "apply the requested operation carefully, and match to the options.\n\n"
                       "VERDICT: (A)–(F)\nRULE CITED: <DU-N> or NONE\nREASONING: show arithmetic.",
        section_title="DATE UNDERSTANDING RULES",
    )


DATE_UNDERSTANDING_TASK = TaskSpec(
    build_scoring_prompt=_du_scoring_prompt,
    is_correct=_mc_correct,
    answer_label=_mc_label,
    parse_verdict=_parse_mc,
    extract_post_think=_extract_reasoning,
    partition_key=_du_partition_key,
    partition_key_to_conditions=_du_key_to_conds,
    format_failure=_format_failure,
    generation_prompt_template=_DU_GEN_PROMPT,
    build_polarity_instruction=_generic_polarity,
    task_name="date_understanding",
    build_rule_scoring_prompt=_rule_score_prompt,
    bootstrap_ruleset=_du_bootstrap,
)


# ─────────────────────────────────────────────────────────────────────────────
# 5. NAVIGATE
# ─────────────────────────────────────────────────────────────────────────────

_NAV_SCORING = """\
You are determining whether a sequence of movement instructions returns to the starting point.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Always face forward. Track net displacement: forward/backward on one axis, left/right on the other.
You return to start only if BOTH net displacements are zero.

CRITICAL: Your VERY FIRST LINE must be:
  VERDICT: Yes
  VERDICT: No

VERDICT: Yes or No  ← FIRST LINE.
RULE CITED: <rule ID, e.g. NV-2> or NONE
REASONING: Show the running totals for each axis then give verdict.\
"""

_NAV_SCORING_COT = """\
You are determining whether a sequence of movement instructions returns to the starting point.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Think step by step: track forward/backward total and left/right total separately.
Return to start iff both totals are zero.

CRITICAL: Your VERY FIRST LINE must be:
  VERDICT: Yes
  VERDICT: No

VERDICT: Yes or No  ← FIRST LINE.
RULE CITED: <rule ID, e.g. NV-2> or NONE
REASONING: Show each axis total then give verdict.\
"""


def _nav_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _NAV_SCORING_COT if cot else _NAV_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


def _nav_partition_key(item: dict) -> tuple:
    t = item["input"].lower()
    # Compute net forward/back and left/right from the instructions
    fb = 0
    lr = 0
    for m in re.finditer(r"take (\d+) steps? (forward|backward|left|right)", t):
        n = int(m.group(1))
        d = m.group(2)
        if d == "forward":  fb += n
        elif d == "backward": fb -= n
        elif d == "left":   lr -= n
        elif d == "right":  lr += n
    fb_balanced = (fb == 0)
    lr_balanced = (lr == 0)
    return (fb_balanced, lr_balanced)


def _nav_key_to_conds(key: tuple) -> list[str]:
    fb_bal, lr_bal = key
    conds = []
    conds.append("forward/backward steps are balanced (net=0)" if fb_bal
                 else "forward/backward steps are NOT balanced")
    conds.append("left/right steps are balanced (net=0)" if lr_bal
                 else "left/right steps are NOT balanced")
    return conds


_NAV_GEN_PROMPT = (
    "You are an expert in spatial reasoning helping a model that fails at navigation return-to-start problems.\n\n"
    "=== REASONING ROADMAP ===\n{roadmap}\n=== END ROADMAP ===\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "KEY INSIGHT: Track two independent axes. Forward/backward is one axis; left/right is another.\n"
    "Both must sum to zero to return to start. 'Always face forward' means turning is irrelevant.\n\n"
    "DIAGNOSE:\n"
    "  1. Which axis did the model compute wrong?\n"
    "  2. Did it forget to track one axis? Miscalculate a subtraction? Confuse forward with right?\n"
    "  3. What check would have caught the error?\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [Navigation Error Type] ===\n"
    "FAILURE_TYPE: A (wrong axis calculation) or B (correct totals, wrong conclusion)\n"
    "ACTIVATE IF:\n"
    "  - the error pattern is: [describe what the model does wrong — e.g. 'only checks one axis']\n"
    "  - the instruction sequence has: [describe the structure that triggers the error]\n"
    "DO NOT ACTIVATE IF: [the simple case where steps obviously cancel]\n"
    "COMMON WRONG MOVE: [exactly what the model miscalculates]\n"
    "NEXT CHECK: [compute forward−backward total AND left−right total → both zero = Yes, else No]\n"
    "WHY THIS WORKS: [1-2 sentences on two-axis independence]\n"
    "SUPPORT:\n"
    "  • [short instruction sequence]  |  Answer: Yes/No  — [axis totals]\n"
    "TARGET_STEP: [roadmap step this corrects]\n\n"
    "OUTPUT 2 — ROADMAP PATCH\n"
    "=== ROADMAP PATCH ===\n"
    "[one-line rule addition, or '(none)']\n"
    "=== END ROADMAP PATCH ===\n"
    "{retry_context}"
)


def _nav_bootstrap(failures, model, api_key):
    return _bootstrap_ruleset(
        failures, model, api_key,
        task_desc="navigation — does a sequence of steps return to the starting point?",
        rule_prefix="NV",
        concepts="two-axis tracking (forward/back, left/right), net displacement, "
                 "both axes must be zero to return to start, 'always face forward' means no turning",
        verdict_fmt="Yes or No",
        ruleset_intro="Track forward/backward and left/right totals independently. Rules:",
        ruleset_footer="\nIf no rule applies, sum forward−backward and left−right separately. "
                       "Return to start iff both sums are zero.\n\n"
                       "VERDICT: Yes or No\nRULE CITED: <NV-N> or NONE\nREASONING: show both axis totals.",
        section_title="NAVIGATE RULES",
    )


NAVIGATE_TASK = TaskSpec(
    build_scoring_prompt=_nav_scoring_prompt,
    is_correct=_yesno_correct,
    answer_label=_yesno_label,
    parse_verdict=_parse_yesno,
    extract_post_think=_extract_reasoning,
    partition_key=_nav_partition_key,
    partition_key_to_conditions=_nav_key_to_conds,
    format_failure=_format_failure,
    generation_prompt_template=_NAV_GEN_PROMPT,
    build_polarity_instruction=_generic_polarity,
    task_name="navigate",
    build_rule_scoring_prompt=_rule_score_prompt,
    bootstrap_ruleset=_nav_bootstrap,
)


# ─────────────────────────────────────────────────────────────────────────────
# 6. SNARKS
# ─────────────────────────────────────────────────────────────────────────────

_SNARKS_SCORING = """\
You are identifying which of two statements is sarcastic.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

A sarcastic statement says the opposite of what it literally means, \
using positive language in a negative context or vice versa.

CRITICAL: Your VERY FIRST LINE must be:
  VERDICT: (A)
  VERDICT: (B)

VERDICT: (A) or (B)  ← FIRST LINE.
RULE CITED: <SK-N> or NONE
REASONING: Explain what makes the chosen statement sarcastic and the other literal.\
"""

_SNARKS_SCORING_COT = """\
You are identifying which of two statements is sarcastic.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Think step by step: look for a mismatch between the literal meaning and the context — \
positive wording used for something bad, or sincere praise that would be absurd in context.

CRITICAL: Your VERY FIRST LINE must be:
  VERDICT: (A)
  VERDICT: (B)

VERDICT: (A) or (B)  ← FIRST LINE.
RULE CITED: <SK-N> or NONE
REASONING: Identify the mismatch in each option then pick.\
"""


def _snarks_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _SNARKS_SCORING_COT if cot else _SNARKS_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


def _snarks_partition_key(item: dict) -> tuple:
    t = item["input"].lower()
    # Detect common sarcasm signal types
    has_praise_bad = any(w in t for w in ["terrible", "awful", "horrible", "worst", "useless", "idiot"])
    has_ironic_positive = any(w in t for w in ["genius", "brilliant", "amazing", "great job", "well done"])
    return (has_praise_bad, has_ironic_positive)


def _snarks_key_to_conds(key: tuple) -> list[str]:
    praise_bad, ironic_pos = key
    conds = []
    if praise_bad:
        conds.append("one option uses clearly negative terms (terrible, awful, worst)")
    if ironic_pos:
        conds.append("one option uses exaggerated praise (genius, brilliant, great job)")
    if not conds:
        conds.append("sarcasm signal is subtle — context mismatch rather than explicit negative terms")
    return conds


_SNARKS_GEN_PROMPT = (
    "You are an expert in pragmatics and irony helping a model that fails at sarcasm detection.\n\n"
    "=== REASONING ROADMAP ===\n{roadmap}\n=== END ROADMAP ===\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "DIAGNOSE:\n"
    "  1. What sarcasm signal is present in the correct answer that the model missed?\n"
    "     Think in terms of: context mismatch, exaggerated praise, self-evident absurdity, \n"
    "     literal reading that contradicts common sense\n"
    "  2. What made the wrong option look sarcastic to the model instead?\n"
    "  3. What everyday analogy captures the sarcasm pattern here?\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [Sarcasm Pattern]: [Memorable Name] ===\n"
    "FAILURE_TYPE: A (missed the sarcasm signal) or B (right signal, wrong option)\n"
    "ACTIVATE IF:\n"
    "  - scenario feels like: [describe the context mismatch pattern in plain language]\n"
    "  - the giveaway is: [what makes the sarcastic option recognizable once you see it]\n"
    "DO NOT ACTIVATE IF: [the case where the positive/negative wording is genuinely literal]\n"
    "COMMON WRONG MOVE: [what the model picks instead and why it's fooled]\n"
    "NEXT CHECK: [plain question to identify the sarcasm → (A) or (B)]\n"
    "WHY THIS WORKS: [1-2 sentences on the irony/context mismatch, in everyday terms]\n"
    "SUPPORT:\n"
    "  • [concrete sarcastic sentence example]  |  Answer: (X)  — [brief note on the mismatch]\n"
    "TARGET_STEP: [roadmap step this corrects]\n\n"
    "OUTPUT 2 — ROADMAP PATCH\n"
    "=== ROADMAP PATCH ===\n"
    "[one-line rule addition, or '(none)']\n"
    "=== END ROADMAP PATCH ===\n"
    "{retry_context}"
)


def _snarks_bootstrap(failures, model, api_key):
    return _bootstrap_ruleset(
        failures, model, api_key,
        task_desc="snarks — identifying which of two statements is sarcastic",
        rule_prefix="SK",
        concepts="context mismatch, irony, exaggerated praise for negative situation, "
                 "literal vs intended meaning, self-evident absurdity",
        verdict_fmt="(A) or (B)",
        ruleset_intro="Find the statement where the literal wording contradicts the implied meaning. Rules:",
        ruleset_footer="\nIf no rule applies, ask: which statement would be absurd or insincere if taken literally? "
                       "That is the sarcastic one.\n\nVERDICT: (A) or (B)\n"
                       "RULE CITED: <SK-N> or NONE\nREASONING: explain the mismatch.",
        section_title="SNARKS RULES",
    )


SNARKS_TASK = TaskSpec(
    build_scoring_prompt=_snarks_scoring_prompt,
    is_correct=_mc_correct,
    answer_label=_mc_label,
    parse_verdict=_parse_mc,
    extract_post_think=_extract_reasoning,
    partition_key=_snarks_partition_key,
    partition_key_to_conditions=_snarks_key_to_conds,
    format_failure=_format_failure,
    generation_prompt_template=_SNARKS_GEN_PROMPT,
    build_polarity_instruction=_generic_polarity,
    task_name="snarks",
    build_rule_scoring_prompt=_rule_score_prompt,
    bootstrap_ruleset=_snarks_bootstrap,
)
