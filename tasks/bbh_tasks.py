"""tasks/bbh_tasks.py — TaskSpec implementations for BBH tasks beyond boolean_expressions.

Tasks
-----
  CAUSAL_JUDGEMENT_TASK    — causal_judgement      (Yes / No)
  SPORTS_TASK              — sports_understanding   (yes / no)
  DISAMBIGUATION_TASK      — disambiguation_qa      ((A) / (B) / (C))
  MOVIE_TASK               — movie_recommendation   ((A) / (B) / (C) / (D))
  GEOMETRIC_TASK           — geometric_shapes       ((A) … (J))
"""

from __future__ import annotations

import re
from utils.task_spec import TaskSpec
from ICR_rules.rules.rule import RuleSet


# ─────────────────────────────────────────────────────────────────────────────
# Shared low-level parsers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_yesno(content: str) -> str | None:
    m = re.search(r"VERDICT:\s*(YES|NO)", content, re.IGNORECASE)
    return m.group(1).upper() if m else None


def _parse_mc(content: str) -> str | None:
    m = re.search(r"VERDICT:\s*\(?([A-Z])\)?", content, re.IGNORECASE)
    return f"({m.group(1).upper()})" if m else None


def _extract_reasoning(content: str) -> str:
    m = re.search(r"REASONING:\s*(.*)", content, re.DOTALL)
    return m.group(1).strip() if m else content.strip()


def _yesno_correct(predicted: str | None, item: dict) -> bool:
    return predicted is not None and predicted == item["answer"].strip().upper()


def _yesno_label(item: dict) -> str:
    return item["answer"].strip().upper()


def _mc_correct(predicted: str | None, item: dict) -> bool:
    return predicted is not None and predicted.upper() == item["answer"].strip().upper()


def _mc_label(item: dict) -> str:
    return item["answer"].strip().upper()


# ─────────────────────────────────────────────────────────────────────────────
# Shared rule-scoring prompt builder (all tasks use item["input"] as {{ question }})
# ─────────────────────────────────────────────────────────────────────────────

def _rule_score_prompt(template_text: str, item: dict) -> str:
    try:
        from jinja2 import Template as _T
        rendered = _T(template_text).render(question=item["input"])
    except Exception:
        rendered = template_text.replace("{{ question }}", item["input"])
    if item["input"][:40] not in rendered:
        rendered = rendered.rstrip() + f"\n\n{item['input']}\n"
    return rendered


# ─────────────────────────────────────────────────────────────────────────────
# Generation prompt factory
# ─────────────────────────────────────────────────────────────────────────────

def _gen_prompt(
    domain: str, task_desc: str,
    type_a: str, type_b: str,
    feature_vocab: str, verdict_fmt: str,
) -> str:
    """Return a generation prompt template (uses {roadmap}, {case_studies}, etc.)."""
    return (
        f"You are an expert in {domain} working on automated reasoning evaluation.\n\n"
        f"A weaker reasoning model keeps making the same mistake on: {task_desc}.\n"
        "Write a TEACHING NOTE diagnosing WHY it fails and teaching the exact fix.\n\n"
        "=== REASONING ROADMAP ===\n{roadmap}\n=== END ROADMAP ===\n\n"
        "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
        "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
        "Your case study MUST address a gap NOT covered above.\n"
        "=== END ALREADY COVERED ===\n\n"
        "=== FAILURES WITH INCORRECT MODEL REASONING ===\n{failure_lines}\n\n"
        "=== YOUR TASK ===\n{polarity_instruction}\n\n"
        "Step 0 — DIAGNOSE:\n"
        f"  TYPE A — MISSING KNOWLEDGE: {type_a}\n"
        f"  TYPE B — WRONG REASONING PATTERN: {type_b}\n\n"
        "Step 1 — State the missing fact (TYPE A) or wrong move (TYPE B) precisely.\n\n"
        "Step 2 — CORRECT MOVE: the specific check that gives the right answer.\n\n"
        "Step 3 — TRIGGER: precise structural conditions.\n"
        "  FEATURE VOCABULARY — use these exact terms in ACTIVATE IF:\n"
        f"{feature_vocab}\n\n"
        "Step 4 — ANTI-TRIGGER: 1-2 cases where the model's approach is correct.\n\n"
        "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
        "=== CASE STUDY: [short title] ===\n"
        "FAILURE_TYPE: A or B\n"
        "ACTIVATE IF:\n"
        "  - [condition 1]\n"
        "DO NOT ACTIVATE IF: [closest case where model is correct]\n"
        "COMMON WRONG MOVE: [1 sentence]\n"
        f"NEXT CHECK: [what to verify → {verdict_fmt}]\n"
        "WHY THIS WORKS: [1-2 sentences]\n"
        "SUPPORT:\n"
        f"  • [example]  |  Answer: {verdict_fmt}  — [brief note]\n"
        "TARGET_STEP: [roadmap step this corrects]\n\n"
        "OUTPUT 2 — ROADMAP PATCH\n"
        "=== ROADMAP PATCH ===\n"
        "[one-line addition, or '(none)']\n"
        "=== END ROADMAP PATCH ===\n"
        "{retry_context}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap factory
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_ruleset(
    failures: list[dict],
    model: str,
    api_key: str,
    *,
    task_desc: str,
    rule_prefix: str,
    concepts: str,
    verdict_fmt: str,
    ruleset_intro: str,
    ruleset_footer: str,
    section_title: str,
) -> RuleSet:
    from utils.llm_client import call_llm
    from ICR_rules.rules.rule import Rule, Section, RuleSet as _RS, _infer_verdict

    failure_lines = "\n".join(
        f"  [{i}] {it.get('input', '?')[:200].strip()}"
        f"\n      Expected: {it.get('answer','?').strip()}  Got: {it.get('predicted','?')}"
        + (f"\n      Correct reasoning: {it['reason'][:300]}" if it.get("reason") else "")
        for i, it in enumerate(failures[:15], 1)
    )
    prompt = (
        f"You are an expert in {task_desc}. A weaker model keeps failing.\n"
        f"Key concepts: {concepts}\n\n"
        f"Incorrectly predicted items:\n{failure_lines}\n\n"
        f"Write 6-10 named rules ({rule_prefix}-1, {rule_prefix}-2, ...) capturing the error patterns.\n"
        "Requirements:\n"
        "  - Each rule: structural condition + verdict\n"
        "  - Order: most specific first\n"
        f"  - End every rule with  →  <verdict>  (format: {verdict_fmt})\n\n"
        f"Output ONLY the rules, one per line:\n"
        f"{rule_prefix}-1: <condition> →  <verdict>\n..."
    )
    response = call_llm(prompt, model=model, api_key=api_key, max_tokens=600, temperature=0.3)

    raw_rules: list[tuple[str, str]] = []
    for line in response.content.splitlines():
        line = line.strip()
        m = re.match(rf"({rule_prefix}-\w+):\s*(.+)", line)
        if m:
            raw_rules.append((m.group(1), line))

    if not raw_rules:
        raw_rules = [(f"{rule_prefix}-1", f"{rule_prefix}-1: fallback rule →  YES")]

    rule_objects = [
        Rule(id=rid, section="main", text=rtext, verdict=_infer_verdict(rtext))
        for rid, rtext in raw_rules
    ]
    section = Section(name="main", title=section_title,
                      preamble="", rules=rule_objects, postamble="")
    return _RS(intro=ruleset_intro, sections=[section],
               footer=ruleset_footer, source_path="")


def _format_failure(item: dict, max_input: int = 300) -> str:
    expected = item.get("answer", "?").strip()
    predicted = item.get("predicted", "?")
    reasoning = (item.get("post_think") or item.get("reasoning") or "").strip()
    block = (
        f"  Input: {item.get('input', '?')[:max_input]}\n"
        f"  expected={expected}  predicted={predicted}\n"
        f"  Model's wrong reasoning:\n"
        f"    {reasoning[:300] if reasoning else '(not captured)'}"
    )
    exact = item.get("_oracle_exact", "")
    if exact:
        block += f"\n  Correct reasoning:\n    {exact[:200]}"
    return block


# ─────────────────────────────────────────────────────────────────────────────
# 1. CAUSAL JUDGEMENT
# ─────────────────────────────────────────────────────────────────────────────

_CAUSAL_SCORING = """\
You are answering causal reasoning questions from the perspective of a typical person.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

CRITICAL: Your VERY FIRST LINE must be:
  VERDICT: YES
  VERDICT: NO
Nothing before this line. No exceptions.

VERDICT: YES or NO  ← FIRST LINE.
RULE CITED: <rule ID, e.g. CJ-2> or NONE
REASONING: You MUST begin with the rule you are applying (e.g. "CJ-3 applies: this is overdetermination because..."). \
If no rule matched, start with "No rule matched.".\
"""

_CAUSAL_SCORING_COT = """\
You are answering causal reasoning questions from the perspective of a typical person.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Think through the causal structure step by step, then give your verdict.

CRITICAL: Your VERY FIRST LINE must be:
  VERDICT: YES
  VERDICT: NO

VERDICT: YES or NO  ← FIRST LINE.
RULE CITED: <rule ID, e.g. CJ-2> or NONE
REASONING: Begin with the rule applied (e.g. "CJ-3 applies: ...") or "No rule matched.".\
"""


def _causal_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _CAUSAL_SCORING_COT if cot else _CAUSAL_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


def _causal_partition_key(item: dict) -> tuple:
    t = item["input"].lower()
    has_cf = any(w in t for w in ["would have", "could have", "had not", "hadn't", "if not"])
    has_od = ("both" in t or "each" in t) and ("sufficient" in t or "alone" in t or "independently" in t)
    has_pp = "prevent" in t and "prevent" in t[t.index("prevent") + 7:]
    return (has_cf, has_od or has_pp)


def _causal_key_to_conds(key: tuple) -> list[str]:
    has_cf, has_complex = key
    conds = []
    conds.append("scenario involves counterfactual reasoning" if has_cf
                 else "no explicit counterfactuals in scenario")
    if has_complex:
        conds.append("scenario involves overdetermination or double-prevention")
    return conds


def _causal_polarity(polarity: str, failure_type: str, divergence_step: str) -> str:
    base = {
        "YES": ("POLARITY — FALSE NEGATIVE (model said NO, correct is YES):\n"
                "Model failed to recognise causation. Identify the missing causal principle."),
        "NO":  ("POLARITY — FALSE POSITIVE (model said YES, correct is NO):\n"
                "Model incorrectly attributed causation. Focus on overdetermination, "
                "preemption, double-prevention, or proximate vs distal cause."),
    }.get(polarity.upper(),
          "Diagnose whether failures are TYPE A (missing principle) or TYPE B (wrong application).")
    if failure_type == "ABANDONMENT":
        base = "STRATEGY — ABANDONMENT: model gave up. Show the next step.\n\n" + base
    return base


def _causal_identify_rule(reasoning: str) -> str | None:
    m = re.search(r"RULE CITED:\s*(CJ-\w+)", reasoning)
    if m:
        return m.group(1)
    m = re.search(r"\b(CJ-\w+)", reasoning)
    return m.group(1) if m else None


_CAUSAL_INTRO = ("You are answering causal judgment questions as a typical person.\n"
                 "Apply these rules in order (stop at the first match):")
_CAUSAL_FOOTER = (
    "\nIf no rule applies, reason from first principles: identify the proximate cause "
    "and consider whether the outcome depends counterfactually on the actor's action.\n\n"
    "VERDICT: YES or NO\n"
    "RULE CITED: <rule ID, e.g. CJ-2> or NONE\n"
    "REASONING: Begin with the rule applied (e.g. 'CJ-3 applies: ...') or 'No rule matched.'."
)


def _causal_bootstrap(failures, model, api_key):
    return _bootstrap_ruleset(
        failures, model, api_key,
        task_desc="causal reasoning and judgment",
        rule_prefix="CJ",
        concepts="proximate vs distal causation, overdetermination, preemption, "
                 "double-prevention, counterfactual dependence, typical person's intuition",
        verdict_fmt="YES or NO",
        ruleset_intro=_CAUSAL_INTRO,
        ruleset_footer=_CAUSAL_FOOTER,
        section_title="CAUSAL JUDGMENT RULES",
    )


_CAUSAL_CONCRETE_BOOTSTRAP_PROMPT = """\
You are creating a cheat sheet for a language model to answer causal judgment questions.

The model answers from the perspective of a typical person.

Below are {n} questions the model answered WRONG:
{failure_lines}

Your task: write a set of named concrete scenario examples that teach the key causal \
reasoning patterns a typical person uses. Each example is a short named scenario \
(like "Two Wires", "Two Gardeners") with a clear causal structure, verdict, and explanation.

Cover these causal types (use the failures above to determine which matter most):
1. Overdetermination — two independent causes each sufficient alone
2. Preemption — one cause takes over before another can act
3. Double prevention — X prevents Y which would have prevented the outcome
4. Proximate vs distal causation — immediate vs background cause
5. Counterfactual dependence — outcome would not have occurred without this action
6. Joint sufficiency — two factors both required; neither alone caused the outcome

For each scenario, use this exact format:

=== [Causal Type]: [Scenario Name] ===
Scenario: <1-2 sentence concrete description>
Causal structure: <what role each actor/factor plays>
Verdict: YES / NO
Why a typical person says this: <1 sentence>
Apply when: <structural cue to recognise this pattern>

Write 4-6 scenarios. Choose concrete everyday situations. \
Do NOT reuse the failure examples above — invent fresh scenarios.
"""


def _causal_concrete_bootstrap(failures: list[dict], model: str, api_key: str) -> str:
    from utils.llm_client import call_llm
    failure_lines = "\n".join(
        f"  [{i}] {it.get('input', '?')[:250].strip()}"
        f"\n      Expected: {it.get('answer','?').strip()}  Got: {it.get('predicted','?')}"
        for i, it in enumerate(failures[:15], 1)
    )
    prompt = _CAUSAL_CONCRETE_BOOTSTRAP_PROMPT.format(
        n=min(len(failures), 15),
        failure_lines=failure_lines,
    )
    response = call_llm(prompt, model=model, api_key=api_key, max_tokens=1200, temperature=0.3)
    return response.content.strip()


_CAUSAL_CONCRETE_CS_PROMPT = """\
You are improving a cheat sheet used by a language model to answer causal judgment questions.

The model answers from the perspective of a typical person.

=== EXISTING CHEAT SHEET ===
{cheatsheet}
=== END CHEAT SHEET ===

Below are {n} questions the model answered WRONG in a specific failure cluster:
{failure_lines}

Your task: write ONE new named concrete scenario example (like "Two Wires", "Traffic Light") \
that teaches the causal reasoning pattern the model is missing in these failures.

Requirements:
- The scenario must be concrete and everyday (not abstract)
- It must be different from any scenarios already in the cheat sheet
- It must directly address the error pattern shown in the failures above
- It should generalise to novel scenarios, not just fix these specific items

Use this exact format:

=== [Causal Type]: [Scenario Name] ===
Scenario: <1-2 sentence concrete description>
Causal structure: <what role each actor/factor plays>
Verdict: YES / NO
Why a typical person says this: <1 sentence>
Apply when: <structural cue to recognise this pattern>
"""


def _causal_concrete_cs_gen(
    failures: list[dict],
    cheatsheet_text: str,
    model: str,
    api_key: str,
) -> str | None:
    from utils.llm_client import call_llm
    failure_lines = "\n".join(
        f"  [{i}] {it.get('input', '?')[:250].strip()}"
        f"\n      Expected: {it.get('answer','?').strip()}  Got: {it.get('predicted','?')}"
        + (f"\n      Wrong reasoning: {it.get('post_think','')[:200]}" if it.get("post_think") else "")
        for i, it in enumerate(failures[:8], 1)
    )
    prompt = _CAUSAL_CONCRETE_CS_PROMPT.format(
        n=min(len(failures), 8),
        cheatsheet=cheatsheet_text[:3000],
        failure_lines=failure_lines,
    )
    response = call_llm(prompt, model=model, api_key=api_key, max_tokens=500, temperature=0.4)
    text = response.content.strip()
    return text if text else None


_CAUSAL_GEN_PROMPT = (
    "You are an expert in causal reasoning helping a model that keeps failing on causal judgment questions.\n"
    "The model answers from the perspective of a typical person.\n\n"
    "=== REASONING ROADMAP ===\n{roadmap}\n=== END ROADMAP ===\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES WITH INCORRECT MODEL REASONING ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "DIAGNOSE the failures by answering:\n"
    "  1. What everyday story shape do these failures share?\n"
    "     Think in terms of concrete situations: two people both trying, one blocking another,\n"
    "     a backup that wasn't needed, a chain of events where something was prevented, etc.\n"
    "  2. What question is a typical person really asking when they judge causation here?\n"
    "  3. What intuition is the model missing — not as an abstract label, but as a plain feeling\n"
    "     a non-expert would have about who is really responsible?\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [Causal Pattern]: [Memorable Scenario Name] ===\n"
    "FAILURE_TYPE: A (model used wrong causal intuition) or B (right intuition, wrong actor)\n"
    "ACTIVATE IF:\n"
    "  - scenario feels like: [1 plain sentence — what shared story shape makes this case study relevant]\n"
    "  - the question is asking: [what the typical person is really judging — e.g. 'whether X's action made a difference']\n"
    "DO NOT ACTIVATE IF: [the superficially similar case where the model's usual reasoning is correct]\n"
    "COMMON WRONG MOVE: [what the model incorrectly concludes and why, in plain language]\n"
    "NEXT CHECK: [a plain question a typical person would ask to resolve this → YES or NO]\n"
    "WHY THIS WORKS: [1-2 sentences on the everyday intuition — avoid abstract causal theory labels]\n"
    "SUPPORT:\n"
    "  • [concrete everyday scenario]  |  Answer: YES/NO  — [brief plain-language note]\n"
    "TARGET_STEP: [roadmap step this corrects]\n\n"
    "OUTPUT 2 — ROADMAP PATCH\n"
    "=== ROADMAP PATCH ===\n"
    "[one-line addition to the roadmap, or '(none)']\n"
    "=== END ROADMAP PATCH ===\n"
    "{retry_context}"
)

CAUSAL_JUDGEMENT_TASK = TaskSpec(
    build_scoring_prompt=_causal_scoring_prompt,
    is_correct=_yesno_correct,
    answer_label=_yesno_label,
    parse_verdict=_parse_yesno,
    extract_post_think=_extract_reasoning,
    partition_key=_causal_partition_key,
    partition_key_to_conditions=_causal_key_to_conds,
    format_failure=_format_failure,
    generation_prompt_template=_CAUSAL_GEN_PROMPT,
    build_polarity_instruction=_causal_polarity,
    task_name="causal_judgement",
    build_rule_scoring_prompt=_rule_score_prompt,
    identify_triggered_rule=_causal_identify_rule,
    rule_id_regex=r"(CJ-\w+)",
    bootstrap_ruleset=_causal_bootstrap,
    bootstrap_cheatsheet_fn=_causal_concrete_bootstrap,
    concrete_cs_gen_fn=_causal_concrete_cs_gen,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SPORTS UNDERSTANDING
# ─────────────────────────────────────────────────────────────────────────────

_SPORTS_SCORING = """\
You are judging whether a sports sentence is plausible.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

CRITICAL: Your VERY FIRST LINE must be:
  VERDICT: YES
  VERDICT: NO

VERDICT: YES or NO  ← FIRST LINE.
RULE CITED: <rule ID, e.g. SP-3> or NONE
REASONING: Begin with the rule applied (e.g. "SP-2 applies: this action is not possible in this sport because...") \
or "No rule matched.".\
"""

_SPORTS_SCORING_COT = """\
You are judging whether a sports sentence is plausible.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Think step by step: identify the athlete, sport, and whether the action described is possible in that sport.

CRITICAL: Your VERY FIRST LINE must be:
  VERDICT: YES
  VERDICT: NO

VERDICT: YES or NO  ← FIRST LINE.
RULE CITED: <rule ID, e.g. SP-3> or NONE
REASONING: Begin with the rule applied or "No rule matched.".\
"""

_SPORTS_KNOWN = {
    "hockey": ["lindholm", "carlson", "ovechkin", "crosby", "mackinnon", "hedman",
               "stamkos", "pastrnak", "draisaitl", "point", "matthews"],
    "basketball": ["lebron", "curry", "durant", "harden", "giannis", "doncic",
                   "kawhi", "paul", "lillard", "tatum", "embiid", "jokic"],
    "football": ["mahomes", "brady", "rodgers", "wilson", "stafford", "burrow",
                 "jackson", "prescott", "murray", "herbert", "allen"],
    "baseball": ["trout", "betts", "judge", "arenado", "freeman", "devers",
                 "guerrero", "alvarez", "soto", "goldschmidt"],
    "soccer": ["messi", "ronaldo", "neymar", "mbappe", "salah", "lewandowski",
               "benzema", "de bruyne", "kante", "modric"],
    "tennis": ["federer", "djokovic", "nadal", "murray", "tsitsipas", "zverev",
               "medvedev", "alcaraz", "swiatek", "osaka"],
}


def _sports_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _SPORTS_SCORING_COT if cot else _SPORTS_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


def _sports_partition_key(item: dict) -> tuple:
    t = item["input"].lower()
    sport = "other"
    for s, players in _SPORTS_KNOWN.items():
        if any(p in t for p in players) or s in t:
            sport = s
            break
    return (sport,)


def _sports_key_to_conds(key: tuple) -> list[str]:
    (sport,) = key
    return [f"sentence is about {sport}"]


def _sports_polarity(polarity: str, failure_type: str, divergence_step: str) -> str:
    base = {
        "YES": ("POLARITY — FALSE NEGATIVE (model said NO, correct is YES):\n"
                "The action IS plausible but model said it isn't. "
                "Likely model doesn't know the player's sport or misidentified the action."),
        "NO":  ("POLARITY — FALSE POSITIVE (model said YES, correct is NO):\n"
                "The action is NOT plausible but model said it is. "
                "Likely model confused sports terminology (e.g. 'beat the buzzer' is basketball, not hockey)."),
    }.get(polarity.upper(), "Diagnose TYPE A (missing sport knowledge) or TYPE B (wrong action/sport mapping).")
    if failure_type == "ABANDONMENT":
        base = "STRATEGY — ABANDONMENT: model gave up. Show the next step.\n\n" + base
    return base


def _sports_identify_rule(reasoning: str) -> str | None:
    m = re.search(r"RULE CITED:\s*(SP-\w+)", reasoning)
    if m:
        return m.group(1)
    m = re.search(r"\b(SP-\w+)", reasoning)
    return m.group(1) if m else None


_SPORTS_INTRO = ("You are judging whether sports sentences are plausible.\n"
                 "Apply these rules in order (stop at the first match):")
_SPORTS_FOOTER = (
    "\nIf no rule applies, identify the athlete's sport and judge whether the described "
    "action is physically and legally possible in that sport.\n\n"
    "VERDICT: YES or NO\n"
    "RULE CITED: <rule ID, e.g. SP-2> or NONE\n"
    "REASONING: Begin with the rule applied or 'No rule matched.'."
)


def _sports_bootstrap(failures, model, api_key):
    return _bootstrap_ruleset(
        failures, model, api_key,
        task_desc="sports plausibility judgment",
        rule_prefix="SP",
        concepts="sport-specific actions and terminology, athlete-sport mapping, "
                 "what actions are possible in which sports",
        verdict_fmt="YES or NO",
        ruleset_intro=_SPORTS_INTRO,
        ruleset_footer=_SPORTS_FOOTER,
        section_title="SPORTS PLAUSIBILITY RULES",
    )


_SPORTS_GEN_PROMPT = (
    "You are an expert in sports rules helping a model that fails on sports plausibility questions.\n"
    "The task: decide if a sentence about an athlete's action is plausible given their sport.\n\n"
    "=== REASONING ROADMAP ===\n{roadmap}\n=== END ROADMAP ===\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES WITH INCORRECT MODEL REASONING ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "DIAGNOSE the failures by answering:\n"
    "  1. Which sport is mentioned? Is the athlete's name a reliable signal for their sport?\n"
    "  2. What action is described? (scoring / movement / equipment use / position / rule violation)\n"
    "  3. Is this action physically or rule-wise possible in that sport?\n"
    "  4. Did the model fail because it doesn't know the sport's rules (TYPE A),\n"
    "     or because it confused rules from a different sport (TYPE B)?\n\n"
    "STRUCTURAL VOCABULARY — use these exact terms in ACTIVATE IF:\n"
    "    sport: hockey / basketball / football / baseball / soccer / tennis / golf / swimming / other\n"
    "    action_type: scoring / movement / equipment / position / rule_violation\n"
    "    error: unknown_sport_rule (TYPE A) / cross_sport_confusion (TYPE B)\n"
    "    answer_is_yes: the sentence IS plausible in that sport\n"
    "    answer_is_no: the sentence is NOT plausible in that sport\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [short title naming the sport and action type] ===\n"
    "FAILURE_TYPE: A (model doesn't know this sport's rule) or B (confuses rules from another sport)\n"
    "ACTIVATE IF:\n"
    "  - [sport and action_type from vocabulary above]\n"
    "DO NOT ACTIVATE IF: [case where the action is clearly possible/impossible and model is correct]\n"
    "COMMON WRONG MOVE: [which sport rule the model incorrectly applies]\n"
    "NEXT CHECK: [the specific sport rule to verify → sentence IS or IS NOT plausible → YES or NO]\n"
    "WHY THIS WORKS: [1-2 sentences on the sport-specific rule]\n"
    "SUPPORT:\n"
    "  • [athlete name + action sentence]  |  Answer: YES/NO  — [sport rule note]\n"
    "TARGET_STEP: [roadmap step this corrects]\n\n"
    "OUTPUT 2 — ROADMAP PATCH\n"
    "=== ROADMAP PATCH ===\n"
    "[one-line sport rule to add to the roadmap, or '(none)']\n"
    "=== END ROADMAP PATCH ===\n"
    "{retry_context}"
)

SPORTS_TASK = TaskSpec(
    build_scoring_prompt=_sports_scoring_prompt,
    is_correct=_yesno_correct,
    answer_label=_yesno_label,
    parse_verdict=_parse_yesno,
    extract_post_think=_extract_reasoning,
    partition_key=_sports_partition_key,
    partition_key_to_conditions=_sports_key_to_conds,
    format_failure=_format_failure,
    generation_prompt_template=_SPORTS_GEN_PROMPT,
    build_polarity_instruction=_sports_polarity,
    task_name="sports_understanding",
    build_rule_scoring_prompt=_rule_score_prompt,
    identify_triggered_rule=_sports_identify_rule,
    rule_id_regex=r"(SP-\w+)",
    bootstrap_ruleset=_sports_bootstrap,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DISAMBIGUATION QA
# ─────────────────────────────────────────────────────────────────────────────

_DISAMBIG_SCORING = """\
You are resolving pronoun antecedents.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

CRITICAL: Your VERY FIRST LINE must be one of:
  VERDICT: (A)
  VERDICT: (B)
  VERDICT: (C)

VERDICT: (A), (B), or (C)  ← FIRST LINE.
RULE CITED: <rule ID, e.g. DQ-2> or NONE
REASONING: Identify who the pronoun refers to, citing the rule applied.\
"""

_DISAMBIG_SCORING_COT = """\
You are resolving pronoun antecedents.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Think through who each candidate referent is, which pronouns appear, and the pragmatic context.

CRITICAL: Your VERY FIRST LINE must be one of:
  VERDICT: (A)
  VERDICT: (B)
  VERDICT: (C)

VERDICT: (A), (B), or (C)  ← FIRST LINE.
RULE CITED: <rule ID, e.g. DQ-2> or NONE
REASONING: Explain the antecedent reasoning.\
"""


def _disambig_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _DISAMBIG_SCORING_COT if cot else _DISAMBIG_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


def _disambig_partition_key(item: dict) -> tuple:
    t = item["input"].lower()
    pronoun = ("they" if "they " in t or "their " in t else
               "she" if " she " in t or " her " in t else "he")
    has_ambig_opt = "ambiguous" in t
    return (pronoun, has_ambig_opt)


def _disambig_key_to_conds(key: tuple) -> list[str]:
    pronoun, has_ambig = key
    conds = [f"pronoun is '{pronoun}'"]
    if has_ambig:
        conds.append("one option is 'Ambiguous'")
    return conds


def _disambig_polarity(polarity: str, failure_type: str, divergence_step: str) -> str:
    base = (
        f"POLARITY — model selected wrong option (majority correct answer is {polarity}):\n"
        "TYPE A: model lacks pragmatic knowledge about typical pronoun resolution conventions.\n"
        "TYPE B: model has the knowledge but misjudges which entity is the antecedent."
    )
    if failure_type == "ABANDONMENT":
        base = "STRATEGY — ABANDONMENT: model gave up. Show the next step.\n\n" + base
    return base


def _disambig_identify_rule(reasoning: str) -> str | None:
    m = re.search(r"RULE CITED:\s*(DQ-\w+)", reasoning)
    if m:
        return m.group(1)
    m = re.search(r"\b(DQ-\w+)", reasoning)
    return m.group(1) if m else None


_DISAMBIG_GEN_PROMPT = (
    "You are an expert in pronoun resolution helping a model that fails on disambiguation questions.\n"
    "The task: given a sentence and 2-3 candidate referents, decide which entity a pronoun refers to.\n\n"
    "=== REASONING ROADMAP ===\n{roadmap}\n=== END ROADMAP ===\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES WITH INCORRECT MODEL REASONING ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "DIAGNOSE the failures by answering:\n"
    "  1. List the candidate referents and their answer labels (A), (B), (C).\n"
    "  2. What pronoun is being resolved? What gender/number constraints apply?\n"
    "  3. Which resolution cue determines the correct answer?\n"
    "     subjecthood preference / recency (most recent mention) / thematic role / gender agreement / world knowledge\n"
    "  4. Is 'Ambiguous' ever correct here? (only when two cues genuinely conflict with equal strength)\n"
    "  5. Did the model apply the wrong cue (TYPE A) or apply the right cue to the wrong entity (TYPE B)?\n\n"
    "STRUCTURAL VOCABULARY — use these exact terms in ACTIVATE IF:\n"
    "    pronoun: he / she / they / him / her / them / his / her\n"
    "    resolution_cue: subjecthood / recency / gender_agreement / thematic_role / world_knowledge\n"
    "    has_ambiguous_option: one answer option is 'Ambiguous'\n"
    "    n_candidates: number of named referent candidates (2 or 3)\n"
    "    answer_is_A / answer_is_B / answer_is_C\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [short title naming the cue, e.g. 'Subjecthood Beats Recency'] ===\n"
    "FAILURE_TYPE: A (model uses wrong resolution cue) or B (right cue, wrong entity)\n"
    "ACTIVATE IF:\n"
    "  - [pronoun and resolution_cue from vocabulary above]\n"
    "DO NOT ACTIVATE IF: [case where the referent is unambiguous and model is correct]\n"
    "COMMON WRONG MOVE: [which entity the model picks and which cue it incorrectly prioritises]\n"
    "NEXT CHECK: [the cue to apply → answer is (A), (B), or (C)]\n"
    "WHY THIS WORKS: [1-2 sentences on the linguistic/pragmatic principle]\n"
    "SUPPORT:\n"
    "  • [example sentence + candidates]  |  Answer: (A)/(B)/(C)  — [cue note]\n"
    "TARGET_STEP: [roadmap step this corrects]\n\n"
    "OUTPUT 2 — ROADMAP PATCH\n"
    "=== ROADMAP PATCH ===\n"
    "[one-line addition to the pronoun resolution procedure, or '(none)']\n"
    "=== END ROADMAP PATCH ===\n"
    "{retry_context}"
)

DISAMBIGUATION_TASK = TaskSpec(
    build_scoring_prompt=_disambig_scoring_prompt,
    is_correct=_mc_correct,
    answer_label=_mc_label,
    parse_verdict=_parse_mc,
    extract_post_think=_extract_reasoning,
    partition_key=_disambig_partition_key,
    partition_key_to_conditions=_disambig_key_to_conds,
    format_failure=_format_failure,
    generation_prompt_template=_DISAMBIG_GEN_PROMPT,
    build_polarity_instruction=_disambig_polarity,
    task_name="disambiguation_qa",
    build_rule_scoring_prompt=_rule_score_prompt,
    identify_triggered_rule=_disambig_identify_rule,
    rule_id_regex=r"(DQ-\w+)",
    bootstrap_ruleset=None,  # MC tasks: use Phase 2 only by default
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. MOVIE RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────────────

_MOVIE_SCORING = """\
You are recommending movies based on similarity to a set of seed films.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

CRITICAL: Your VERY FIRST LINE must be one of:
  VERDICT: (A)
  VERDICT: (B)
  VERDICT: (C)
  VERDICT: (D)

VERDICT: (A), (B), (C), or (D)  ← FIRST LINE.
RULE CITED: <rule ID, e.g. MR-2> or NONE
REASONING: Identify shared genre/era/tone features and cite the rule.\
"""

_MOVIE_SCORING_COT = """\
You are recommending movies based on similarity to a set of seed films.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Think through: what genre, era, tone, and audience do the seed films share? Which option best matches?

CRITICAL: Your VERY FIRST LINE must be one of:
  VERDICT: (A)
  VERDICT: (B)
  VERDICT: (C)
  VERDICT: (D)

VERDICT: (A), (B), (C), or (D)  ← FIRST LINE.
RULE CITED: <rule ID, e.g. MR-2> or NONE
REASONING: Explain your genre/similarity reasoning.\
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
    "=== REASONING ROADMAP ===\n{roadmap}\n=== END ROADMAP ===\n\n"
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
    "TARGET_STEP: [roadmap step this corrects]\n\n"
    "OUTPUT 2 — ROADMAP PATCH\n"
    "=== ROADMAP PATCH ===\n"
    "[one-line addition to the similarity reasoning procedure, or '(none)']\n"
    "=== END ROADMAP PATCH ===\n"
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
)


# ─────────────────────────────────────────────────────────────────────────────
# 5. GEOMETRIC SHAPES
# ─────────────────────────────────────────────────────────────────────────────

_GEO_SCORING = """\
You are identifying geometric shapes from SVG path descriptions.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

CRITICAL: Your VERY FIRST LINE must be one of:
  VERDICT: (A)  VERDICT: (B)  VERDICT: (C)  … VERDICT: (J)

VERDICT: (A)–(J)  ← FIRST LINE.
RULE CITED: <rule ID, e.g. GS-3> or NONE
REASONING: Count the vertices / line segments, name the shape, cite the rule.\
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

CRITICAL: Your VERY FIRST LINE must be one of:
  VERDICT: (A)  VERDICT: (B)  …  VERDICT: (J)

VERDICT: (A)–(J)  ← FIRST LINE.
RULE CITED: <rule ID, e.g. GS-3> or NONE
REASONING: Count vertices and identify shape, citing the rule.\
"""


def _geo_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _GEO_SCORING_COT if cot else _GEO_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


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
    "RULE CITED: <rule ID, e.g. GS-2> or NONE\n"
    "REASONING: Begin with the rule applied or 'No rule matched. Counted N vertices → <shape>'."
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


_GEO_GEN_PROMPT = (
    "You are an expert in SVG path geometry helping a model that fails on geometric shape identification.\n"
    "The task: given an SVG path string, identify which named shape (triangle, hexagon, etc.) it describes.\n\n"
    "=== REASONING ROADMAP ===\n{roadmap}\n=== END ROADMAP ===\n\n"
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
    "TARGET_STEP: [roadmap step this corrects]\n\n"
    "OUTPUT 2 — ROADMAP PATCH\n"
    "=== ROADMAP PATCH ===\n"
    "[one-line counting rule or name-mapping addition, or '(none)']\n"
    "=== END ROADMAP PATCH ===\n"
    "{retry_context}"
)

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
)
