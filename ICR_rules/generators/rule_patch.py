"""
rule_patch.py — Generate rule patches using a strong oracle model (GPT-5.4).

Given a target rule that is misfiring on a set of failures, the generator
asks the oracle to propose a minimal patch: tighten a condition, split the
rule, replace it, or insert a guard before it.

The oracle receives:
  - The exact current rule text
  - Full cheatsheet context (read-only, for reference)
  - Failing cases with the model's wrong reasoning + GPT-5.4 correct traces
  - Correct cases the rule handles properly (regression anchors)

Output is a RulePatch with the new rule text(s) and reasoning.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from utils.llm_client import call_llm, get_api_key
from ICR_reasoning.core.oracle import OracleDict
from ..rules.rule import Rule, RulePatch, RuleSet

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

RULE_PATCH_PROMPT = """\
You are an expert in universal algebra improving a decision guide used by a weaker \
reasoning model (Gemma 4 31B) to classify magma equation implications (TRUE/FALSE).

The guide applies a sequence of named rules. One specific rule is causing \
classification errors. Your job: propose the smallest possible patch that fixes \
the failing cases without breaking the correct ones.

=== TARGET RULE ===
Rule ID: {rule_id}
Current text:
{rule_text}

=== FULL CHEATSHEET (read-only context — do not patch other rules) ===
{cheatsheet_context}

=== FAILING CASES (this rule fired → wrong verdict) ===
{failure_lines}

=== CORRECT CASES THIS RULE HANDLES (do not break) ===
{correct_lines}

=== YOUR TASK ===
Choose exactly one patch type:

  TIGHTEN   — add one or two conditions to the existing rule so it no longer
               fires on the failing cases above (keep same rule ID)
  SPLIT     — replace with two rules: one for the correct cases, one for what
               was incorrectly matched (use IDs like TR-3A, TR-3B)
  REPLACE   — rewrite the rule entirely with tighter conditions (keep same ID)
  ADD_GUARD — insert a new rule IMMEDIATELY BEFORE this one that catches the
               failing cases first and emits the correct verdict (new rule ID
               ends in -G, e.g. TR-3-G)

Hard requirements:
  1. Use only features already defined in the cheatsheet:
     bare, vars, size, imb, vA, vB, sA, sB, LP, RP, SET, XOR, AB,
     rhsVars, rhsTotals, Lx, Rx, xTop, topShape, square
  2. Keep rule text concise — match the style of existing rules exactly
  3. Every new rule must end with →  TRUE or →  FALSE
  4. For SPLIT/ADD_GUARD: list rules in the order they should appear

Output EXACTLY this format (no extra text before or after):
PATCH_TYPE: <TIGHTEN|SPLIT|REPLACE|ADD_GUARD>
NEW_RULES:
  <rule_id>: <rule_text>
  [<rule_id_B>: <rule_text_B>]
REASONING: <why this patch fixes failures without breaking correct cases>
VERIFY: <for each failing case, state which new condition fails to match>
"""

# Tokens for patch generation
PATCH_MAX_TOKENS = 600
PATCH_TEMPS = [0.3, 0.7]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_rule_patch(
    target_rule: Rule,
    rule_set: RuleSet,
    failures: list[dict],
    correct_pool: list[dict],
    oracle: OracleDict,
    model: str,
    api_key: str,
    n: int = 2,
    task_spec=None,
) -> Optional[RulePatch]:
    """
    Generate up to *n* candidate patches in parallel; return the first
    successfully parsed one (lowest temperature wins ties).
    """
    if task_spec is not None and task_spec.build_rule_patch_prompt is not None:
        prompt = task_spec.build_rule_patch_prompt(
            target_rule, rule_set, failures, list(correct_pool), oracle
        )
    else:
        cheatsheet_context = rule_set.render_decision_guide()
        failure_lines = _format_failures(failures, oracle, task_spec=task_spec)
        correct_lines = _format_correct(correct_pool, task_spec=task_spec)
        prompt = RULE_PATCH_PROMPT.format(
            rule_id=target_rule.id,
            rule_text=target_rule.text.strip(),
            cheatsheet_context=cheatsheet_context,
            failure_lines=failure_lines,
            correct_lines=correct_lines,
        )

    rule_id_regex = task_spec.rule_id_regex if task_spec is not None else None
    temps = PATCH_TEMPS[:n]

    with ThreadPoolExecutor(max_workers=len(temps)) as pool:
        futures = {
            pool.submit(
                call_llm, prompt, model=model, api_key=api_key,
                max_tokens=PATCH_MAX_TOKENS, temperature=t,
            ): t
            for t in temps
        }
        results = []
        for future in as_completed(futures):
            temp = futures[future]
            try:
                response = future.result()
                patch = _parse_patch_response(response.content, target_rule.id,
                                              rule_id_regex=rule_id_regex)
                if patch is not None:
                    results.append((temp, patch))
            except Exception:
                pass

    if not results:
        return None
    results.sort(key=lambda x: x[0])  # prefer lowest temperature
    return results[0][1]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_failures(failures: list[dict], oracle: OracleDict, task_spec=None) -> str:
    lines = []
    for i, item in enumerate(failures[:10], 1):  # cap at 10
        if "equation1" not in item:
            # Bake gold reasoning into _oracle_exact so format_failure can show it
            if "_oracle_exact" not in item and item.get("reason"):
                item = {**item, "_oracle_exact": item["reason"]}
            # Non-magma task: use task_spec.format_failure when available
            if task_spec is not None and callable(getattr(task_spec, "format_failure", None)):
                lines.append(f"[{i}]\n{task_spec.format_failure(item)}")
            else:
                lines.append(f"[{i}] {item.get('input', '?')[:300]}")
                lines.append(f"    Expected: {item.get('answer', '?')}")
                lines.append(f"    Model predicted: {item.get('predicted', '?')}")
                reasoning = item.get("reasoning") or item.get("post_think") or ""
                if reasoning:
                    lines.append(f"    Wrong reasoning: ...{reasoning[-400:].replace(chr(10), ' ').strip()}")
            lines.append("")
            continue
        lines.append(f"[{i}] E1 = {item['equation1']}  |  E2 = {item['equation2']}")
        lines.append(f"    Expected: {'TRUE' if item.get('answer') else 'FALSE'}")
        lines.append(f"    Model predicted: {item.get('predicted', item.get('verdict', '?'))}")
        reasoning = item.get("reasoning") or item.get("post_think") or ""
        if reasoning:
            snippet = reasoning[-400:].replace("\n", " ").strip()
            lines.append(f"    Wrong reasoning: ...{snippet}")
        oracle_key = (item["equation1"], item["equation2"])
        oracle_reasoning = oracle.get(oracle_key, "")
        if oracle_reasoning:
            snippet = oracle_reasoning[:400].replace("\n", " ").strip()
            lines.append(f"    Oracle (correct): {snippet}...")
        nn = item.get("oracle_nearest")
        if nn and not oracle_reasoning:
            snippet = nn.get("reasoning", "")[:300].replace("\n", " ").strip()
            lines.append(f"    Similar oracle: {snippet}...")
        lines.append("")
    return "\n".join(lines)


def _format_correct(correct_pool: list[dict], task_spec=None) -> str:
    lines = []
    for item in list(correct_pool)[:6]:  # cap at 6
        if "equation1" not in item:
            answer = item.get("answer", "?")
            lines.append(f"  {item.get('input', '?')[:150]}  |  Answer: {answer}")
        else:
            lines.append(
                f"  E1 = {item['equation1']}  |  E2 = {item['equation2']}  "
                f"|  Answer: {'TRUE' if item.get('answer') else 'FALSE'}"
            )
    return "\n".join(lines) if lines else "  (none available)"


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

_MAGMA_RULE_ID_PATTERN = r"(TR-\w+|FR-\w+|FG-\w+|ER-\w+|M\d+\w*|SEP-\w+|PROJ-\w+|CPLEMMA\w*)"


def _parse_patch_response(
    text: str,
    target_rule_id: str,
    rule_id_regex: str | None = None,
) -> Optional[RulePatch]:
    """Parse LLM output into a RulePatch. Returns None if malformed."""
    id_pattern = rule_id_regex or _MAGMA_RULE_ID_PATTERN

    # Extract PATCH_TYPE
    m = re.search(r"PATCH_TYPE:\s*(TIGHTEN|SPLIT|REPLACE|ADD_GUARD)", text, re.IGNORECASE)
    if not m:
        return None
    patch_type = m.group(1).upper()

    # Extract NEW_RULES block
    nr_match = re.search(r"NEW_RULES:\s*\n(.*?)(?:\nREASONING:|\Z)", text, re.DOTALL)
    if not nr_match:
        return None

    new_rules: list[tuple[str, str]] = []
    for line in nr_match.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        # Match "RULE_ID: rule text..."
        rm = re.match(rf"{id_pattern}:\s*(.+)", line)
        if rm:
            new_rules.append((rm.group(1), line))
        elif new_rules:
            # Continuation of previous rule (multi-line)
            prev_id, prev_text = new_rules[-1]
            new_rules[-1] = (prev_id, prev_text + "\n" + "         " + line)

    if not new_rules:
        return None

    # Extract REASONING
    reas_match = re.search(r"REASONING:\s*(.*?)(?:\nVERIFY:|\Z)", text, re.DOTALL)
    reasoning = reas_match.group(1).strip() if reas_match else ""

    # Extract VERIFY
    ver_match = re.search(r"VERIFY:\s*(.*)", text, re.DOTALL)
    verify = ver_match.group(1).strip() if ver_match else ""

    return RulePatch(
        target_rule_id=target_rule_id,
        patch_type=patch_type,
        new_rules=new_rules,
        reasoning=reasoning,
        verify=verify,
    )
