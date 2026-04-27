"""
generator.py — Generate new PK sections targeting specific failure partitions.

The generator is given:
  - The structural class description (form_e1, form_e2, polarity, depth)
  - A set of failing examples
  - A summary of existing sections (to avoid duplication)

It produces a new general rule or heuristic section in === TITLE === format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from utils.llm_client import call_llm, LLMResponse


_GEN_PROMPT = """\
You are improving a mathematical reasoning guide for equational theories of magmas.
A magma is a set with a binary operation * with no other axioms.

TARGET FAILURE CLASS
  Structural partition : {partition_label}
  E1 form              : {form_e1}
  E2 form              : {form_e2}
  Expected answer      : {polarity}
  Depth bucket         : {depth_desc}

The following {n_examples} items ALL belong to this class and are CONSISTENTLY WRONG:
{failure_examples}

EXISTING PRIOR KNOWLEDGE SECTIONS (titles only — do NOT duplicate):
{existing_titles}

YOUR TASK
Write ONE new prior knowledge section — a GENERAL RULE or HEURISTIC — for this
structural class.  This is NOT a case study.  Do not use IDENTIFY/ACTION/WHY/EXAMPLES
fields.  Instead, write in the same style as the existing sections above: a numbered
or bulleted set of decision rules with brief mathematical justification.

CORRECT FORMAT (imitate this style exactly):
  === STEP N: YOUR RULE TITLE ===

  RULE X: <statement of the rule> → TRUE or FALSE.
    WHY: <one-sentence mathematical justification>.
    Example: E1 = "..." | E2 = "..." → <answer and reason>.

  RULE X+1: <another rule if needed>
    WHY: ...

Requirements:
  • Open with === YOUR SECTION TITLE ===
  • State rules that apply to EVERY item in this structural class, not just the examples
  • Each rule must have a WHY line explaining the mathematical reason
  • Include at least one worked example inline (not as a separate block)
  • Length: 100–300 words
  • Do NOT use IDENTIFY / ACTION / DOES NOT APPLY TO headings

Write ONLY the new section text, starting with ===.\
"""


@dataclass
class GeneratedSection:
    title: str
    content: str
    partition_label: str
    raw_prompt: str = ""


_DEPTH_DESCRIPTIONS = {
    0: "depth-0 (no * operators)",
    1: "depth-1 (exactly one * operator)",
    2: "depth-2+ (two or more * operators / deeply nested)",
}


def generate_section(
    partition_label: str,
    form_e1: str,
    form_e2: str,
    polarity: str,
    depth_bucket: int,
    failures: list[dict],
    existing_titles: list[str],
    model: str,
    api_key: str,
    reasoning_effort: str | None = "low",
    max_examples: int = 8,
) -> GeneratedSection | None:
    """
    Generate a new PK section for the given failure partition.
    Returns None on API error or if the response is empty.
    """
    examples = failures[:max_examples]
    failure_text = "\n".join(
        f"  E1 = {item['equation1']}  |  E2 = {item['equation2']}  |  answer = {item.get('answer', '?')}"
        for item in examples
    )
    titles_text = "\n".join(f"  - {t}" for t in existing_titles) or "  (none yet)"
    depth_desc  = _DEPTH_DESCRIPTIONS.get(depth_bucket, f"depth-{depth_bucket}+")

    prompt = _GEN_PROMPT.format(
        partition_label=partition_label,
        form_e1=form_e1,
        form_e2=form_e2,
        polarity=polarity,
        depth_desc=depth_desc,
        n_examples=len(examples),
        failure_examples=failure_text,
        existing_titles=titles_text,
    )

    try:
        resp: LLMResponse = call_llm(
            prompt=prompt,
            model=model,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            max_tokens=1500,
        )
        content = resp.content.strip()
        if not content:
            return None

        # Extract title from === ... ===
        m = re.match(r"===\s*(.+?)\s*===", content)
        title = m.group(1).strip() if m else f"Rule for {partition_label}"

        return GeneratedSection(
            title=title,
            content=content,
            partition_label=partition_label,
            raw_prompt=prompt,
        )

    except Exception as exc:
        print(f"  [generator] Error for partition '{partition_label}': {exc}")
        return None
