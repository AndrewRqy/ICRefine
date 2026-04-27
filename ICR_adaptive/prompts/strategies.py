"""
ICR_adaptive/prompts/strategies.py

Prompt-building strategies for the adaptive generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ICR_adaptive.config import TaskConfig
from ICR_adaptive.components.failure_classifier import FailureType


class PromptStrategy(str, Enum):
    DIRECT_FIX = "direct_fix"
    ORACLE_GUIDED = "oracle_guided"
    CONTRAST = "contrast"


@dataclass
class GenerationContext:
    task_cfg: TaskConfig
    cheatsheet_text: str
    item: dict
    model_response: str
    failure_type: FailureType
    divergence_step: str
    divergence_rule: str
    oracle_trace: Optional[str] = None
    related_case: Optional[str] = None


def build_prompt(ctx: GenerationContext, strategy: PromptStrategy) -> str:
    if strategy == PromptStrategy.DIRECT_FIX:
        return _direct_fix(ctx)
    if strategy == PromptStrategy.ORACLE_GUIDED:
        return _oracle_guided(ctx)
    if strategy == PromptStrategy.CONTRAST:
        return _contrast(ctx)
    raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Shared format block injected into every strategy
# ---------------------------------------------------------------------------

_FORMAT_BLOCK = """\
REQUIRED OUTPUT FORMAT — follow exactly:
  1. Plain text only. No Markdown (#, ##, **bold**, tables), no LaTeX (\\[, \\frac).
  2. Mark every distinct reasoning phase with:
         [STEP: step_name]   — e.g. [STEP: bare_check], [STEP: motif_table], [STEP: affine_probes]
  3. Mark every rule that fires (or explicitly does not fire) with:
         [RULE: rule_name]   — e.g. [RULE: M2], [RULE: CPLEMMA], [RULE: P1]
     Every step that checks or applies a rule MUST emit at least one [RULE:] tag.
  4. End with exactly these four lines:
         REASONING: <one-line summary>
         PROOF: <if TRUE, one sentence; else leave blank>
         COUNTEREXAMPLE: <if FALSE, one sentence; else leave blank>
         VERDICT: TRUE   (or FALSE)
  5. Keep total length under 600 words.

FORMAT EXAMPLE A — TRUE case (motif rule fires):
[STEP: bare_check] Left side is lone variable x; right side has *. bare(A)=TRUE.
[STEP: motif_table] Product side: rhsVars=3, rhsTotals=122, Lx=F, Rx=T, topShape=m-m, xTop=both.
[RULE: M9] rhsTotals=122 OK  Lx=F OK  Rx=T OK  x appears 2 times OK  topShape=m-m!=v-m OK -> M9 fires.
[STEP: sanity_probes] Run P1 and P2 on B.
[RULE: P1] B: x*x = (x*x)*x. Under u*v=v+1 mod 3: lhs=x+1, rhs=x+1. Pass.
[RULE: P2] Under u*v=u+1 mod 3: A sides differ. Skip.
Neither probe refutes B.
REASONING: bare(A)=TRUE, M9 fires, sanity probes pass.
PROOF: A forces right-preserving constant; B holds.
COUNTEREXAMPLE:
VERDICT: TRUE

FORMAT EXAMPLE B — FALSE case (counterexample search also uses [RULE:]):
[STEP: parse_equations] A: x*y = x, B: x = y*(y*x).
[STEP: bare_check] A right side is lone x? No. A is left-absorption. B left side bare=TRUE.
[RULE: left_absorb_flag] A forces x*y=x for all x,y; check what this implies for B.
[STEP: probe_constant] Constant magma u*v=c: A: c=c yes. B: c=c*(c*c)=c yes. Not refuted.
[RULE: constant_pass] Constant magma trivially satisfies both; try richer structure.
[STEP: probe_left_zero] {0,1} with u*v=u: A: x*y=x yes. B: x=y*(y*x)=y*y=y; need x=y — fails for x=0, y=1.
[RULE: left_zero_refutes_B] Under left-zero, A holds but B requires x=y, which is false when x≠y.
[STEP: verify] x=0, y=1: A: 0*1=0 yes. B: 0=1*(1*0)=1*1=1 — mismatch confirmed.
[RULE: counterexample_confirmed] {0,1} with u*v=u is a valid counterexample.
REASONING: left-zero satisfies A but refutes B at x=0, y=1.
PROOF:
COUNTEREXAMPLE: {0,1}, u*v=u (left-zero), x=0, y=1: B gives 0=1*(1*0)=1 which is false.
VERDICT: FALSE
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item_display(ctx: GenerationContext) -> str:
    lines = []
    for f in ctx.task_cfg.input_fields:
        lines.append(f"  {f}: {ctx.item.get(f, '?')}")
    af = ctx.task_cfg.answer_field
    lines.append(f"  {af} (ground truth): {ctx.item.get(af, '?')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _direct_fix(ctx: GenerationContext) -> str:
    fields = ctx.task_cfg.input_fields
    eq1 = ctx.item.get(fields[0], "?") if len(fields) > 0 else "?"
    eq2 = ctx.item.get(fields[1], "?") if len(fields) > 1 else "?"
    expected = str(ctx.item.get(ctx.task_cfg.answer_field, "?")).upper()
    f0 = fields[0] if len(fields) > 0 else "input_1"
    f1 = fields[1] if len(fields) > 1 else "input_2"
    return f"""You are improving a reasoning cheatsheet. A model failed on the problem below.
Your job is to write a worked example (case study) that shows the correct execution path
so that a model following the cheatsheet will get it right next time.

DOMAIN: {ctx.task_cfg.domain_description}

PROBLEM:
{_item_display(ctx)}

CRITICAL CONSTRAINT: Your case study MUST use EXACTLY the equations above.
Do NOT substitute, invent, or use any different equations. The first step of your
case study must be:
[STEP: parse_equations]
{f0}: {eq1}
{f1}: {eq2}
Expected answer: {expected}

FAILURE TYPE: {ctx.failure_type.value}
DIVERGENCE POINT: step={ctx.divergence_step}, rule={ctx.divergence_rule}

MODEL RESPONSE (incorrect):
{ctx.model_response[:800]}

CURRENT CHEATSHEET:
{ctx.cheatsheet_text[:1200] if ctx.cheatsheet_text.strip() else "(blank — no protocol yet)"}

{f"RELATED CASE STUDY:{chr(10)}{ctx.related_case[:600]}" if ctx.related_case else ""}

{_FORMAT_BLOCK}
Write the case study now:
"""


def _oracle_guided(ctx: GenerationContext) -> str:
    oracle_section = (
        f"ORACLE TRACE (correct reasoning):\n{ctx.oracle_trace[:800]}"
        if ctx.oracle_trace
        else "ORACLE TRACE: (not available)"
    )
    return f"""You are improving a reasoning cheatsheet using a reference oracle trace.

DOMAIN: {ctx.task_cfg.domain_description}

PROBLEM:
{_item_display(ctx)}

{oracle_section}

MODEL RESPONSE (incorrect):
{ctx.model_response[:800]}

FAILURE TYPE: {ctx.failure_type.value}
DIVERGENCE POINT: step={ctx.divergence_step}, rule={ctx.divergence_rule}

CURRENT CHEATSHEET:
{ctx.cheatsheet_text[:1200] if ctx.cheatsheet_text.strip() else "(blank)"}

{_FORMAT_BLOCK}
Write the case study now, focusing on step '{ctx.divergence_step}':
"""


def _contrast(ctx: GenerationContext) -> str:
    return f"""You are improving a reasoning cheatsheet by creating a contrastive example.

DOMAIN: {ctx.task_cfg.domain_description}

PROBLEM:
{_item_display(ctx)}

INCORRECT MODEL RESPONSE:
{ctx.model_response[:800]}

FAILURE TYPE: {ctx.failure_type.value}
DIVERGENCE POINT: step={ctx.divergence_step}, rule={ctx.divergence_rule}

CURRENT CHEATSHEET:
{ctx.cheatsheet_text[:1200] if ctx.cheatsheet_text.strip() else "(blank)"}

{_FORMAT_BLOCK}
Write a contrastive case study that shows BOTH the wrong path (WRONG PATH:) and
the correct path (CORRECT PATH:), making the distinction explicit at step '{ctx.divergence_step}'.
"""
