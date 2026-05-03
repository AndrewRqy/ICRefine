"""tasks/magma.py — TaskSpec for equational implication over magmas.

Wraps all domain-specific logic that was previously hard-coded inside
utils/scorer.py, ICR_partition/training/partition.py, and the prompt templates.

Singleton usage:

    from tasks.magma import MAGMA_TASK
    result = run_partition_loop(..., task_spec=MAGMA_TASK)
"""

from __future__ import annotations

from utils.task_spec import TaskSpec
from utils.data import is_true
from utils.equation_features import compute_pair_features
from utils.parser import parse_response as _parse_response, normalize as _normalize
from ICR_naive.prompts.templates import SCORING_PROMPT, SCORING_PROMPT_COT_FIRST
from ICR_reasoning.prompts.templates import CASE_STUDY_WITH_REASONING_PROMPT
from ICR_partition.training.partition import (
    item_partition_key as _item_partition_key,
    partition_key_to_conditions as _partition_key_to_conditions,
)


# ---------------------------------------------------------------------------
# Scoring prompt
# ---------------------------------------------------------------------------

def _render_features_block(item: dict) -> str:
    """Pre-compute the structural feature vector block for one item."""
    try:
        pf = compute_pair_features(item["equation1"], item["equation2"])
        f1, f2 = pf.e1, pf.e2

        def _b(v: bool) -> str:
            return "TRUE" if v else "FALSE"

        lines = [
            "=== PRECOMPUTED FEATURES ===",
            "",
            f"  E1: size={f1.size}  vars={f1.vars}  imb={f1.imb}  bare={_b(f1.bare)}",
            f"      LP={_b(f1.lp)}  RP={_b(f1.rp)}  SET={_b(f1.set_eq)}  "
            f"XOR={_b(f1.xor)}  AB={_b(f1.ab)}",
            f"  E2: size={f2.size}  vars={f2.vars}  imb={f2.imb}  bare={_b(f2.bare)}",
            f"      LP={_b(f2.lp)}  RP={_b(f2.rp)}  SET={_b(f2.set_eq)}  "
            f"XOR={_b(f2.xor)}  AB={_b(f2.ab)}",
            "",
        ]

        if f1.bare:
            lines.append(
                f"  E1 STEP0B: Lx={_b(f1.lx)}  Rx={_b(f1.rx)}  "
                f"xTop={f1.xtop}  topShape={f1.top_shape}  "
                f"square={_b(f1.square)}  rhsVars={f1.rhs_vars}"
            )
            lines.append("")

        if pf.sep_fires != "none":
            lines.append(
                f"  SEPARATOR: {pf.sep_fires}(E1)=TRUE but {pf.sep_fires}(E2)=FALSE "
                f"→ structural invariant violated → FALSE."
            )
        else:
            lines.append("  SEPARATOR: none fire.")

        if pf.collapse_type == "left_proj":
            lines.append(
                "  COLLAPSE: E1 matches a left-projection canonical form. "
                "Evaluate E2 by replacing every product (a*b) with a. "
                "If both sides reduce to the same variable → TRUE; otherwise → FALSE."
            )
        elif pf.collapse_type == "right_proj":
            lines.append(
                "  COLLAPSE: E1 matches a right-projection canonical form. "
                "Evaluate E2 by replacing every product (a*b) with b. "
                "If both sides reduce to the same variable → TRUE; otherwise → FALSE."
            )
        else:
            lines.append("  COLLAPSE: E1 does not match any canonical collapse form.")

        lines.append("")
        return "\n".join(lines) + "\n"
    except Exception:
        return ""


def _magma_build_scoring_prompt(cheatsheet_text: str, item: dict, cot_first: bool = False) -> str:
    template = SCORING_PROMPT_COT_FIRST if cot_first else SCORING_PROMPT
    return template.format(
        cheatsheet=cheatsheet_text,
        equation1=item["equation1"],
        equation2=item["equation2"],
        features_block=_render_features_block(item),
    )


# ---------------------------------------------------------------------------
# Answer checking
# ---------------------------------------------------------------------------

def _magma_is_correct(predicted: str | None, item: dict) -> bool:
    if predicted is None:
        return False
    return (predicted == "TRUE") == is_true(item["answer"])


def _magma_answer_label(item: dict) -> str:
    return "TRUE" if is_true(item["answer"]) else "FALSE"


# ---------------------------------------------------------------------------
# Verdict / post-think parsing
# ---------------------------------------------------------------------------

def _magma_parse_verdict(content: str) -> str | None:
    return _parse_response(_normalize(content))["verdict"]


def _magma_extract_post_think(content: str) -> str:
    parsed = _parse_response(_normalize(content))
    return parsed["reasoning"] or content.strip()


# ---------------------------------------------------------------------------
# Failure display
# ---------------------------------------------------------------------------

def _magma_format_failure(item: dict) -> str:
    """Format one magma equation-pair failure for the generation prompt."""
    expected   = "TRUE" if is_true(item.get("answer", False)) else "FALSE"
    predicted  = item.get("predicted", "?")
    post_think = item.get("post_think", "").strip()

    block = (
        f"  E1 = {item.get('equation1', '?')}\n"
        f"  E2 = {item.get('equation2', '?')}\n"
        f"  expected={expected}  predicted={predicted}\n"
        f"  WRONG reasoning (model's post-think):\n"
        f"    {post_think if post_think else '(not captured)'}"
    )

    # Exact oracle reasoning (pre-baked by generate_candidates before calling format_failure)
    exact = item.get("_oracle_exact", "")
    if exact:
        block += f"\n  CORRECT reasoning (oracle — exact same pair):\n    {exact}"

    # Nearest-neighbour oracle (different pair, similar structure — set by _solve_bin)
    nearest = item.get("oracle_nearest")
    if nearest and not exact:
        nn_e1       = nearest.get("equation1", "?")
        nn_e2       = nearest.get("equation2", "?")
        nn_ans      = "TRUE" if is_true(nearest.get("answer", False)) else "FALSE"
        nn_reasoning = nearest.get("reasoning", "").strip()
        sim         = item.get("oracle_sim", 0.0)
        block += (
            f"\n  CORRECT reasoning (oracle — nearest structural match, sim={sim:.2f}):\n"
            f"  E1 = {nn_e1}  |  E2 = {nn_e2}  |  {nn_ans}\n"
            f"    {nn_reasoning if nn_reasoning else '(not captured)'}"
        )

    return block


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

MAGMA_TASK = TaskSpec(
    build_scoring_prompt=_magma_build_scoring_prompt,
    is_correct=_magma_is_correct,
    answer_label=_magma_answer_label,
    parse_verdict=_magma_parse_verdict,
    extract_post_think=_magma_extract_post_think,
    partition_key=_item_partition_key,
    partition_key_to_conditions=_partition_key_to_conditions,
    format_failure=_magma_format_failure,
    generation_prompt_template=CASE_STUDY_WITH_REASONING_PROMPT,
    task_name="magma_implication",
)
