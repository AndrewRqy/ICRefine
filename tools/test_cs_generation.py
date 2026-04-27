"""
tools/test_cs_generation.py — Score the mixed dataset with gpt-oss-120b,
pair failures with oracle reasoning, group by partition, and run the case
study generation prompt on the top bins so we can study the output.

Usage:
  cd ICRefine
  python3 tools/test_cs_generation.py \\
      --dataset datasets/mixed_n100_h1_h2.jsonl \\
      --oracle-csv gpt5.4_mixed_oracle.csv \\
      --cheatsheet runs/partition_neurico_v2_gpt120b/cheatsheet_current \\
      --scoring-model openai/gpt-oss-120b \\
      --cs-model openai/gpt-4o \\
      --concurrency 50 \\
      --top-bins 3 \\
      --failures-per-bin 10 \\
      --out tools/cs_generation_test.md

The script saves scored failures to --cache-file so re-runs skip scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from utils.llm_client import call_llm, get_api_key
from utils.scorer import score_batch
from utils.cheatsheet import Cheatsheet
from utils.equation_features import compute_pair_features
from ICR_partition.training.partition import item_partition_key, partition_label as _pl
from ICR_reasoning.generators.case_study import (
    _format_failures_with_reasoning,
    _render_case_studies_text,
    _parse_response,
)
from ICR_select.generators.case_study import _format_already_covered
from ICR_reasoning.prompts.templates import CASE_STUDY_WITH_REASONING_PROMPT


# ---------------------------------------------------------------------------
# Oracle loading
# ---------------------------------------------------------------------------

def load_oracle_map(oracle_csv: str) -> dict[tuple[str, str], str]:
    """
    Returns {(eq1, eq2): oracle_reasoning} for every correct gpt-5.4 response.
    Correct responses are the oracle trace — they show what the right reasoning
    looks like, giving the case-study generator a contrast signal.
    """
    omap: dict[tuple[str, str], str] = {}
    with open(oracle_csv) as f:
        for row in csv.DictReader(f):
            # Keep correct responses as positive oracle; also keep wrong ones
            # (they show failure modes at oracle-model level, still useful signal)
            key = (row["equation1"].strip(), row["equation2"].strip())
            if key not in omap:
                omap[key] = row["response"]
    return omap


# ---------------------------------------------------------------------------
# Failure enrichment
# ---------------------------------------------------------------------------

def enrich_with_oracle(
    wrong: list[dict],
    oracle_map: dict[tuple[str, str], str],
) -> list[dict]:
    """
    Attach oracle_reasoning to each failure where available.
    The ICR_reasoning generator's _format_failures_with_reasoning will use it.
    """
    enriched = []
    for item in wrong:
        key = (item["equation1"].strip(), item["equation2"].strip())
        oracle = oracle_map.get(key, "")
        enriched.append({**item, "oracle_reasoning": oracle})
    return enriched


# ---------------------------------------------------------------------------
# Partition grouping
# ---------------------------------------------------------------------------

def group_by_partition(items: list[dict]) -> dict[str, list[dict]]:
    bins: dict[str, list[dict]] = {}
    for item in items:
        try:
            label = _pl(item_partition_key(item))
        except Exception:
            label = "UNKNOWN"
        bins.setdefault(label, []).append(item)
    return bins


# ---------------------------------------------------------------------------
# Feature block renderer (compact, for the report)
# ---------------------------------------------------------------------------

def _feature_summary(item: dict) -> str:
    try:
        pf = compute_pair_features(item["equation1"], item["equation2"])
        f1, f2 = pf.e1, pf.e2

        def b(v: bool) -> str:
            return "T" if v else "F"

        sep = pf.sep_fires if pf.sep_fires != "none" else "-"
        col = pf.collapse_type if pf.collapse_type != "none" else "-"
        return (
            f"E1: sz={f1.size} v={f1.vars} imb={f1.imb} bare={b(f1.bare)} "
            f"LP={b(f1.lp)} RP={b(f1.rp)} XOR={b(f1.xor)} AB={b(f1.ab)}  |  "
            f"E2: sz={f2.size} v={f2.vars} imb={f2.imb} bare={b(f2.bare)} "
            f"LP={b(f2.lp)} RP={b(f2.rp)} XOR={b(f2.xor)} AB={b(f2.ab)}  |  "
            f"sep={sep} collapse={col}"
        )
    except Exception as e:
        return f"(features unavailable: {e})"


# ---------------------------------------------------------------------------
# Case study generation
# ---------------------------------------------------------------------------

def generate_case_study(
    failures: list[dict],
    cheatsheet: Cheatsheet,
    model: str,
    api_key: str,
    polarity: str = "",
) -> str:
    from ICR_select.generators.case_study import _format_already_covered

    failure_lines = _format_failures_with_reasoning(failures, oracle=None)
    case_studies_text = _render_case_studies_text(cheatsheet)
    already_covered = _format_already_covered(cheatsheet)

    _p = polarity.strip().upper()
    if _p == "TRUE":
        polarity_instruction = (
            "POLARITY DIRECTIVE — FALSE NEGATIVE bin (model said FALSE, correct answer is TRUE):\n"
            "Prioritize TYPE A (MISSING KNOWLEDGE). Distill the missing lemma or algebraic fact "
            "from the oracle traces — something the weaker model never considers even when following "
            "a plausible path."
        )
    elif _p == "FALSE":
        polarity_instruction = (
            "POLARITY DIRECTIVE — FALSE POSITIVE bin (model said TRUE, correct answer is FALSE):\n"
            "Prioritize TYPE B (WRONG/MISSING REASONING PATTERN). Identify the exact wrong move "
            "and the correct structural test the stronger model performs instead."
        )
    else:
        polarity_instruction = (
            "Diagnose whether these failures are TYPE A (missing algebraic knowledge) or TYPE B "
            "(wrong reasoning pattern), choosing the type that best explains the majority of cases."
        )

    prompt = CASE_STUDY_WITH_REASONING_PROMPT.format(
        roadmap=cheatsheet.roadmap.strip(),
        case_studies=case_studies_text,
        failure_lines=failure_lines,
        already_covered=already_covered,
        polarity_instruction=polarity_instruction,
    )

    try:
        resp = call_llm(prompt, model, api_key, temperature=0.3, max_tokens=1400)
        return resp.content.strip()
    except Exception as exc:
        return f"[ERROR: {exc}]"


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _format_bin_failures(failures: list[dict], n: int = 12) -> str:
    parts = []
    for i, f in enumerate(failures[:n], 1):
        oracle = f.get("oracle_reasoning", "").strip()
        think = f.get("post_think", "").strip()
        parts.append(
            f"[{i}] E1 = {f['equation1']}\n"
            f"     E2 = {f['equation2']}\n"
            f"     Expected={f.get('expected','?')}  Predicted={f.get('predicted','?')}\n"
            f"     Features: {_feature_summary(f)}\n"
            f"     Model reasoning: {textwrap.shorten(think, 300, placeholder='...')}\n"
            + (f"     Oracle:          {textwrap.shorten(oracle, 300, placeholder='...')}\n" if oracle else "")
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",         default="datasets/mixed_n100_h1_h2.jsonl")
    p.add_argument("--oracle-csv",      default="gpt5.4_mixed_oracle.csv")
    p.add_argument("--cheatsheet",      default="runs/partition_neurico_v2_gpt120b/cheatsheet_current")
    p.add_argument("--scoring-model",   default="openai/gpt-oss-120b")
    p.add_argument("--cs-model",        default="openai/gpt-4o",
                   help="Model used to generate case studies from the failure bins")
    p.add_argument("--concurrency",     type=int, default=50)
    p.add_argument("--top-bins",        type=int, default=3,
                   help="Number of largest failure bins to generate case studies for")
    p.add_argument("--failures-per-bin", type=int, default=10,
                   help="Max failures to feed into each case study generation call")
    p.add_argument("--min-bin-size",    type=int, default=4,
                   help="Skip bins smaller than this")
    p.add_argument("--cache-file",      default="tools/scored_failures_cache.jsonl",
                   help="Cache file for scored failures — skip scoring if exists")
    p.add_argument("--force-rescore",   action="store_true",
                   help="Ignore cache and re-score")
    p.add_argument("--out",             default="tools/cs_generation_test.md")
    p.add_argument("--api-key",         default=None)
    args = p.parse_args()

    api_key = args.api_key or get_api_key()

    # ---- Load cheatsheet ----
    cs_path = _ROOT / args.cheatsheet
    cheatsheet = Cheatsheet.load(cs_path)
    print(f"[cheatsheet] {cs_path} — "
          f"{len(cheatsheet.case_studies)} case studies, "
          f"roadmap={len(cheatsheet.roadmap)} chars")

    # ---- Load oracle map ----
    oracle_map = load_oracle_map(str(_ROOT / args.oracle_csv))
    print(f"[oracle] {len(oracle_map)} entries from {args.oracle_csv}")

    # ---- Score or load from cache ----
    cache_path = _ROOT / args.cache_file
    if cache_path.exists() and not args.force_rescore:
        print(f"[score] Loading from cache: {cache_path}")
        with open(cache_path) as f:
            wrong = [json.loads(l) for l in f]
        print(f"[score] {len(wrong)} cached failures")
    else:
        print(f"[score] Scoring {args.dataset} with {args.scoring_model} ...")
        with open(_ROOT / args.dataset) as f:
            all_items = [json.loads(l) for l in f]

        cs_text = cheatsheet.render()
        correct, wrong = score_batch(
            all_items, cs_text, args.scoring_model, api_key,
            concurrency=args.concurrency,
            temperature=0.0,
            reasoning_effort="low",
            progress_label="scoring",
        )
        print(f"[score] correct={len(correct)}  wrong={len(wrong)}  "
              f"accuracy={len(correct)/(len(correct)+len(wrong)):.1%}")

        cache_path.parent.mkdir(exist_ok=True)
        with open(cache_path, "w") as f:
            for item in wrong:
                f.write(json.dumps(item) + "\n")
        print(f"[score] Failures cached to {cache_path}")

    # ---- Enrich with oracle ----
    wrong = enrich_with_oracle(wrong, oracle_map)
    n_oracle = sum(1 for w in wrong if w.get("oracle_reasoning"))
    print(f"[oracle] {n_oracle}/{len(wrong)} failures have oracle reasoning")

    # ---- Group by partition ----
    bins = group_by_partition(wrong)
    sorted_bins = sorted(
        [(label, items) for label, items in bins.items() if len(items) >= args.min_bin_size],
        key=lambda x: -len(x[1]),
    )
    print(f"\n[bins] {len(sorted_bins)} bins with >= {args.min_bin_size} failures:")
    for label, items in sorted_bins:
        n_ora = sum(1 for it in items if it.get("oracle_reasoning"))
        polarity_set = set(it.get("expected", "?") for it in items)
        print(f"  {label}: {len(items)} failures  oracle={n_ora}  polarity={polarity_set}")

    # ---- Generate case studies for top bins ----
    report_parts: list[str] = [
        f"# Case Study Generation Test\n\n"
        f"**Scoring model**: {args.scoring_model}  \n"
        f"**CS model**: {args.cs_model}  \n"
        f"**Cheatsheet**: {args.cheatsheet}  \n"
        f"**Total failures**: {len(wrong)}  \n"
        f"**Oracle-paired**: {n_oracle}/{len(wrong)}  \n\n"
    ]

    top = sorted_bins[:args.top_bins]
    for rank, (label, items) in enumerate(top, 1):
        polarity_set = set(it.get("expected", "?") for it in items)
        polarity = list(polarity_set)[0] if len(polarity_set) == 1 else ""
        n_ora = sum(1 for it in items if it.get("oracle_reasoning"))

        sample = items[: args.failures_per_bin]

        print(f"\n[gen] Bin {rank}/{len(top)}: {label} ({len(items)} failures, "
              f"{n_ora} oracle, polarity={polarity!r})")

        cs_output = generate_case_study(sample, cheatsheet, args.cs_model, api_key, polarity)

        report_parts.append(
            f"{'='*80}\n"
            f"## Bin {rank}: `{label}`\n"
            f"**Failures**: {len(items)}  **Oracle-paired**: {n_ora}  "
            f"**Polarity**: {polarity or 'mixed'}  \n\n"
        )
        report_parts.append(f"### Failures fed in ({len(sample)} items)\n\n```\n")
        report_parts.append(_format_bin_failures(sample, n=args.failures_per_bin))
        report_parts.append("\n```\n\n")
        report_parts.append(f"### Generated case study\n\n```\n{cs_output}\n```\n\n")

    # ---- Write report ----
    out_path = _ROOT / args.out
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("".join(report_parts))
    print(f"\n[done] Report written to {out_path}")


if __name__ == "__main__":
    main()
