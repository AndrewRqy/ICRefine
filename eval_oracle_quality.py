"""
eval_oracle_quality.py — Compare case study generation with vs without oracle.

Scores N items against the baseline DT to collect real failures (with post_think),
then generates one case study with oracle contrast and one without.  Print both so
you can judge whether the oracle-guided output is more specific.

Usage (from ICRefine/):
    # Full run — score items, generate, evaluate, save state
    python eval_oracle_quality.py \
        --dataset path/to/dataset.jsonl \
        --cheatsheet path/to/prior_knowledge.txt \
        --oracle-csv path/to/oracle.csv \
        --model-score openai/gpt-oss-120b \
        --model-casestudy openai/gpt-4o \
        --n-items 40

    # Re-evaluate only — skip scoring + generation, use saved state
    python eval_oracle_quality.py \
        --from-bin runs/oracle_eval/bin_state.json \
        --cheatsheet path/to/prior_knowledge.txt \
        --model-score openai/gpt-oss-120b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

sys.path.insert(0, str(Path(__file__).parent))

from ICR_naive.core.cheatsheet import Cheatsheet
from ICR_naive.core.data import load_jsonl
from ICR_reasoning.core.llm_client import get_api_key
from ICR_reasoning.core.oracle import load_oracle_csv
from ICR_reasoning.training.scorer import score_batch
from ICR_select.generators.case_study import generate_candidates


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--from-bin",     default=None, metavar="FILE",
                   help="Load saved bin_state.json and skip straight to evaluation.")
    p.add_argument("--cheatsheet",   required=True)
    p.add_argument("--oracle-csv",   default=None)
    p.add_argument("--dataset",      default=None)
    p.add_argument("--model-score",     default="openai/gpt-oss-120b")
    p.add_argument("--model-casestudy", default="openai/gpt-4o")
    p.add_argument("--n-items",      type=int, default=40)
    p.add_argument("--bin-size",     type=int, default=5)
    p.add_argument("--concurrency",  type=int, default=10)
    p.add_argument("--reasoning-effort", default="low")
    p.add_argument("--save-bin",     default="runs/oracle_eval/bin_state.json",
                   help="Where to save failures + generated CSes after a full run.")
    args = p.parse_args()

    api_key = get_api_key()

    cheatsheet = Cheatsheet(
        decision_tree=Path(args.cheatsheet).read_text(encoding="utf-8").strip()
    )

    # ── Load from saved state OR run full pipeline ────────────────────────────
    if args.from_bin:
        state = json.loads(Path(args.from_bin).read_text(encoding="utf-8"))
        failures   = state["failures"]
        without_cs = state["without_oracle_cs"]
        with_cs    = state["with_oracle_cs"]
        oracle     = load_oracle_csv(Path(args.oracle_csv)) if args.oracle_csv else {}
        print(f"\nLoaded {len(failures)} failures and both CSes from {args.from_bin}", file=sys.stderr)

    else:
        if not args.dataset or not args.oracle_csv:
            p.error("--dataset and --oracle-csv are required unless --from-bin is used.")

        oracle = load_oracle_csv(Path(args.oracle_csv))
        items  = load_jsonl(Path(args.dataset))[: args.n_items]

        print(f"\nScoring {len(items)} items to collect failures ...", file=sys.stderr)
        _, wrong = score_batch(
            items, cheatsheet.render(),
            args.model_score, api_key,
            concurrency=args.concurrency,
            reasoning_effort=args.reasoning_effort,
            cot_first=False,
            progress_label="baseline-score",
        )

        failures = wrong[: args.bin_size]
        if not failures:
            print("No failures found — try more items or a harder dataset.")
            return

        n_covered = sum(
            1 for it in failures
            if (it["equation1"].strip(), it["equation2"].strip()) in oracle
        )
        print(
            f"\n{len(failures)} failures collected, "
            f"{n_covered}/{len(failures)} covered by oracle.\n",
            file=sys.stderr,
        )

        print("=" * 70)
        print("GENERATING (without oracle) ...")
        print("=" * 70, file=sys.stderr)
        without = generate_candidates(
            failures, cheatsheet, args.model_casestudy, api_key,
            n=1, oracle=None,
        )

        print("=" * 70)
        print("GENERATING (with oracle) ...")
        print("=" * 70, file=sys.stderr)
        with_oracle_list = generate_candidates(
            failures, cheatsheet, args.model_casestudy, api_key,
            n=1, oracle=oracle,
        )

        without_cs = without[0]          if without          else None
        with_cs    = with_oracle_list[0] if with_oracle_list else None

        # Save state so we can re-evaluate without regenerating
        save_path = Path(args.save_bin)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps({
            "failures":         failures,
            "without_oracle_cs": without_cs,
            "with_oracle_cs":   with_cs,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved bin state → {save_path}", file=sys.stderr)

    # ── Test both case studies on the failure bin ────────────────────────────
    def _eval_cs(cs: str, label: str) -> tuple[int, int]:
        """Score failures with this CS appended. Returns (n_fixed, n_total)."""
        updated = Cheatsheet(
            decision_tree=cheatsheet.decision_tree,
            case_studies=cheatsheet.case_studies + [cs],
        )
        correct, _ = score_batch(
            failures, updated.render(),
            args.model_score, api_key,
            concurrency=args.concurrency,
            reasoning_effort=args.reasoning_effort,
            cot_first=False,
            progress_label=label,
        )
        return len(correct), len(failures)

    print("\nTesting case studies on failure bin ...", file=sys.stderr)
    without_fixed = _eval_cs(without_cs,  "test-without-oracle") if without_cs  else (0, len(failures))
    with_fixed    = _eval_cs(with_cs,     "test-with-oracle")    if with_cs     else (0, len(failures))

    # ── Print comparison ──────────────────────────────────────────────────────
    sep = "\n" + "─" * 70 + "\n"

    print("\n\n" + "=" * 70)
    print("FAILURE BIN (what both versions saw)")
    print("=" * 70)
    for i, it in enumerate(failures, 1):
        key = (it["equation1"].strip(), it["equation2"].strip())
        has_oracle = "✓ oracle" if key in oracle else "✗ no oracle"
        print(f"  [{i}] E1={it['equation1']}")
        print(f"       E2={it['equation2']}")
        print(f"       expected={'TRUE' if it['answer'] else 'FALSE'}  ({has_oracle})")
        if it.get("post_think"):
            print(f"       post_think: {it['post_think'][:120]} ...")
        print()

    print(sep)
    print("WITHOUT ORACLE")
    print(sep)
    print(without_cs if without_cs else "(generation failed)")
    n, total = without_fixed
    print(f"\n  >> Fix rate on failure bin: {n}/{total} ({n/total:.0%})")

    print(sep)
    print("WITH ORACLE")
    print(sep)
    print(with_cs if with_cs else "(generation failed)")
    n, total = with_fixed
    print(f"\n  >> Fix rate on failure bin: {n}/{total} ({n/total:.0%})")

    print(sep)
    n_wo, n_w = without_fixed[0], with_fixed[0]
    winner = "WITH ORACLE" if n_w > n_wo else ("WITHOUT ORACLE" if n_wo > n_w else "TIE")
    print(f"RESULT: {winner}  (without={n_wo}/{total}  with={n_w}/{total})")
    print()
    print("WHAT TO LOOK FOR:")
    print("  Good: IDENTIFY has precise structural conditions (e.g. 'E1 contains x*y sub-term')")
    print("  Good: DOES NOT APPLY TO covers close misses that shouldn't trigger")
    print("  Good: EXAMPLES match the actual failure equations above")
    print("  Bad:  IDENTIFY is a broad category ('when transitivity fails')")
    print("  Bad:  WHY section is vague or generic")


if __name__ == "__main__":
    main()
