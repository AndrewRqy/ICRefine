"""
ICR_pk/pipeline.py — Prior Knowledge section rating and refinement.

Ablates each existing section of the prior knowledge to measure its
contribution to accuracy, then generates new sections for structural
failure partitions that are underserved by the current PK.

Usage
-----
    python -m ICR_pk.pipeline \\
        --dataset          path/to/dataset.jsonl \\
        --prior-knowledge  path/to/NeuriCo_cheatsheet.txt \\
        --model-score      openai/gpt-oss-120b \\
        --model-casestudy  o4-mini \\
        --max-outer-iters  3 \\
        --concurrency      100 \\
        --output-dir       runs/pk_run \\
        --pk-out           path/to/updated_pk.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from utils.data import load_jsonl
from utils.llm_client import get_api_key

from .training.loop import PKLoopConfig, run_pk_loop
from .training.section_parser import render_pk

load_dotenv(Path(__file__).parent.parent / ".env")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ICR_pk — Prior Knowledge section ablation and refinement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Data")
    g.add_argument("--dataset", required=True, metavar="FILE",
                   help="JSONL dataset of items to score (equation1, equation2, answer).")
    g.add_argument("--limit", type=int, default=None, metavar="N",
                   help="Cap to first N items.")

    g = p.add_argument_group("Prior Knowledge")
    g.add_argument("--prior-knowledge", required=True, metavar="FILE",
                   help="Prior knowledge .txt file to ablate and extend.")

    g = p.add_argument_group("Models")
    g.add_argument("--model-score",      default="openai/gpt-oss-120b", metavar="MODEL",
                   help="Scoring model (used for ablation and validation).")
    g.add_argument("--model-casestudy",  default="o4-mini",             metavar="MODEL",
                   help="Generation model (used for new section drafting).")
    g.add_argument("--reasoning-effort", default="low",
                   choices=["low", "medium", "high", "none"])

    g = p.add_argument_group("Loop")
    g.add_argument("--max-outer-iters",       type=int,   default=3,    metavar="N")
    g.add_argument("--concurrency",           type=int,   default=100,  metavar="N",
                   help="Parallel API calls for scoring.")
    g.add_argument("--ablation-sample-size",  type=int,   default=None, metavar="N",
                   help="Subsample N items per section ablation (None = all items).")
    g.add_argument("--gen-trigger-failures",  type=int,   default=5,    metavar="N",
                   help="Min failures in a partition to attempt generation.")
    g.add_argument("--acceptance-threshold",  type=float, default=0.15, metavar="F",
                   help="Min accuracy improvement on partition failures to accept a new section.")
    g.add_argument("--regression-threshold",  type=float, default=0.05, metavar="F",
                   help="Max allowed regression on correct items. 0 = skip regression check.")
    g.add_argument("--contribution-threshold",type=float, default=-0.02,metavar="F",
                   help="Sections with contribution below this are flagged HARMFUL.")
    g.add_argument("--prune-harmful",         action="store_true", default=False,
                   help="Automatically remove sections flagged as HARMFUL.")
    g.add_argument("--max-gen-attempts",      type=int,   default=2,    metavar="N",
                   help="Max candidate generation attempts per partition per iteration.")

    g = p.add_argument_group("Output")
    g.add_argument("--output-dir", default="runs/pk_run", metavar="DIR",
                   help="Directory for checkpoints and logs.")
    g.add_argument("--pk-out",     default=None, metavar="FILE",
                   help="Write final refined prior knowledge to this path.")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args    = _build_parser().parse_args()
    api_key = get_api_key()

    # Load dataset
    items = load_jsonl(Path(args.dataset))
    if args.limit:
        items = items[:args.limit]
    _log(f"\n[ICR_pk] Dataset: {len(items)} items  ({args.dataset})")

    # Load prior knowledge
    pk_path = Path(args.prior_knowledge)
    if not pk_path.exists():
        raise SystemExit(f"Error: --prior-knowledge not found: {pk_path}")
    prior_knowledge = pk_path.read_text(encoding="utf-8").strip()
    _log(f"[ICR_pk] Prior knowledge: {pk_path.name}  ({len(prior_knowledge)} chars)")

    reasoning_effort = None if args.reasoning_effort == "none" else args.reasoning_effort

    cfg = PKLoopConfig(
        model_score=args.model_score,
        model_casestudy=args.model_casestudy,
        api_key=api_key,
        concurrency=args.concurrency,
        reasoning_effort=reasoning_effort,
        max_outer_iters=args.max_outer_iters,
        ablation_sample_size=args.ablation_sample_size,
        contribution_threshold=args.contribution_threshold,
        gen_trigger_failures=args.gen_trigger_failures,
        acceptance_threshold=args.acceptance_threshold,
        regression_threshold=args.regression_threshold,
        max_gen_attempts=args.max_gen_attempts,
        prune_harmful=args.prune_harmful,
    )

    output_dir = Path(args.output_dir)
    result = run_pk_loop(items, prior_knowledge, cfg, output_dir)

    # ── Write final outputs ───────────────────────────────────────────────
    final_pk = render_pk(result.sections)
    (output_dir / "pk_final.txt").write_text(final_pk, encoding="utf-8")
    (output_dir / "update_log.json").write_text(
        json.dumps(result.update_log, indent=2), encoding="utf-8"
    )
    (output_dir / "iter_summaries.json").write_text(
        json.dumps(result.iter_summaries, indent=2), encoding="utf-8"
    )

    if args.pk_out:
        out = Path(args.pk_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(final_pk, encoding="utf-8")
        _log(f"\n[ICR_pk] Final PK written to: {out}")

    # ── Print section summary ─────────────────────────────────────────────
    _log(f"\n[ICR_pk] Completed. Active sections: "
         f"{len([s for s in result.sections if not s.pruned])}")

    if result.iter_summaries:
        last_iter = result.iter_summaries[-1]["outer_iter"]
        _log("\nFinal section ratings (last iteration):")
        for entry in result.update_log:
            if entry.get("event") == "section_rated" and entry.get("outer_iter") == last_iter:
                _log(f"  [{entry['contribution']:+.1%}  {entry['label']:7s}] "
                     f"{entry['section_title']}")

    _log(f"[ICR_pk] Run artefacts: {output_dir}/")


if __name__ == "__main__":
    main()
