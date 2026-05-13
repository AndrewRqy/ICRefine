"""
sfcr_e1_stability.py — SFCR v2 Experiment E1: Stability Calibration

For CJ, GS, and LogiQA, across 3 data-split seeds:
  - Score source + proxy models under fixed CS-ICL anchor with n_evals=3
    (soft probability mode) and n_evals=1 (hard binary) for comparison.
  - Log p_M(x) per model per item, Jaccard matrix, region IDs and sizes,
    and per-region denominators.

Goal: verify that V_shared / V_private / V_easy are stable enough across
seeds for rate-based gate judgment. If |V_private| < 5 or |V_easy| < 20
consistently, the count-aware gate must be used.

Fixed setup:
  Source   : gpt-4.1-mini
  Proxies  : gpt-4.1, gemini-2.0-flash, llama-3.3-70b
  Anchor   : gen_gpt-4.1-mini_0_1000.txt  (same for all seeds)
  Seeds    : 1000, 2000, 3000  (rule_gen / gate split shuffle seeds)

Output:
  ICR_paper_prep/experiment_runs/sfcr_e1_calibration.json

Usage:
    python3 scripts/experiments/sfcr_e1_stability.py
    python3 scripts/experiments/sfcr_e1_stability.py --tasks causal_judgement geometric_shapes
    python3 scripts/experiments/sfcr_e1_stability.py --seeds 1000 --n-evals 1
    python3 scripts/experiments/sfcr_e1_stability.py --concurrency 30 --workers 3
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from utils.data import load_jsonl
from utils.llm_client import get_api_key
from utils.results_io import locked_save
from tasks.registry import get_task

from ICR_sfcr.failure_regions import compute_failure_regions, _tag_ids
from ICR_sfcr.splits import make_splits

# ---------------------------------------------------------------------------
# Task config
# ---------------------------------------------------------------------------

TASKS = {
    "causal_judgement": {
        "task_name":  "causal_judgement",
        "dataset":    ROOT / "datasets/bbh/causal_judgement_train_labeled.jsonl",
        "anchor":     ROOT / "CS_ICL_Initial_Prompt/bbh_causal_judgement/gen_gpt-4.1-mini_0_1000.txt",
        "rule_gen_n": 60,
        "gate_n":     40,
    },
    "geometric_shapes": {
        "task_name":  "geometric_shapes",
        "dataset":    ROOT / "datasets/bbh/geometric_shapes_train_labeled.jsonl",
        "anchor":     ROOT / "CS_ICL_Initial_Prompt/bbh_geometric_shapes/gen_gpt-4.1-mini_0_1000.txt",
        "rule_gen_n": 90,
        "gate_n":     60,
    },
    "agieval_logiqa_en": {
        "task_name":  "agieval_logiqa_en",
        "dataset":    ROOT / "datasets/agieval/logiqa_en_train.jsonl",
        "anchor":     ROOT / "CS_ICL_Initial_Prompt/agieval_logiqa_en/gen_gpt-4.1-mini_0_1000.txt",
        "rule_gen_n": 150,
        "gate_n":     100,
    },
}

MODELS = [
    ("openai/gpt-4.1-mini",               "gpt-4.1-mini"),    # source
    ("openai/gpt-4.1",                    "gpt-4.1"),
    ("google/gemini-2.0-flash-001",       "gemini-2.0-flash"),
    ("meta-llama/llama-3.3-70b-instruct", "llama-3.3-70b"),
]
SOURCE_MODEL  = MODELS[0][0]
PROXY_MODELS  = [m for m, _ in MODELS[1:]]

RESULTS_FILE = ROOT / "ICR_paper_prep/experiment_runs/sfcr_e1_calibration.json"

# ---------------------------------------------------------------------------
# Core calibration function
# ---------------------------------------------------------------------------

def calibrate_one(
    task_key: str,
    cfg: dict,
    seed: int,
    n_evals: int,
    api_key: str,
    concurrency: int,
) -> dict:
    """
    Run failure region computation for one (task, split-seed, n_evals) triple.

    Returns a record with region sizes, Jaccard matrix, per-region IDs,
    p_M values per item, and denominator flags for gate selection.
    """
    task_spec   = get_task(cfg["task_name"])
    anchor_text = cfg["anchor"].read_text(encoding="utf-8").strip()
    all_items   = load_jsonl(cfg["dataset"])

    splits = make_splits(
        all_items,
        rule_gen_n=cfg["rule_gen_n"],
        gate_n=cfg["gate_n"],
        seed=seed,
    )
    _tag_ids(splits.rule_gen)

    label = f"{task_key}/seed{seed}/n_evals{n_evals}"
    print(f"\n[E1] {label}  ({len(splits.rule_gen)} rule_gen items)")

    regions = compute_failure_regions(
        items=splits.rule_gen,
        anchor_cheatsheet=anchor_text,
        source_model=SOURCE_MODEL,
        proxy_models=PROXY_MODELS,
        api_key=api_key,
        task_spec=task_spec,
        concurrency=concurrency,
        label=label,
        n_evals=n_evals,
        eval_temperature=0.7,
        tau_s=0.5,
        tau_p=0.5,
        tau_low=0.33,
    )

    # Gate denominator flags
    n_prv  = len(regions.V_private)
    n_easy = len(regions.V_easy)
    count_gate_triggered = (n_prv < 5 or n_easy < 20)

    # Serialize p_M per model (sfcr_id → probability)
    p_by_model_serial = {
        model.split("/")[-1]: dict(p_map)
        for model, p_map in regions.p_by_model.items()
    }

    record = {
        "task":           task_key,
        "seed":           seed,
        "n_evals":        n_evals,
        "rule_gen_n":     len(splits.rule_gen),
        "gate_n":         len(splits.gate),
        # Region sizes (denominators used by gate)
        "v_shared_size":  len(regions.V_shared),
        "v_private_size": n_prv,
        "v_easy_size":    n_easy,
        "f_s_size":       len(regions.F_s),
        "source_accuracy": round(regions.source_accuracy, 4),
        # Jaccard per proxy
        "jaccard": {
            f"{s}↔{p}": round(v, 4)
            for (s, p), v in regions.jaccard_matrix.items()
        },
        # Region item IDs (for cross-seed overlap analysis)
        "v_shared_ids":  [it["_sfcr_id"] for it in regions.V_shared],
        "v_private_ids": [it["_sfcr_id"] for it in regions.V_private],
        "v_easy_ids":    [it["_sfcr_id"] for it in regions.V_easy],
        # Soft probabilities per model per item
        "p_by_model": p_by_model_serial,
        # Gate selection flag
        "count_gate_triggered": count_gate_triggered,
        "skip_reason":    regions.skip_reason,
    }

    # Cross-seed overlap can be computed offline from IDs
    flag = " *** COUNT-GATE" if count_gate_triggered else ""
    print(
        f"[E1] {label}  "
        f"|V_shared|={len(regions.V_shared)}  "
        f"|V_private|={n_prv}  |V_easy|={n_easy}"
        + flag
    )
    return record


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(results: dict) -> None:
    print(f"\n{'='*90}")
    print(" E1 Stability Calibration Summary")
    print(f"{'='*90}")

    header = f"{'Task':<22} {'Seed':>6} {'nEv':>4} {'|F_s|':>6} {'|Vsh|':>6} {'|Vpr|':>6} {'|Vey|':>7} {'srcAcc':>7} {'CountGate':>10}"
    print(header)
    print("-" * 90)

    for key in sorted(results):
        r = results[key]
        cg = "YES ***" if r.get("count_gate_triggered") else "no"
        skip = f"  SKIP: {r['skip_reason']}" if r.get("skip_reason") else ""
        print(
            f"{r['task']:<22} {r['seed']:>6} {r['n_evals']:>4} "
            f"{r['f_s_size']:>6} {r['v_shared_size']:>6} {r['v_private_size']:>6} "
            f"{r['v_easy_size']:>7} {r['source_accuracy']:>7.1%} {cg:>10}"
            + skip
        )

    # Cross-seed V_shared overlap (Jaccard) per task × n_evals pair
    print(f"\n{'='*90}")
    print(" Cross-seed V_shared Jaccard (stability check)")
    print(f"{'='*90}")

    import itertools
    for task_key in TASKS:
        for n_evals in (1, 3):
            seed_recs = [
                results.get(f"{task_key}|seed{s}|n{n_evals}")
                for s in [1000, 2000, 3000]
                if results.get(f"{task_key}|seed{s}|n{n_evals}")
            ]
            if len(seed_recs) < 2:
                continue
            pairs = list(itertools.combinations(seed_recs, 2))
            jaccs = []
            for r1, r2 in pairs:
                s1 = set(r1["v_shared_ids"])
                s2 = set(r2["v_shared_ids"])
                if s1 | s2:
                    jaccs.append(len(s1 & s2) / len(s1 | s2))
            if jaccs:
                mean_j = sum(jaccs) / len(jaccs)
                print(
                    f"  {task_key:<22} n_evals={n_evals}  "
                    f"V_shared Jaccard across seeds: "
                    + ", ".join(f"{j:.3f}" for j in jaccs)
                    + f"  (mean={mean_j:.3f})"
                )

    print(f"\nResults saved to {RESULTS_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="SFCR E1 Stability Calibration"
    )
    ap.add_argument("--tasks",        nargs="+", default=list(TASKS.keys()),
                    choices=list(TASKS.keys()),
                    help="Tasks to calibrate (default: all 3)")
    ap.add_argument("--seeds",        nargs="+", type=int, default=[1000, 2000, 3000],
                    help="Data-split seeds (default: 1000 2000 3000)")
    ap.add_argument("--n-evals",      nargs="+", type=int, default=[1, 3],
                    help="n_evals values to run (1=hard-binary, 3=soft). Default: both")
    ap.add_argument("--concurrency",  type=int, default=30,
                    help="API concurrency per score_batch call")
    ap.add_argument("--workers",      type=int, default=3,
                    help="Parallel (task, seed, n_evals) jobs")
    args = ap.parse_args()

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    api_key = get_api_key()

    # Load existing results
    results: dict = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else {}

    # Build job list
    jobs = []
    for task_key in args.tasks:
        cfg = TASKS[task_key]
        for seed in args.seeds:
            for n_evals in args.n_evals:
                rec_key = f"{task_key}|seed{seed}|n{n_evals}"
                if rec_key in results:
                    print(f"[skip] {rec_key} already in results")
                    continue
                jobs.append((task_key, cfg, seed, n_evals, rec_key))

    print(f"\n[E1] {len(jobs)} jobs to run ({args.workers} parallel workers)\n")

    def _run(job):
        task_key, cfg, seed, n_evals, rec_key = job
        record = calibrate_one(task_key, cfg, seed, n_evals, api_key, args.concurrency)
        return rec_key, record

    # Run with thread pool — each job calls score_batch internally which manages
    # its own concurrency; outer workers parallelise across (task, seed, n_evals).
    # Keep workers low (2-3) to avoid rate-limit collisions across jobs.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, j): j for j in jobs}
        for fut in as_completed(futures):
            rec_key, record = fut.result()
            results.update(locked_save(RESULTS_FILE, {rec_key: record}))
            print(f"[E1] saved: {rec_key}")

    _print_summary(results)


if __name__ == "__main__":
    main()
