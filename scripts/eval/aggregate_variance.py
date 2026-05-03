"""
aggregate_variance.py — Compute 3-seed means from variance eval results.

For each (seed1, seed2, seed3) group, averages full / pk_only / delta_cs per
task per model.  delta_cs is recomputed from (mean_full - mean_pk_only) rather
than averaged directly.

Usage:
    # v3: average seed1 (canonical) with variance seeds 2 & 3
    python3 scripts/eval/aggregate_variance.py \\
        --condition v3 \\
        --seed1 runs/rf_transfer_5tasks_v3.json \\
        --seed2 runs/variance/eval_results/v3_seed2_rf.json \\
        --seed3 runs/variance/eval_results/v3_seed3_rf.json \\
        --out   runs/variance/v3_3seed_mean.json

    # e3: CJ oracle-ablation condition
    python3 scripts/eval/aggregate_variance.py \\
        --condition e3 \\
        --seed1 runs/e3_no_oracle_rf.json \\
        --seed2 runs/variance/eval_results/e3_seed2_rf.json \\
        --seed3 runs/variance/eval_results/e3_seed3_rf.json \\
        --out   runs/variance/e3_3seed_mean.json

    # ea: EA Phase-1 condition
    python3 scripts/eval/aggregate_variance.py \\
        --condition ea \\
        --seed1 runs/bbh_ea_phase1_rf.json \\
        --seed2 runs/variance/eval_results/ea_seed2_rf.json \\
        --seed3 runs/variance/eval_results/ea_seed3_rf.json \\
        --out   runs/variance/ea_3seed_mean.json

    # convenience: run all three groups at once
    python3 scripts/eval/aggregate_variance.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


GROUPS = {
    "v3": {
        "seed1": "runs/rf_transfer_5tasks_v3.json",
        "seed2": "runs/variance/eval_results/v3_seed2_rf.json",
        "seed3": "runs/variance/eval_results/v3_seed3_rf.json",
        "out":   "runs/variance/v3_3seed_mean.json",
    },
    "e3": {
        "seed1": "runs/e3_no_oracle_rf.json",
        "seed2": "runs/variance/eval_results/e3_seed2_rf.json",
        "seed3": "runs/variance/eval_results/e3_seed3_rf.json",
        "out":   "runs/variance/e3_3seed_mean.json",
    },
    "ea": {
        "seed1": "runs/bbh_ea_phase1_rf.json",
        "seed2": "runs/variance/eval_results/ea_seed2_rf.json",
        "seed3": "runs/variance/eval_results/ea_seed3_rf.json",
        "out":   "runs/variance/ea_3seed_mean.json",
    },
}


def _load(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[warn] missing: {p}", file=sys.stderr)
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def aggregate(seed_files: list[str | Path]) -> dict:
    """Average full / pk_only across all seed files that contain each task/model."""
    loaded = [_load(f) for f in seed_files]

    # Collect all task names present in at least one seed
    all_tasks = set()
    for d in loaded:
        all_tasks.update(d.keys())

    result: dict[str, dict] = {}

    for task in sorted(all_tasks):
        # Which seeds have this task?
        task_seeds = [d[task] for d in loaded if task in d]
        if not task_seeds:
            continue

        # Collect all model IDs
        all_models = set()
        for ts in task_seeds:
            all_models.update(ts.keys())

        task_result: dict[str, dict] = {}
        for model_id in all_models:
            model_seeds = [ts[model_id] for ts in task_seeds if model_id in ts]
            if not model_seeds:
                continue

            n = len(model_seeds)
            label = model_seeds[0].get("label", model_id)

            # Average full
            fulls = [s["full"] for s in model_seeds if "full" in s]
            mean_full = sum(fulls) / len(fulls) if fulls else None

            # Average pk_only (may not be present in all conditions)
            pk_onlys = [s["pk_only"] for s in model_seeds if "pk_only" in s]
            mean_pk = sum(pk_onlys) / len(pk_onlys) if pk_onlys else None

            entry: dict = {"label": label, "n_seeds": n}
            if mean_full is not None:
                entry["full"] = round(mean_full, 4)
            if mean_pk is not None:
                entry["pk_only"] = round(mean_pk, 4)
            if mean_full is not None and mean_pk is not None:
                entry["delta_cs"] = round(mean_full - mean_pk, 4)

            # Carry forward cs_icl from seed1 only (it's a static reference, not per-seed)
            if "cs_icl" in model_seeds[0]:
                entry["cs_icl"] = model_seeds[0]["cs_icl"]

            task_result[model_id] = entry

        result[task] = task_result

    return result


def print_summary(result: dict, condition: str) -> None:
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"3-seed mean — {condition}", file=sys.stderr)
    print(f"  {'Task':<22} {'Model':<28} {'full':>7} {'pk_only':>8} {'Δ_cs':>7} {'n':>3}",
          file=sys.stderr)
    for task, models in sorted(result.items()):
        for model_id, r in models.items():
            full    = f"{r['full']:.1%}"    if "full"    in r else "  ---  "
            pk_only = f"{r['pk_only']:.1%}" if "pk_only" in r else "   ---  "
            delta   = f"{r['delta_cs']:+.1%}" if "delta_cs" in r else "   ---"
            n       = r.get("n_seeds", "?")
            print(f"  {task:<22} {r['label']:<28} {full:>7} {pk_only:>8} {delta:>7} {n:>3}",
                  file=sys.stderr)


def run_group(condition: str, seed1: str, seed2: str, seed3: str, out: str) -> None:
    missing = [f for f in (seed1, seed2, seed3) if not Path(f).exists()]
    if missing:
        print(f"[{condition}] Skipping — missing files: {missing}", file=sys.stderr)
        return

    result = aggregate([seed1, seed2, seed3])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_summary(result, condition)
    print(f"\n  → {out}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", default=None)
    ap.add_argument("--seed1", default=None)
    ap.add_argument("--seed2", default=None)
    ap.add_argument("--seed3", default=None)
    ap.add_argument("--out",   default=None)
    ap.add_argument("--all",   action="store_true")
    args = ap.parse_args()

    if args.all or args.condition == "all":
        for cond, cfg in GROUPS.items():
            run_group(cond, cfg["seed1"], cfg["seed2"], cfg["seed3"], cfg["out"])
        return

    if args.condition and not (args.seed1 or args.seed2 or args.seed3):
        cfg = GROUPS[args.condition]
        run_group(args.condition,
                  cfg["seed1"], cfg["seed2"], cfg["seed3"], cfg["out"])
        return

    if not all([args.seed1, args.seed2, args.seed3, args.out]):
        ap.error("Provide --all, --condition alone, or --condition + --seed1/2/3 + --out")

    result = aggregate([args.seed1, args.seed2, args.seed3])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_summary(result, args.condition or "custom")
    print(f"\n  → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
