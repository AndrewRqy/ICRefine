"""
bootstrap_ci.py — Bootstrap CIs for the headline Table 1 results.

Bootstraps over the 5 non-ceiling tasks (resampling with replacement)
to compute 95% CIs for the 5-task mean accuracy and for the delta
(v3 full minus CS-ICL) for each model.

Uses existing 3-seed mean files for v3-full/PK-only and the single-run
CS-ICL values embedded in the eval result JSONs.
Optionally accepts CS-ICL seed2/seed3 rescore files (once available)
to also bootstrap over CS-ICL evaluation variance.

Run from ICRefine root:
    python3 scripts/eval/bootstrap_ci.py
    python3 scripts/eval/bootstrap_ci.py --csicl-seeds runs/csicl_seed2_rf.json runs/csicl_seed3_rf.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

TASKS = [
    "causal_judgement",
    "geometric_shapes",
    "formal_fallacies",
    "disambiguation_qa",
    "snarks",
]

MODELS = [
    ("openai/gpt-4.1-mini",                 "mini*"),
    ("openai/gpt-4.1",                       "GPT-4.1"),
    ("anthropic/claude-3-7-sonnet-20250219", "Claude"),
    ("google/gemini-2.0-flash-001",          "Gemini"),
    ("meta-llama/llama-3.3-70b-instruct",    "Llama"),
]

# CS-ICL source priority: canonical file (RF-scored) first, then v3-seed1 embedded
# canonical has CJ/GS/DQ/FF; v3-seed1 file has CJ/GS/DQ/Snarks
# canonical_5models has FF cs_icl values; rf_transfer has Snarks cs_icl
CSICL_SOURCES = [
    "runs/bbh_v3_rf_canonical_5models.json",   # CJ, GS, DQ, FF
    "runs/rf_transfer_5tasks_v3.json",          # CJ, GS, DQ, Snarks (fallback / Snarks only)
]

N_BOOT = 10_000
CI_LEVEL = 0.95


def load(path):
    p = ROOT / path
    if not p.exists():
        return None
    return json.loads(p.read_text())


def bootstrap_mean(values, n=N_BOOT, ci=CI_LEVEL):
    """Bootstrap CI for the mean of `values` by resampling with replacement."""
    rng = random.Random(42)
    k = len(values)
    means = []
    for _ in range(n):
        sample = [values[rng.randrange(k)] for _ in range(k)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = int((1 - ci) / 2 * n)
    hi = int((1 + ci) / 2 * n)
    return statistics.mean(values), means[lo], means[hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csicl-seeds", nargs="+", default=[], metavar="JSON",
                    help="Extra CS-ICL rescore JSONs (seed2, seed3) for multi-seed CS-ICL CI")
    args = ap.parse_args()

    # v3 full/PK-only per-task: use 3-seed means file (covers all 5 tasks)
    v3_mean = load("runs/variance/v3_3seed_mean.json")

    # For per-task per-seed (delta bootstrap over both seeds and tasks)
    v3_s1 = load("runs/rf_transfer_5tasks_v3.json")
    v3_s2 = load("runs/variance/eval_results/v3_seed2_rf.json")
    v3_s3 = load("runs/variance/eval_results/v3_seed3_rf.json")

    # CS-ICL per-task: merge from multiple canonical sources
    csicl_seed1: dict = {}  # {task: {model_id: acc}}
    for src_path in CSICL_SOURCES:
        src = load(src_path)
        if src is None:
            continue
        for task, models in src.items():
            if task not in TASKS:
                continue
            for mid, vals in models.items():
                v = vals.get("cs_icl")
                if v is not None and task not in csicl_seed1:
                    csicl_seed1.setdefault(task, {})[mid] = v
                elif v is not None and mid not in csicl_seed1.get(task, {}):
                    csicl_seed1.setdefault(task, {})[mid] = v

    # Extra CS-ICL seeds (from rescore runs)
    csicl_extra = []
    for path in args.csicl_seeds:
        d = load(path)
        if d:
            csicl_extra.append(d)
    print(f"CS-ICL seeds available: 1 (canonical) + {len(csicl_extra)} extra\n")
    for task in TASKS:
        covered = list(csicl_seed1.get(task, {}).keys())
        print(f"  CS-ICL {task}: {len(covered)} models covered")

    print("=" * 70)
    print("BOOTSTRAP CIs — 5-task mean RF accuracy (95%, task-resampling)")
    print("=" * 70)

    for model_id, model_lbl in MODELS:
        print(f"\n── {model_lbl} ({model_id}) ──")

        # v3 full: per-task 3-seed means from aggregated file
        v3_full_task = []
        for task in TASKS:
            if v3_mean and task in v3_mean and model_id in v3_mean[task]:
                v = v3_mean[task][model_id].get("full")
                v3_full_task.append(v)
            else:
                v3_full_task.append(None)

        v3_full_task_clean = [v for v in v3_full_task if v is not None]
        if v3_full_task_clean:
            mn, lo, hi = bootstrap_mean(v3_full_task_clean)
            print(f"  v3 full (5-task mean):   {mn:.1%}  95% CI [{lo:.1%}, {hi:.1%}]")

        # v3 PK-only: per-task 3-seed means from aggregated file
        v3_pk_task = []
        for task in TASKS:
            if v3_mean and task in v3_mean and model_id in v3_mean[task]:
                v = v3_mean[task][model_id].get("pk_only")
                v3_pk_task.append(v)
            else:
                v3_pk_task.append(None)

        v3_pk_task_clean = [v for v in v3_pk_task if v is not None]
        if v3_pk_task_clean:
            mn, lo, hi = bootstrap_mean(v3_pk_task_clean)
            print(f"  v3 PK-only (5-task mean): {mn:.1%}  95% CI [{lo:.1%}, {hi:.1%}]")

        # CS-ICL: from canonical seed1 source, optionally averaged with extra seeds
        csicl_task = []
        for task in TASKS:
            task_vals = []
            v = csicl_seed1.get(task, {}).get(model_id)
            if v is not None:
                task_vals.append(v)
            for extra in csicl_extra:
                if task in extra and model_id in extra[task]:
                    v2 = extra[task][model_id].get("cs_icl")
                    if v2 is not None:
                        task_vals.append(v2)
            if task_vals:
                csicl_task.append(statistics.mean(task_vals))
            else:
                csicl_task.append(None)

        csicl_task_clean = [v for v in csicl_task if v is not None]
        if csicl_task_clean:
            mn, lo, hi = bootstrap_mean(csicl_task_clean)
            n_seeds = 1 + len(csicl_extra)
            print(f"  CS-ICL ({n_seeds}-seed avg, 5-task): {mn:.1%}  95% CI [{lo:.1%}, {hi:.1%}]")

        # Delta (v3 full - CS-ICL), per-task, then bootstrapped
        if len(v3_full_task) == len(csicl_task):
            delta_task = []
            for vf, vc in zip(v3_full_task, csicl_task):
                if vf is not None and vc is not None:
                    delta_task.append(vf - vc)
            if delta_task:
                mn, lo, hi = bootstrap_mean(delta_task)
                print(f"  Δ (full − CS-ICL):        {mn:+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]")
                sig = "**significant**" if lo > 0 or hi < 0 else "not significant (CI crosses 0)"
                print(f"    → {sig}")

    print("\n" + "=" * 70)
    print("Per-task breakdown (v3 full 3-seed mean vs CS-ICL)")
    print("=" * 70)
    for task in TASKS:
        print(f"\n  Task: {task}")
        for model_id, model_lbl in MODELS:
            v3_full = None
            if v3_mean and task in v3_mean and model_id in v3_mean[task]:
                v3_full = v3_mean[task][model_id].get("full")

            csicl = csicl_seed1.get(task, {}).get(model_id)

            if v3_full is not None and csicl is not None:
                delta = v3_full - csicl
                print(f"    {model_lbl:10s}: v3={v3_full:.1%}  CS-ICL={csicl:.1%}  Δ={delta:+.1%}")


if __name__ == "__main__":
    main()
