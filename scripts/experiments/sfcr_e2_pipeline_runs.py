"""
sfcr_e2_pipeline_runs.py — SFCR v2 Experiment E2: Parallel Pipeline Runs

Launches 7 SFCR pipeline jobs in parallel, one per (task, seed) configuration
derived from E1 stability calibration:

  CJ       seeds 1000/2000/3000  — count-aware gate, subtypes, repair_attempts=1
  GS       seed 1000 only        — count-aware gate, no subtypes, repair_attempts=0
  LogiQA   seeds 1000/2000/3000  — U_LCB gate (|V_private|≥5, |V_easy|≥20 expected),
                                   subtypes, repair_attempts=1

Each job writes its output to:
  ICR_paper_prep/sfcr_e2/{task}_seed{seed}/

Log files per job:
  ICR_paper_prep/sfcr_e2/logs/{task}_seed{seed}.log

Usage:
    python3 scripts/experiments/sfcr_e2_pipeline_runs.py
    python3 scripts/experiments/sfcr_e2_pipeline_runs.py --tasks causal_judgement agieval_logiqa_en
    python3 scripts/experiments/sfcr_e2_pipeline_runs.py --seeds 1000
    python3 scripts/experiments/sfcr_e2_pipeline_runs.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Job configurations (derived from E1 calibration)
# ---------------------------------------------------------------------------

PROXY_MODELS = ",".join([
    "openai/gpt-4.1",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.3-70b-instruct",
])

JOBS = [
    # ── Causal Judgement ────────────────────────────────────────────────────
    # V_private=0-1 everywhere → count-aware gate fires automatically.
    # Cross-seed V_shared Jaccard mean=0.36 → run all 3 seeds.
    # Subtypes viable (V_shared=20-27). Repair helps narrow overfitting.
    {
        "task":         "causal_judgement",
        "seed":         1000,
        "dataset":      "datasets/bbh/causal_judgement_train_labeled.jsonl",
        "anchor":       "CS_ICL_Initial_Prompt/bbh_causal_judgement/gen_gpt-4.1-mini_0_1000.txt",
        "rule_gen_n":   60,
        "gate_n":       40,
        "n_candidates": 9,
        "candidates_per_subtype": 3,
        "repair_attempts": 1,
        "use_subtypes": True,
    },
    {
        "task":         "causal_judgement",
        "seed":         2000,
        "dataset":      "datasets/bbh/causal_judgement_train_labeled.jsonl",
        "anchor":       "CS_ICL_Initial_Prompt/bbh_causal_judgement/gen_gpt-4.1-mini_0_1000.txt",
        "rule_gen_n":   60,
        "gate_n":       40,
        "n_candidates": 9,
        "candidates_per_subtype": 3,
        "repair_attempts": 1,
        "use_subtypes": True,
    },
    {
        "task":         "causal_judgement",
        "seed":         3000,
        "dataset":      "datasets/bbh/causal_judgement_train_labeled.jsonl",
        "anchor":       "CS_ICL_Initial_Prompt/bbh_causal_judgement/gen_gpt-4.1-mini_0_1000.txt",
        "rule_gen_n":   60,
        "gate_n":       40,
        "n_candidates": 9,
        "candidates_per_subtype": 3,
        "repair_attempts": 1,
        "use_subtypes": True,
    },
    # ── Geometric Shapes ────────────────────────────────────────────────────
    # Source accuracy 84-90% (near ceiling), V_shared=8-12, gpt-4.1 Jaccard
    # only 0.15 (misaligned proxy). Single seed only; no subtypes (pool too
    # small to cluster meaningfully). No repair (low V_shared count).
    {
        "task":         "geometric_shapes",
        "seed":         1000,
        "dataset":      "datasets/bbh/geometric_shapes_train_labeled.jsonl",
        "anchor":       "CS_ICL_Initial_Prompt/bbh_geometric_shapes/gen_gpt-4.1-mini_0_1000.txt",
        "rule_gen_n":   90,
        "gate_n":       60,
        "n_candidates": 6,
        "candidates_per_subtype": 2,
        "repair_attempts": 0,
        "use_subtypes": False,
    },
    # ── LogiQA ──────────────────────────────────────────────────────────────
    # Only task where U_LCB gate is valid (|V_private|≥5, |V_easy|≥56).
    # Cross-seed Jaccard mean=0.25 (lower stability) → run all 3 seeds and
    # aggregate. Subtypes viable (V_shared=49-58). Repair enabled.
    {
        "task":         "agieval_logiqa_en",
        "seed":         1000,
        "dataset":      "datasets/agieval/logiqa_en_train.jsonl",
        "anchor":       "CS_ICL_Initial_Prompt/agieval_logiqa_en/gen_gpt-4.1-mini_0_1000.txt",
        "rule_gen_n":   150,
        "gate_n":       100,
        "n_candidates": 9,
        "candidates_per_subtype": 3,
        "repair_attempts": 1,
        "use_subtypes": True,
    },
    {
        "task":         "agieval_logiqa_en",
        "seed":         2000,
        "dataset":      "datasets/agieval/logiqa_en_train.jsonl",
        "anchor":       "CS_ICL_Initial_Prompt/agieval_logiqa_en/gen_gpt-4.1-mini_0_1000.txt",
        "rule_gen_n":   150,
        "gate_n":       100,
        "n_candidates": 9,
        "candidates_per_subtype": 3,
        "repair_attempts": 1,
        "use_subtypes": True,
    },
    {
        "task":         "agieval_logiqa_en",
        "seed":         3000,
        "dataset":      "datasets/agieval/logiqa_en_train.jsonl",
        "anchor":       "CS_ICL_Initial_Prompt/agieval_logiqa_en/gen_gpt-4.1-mini_0_1000.txt",
        "rule_gen_n":   150,
        "gate_n":       100,
        "n_candidates": 9,
        "candidates_per_subtype": 3,
        "repair_attempts": 1,
        "use_subtypes": True,
    },
]


# ---------------------------------------------------------------------------
# Build subprocess command
# ---------------------------------------------------------------------------

def _build_cmd(job: dict, concurrency: int) -> list[str]:
    task = job["task"]
    seed = job["seed"]
    output_dir = ROOT / f"ICR_paper_prep/sfcr_e2/{task}_seed{seed}"

    cmd = [
        sys.executable, "-m", "ICR_sfcr.pipeline",
        "--task",               task,
        "--dataset",            str(ROOT / job["dataset"]),
        "--anchor-cheatsheet",  str(ROOT / job["anchor"]),
        "--output-dir",         str(output_dir),
        "--model-source",       "openai/gpt-4.1-mini",
        "--models-proxy",       PROXY_MODELS,
        "--oracle-mode",        "label_only",
        "--routing-mode",       "routed",
        "--seed",               str(seed),
        "--rule-gen-n",         str(job["rule_gen_n"]),
        "--gate-n",             str(job["gate_n"]),
        "--n-candidates",       str(job["n_candidates"]),
        "--candidates-per-subtype", str(job["candidates_per_subtype"]),
        "--repair-attempts",    str(job["repair_attempts"]),
        "--n-eval-seeds",       "1",
        "--concurrency",        str(concurrency),
    ]
    if not job["use_subtypes"]:
        cmd.append("--no-subtypes")
    return cmd


# ---------------------------------------------------------------------------
# Run one job
# ---------------------------------------------------------------------------

def _run_job(job: dict, concurrency: int, log_dir: Path, dry_run: bool) -> dict:
    task = job["task"]
    seed = job["seed"]
    label = f"{task}_seed{seed}"
    log_path = log_dir / f"{label}.log"

    cmd = _build_cmd(job, concurrency)

    if dry_run:
        print(f"[DRY-RUN] {label}")
        print("  " + " ".join(cmd))
        return {"label": label, "status": "dry-run", "returncode": 0, "log": str(log_path)}

    print(f"[E2] START  {label}  (log → {log_path.relative_to(ROOT)})")
    t0 = time.monotonic()

    with log_path.open("w") as fh:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )

    elapsed = time.monotonic() - t0
    status = "OK" if result.returncode == 0 else f"FAILED(rc={result.returncode})"
    print(f"[E2] {status:<16} {label}  ({elapsed/60:.1f} min)")

    return {
        "label":      label,
        "status":     status,
        "returncode": result.returncode,
        "log":        str(log_path),
        "elapsed_s":  round(elapsed),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(outcomes: list[dict]) -> None:
    print(f"\n{'='*70}")
    print(" E2 Pipeline Run Summary")
    print(f"{'='*70}")
    print(f"  {'Job':<35} {'Status':<20} {'Time':>8}")
    print("-" * 70)
    for o in outcomes:
        mins = f"{o.get('elapsed_s', 0) // 60}m{o.get('elapsed_s', 0) % 60:02d}s"
        print(f"  {o['label']:<35} {o['status']:<20} {mins:>8}")

    failed = [o for o in outcomes if o["returncode"] != 0]
    if failed:
        print(f"\n  *** {len(failed)} job(s) FAILED ***")
        for o in failed:
            print(f"    {o['label']}  →  {o['log']}")
    else:
        print(f"\n  All {len(outcomes)} jobs completed successfully.")

    print(f"\n  Output root: ICR_paper_prep/sfcr_e2/")
    print(f"  Logs:        ICR_paper_prep/sfcr_e2/logs/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="SFCR E2 parallel pipeline runs")
    ap.add_argument("--tasks",       nargs="+",
                    default=["causal_judgement", "geometric_shapes", "agieval_logiqa_en"],
                    help="Tasks to run (default: all three)")
    ap.add_argument("--seeds",       nargs="+", type=int, default=[1000, 2000, 3000],
                    help="Seeds to run (default: 1000 2000 3000)")
    ap.add_argument("--concurrency", type=int, default=20,
                    help="API concurrency per score_batch call (default: 20)")
    ap.add_argument("--workers",     type=int, default=7,
                    help="Parallel job workers (default: 7 = all jobs)")
    ap.add_argument("--dry-run",     action="store_true",
                    help="Print commands without executing")
    args = ap.parse_args()

    # Filter jobs
    jobs = [
        j for j in JOBS
        if j["task"] in args.tasks and j["seed"] in args.seeds
    ]
    if not jobs:
        print("No jobs match the given --tasks / --seeds filter.")
        sys.exit(1)

    log_dir = ROOT / "ICR_paper_prep/sfcr_e2/logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[E2] {len(jobs)} job(s) to run  ({args.workers} parallel workers)")
    print(f"     Concurrency per job: {args.concurrency}")
    if args.dry_run:
        print("     *** DRY RUN — no API calls will be made ***\n")

    outcomes: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_job, j, args.concurrency, log_dir, args.dry_run): j
            for j in jobs
        }
        for fut in as_completed(futures):
            outcomes.append(fut.result())

    _print_summary(outcomes)

    # Exit non-zero if any job failed
    if any(o["returncode"] != 0 for o in outcomes):
        sys.exit(1)


if __name__ == "__main__":
    main()
