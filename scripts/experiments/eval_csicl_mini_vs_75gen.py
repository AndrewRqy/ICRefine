"""
eval_csicl_mini_vs_75gen.py — Overnight eval: gpt-4.1-mini CS-ICL vs 75gen bootstrap

Scores two families of cheatsheets across all 8 tasks (5 BBH + 3 AGIEval),
3 seeds each, on 4 models (no Claude):

  mini_csicl   : CS_ICL_Initial_Prompt/{folder}/gen_gpt-4.1-mini_0_{seed}.txt
  75gen        : CS_ICL_Initial_Prompt/{folder}/gen_gpt-4.1-mini_75gen_{seed}.txt

Total jobs: 8 tasks × 2 variants × 3 seeds × 4 models = 192

Result keys:
  {task}|mini_csicl|s{seed}|{model}    per-seed accuracy
  {task}|75gen|s{seed}|{model}
  {task}|mini_csicl|{model}            3-seed mean
  {task}|75gen|{model}

Output:
  ICR_paper_prep/experiment_runs/csicl_mini_vs_75gen_results.json

Usage:
    python3 scripts/experiments/eval_csicl_mini_vs_75gen.py
    python3 scripts/experiments/eval_csicl_mini_vs_75gen.py --concurrency 50 --workers 8
    python3 scripts/experiments/eval_csicl_mini_vs_75gen.py --tasks causal_judgement agieval_logiqa_en
    python3 scripts/experiments/eval_csicl_mini_vs_75gen.py --variants 75gen
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
from utils.scorer import score_batch
from tasks.registry import get_task

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODELS = [
    ("openai/gpt-4.1-mini",               "gpt-4.1-mini"),
    ("openai/gpt-4.1",                    "gpt-4.1"),
    ("google/gemini-2.0-flash-001",       "gemini-2.0-flash"),
    ("meta-llama/llama-3.3-70b-instruct", "llama-3.3-70b"),
]

# ---------------------------------------------------------------------------
# Task config
# ---------------------------------------------------------------------------

CS_ICL_DIR = ROOT / "CS_ICL_Initial_Prompt"

TASKS = {
    # ── BBH ──────────────────────────────────────────────────────────────────
    "causal_judgement": {
        "test":       ROOT / "datasets/bbh/causal_judgement_test.jsonl",
        "cs_icl_dir": CS_ICL_DIR / "bbh_causal_judgement",
    },
    "disambiguation_qa": {
        "test":       ROOT / "datasets/bbh/disambiguation_qa_test.jsonl",
        "cs_icl_dir": CS_ICL_DIR / "bbh_disambiguation_qa",
    },
    "formal_fallacies": {
        "test":       ROOT / "datasets/bbh/formal_fallacies_test.jsonl",
        "cs_icl_dir": CS_ICL_DIR / "bbh_formal_fallacies",
    },
    "geometric_shapes": {
        "test":       ROOT / "datasets/bbh/geometric_shapes_test.jsonl",
        "cs_icl_dir": CS_ICL_DIR / "bbh_geometric_shapes",
    },
    "snarks": {
        "test":       ROOT / "datasets/bbh/snarks_test.jsonl",
        "cs_icl_dir": CS_ICL_DIR / "bbh_snarks",
    },
    # ── AGIEval ──────────────────────────────────────────────────────────────
    # mini_csicl results already exist for AGIEval — only 75gen needed here.
    "agieval_logiqa_en": {
        "test":       ROOT / "datasets/agieval/logiqa_en_test.jsonl",
        "cs_icl_dir": CS_ICL_DIR / "agieval_logiqa_en",
        "variants":   ["75gen"],
    },
    "agieval_lsat_ar": {
        "test":       ROOT / "datasets/agieval/lsat_ar_test.jsonl",
        "cs_icl_dir": CS_ICL_DIR / "agieval_lsat_ar",
        "variants":   ["75gen"],
    },
    "agieval_lsat_lr": {
        "test":       ROOT / "datasets/agieval/lsat_lr_test.jsonl",
        "cs_icl_dir": CS_ICL_DIR / "agieval_lsat_lr",
        "variants":   ["75gen"],
    },
}

VARIANTS = {
    "mini_csicl": "gen_gpt-4.1-mini_0_{seed}.txt",
    "75gen":      "gen_gpt-4.1-mini_75gen_{seed}.txt",
}

SEEDS = [1000, 2000, 3000]

RESULTS_FILE = ROOT / "ICR_paper_prep/experiment_runs/csicl_mini_vs_75gen_results.json"

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(items, cs_text, task_spec, model_id, api_key, concurrency, label) -> float:
    correct, wrong = score_batch(
        items=items,
        cheatsheet_text=cs_text,
        model=model_id,
        api_key=api_key,
        concurrency=concurrency,
        temperature=0.0,
        progress_label=f"  [{label}]",
        reasoning_effort=None,
        cot_first=True,
        task_spec=task_spec,
        prefer_rf=True,
    )
    n = len(correct) + len(wrong)
    acc = len(correct) / n if n else 0.0
    print(f"  [{label}]  {len(correct)}/{n} = {acc:.1%}")
    return acc


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(results: dict) -> None:
    print(f"\n{'='*90}")
    print(" CS-ICL mini vs 75gen — 3-seed means")
    print(f"{'='*90}")
    header = f"{'Task':<22} {'Variant':<12} {'gpt-4.1-mini':>14} {'gpt-4.1':>10} {'gemini':>10} {'llama':>10}"
    print(header)
    print("-" * 90)
    model_labels = [ml for _, ml in MODELS]
    for task in TASKS:
        for variant in VARIANTS:
            row = f"{task:<22} {variant:<12}"
            for ml in model_labels:
                k = f"{task}|{variant}|{ml}"
                v = results.get(k)
                row += f"  {v:.1%}" if v is not None else "        —"
            print(row)
        print()
    print(f"\nResults: {RESULTS_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Eval gpt-4.1-mini CS-ICL vs 75gen bootstrap")
    ap.add_argument("--tasks",       nargs="+", default=list(TASKS.keys()),
                    choices=list(TASKS.keys()))
    ap.add_argument("--variants",    nargs="+", default=list(VARIANTS.keys()),
                    choices=list(VARIANTS.keys()))
    ap.add_argument("--seeds",       nargs="+", type=int, default=SEEDS)
    ap.add_argument("--models",      nargs="+", default=None,
                    help="Model labels to run (default: all 4)")
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--workers",     type=int, default=8)
    args = ap.parse_args()

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    api_key = get_api_key()

    model_filter = set(args.models) if args.models else None
    active_models = [(mid, ml) for mid, ml in MODELS
                     if model_filter is None or ml in model_filter]

    results: dict = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else {}

    # Build job list
    jobs = []
    for task_name in args.tasks:
        cfg        = TASKS[task_name]
        test_items = load_jsonl(cfg["test"])
        task_spec  = get_task(task_name)
        task_variants = cfg.get("variants", args.variants)
        for variant in task_variants:
            fname_tpl = VARIANTS[variant]
            for seed in args.seeds:
                cs_path = cfg["cs_icl_dir"] / fname_tpl.format(seed=seed)
                if not cs_path.exists():
                    print(f"[skip] missing {cs_path.relative_to(ROOT)}")
                    continue
                for model_id, model_label in active_models:
                    k = f"{task_name}|{variant}|s{seed}|{model_label}"
                    if k in results:
                        print(f"[skip] {k}")
                        continue
                    jobs.append(dict(
                        key=k,
                        label=f"{task_name}/{variant}/s{seed}/{model_label}",
                        cs_path=cs_path,
                        test_items=test_items,
                        task_spec=task_spec,
                        model_id=model_id,
                        api_key=api_key,
                        concurrency=args.concurrency,
                    ))

    print(f"\n[eval] {len(jobs)} jobs to run ({args.workers} parallel workers, "
          f"concurrency={args.concurrency})\n")

    def _run(job: dict) -> tuple[str, float]:
        cs_text = job["cs_path"].read_text(encoding="utf-8").strip()
        acc = _score(job["test_items"], cs_text, job["task_spec"],
                     job["model_id"], job["api_key"], job["concurrency"], job["label"])
        return job["key"], acc

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, j): j for j in jobs}
        for fut in as_completed(futures):
            key, acc = fut.result()
            results.update(locked_save(RESULTS_FILE, {key: acc}))
            print(f"[saved] {key} = {acc:.1%}")

    # Compute 3-seed means
    print("\nComputing 3-seed means...")
    means: dict[str, float] = {}
    for task_name in args.tasks:
        for variant in VARIANTS:
            for _, model_label in active_models:
                seed_vals = [
                    results[k]
                    for seed in SEEDS
                    if (k := f"{task_name}|{variant}|s{seed}|{model_label}") in results
                ]
                if seed_vals:
                    means[f"{task_name}|{variant}|{model_label}"] = round(
                        sum(seed_vals) / len(seed_vals), 6
                    )

    results.update(locked_save(RESULTS_FILE, means))
    _print_summary(results)


if __name__ == "__main__":
    main()
