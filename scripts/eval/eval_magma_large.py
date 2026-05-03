"""
eval_magma_large.py — Test-set evaluation for the magma_large pipeline run.

Scores the cheatsheet from runs/magma_large/ on datasets/magma_large_test.jsonl
(636 items, 318T/318F, combined hard2+hard3+normal tiers) using multiple model
families. Reports full (pk+cs), pk_only, delta_cs, and cs_icl_baseline accuracy.

Usage:
    python3 scripts/eval/eval_magma_large.py
    python3 scripts/eval/eval_magma_large.py --run-dir runs/magma_large --concurrency 20
    python3 scripts/eval/eval_magma_large.py --out runs/magma_large_rf.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / "ICR_partition" / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.data import load_jsonl
from utils.llm_client import get_api_key
from utils.scorer import score_batch
from tasks.magma import MAGMA_TASK

ROOT = Path(__file__).resolve().parent.parent.parent

MODELS = [
    ("openai/gpt-4.1-mini",                  "gpt-4.1-mini [train model]"),
    ("openai/gpt-4.1",                        "gpt-4.1"),
    ("anthropic/claude-3-7-sonnet-20250219",  "claude-3.7-sonnet"),
    ("google/gemini-2.0-flash-001",           "gemini-2.0-flash"),
    ("meta-llama/llama-3.3-70b-instruct",     "llama-3.3-70b"),
]

DEFAULT_RUN_DIR  = ROOT / "runs/magma_large"
DEFAULT_TEST     = ROOT / "datasets/magma_large_test.jsonl"
DEFAULT_OUT      = ROOT / "runs/magma_large_rf.json"


def _score(items: list[dict], cs_text: str, model: str, api_key: str,
           concurrency: int, label: str) -> float:
    correct_items, _ = score_batch(
        items, cs_text, model, api_key,
        concurrency=concurrency,
        temperature=0.0,
        reasoning_effort=None,
        cot_first=True,
        task_spec=MAGMA_TASK,
    )
    acc = len(correct_items) / len(items) if items else 0.0
    print(f"  [{label}] {acc:.1%}", file=sys.stderr)
    return acc


def _score_model(model_id: str, label: str, items: list[dict],
                 full_txt: str, pk_txt: str,
                 api_key: str, concurrency: int) -> dict:
    print(f"\n[{label}]", file=sys.stderr)
    full_acc  = _score(items, full_txt,  model_id, api_key, concurrency, "full   ")
    pk_acc    = _score(items, pk_txt,    model_id, api_key, concurrency, "pk_only")
    return {
        "label":    label,
        "full":     full_acc,
        "pk_only":  pk_acc,
        "delta_cs": round(full_acc - pk_acc, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",     default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--test-file",   default=str(DEFAULT_TEST))
    ap.add_argument("--out",         default=str(DEFAULT_OUT))
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--models",      nargs="+", default=None,
                    help="Subset of model IDs to evaluate (default: all 5)")
    args = ap.parse_args()

    run_dir   = Path(args.run_dir)
    test_file = Path(args.test_file)
    out_path  = Path(args.out)
    api_key   = get_api_key()

    # Load test items
    items = load_jsonl(str(test_file))
    print(f"[data] {len(items)} test items from {test_file}", file=sys.stderr)

    # Load cheatsheets
    full_txt = (run_dir / "cheatsheet_final.txt").read_text(encoding="utf-8").strip()
    pk_txt   = (run_dir / "cheatsheet_phase1_pk_final.txt").read_text(encoding="utf-8").strip()
    print(f"[cheatsheet] full={len(full_txt)} chars  pk_only={len(pk_txt)} chars", file=sys.stderr)

    # Filter models
    models = MODELS
    if args.models:
        model_set = set(args.models)
        models = [(m, l) for m, l in MODELS if m in model_set]

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        futures = {
            pool.submit(
                _score_model,
                model_id, label, items, full_txt, pk_txt,
                api_key, args.concurrency
            ): model_id
            for model_id, label in models
        }
        for future in as_completed(futures):
            model_id = futures[future]
            try:
                results[model_id] = future.result()
            except Exception as e:
                print(f"[error] {model_id}: {e}", file=sys.stderr)

    # Print summary table
    print(f"\n{'='*65}", file=sys.stderr)
    print(f"MAGMA LARGE TEST EVAL  (N={len(items)})", file=sys.stderr)
    print(f"{'='*65}", file=sys.stderr)
    hdr = f"  {'Model':<30}  {'full':>6}  {'pk_only':>7}  {'Δ_cs':>6}"
    print(hdr, file=sys.stderr)
    print(f"  {'-'*58}", file=sys.stderr)
    for model_id, label in models:
        if model_id not in results:
            continue
        r = results[model_id]
        train_marker = " ← train model" if "train model" in r["label"] else ""
        print(
            f"  {r['label']:<30}  {r['full']*100:>5.1f}%  {r['pk_only']*100:>6.1f}%  "
            f"{r['delta_cs']*100:>+5.1f}%{train_marker}",
            file=sys.stderr,
        )
    print(f"{'='*65}\n", file=sys.stderr)

    # Save JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"magma": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Results → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
