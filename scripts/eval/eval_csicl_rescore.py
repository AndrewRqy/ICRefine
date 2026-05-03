"""
eval_csicl_rescore.py — Score the static CS-ICL cheatsheet independently.

Runs RF scoring for the five non-ceiling tasks against all five evaluation
models using only the CS-ICL cheatsheet (no ICRefine pipeline output needed).
Run 3 times with different --out paths to get 3-seed CS-ICL means.

Usage (from ICRefine root):
    python3 scripts/eval/eval_csicl_rescore.py --out runs/csicl_seed2_rf.json
    python3 scripts/eval/eval_csicl_rescore.py --out runs/csicl_seed3_rf.json
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

ROOT = Path(__file__).resolve().parent.parent.parent
CSICL_ROOT = ROOT.parent / "cheat-sheet-icl" / "data" / "cheat_prompt"

TASKS = {
    "causal_judgement": {
        "module": "tasks.causal_judgement",
        "attr":   "CAUSAL_JUDGEMENT_TASK",
        "test":   "datasets/bbh/causal_judgement_test.jsonl",
        "csicl":  ("bbh_causal_judgement", "gen_gpt-4.1-2025-04-14_0_1000.txt"),
    },
    "geometric_shapes": {
        "module": "tasks.geometric_shapes",
        "attr":   "GEOMETRIC_TASK",
        "test":   "datasets/bbh/geometric_shapes_test.jsonl",
        "csicl":  ("bbh_geometric_shapes", "gen_gpt-4.1-2025-04-14_0_1000.txt"),
    },
    "formal_fallacies": {
        "module": "tasks.formal_fallacies",
        "attr":   "FORMAL_FALLACIES_TASK",
        "test":   "datasets/bbh/formal_fallacies_test.jsonl",
        "csicl":  ("bbh_formal_fallacies", "gen_gpt-4.1_0_1000.txt"),
    },
    "disambiguation_qa": {
        "module": "tasks.disambiguation_qa",
        "attr":   "DISAMBIGUATION_TASK",
        "test":   "datasets/bbh/disambiguation_qa_test.jsonl",
        "csicl":  ("bbh_disambiguation_qa", "gen_gpt-4.1-2025-04-14_0_1000.txt"),
    },
    "snarks": {
        "module": "tasks.snarks",
        "attr":   "SNARKS_TASK",
        "test":   "datasets/bbh/snarks_test.jsonl",
        "csicl":  ("bbh_snarks", "gen_gpt-4.1_0_1000.txt"),
    },
}

MODELS = [
    ("openai/gpt-4.1-mini",                 "gpt-4.1-mini [train model]"),
    ("openai/gpt-4.1",                      "gpt-4.1"),
    ("anthropic/claude-3-7-sonnet-20250219","claude-3.7-sonnet"),
    ("google/gemini-2.0-flash-001",         "gemini-2.0-flash"),
    ("meta-llama/llama-3.3-70b-instruct",   "llama-3.3-70b"),
]


def score_csicl(task_name, cfg, model_id, model_label, api_key, concurrency):
    import importlib
    import copy
    mod = importlib.import_module(cfg["module"])
    task_spec = copy.copy(getattr(mod, cfg["attr"]))

    # Apply the same RF-prompt patch as eval_cs_ablation.py --reasoning-first:
    # replace scoring prompt and parser with the RF variants, clear build_eval_prompt.
    rf_builder = getattr(mod, f"_{task_name}_scoring_prompt_rf", None)
    rf_parser  = getattr(mod, f"_parse_{task_name}_rf", None)
    if rf_builder is not None and rf_parser is not None:
        task_spec.build_scoring_prompt = rf_builder
        task_spec.parse_verdict        = rf_parser
        task_spec.build_eval_prompt    = None

    cs_path = CSICL_ROOT / cfg["csicl"][0] / cfg["csicl"][1]
    if not cs_path.exists():
        print(f"  [SKIP] CS-ICL cheatsheet not found: {cs_path}", file=sys.stderr)
        return None

    cs_text = cs_path.read_text(encoding="utf-8")
    items = load_jsonl(ROOT / cfg["test"])

    label = f"{task_name}/{model_label}"
    correct_items, _ = score_batch(
        items, cs_text, model_id, api_key,
        concurrency=concurrency,
        temperature=0.0,
        reasoning_effort=None,
        cot_first=True,
        task_spec=task_spec,
    )
    acc = len(correct_items) / len(items) if items else 0.0
    print(f"  [{label}] {acc:.1%}", file=sys.stderr)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",         default="runs/csicl_rescore.json")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--tasks",       nargs="+", default=list(TASKS))
    ap.add_argument("--models",      nargs="+", default=None)
    args = ap.parse_args()

    api_key = get_api_key()
    tasks_to_run = {k: v for k, v in TASKS.items() if k in args.tasks}
    models_to_run = [(mid, mlbl) for mid, mlbl in MODELS
                     if args.models is None or mid in args.models]

    results: dict = {}
    futures = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for task_name, cfg in tasks_to_run.items():
            for model_id, model_label in models_to_run:
                fut = ex.submit(
                    score_csicl, task_name, cfg, model_id, model_label,
                    api_key, args.concurrency,
                )
                futures[fut] = (task_name, model_id, model_label)

        for fut in as_completed(futures):
            task_name, model_id, model_label = futures[fut]
            acc = fut.result()
            if acc is not None:
                results.setdefault(task_name, {})[model_id] = {
                    "label": model_label,
                    "cs_icl": acc,
                }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved → {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
