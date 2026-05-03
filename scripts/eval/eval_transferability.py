"""
eval_transferability.py — Cross-model transferability of v3 ICR cheatsheets.

Scores the v3 cheatsheets (generated with gpt-4.1-mini) against the BBH test
sets using a diverse suite of scoring models. Produces a task × model accuracy
matrix to measure how well cheatsheet knowledge transfers across model families.

Usage:
    python3 eval_transferability.py \\
        --run-dir runs/bbh_v3 \\
        --concurrency 16 \\
        --out runs/transferability_results.json

    # Score a single model:
    python3 eval_transferability.py --models openai/gpt-4.1

    # Score all tasks with two specific models:
    python3 eval_transferability.py \\
        --models openai/gpt-4.1-mini anthropic/claude-3-5-haiku-20241022
"""

from __future__ import annotations

import argparse
import importlib
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


# ---------------------------------------------------------------------------
# Task registry (matches eval_bbh_comparison.py)
# ---------------------------------------------------------------------------

TASKS: dict[str, dict] = {
    "boolean_expressions": {
        "task_flag":  "bbh_boolean",
        "module":     "tasks.bbh_boolean",
        "attr":       "BBH_BOOLEAN_TASK",
        "test_jsonl": "datasets/bbh/boolean_expressions_test.jsonl",
        "cs_icl_dir": "bbh_boolean_expressions",
        "cs_icl_file":"gen_gpt-4.1-2025-04-14_0_1000.txt",
    },
    "causal_judgement": {
        "task_flag":  "causal_judgement",
        "module":     "tasks.causal_judgement",
        "attr":       "CAUSAL_JUDGEMENT_TASK",
        "test_jsonl": "datasets/bbh/causal_judgement_test.jsonl",
        "cs_icl_dir": "bbh_causal_judgement",
        "cs_icl_file":"gen_gpt-4.1-2025-04-14_0_1000.txt",
    },
    "date_understanding": {
        "task_flag":  "date_understanding",
        "module":     "tasks.date_understanding",
        "attr":       "DATE_UNDERSTANDING_TASK",
        "test_jsonl": "datasets/bbh/date_understanding_test.jsonl",
        "cs_icl_dir": "bbh_date_understanding",
        "cs_icl_file":"gen_gpt-4.1-mini_0_1000.txt",
    },
    "disambiguation_qa": {
        "task_flag":  "disambiguation_qa",
        "module":     "tasks.disambiguation_qa",
        "attr":       "DISAMBIGUATION_TASK",
        "test_jsonl": "datasets/bbh/disambiguation_qa_test.jsonl",
        "cs_icl_dir": "bbh_disambiguation_qa",
        "cs_icl_file":"gen_gpt-4.1-2025-04-14_0_1000.txt",
    },
    "formal_fallacies": {
        "task_flag":  "formal_fallacies",
        "module":     "tasks.formal_fallacies",
        "attr":       "FORMAL_FALLACIES_TASK",
        "test_jsonl": "datasets/bbh/formal_fallacies_test.jsonl",
        "cs_icl_dir": "bbh_formal_fallacies",
        "cs_icl_file":"gen_gpt-4.1-mini_0_1000.txt",
    },
    "geometric_shapes": {
        "task_flag":  "geometric_shapes",
        "module":     "tasks.geometric_shapes",
        "attr":       "GEOMETRIC_TASK",
        "test_jsonl": "datasets/bbh/geometric_shapes_test.jsonl",
        "cs_icl_dir": "bbh_geometric_shapes",
        "cs_icl_file":"gen_gpt-4.1-2025-04-14_0_1000.txt",
    },
    "logical_deduction_three": {
        "task_flag":  "logical_deduction_three",
        "module":     "tasks.logical_deduction",
        "attr":       "LOGICAL_DEDUCTION_3_TASK",
        "test_jsonl": "datasets/bbh/logical_deduction_three_objects_test.jsonl",
        "cs_icl_dir": "bbh_logical_deduction_three_objects",
        "cs_icl_file":"gen_gpt-4.1-mini_0_1000.txt",
    },
    "navigate": {
        "task_flag":  "navigate",
        "module":     "tasks.navigate",
        "attr":       "NAVIGATE_TASK",
        "test_jsonl": "datasets/bbh/navigate_test.jsonl",
        "cs_icl_dir": "bbh_navigate",
        "cs_icl_file":"gen_gpt-4.1-mini_0_1000.txt",
    },
    "snarks": {
        "task_flag":  "snarks",
        "module":     "tasks.snarks",
        "attr":       "SNARKS_TASK",
        "test_jsonl": "datasets/bbh/snarks_test.jsonl",
        "cs_icl_dir": "bbh_snarks",
        "cs_icl_file":"gen_gpt-4.1-mini_0_1000.txt",
    },
    "sports_understanding": {
        "task_flag":  "sports_understanding",
        "module":     "tasks.sports_understanding",
        "attr":       "SPORTS_TASK",
        "test_jsonl": "datasets/bbh/sports_understanding_test.jsonl",
        "cs_icl_dir": "bbh_sports_understanding",
        "cs_icl_file":"gen_gpt-4.1-2025-04-14_0_1000.txt",
    },
    "web_of_lies": {
        "task_flag":  "web_of_lies",
        "module":     "tasks.bbh_tasks_ext",
        "attr":       "WEB_OF_LIES_TASK",
        "test_jsonl": "datasets/bbh/web_of_lies_test.jsonl",
        "cs_icl_dir": "bbh_web_of_lies",
        "cs_icl_file":"gen_gpt-4.1-mini_0_1000.txt",
    },
}

# ---------------------------------------------------------------------------
# Model suite: producer × capability × training paradigm
# ---------------------------------------------------------------------------
# Each entry: (model_id, label for display, tier, producer, training_style)

MODEL_SUITE = [
    # OpenAI — RLHF / instruction-tuned
    ("openai/gpt-4.1-mini",                "gpt-4.1-mini",     "small",    "OpenAI",     "RLHF"),
    ("openai/gpt-4.1",                     "gpt-4.1",          "frontier", "OpenAI",     "RLHF"),
    # Anthropic — Constitutional AI
    ("anthropic/claude-3-5-haiku-20241022","claude-3.5-haiku", "small",    "Anthropic",  "ConstitutionalAI"),
    ("anthropic/claude-3-7-sonnet-20250219","claude-3.7-sonnet","frontier", "Anthropic",  "ConstitutionalAI"),
    # Google — Gemini training
    ("google/gemini-2.0-flash-001",        "gemini-2.0-flash", "small",    "Google",     "Gemini"),
    # Meta — open-source SFT + RLHF
    ("meta-llama/llama-3.3-70b-instruct",  "llama-3.3-70b",   "mid",      "Meta",       "OpenRLHF"),
]

MODEL_SUITE_MAP = {m[0]: m for m in MODEL_SUITE}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_task(module: str, attr: str):
    return getattr(importlib.import_module(module), attr)


def _load_v3_cheatsheet(run_dir: Path, task_name: str) -> str | None:
    for name in ("cheatsheet_final.txt", "cheatsheet_current.txt"):
        p = run_dir / task_name / name
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return None


def _load_cs_icl_cheatsheet(cs_icl_dir: Path, cfg: dict) -> str | None:
    if not cfg.get("cs_icl_dir") or not cfg.get("cs_icl_file"):
        return None
    p = cs_icl_dir / cfg["cs_icl_dir"] / cfg["cs_icl_file"]
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


def _score_cheatsheet(items, cs_text, task_spec, model, api_key, concurrency, label):
    # temperature=0.0 is pinned explicitly on every call to prevent OpenRouter
    # or provider defaults from introducing sampling variance across models.
    eval_fn = getattr(task_spec, "build_eval_prompt", None)
    if eval_fn is not None:
        from utils.llm_client import call_llm_batch
        prompts = [eval_fn(cs_text, item) for item in items]
        responses = call_llm_batch(
            prompts, model=model, api_key=api_key,
            temperature=0.0, max_tokens=32,
            concurrency=concurrency,
            progress_label=label,
            reasoning_effort=None,
        )

        def _check(item, resp):
            if resp is None:
                return False
            predicted = task_spec.parse_verdict(resp.content)
            if predicted is None:
                predicted = resp.content.strip()
            return task_spec.is_correct(predicted, item)

        correct = sum(_check(item, resp) for item, resp in zip(items, responses))
    else:
        correct_items, _ = score_batch(
            items, cs_text, model, api_key,
            concurrency=concurrency,
            reasoning_effort=None,
            cot_first=True,
            task_spec=task_spec,
            temperature=0.0,
        )
        correct = len(correct_items)

    acc = correct / len(items) if items else 0.0
    print(f"  [{label}] acc = {acc:.1%}", file=sys.stderr)
    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",     default="runs/bbh_v3",
                    help="Directory containing per-task v3 run outputs")
    ap.add_argument("--cs-icl-dir",  default="../cheat-sheet-icl/data/cheat_prompt",
                    help="Directory containing CS-ICL cheatsheets for comparison")
    ap.add_argument("--no-csicl",    action="store_true",
                    help="Skip CS-ICL comparison scoring")
    ap.add_argument("--models",      nargs="+",
                    default=[m[0] for m in MODEL_SUITE],
                    help="Space-separated list of model IDs to evaluate")
    ap.add_argument("--tasks",       nargs="+",
                    default=list(TASKS.keys()),
                    help="Subset of tasks to score (default: all 11)")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="Parallel scoring calls per model")
    ap.add_argument("--out",         default="runs/transferability_results.json",
                    help="Output JSON file")
    args = ap.parse_args()

    api_key = get_api_key()
    from utils.run_logger import RunLogger, make_run_id, set_logger
    _run_logger = RunLogger(log_base="runs/logs/eval", run_id=make_run_id("transferability"), config=vars(args))
    set_logger(_run_logger)
    print(f"[log] {_run_logger.log_dir}", file=sys.stderr)
    run_dir = Path(args.run_dir)

    # Load existing results so partial runs can be resumed
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float | None]] = {}
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text(encoding="utf-8"))
            print(f"Resuming from {out_path} ({len(results)} tasks cached)", file=sys.stderr)
        except Exception:
            pass

    tasks_to_run = [t for t in args.tasks if t in TASKS]
    models_to_run = args.models

    cs_icl_base = Path(args.cs_icl_dir)

    # Pre-load task specs and cheatsheets
    task_data: dict[str, dict] = {}
    for task_name in tasks_to_run:
        cfg        = TASKS[task_name]
        cs_text    = _load_v3_cheatsheet(run_dir, task_name)
        cs_icl_text = (
            None if args.no_csicl
            else _load_cs_icl_cheatsheet(cs_icl_base, cfg)
        )
        test_items = load_jsonl(Path(cfg["test_jsonl"]))
        task_spec  = _load_task(cfg["module"], cfg["attr"])
        task_data[task_name] = {
            "cfg":         cfg,
            "cs_text":     cs_text,
            "cs_icl_text": cs_icl_text,
            "test_items":  test_items,
            "task_spec":   task_spec,
        }
        if cs_text is None:
            print(f"[warn] No cheatsheet found for {task_name} in {run_dir}", file=sys.stderr)
        if cs_icl_text is None and not args.no_csicl:
            print(f"[warn] No CS-ICL cheatsheet found for {task_name}", file=sys.stderr)

    # Each task × model scores both "ours" and "cs_icl" cheatsheets.
    # temperature=0.0 is pinned for all models to prevent OpenRouter from
    # applying arbitrary sampling defaults that vary across providers.
    n_sheets    = 1 if args.no_csicl else 2
    total_cells = len(tasks_to_run) * len(models_to_run) * n_sheets
    done_cells  = 0

    def _score(items, cs_text, task_spec, model_id, label):
        return _score_cheatsheet(items, cs_text, task_spec, model_id,
                                 api_key, args.concurrency, label)

    for model_id in models_to_run:
        model_meta  = MODEL_SUITE_MAP.get(model_id, (model_id, model_id, "?", "?", "?"))
        model_label = model_meta[1]

        print(f"\n{'='*65}", file=sys.stderr)
        print(f"  Model: {model_id}  [{model_meta[3]} | {model_meta[2]} | {model_meta[4]}]",
              file=sys.stderr)
        print(f"  temperature=0.0 pinned (stable scoring)", file=sys.stderr)
        print(f"{'='*65}", file=sys.stderr)

        for task_name in tasks_to_run:
            td = task_data[task_name]
            print(f"\n  Task: {task_name}  ({len(td['test_items'])} test items)", file=sys.stderr)

            if task_name not in results:
                results[task_name] = {}

            # ── Score our cheatsheet ──────────────────────────────────────────
            ours_key = f"ours/{model_id}"
            if results[task_name].get(ours_key) is not None:
                print(f"  [ours] already scored — skipping", file=sys.stderr)
            elif td["cs_text"] is None:
                print(f"  [ours] no cheatsheet — skipping", file=sys.stderr)
            else:
                try:
                    results[task_name][ours_key] = _score(
                        td["test_items"], td["cs_text"], td["task_spec"],
                        model_id, f"{task_name}/ours/{model_label}",
                    )
                except Exception as exc:
                    print(f"  [error] ours × {model_id}: {exc}", file=sys.stderr)
                    results[task_name][ours_key] = None
            done_cells += 1

            # ── Score CS-ICL cheatsheet ───────────────────────────────────────
            if not args.no_csicl:
                csicl_key = f"cs_icl/{model_id}"
                if results[task_name].get(csicl_key) is not None:
                    print(f"  [cs_icl] already scored — skipping", file=sys.stderr)
                elif td["cs_icl_text"] is None:
                    print(f"  [cs_icl] no cheatsheet — skipping", file=sys.stderr)
                else:
                    try:
                        results[task_name][csicl_key] = _score(
                            td["test_items"], td["cs_icl_text"], td["task_spec"],
                            model_id, f"{task_name}/cs_icl/{model_label}",
                        )
                    except Exception as exc:
                        print(f"  [error] cs_icl × {model_id}: {exc}", file=sys.stderr)
                        results[task_name][csicl_key] = None
                done_cells += 1

            # Save after every task so partial results persist
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"  Progress: {done_cells}/{total_cells} cells", file=sys.stderr)

    # ── Summary table ────────────────────────────────────────────────────────
    col_w   = 14
    row_w   = 32   # task + sheet label
    separator = "-" * (row_w + len(models_to_run) * (col_w + 2))

    print("\n" + "=" * len(separator))
    print("CROSS-MODEL TRANSFERABILITY  —  accuracy on test sets")
    print("  cheatsheets generated with gpt-4.1-mini | scored with each column model")
    print("  temperature=0.0 pinned for all models")
    print("=" * len(separator))

    header = f"{'Task / Sheet':<{row_w}}"
    for model_id in models_to_run:
        label = MODEL_SUITE_MAP.get(model_id, (model_id, model_id))[1]
        header += f"  {label:>{col_w}}"
    print(header)
    print(separator)

    sheet_configs = [("ours", "ours")]
    if not args.no_csicl:
        sheet_configs.append(("cs_icl", "cs-icl"))

    for task_name in tasks_to_run:
        for sheet_key, sheet_label in sheet_configs:
            row_label = f"{task_name} [{sheet_label}]"
            row_str   = f"{row_label:<{row_w}}"
            for model_id in models_to_run:
                acc = results.get(task_name, {}).get(f"{sheet_key}/{model_id}")
                cell = f"{acc:.1%}" if acc is not None else "N/A"
                row_str += f"  {cell:>{col_w}}"
            print(row_str)
        print(separator)

    # Column averages (ours only)
    avg_str = f"{'AVERAGE (ours)':<{row_w}}"
    for model_id in models_to_run:
        accs = [
            results.get(t, {}).get(f"ours/{model_id}")
            for t in tasks_to_run
            if results.get(t, {}).get(f"ours/{model_id}") is not None
        ]
        avg = sum(accs) / len(accs) if accs else None
        avg_str += f"  {f'{avg:.1%}' if avg is not None else 'N/A':>{col_w}}"
    print(avg_str)
    print("=" * len(separator))

    # Model metadata legend
    print("\nModel legend:")
    for model_id in models_to_run:
        meta = MODEL_SUITE_MAP.get(model_id, (model_id, model_id, "?", "?", "?"))
        print(f"  {meta[1]:<20}  producer={meta[3]:<12}  tier={meta[2]:<9}  training={meta[4]}")

    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    main()
