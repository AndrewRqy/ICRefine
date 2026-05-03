"""
eval_cot_mode.py — Score all v3 cheatsheets (ours + CS-ICL) using the
reasoning scorer (score_batch with cot_first=True), bypassing build_eval_prompt.

Hypothesis: CoT/reasoning mode makes models follow cheatsheet guidance more
closely, potentially increasing accuracy deltas between ours and CS-ICL.

Outputs a comparison table:
  task | ours_cot | csicl_cot | delta_cot | ours_verdict | csicl_verdict | delta_verdict

Usage:
    python3 eval_cot_mode.py --concurrency 20 --out runs/cot_mode_results.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / "ICR_partition" / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.data import load_jsonl
from utils.llm_client import get_api_key
from utils.scorer import score_batch

# Official verdict-only results from runs/bbh_v3/comparison_results.json
VERDICT_ONLY_RESULTS = {
    "boolean_expressions":     {"ours": 0.870, "cs_icl": 0.880},
    "causal_judgement":        {"ours": 0.655, "cs_icl": 0.736},
    "date_understanding":      {"ours": 0.720, "cs_icl": 0.720},
    "disambiguation_qa":       {"ours": 0.770, "cs_icl": 0.750},
    "formal_fallacies":        {"ours": 0.700, "cs_icl": 0.640},
    "geometric_shapes":        {"ours": 0.480, "cs_icl": 0.510},
    "logical_deduction_three": {"ours": 0.950, "cs_icl": 0.950},
    "navigate":                {"ours": 0.800, "cs_icl": 0.770},
    "snarks":                  {"ours": 0.845, "cs_icl": 0.887},
    "sports_understanding":    {"ours": 0.940, "cs_icl": 0.930},
    "web_of_lies":             {"ours": 0.660, "cs_icl": 0.580},
}

TASKS = {
    "boolean_expressions": {
        "module":     "tasks.bbh_boolean",
        "attr":       "BBH_BOOLEAN_TASK",
        "test_jsonl": "datasets/bbh/boolean_expressions_test.jsonl",
        "cs_icl_dir": "bbh_boolean_expressions",
        "cs_icl_file":"gen_gpt-4.1-2025-04-14_0_1000.txt",
    },
    "causal_judgement": {
        "module":     "tasks.bbh_tasks",
        "attr":       "CAUSAL_JUDGEMENT_TASK",
        "test_jsonl": "datasets/bbh/causal_judgement_test.jsonl",
        "cs_icl_dir": "bbh_causal_judgement",
        "cs_icl_file":"gen_gpt-4.1-2025-04-14_0_1000.txt",
    },
    "date_understanding": {
        "module":     "tasks.bbh_tasks_ext",
        "attr":       "DATE_UNDERSTANDING_TASK",
        "test_jsonl": "datasets/bbh/date_understanding_test.jsonl",
        "cs_icl_dir": "bbh_date_understanding",
        "cs_icl_file":"gen_gpt-4.1-mini_0_1000.txt",
    },
    "disambiguation_qa": {
        "module":     "tasks.bbh_tasks",
        "attr":       "DISAMBIGUATION_TASK",
        "test_jsonl": "datasets/bbh/disambiguation_qa_test.jsonl",
        "cs_icl_dir": "bbh_disambiguation_qa",
        "cs_icl_file":"gen_gpt-4.1-2025-04-14_0_1000.txt",
    },
    "formal_fallacies": {
        "module":     "tasks.bbh_tasks_ext",
        "attr":       "FORMAL_FALLACIES_TASK",
        "test_jsonl": "datasets/bbh/formal_fallacies_test.jsonl",
        "cs_icl_dir": "bbh_formal_fallacies",
        "cs_icl_file":"gen_gpt-4.1-mini_0_1000.txt",
    },
    "geometric_shapes": {
        "module":     "tasks.bbh_tasks",
        "attr":       "GEOMETRIC_TASK",
        "test_jsonl": "datasets/bbh/geometric_shapes_test.jsonl",
        "cs_icl_dir": "bbh_geometric_shapes",
        "cs_icl_file":"gen_gpt-4.1-2025-04-14_0_1000.txt",
    },
    "logical_deduction_three": {
        "module":     "tasks.bbh_tasks_ext",
        "attr":       "LOGICAL_DEDUCTION_3_TASK",
        "test_jsonl": "datasets/bbh/logical_deduction_three_objects_test.jsonl",
        "cs_icl_dir": "bbh_logical_deduction_three_objects",
        "cs_icl_file":"gen_gpt-4.1-mini_0_1000.txt",
    },
    "navigate": {
        "module":     "tasks.bbh_tasks_ext",
        "attr":       "NAVIGATE_TASK",
        "test_jsonl": "datasets/bbh/navigate_test.jsonl",
        "cs_icl_dir": "bbh_navigate",
        "cs_icl_file":"gen_gpt-4.1-mini_0_1000.txt",
    },
    "snarks": {
        "module":     "tasks.bbh_tasks_ext",
        "attr":       "SNARKS_TASK",
        "test_jsonl": "datasets/bbh/snarks_test.jsonl",
        "cs_icl_dir": "bbh_snarks",
        "cs_icl_file":"gen_gpt-4.1-mini_0_1000.txt",
    },
    "sports_understanding": {
        "module":     "tasks.bbh_tasks",
        "attr":       "SPORTS_TASK",
        "test_jsonl": "datasets/bbh/sports_understanding_test.jsonl",
        "cs_icl_dir": "bbh_sports_understanding",
        "cs_icl_file":"gen_gpt-4.1-2025-04-14_0_1000.txt",
    },
    "web_of_lies": {
        "module":     "tasks.bbh_tasks_ext",
        "attr":       "WEB_OF_LIES_TASK",
        "test_jsonl": "datasets/bbh/web_of_lies_test.jsonl",
        "cs_icl_dir": "bbh_web_of_lies",
        "cs_icl_file":"gen_gpt-4.1-mini_0_1000.txt",
    },
}


def _score_cot(items, cs_text, task_spec, model, api_key, concurrency, label) -> float:
    correct_items, _ = score_batch(
        items, cs_text, model, api_key,
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
    ap.add_argument("--tasks",       nargs="+", default=list(TASKS))
    ap.add_argument("--model",       default="openai/gpt-4.1-mini")
    ap.add_argument("--concurrency", type=int,  default=20)
    ap.add_argument("--cs-icl-dir",  default="../cheat-sheet-icl/data/cheat_prompt")
    ap.add_argument("--run-dir",     default="runs/bbh_v3")
    ap.add_argument("--out",         default="runs/cot_mode_results.json")
    ap.add_argument("--no-csicl",    action="store_true")
    ap.add_argument("--no-ours",     action="store_true")
    args = ap.parse_args()

    api_key     = get_api_key()
    from utils.run_logger import RunLogger, make_run_id, set_logger
    _run_logger = RunLogger(log_base="runs/logs/eval", run_id=make_run_id("cot_mode"), config=vars(args))
    set_logger(_run_logger)
    print(f"[log] {_run_logger.log_dir}", file=sys.stderr)
    cs_icl_base = Path(args.cs_icl_dir)
    run_dir     = Path(args.run_dir)
    results: dict[str, dict] = {}

    for task_name in args.tasks:
        if task_name not in TASKS:
            print(f"Unknown task: {task_name}", file=sys.stderr)
            continue

        cfg       = TASKS[task_name]
        task_spec = getattr(importlib.import_module(cfg["module"]), cfg["attr"])
        items     = load_jsonl(Path(cfg["test_jsonl"]))

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Task: {task_name}  ({len(items)} test items)", file=sys.stderr)

        row: dict = {
            "n_test":       len(items),
            "ours_cot":     None,
            "csicl_cot":    None,
            "ours_verdict": VERDICT_ONLY_RESULTS.get(task_name, {}).get("ours"),
            "csicl_verdict":VERDICT_ONLY_RESULTS.get(task_name, {}).get("cs_icl"),
        }

        # CS-ICL CoT
        if not args.no_csicl:
            cs_path = cs_icl_base / cfg["cs_icl_dir"] / cfg["cs_icl_file"]
            if cs_path.exists():
                cs_text = cs_path.read_text(encoding="utf-8").strip()
                row["csicl_cot"] = _score_cot(
                    items, cs_text, task_spec, args.model, api_key,
                    args.concurrency, f"{task_name}/csicl_cot",
                )
            else:
                print(f"  [warn] CS-ICL cheatsheet not found: {cs_path}", file=sys.stderr)

        # Ours CoT
        if not args.no_ours:
            our_path = run_dir / task_name / "cheatsheet_final.txt"
            if not our_path.exists():
                our_path = run_dir / task_name / "cheatsheet_current.txt"
            if our_path.exists():
                our_text = our_path.read_text(encoding="utf-8").strip()
                row["ours_cot"] = _score_cot(
                    items, our_text, task_spec, args.model, api_key,
                    args.concurrency, f"{task_name}/ours_cot",
                )
            else:
                print(f"  [warn] Ours cheatsheet not found in {run_dir / task_name}", file=sys.stderr)

        results[task_name] = row
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Summary table
    print(f"\n{'='*90}", file=sys.stderr)
    print(f"{'Task':<28} {'ours_cot':>9} {'csicl_cot':>10} {'Δ_cot':>7} | {'ours_vrd':>9} {'csicl_vrd':>10} {'Δ_vrd':>7}", file=sys.stderr)
    print(f"{'-'*90}", file=sys.stderr)
    for task_name, r in results.items():
        oc = r.get("ours_cot")
        cc = r.get("csicl_cot")
        ov = r.get("ours_verdict")
        cv = r.get("csicl_verdict")
        d_cot = (oc - cc) if (oc is not None and cc is not None) else None
        d_vrd = (ov - cv) if (ov is not None and cv is not None) else None
        def _fmt(v): return f"{v:.1%}" if v is not None else "   N/A"
        def _fmtd(v): return f"{v:+.1%}" if v is not None else "   N/A"
        print(
            f"  {task_name:<26} {_fmt(oc):>9} {_fmt(cc):>10} {_fmtd(d_cot):>7} | "
            f"{_fmt(ov):>9} {_fmt(cv):>10} {_fmtd(d_vrd):>7}",
            file=sys.stderr,
        )

    print(f"\nFull results → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
