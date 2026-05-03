#!/usr/bin/env python3
"""
strip_cs_to_plain_examples.py — E-NEW3 format ablation prep.

Reads cheatsheet_final.json, strips the ACTIVATE IF / IDENTIFY conditional
structure from each case study (keeping only the support_examples as plain
Q→A pairs), and writes a new cheatsheet_final.txt in an output dir suitable
for eval_cs_ablation.py --run-dir-overrides.

This isolates the effect of the ACTIVATE IF conditional wrapper:
  ACTIVATE IF (original) vs. plain worked examples (this script's output)

Usage:
    python3 scripts/eval/strip_cs_to_plain_examples.py \
        --tasks causal_judgement geometric_shapes snarks formal_fallacies \
        --src-run-dir runs/bbh_v3 \
        --out-dir runs/bbh_v3_plain_examples
"""
import argparse
import json
import os
import shutil


def build_plain_cheatsheet(src_json_path: str, src_pk_txt_path: str) -> str:
    """Return cheatsheet text: original PK + CS reformatted as plain worked examples."""
    with open(src_json_path) as f:
        data = json.load(f)

    # PK section — read from the pk_final txt (clean, no truncation)
    with open(src_pk_txt_path) as f:
        pk_txt = f.read().strip()

    case_studies = data.get("case_studies", [])
    if not case_studies:
        # No CS — return PK only (full == pk_only for this task)
        return pk_txt + "\n"

    lines = [pk_txt, "", "=== WORKED EXAMPLES ===", ""]

    for i, cs in enumerate(case_studies, 1):
        title = cs.get("title", f"Example {i}")
        examples = cs.get("support_examples", [])

        lines.append(f"--- Worked Example {i}: {title} ---")
        lines.append(f"=== WORKED EXAMPLE: {title} ===")

        if not examples:
            lines.append("(no examples available)")
        else:
            for ex in examples:
                e1 = ex.get("e1", "").strip()
                e2 = ex.get("e2", "").strip()
                if e1:
                    lines.append(f"  Input:  {e1}")
                if e2:
                    lines.append(f"  Answer: {e2}")
                lines.append("")

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--src-run-dir", default="runs/bbh_v3")
    parser.add_argument("--out-dir", default="runs/bbh_v3_plain_examples")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    for task in args.tasks:
        src_task_dir = os.path.join(root, args.src_run_dir, task)
        out_task_dir = os.path.join(root, args.out_dir, task)
        os.makedirs(out_task_dir, exist_ok=True)

        src_json = os.path.join(src_task_dir, "cheatsheet_final.json")
        src_pk_txt = os.path.join(src_task_dir, "cheatsheet_phase1_pk_final.txt")
        src_pk_json = os.path.join(src_task_dir, "cheatsheet_phase1_pk_final.json")

        if not os.path.exists(src_json):
            print(f"[skip] {task}: cheatsheet_final.json not found")
            continue
        if not os.path.exists(src_pk_txt):
            print(f"[skip] {task}: cheatsheet_phase1_pk_final.txt not found")
            continue

        # Copy pk_final files unchanged (eval uses pk_only condition)
        shutil.copy2(src_pk_txt, out_task_dir)
        if os.path.exists(src_pk_json):
            shutil.copy2(src_pk_json, out_task_dir)

        new_text = build_plain_cheatsheet(src_json, src_pk_txt)
        out_final = os.path.join(out_task_dir, "cheatsheet_final.txt")
        with open(out_final, "w") as f:
            f.write(new_text)

        n_cs = new_text.count("=== WORKED EXAMPLE:")
        print(f"[ok] {task}: {n_cs} worked example(s) → {out_final}")


if __name__ == "__main__":
    main()
