"""
Debug script: run gpt-4.1 on WOL test items with EA PK and log full responses.
Runs the first 30 items sequentially with reasoning-first prompt format.
"""

import importlib, json, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path("ICR_partition/.env"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.data import load_jsonl
from utils.llm_client import get_api_key, call_llm

mod = importlib.import_module("tasks.bbh_tasks_ext")
task_spec = mod.WEB_OF_LIES_TASK

MODEL = "openai/gpt-4.1"
N = 30
OUT = Path("runs/logs/evals/misc/debug_gpt41_wol_ea_reasfirst.jsonl")

api_key = get_api_key()
pk_txt  = Path("runs/bbh_ea_p1/web_of_lies/cheatsheet_phase1_pk_final.txt").read_text().strip()
items   = load_jsonl("datasets/bbh/web_of_lies_test.jsonl")[:N]

correct = 0
results = []

for i, item in enumerate(items):
    prompt = mod._web_of_lies_scoring_prompt_rf(pk_txt, item)
    resp = call_llm(prompt, model=MODEL, api_key=api_key,
                    temperature=0.0, max_tokens=600, reasoning_effort=None)

    raw = resp.content if resp else ""
    predicted = mod._parse_web_of_lies_rf(raw) if raw else None
    ok = task_spec.is_correct(predicted, item)
    correct += ok

    entry = {
        "i": i,
        "question": item["input"],
        "answer": item["answer"],
        "predicted": predicted,
        "correct": ok,
        "response": raw,
    }
    results.append(entry)

    status = "OK" if ok else "WRONG"
    print(f"[{i+1:02d}/{N}] {status}  answer={item['answer']}  predicted={predicted}")
    if not ok:
        print(f"       {raw[:600].replace(chr(10), ' | ')}")
    sys.stdout.flush()

print(f"\nAccuracy: {correct}/{N} = {correct/N:.1%}")

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print(f"Full responses saved → {OUT}")
