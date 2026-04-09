# ICRefine — Iterative Cheatsheet Refinement

Automatically improves a cheatsheet used to prompt LLMs on **magma equation implication** tasks.
ICRefine is a standalone project — it only needs a dataset (`.jsonl`) and an optional starting cheatsheet/prior-knowledge file.

---

## Quick Start

Follow these steps in order. All commands run from inside `ICRefine/`.

ICRefine uses two types of models and each can be served via OpenRouter or vLLM:

| Role | Flag | What it does |
|---|---|---|
| Scoring model | `--model-score` | Evaluates items against the current cheatsheet to find failures |
| Case study model | `--model-casestudy` | Writes new case studies from failure bins |

| Backend | When to use | Key requirement |
|---|---|---|
| **OpenRouter** (default) | Cloud models (GPT, Claude, etc.) | `OPENROUTER_API_KEY` in `.env` |
| **vLLM** | Locally-served open-source models (e.g. DeepSeek-R1-14B on a GPU cluster) | vLLM server running; `VLLM_BASE_URL` + `VLLM_MODEL` in `.env` |

You can mix backends: e.g. use vLLM for scoring (cheap, fast) and OpenRouter for case study generation (higher quality). **API key selection is automatic and per-model** — there is no `--api-key` flag. The routing logic in `ICR_naive/core/llm_client.py` resolves the key for each call based on the model name:

| Model name | Env var used |
|---|---|
| Matches `VLLM_MODEL` | `VLLM_API_KEY` (usually empty) |
| Starts with `gpt-4`, `o1`, `o3`, `o4` and `OPENAI_API_KEY` is set | `OPENAI_API_KEY` → OpenAI directly |
| Everything else | `OPENROUTER_API_KEY` → OpenRouter |

So to use different keys for scoring vs case study generation, simply set whichever keys apply and pass different model names to `--model-score` and `--model-casestudy`:

```
# .env
OPENROUTER_API_KEY=sk-or-v1-xxxx     # used for --model-casestudy openai/gpt-4o
OPENAI_API_KEY=sk-xxxx               # used if --model-casestudy gpt-4o (direct)
VLLM_BASE_URL=http://localhost:8000/v1/chat/completions
VLLM_MODEL=deepseek-r1-14b           # used for --model-score deepseek-r1-14b
VLLM_API_KEY=                        # usually empty
```

**1. Install uv** (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Install ICRefine dependencies**

```bash
uv sync
source .venv/bin/activate
```

All subsequent `python` commands assume the venv is active. Alternatively, prefix any command with `uv run` (e.g. `uv run python -m ICR_select.pipeline ...`).

**3. Configure your backend**

```bash
cp .env.example .env
# Edit .env and add your API keys
```

*OpenRouter (cloud models):*
```
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxx
```

*vLLM (local model — for scoring):*
```
VLLM_BASE_URL=http://localhost:8000/v1/chat/completions
VLLM_MODEL=deepseek-r1-14b
```

**4. Run a smoke test**

*OpenRouter only:*
```bash
python -m ICR_select.pipeline \
    --dataset path/to/dataset.jsonl \
    --prior-knowledge path/to/prior_knowledge.txt \
    --model-score openai/gpt-oss-120b \
    --model-casestudy openai/gpt-4o \
    --limit 20 \
    --output-dir runs/smoke
```

*vLLM for scoring + OpenRouter for case studies (recommended on cluster):*
```bash
python -m ICR_select.pipeline \
    --dataset path/to/dataset.jsonl \
    --prior-knowledge path/to/prior_knowledge.txt \
    --model-score deepseek-r1-14b \
    --model-casestudy openai/gpt-4o \
    --limit 20 \
    --output-dir runs/smoke
```

This runs 20 items and exits. Check `runs/smoke/` for output artifacts.

**5. Run the full pipeline**

```bash
python -m ICR_select.pipeline \
    --dataset path/to/dataset.jsonl \
    --prior-knowledge path/to/prior_knowledge.txt \
    --model-score deepseek-r1-14b \
    --model-casestudy openai/gpt-4o \
    --bin-threshold 5 --batch-size 10 \
    --n-candidates 3 \
    --output-dir runs/select_run \
    --cheatsheet-out path/to/output_cheatsheet.txt
```

---

## Modes

Three refinement modes are available. **ICR_select is recommended** for all production runs.

| Mode | Entry point | Description |
|---|---|---|
| `ICR_naive` | `python -m ICR_naive.pipeline` | Basic loop — collect failures, generate one case study, append |
| `ICR_reasoning` | `python -m ICR_reasoning.pipeline` | Same, but feeds the model's chain-of-thought to the case study generator |
| `ICR_select` | `python -m ICR_select.pipeline` | Full quality-gated loop with candidate competition, fix-rate / regression / similarity gates, pruning, condensation |

---

## Cheatsheet Init Modes

All pipelines support these initialisation options (mutually exclusive except `--prior-knowledge`):

| Flag | Behaviour |
|---|---|
| _(none)_ | LLM generates a reasoning roadmap from seed examples |
| `--init-roadmap FILE` | Load a plain `.txt` as the trainable roadmap; case studies start empty |
| `--init-cheatsheet PATH` | Load a previously saved cheatsheet (`.json` sidecar); full state restored |
| `--prior-knowledge FILE` | Load a frozen plain-text file into the `prior_knowledge` field — can be combined with any option above, or used alone to start with an empty trainable roadmap |

---

## ICR_naive

```bash
python -m ICR_naive.pipeline \
    --dataset path/to/dataset.jsonl \
    --init-txt path/to/prior_knowledge.txt \
    --model-score openai/gpt-oss-120b \
    --model-casestudy openai/gpt-4o \
    --bin-threshold 5 --batch-size 20 \
    --output-dir runs/naive_run \
    --cheatsheet-out path/to/output_cheatsheet.txt
```

| Flag | Default | Description |
|---|---|---|
| `--bin-threshold N` | `5` | Failures needed to trigger case study generation |
| `--batch-size N` | `20` | Items scored per mini-batch |
| `--val-split FRAC` | `0.2` | Fraction held out for validation |
| `--reasoning-effort` | `low` | `low` / `medium` / `high` / `none` |

---

## ICR_reasoning

```bash
python -m ICR_reasoning.pipeline \
    --dataset path/to/dataset.jsonl \
    --init-txt path/to/prior_knowledge.txt \
    --model-score openai/gpt-oss-120b \
    --model-casestudy openai/gpt-4o \
    --bin-threshold 3 --batch-size 10 \
    --output-dir runs/reasoning_run \
    --cheatsheet-out path/to/output_cheatsheet.txt
```

Additional flags beyond ICR_naive:

| Flag | Default | Description |
|---|---|---|
| `--cot-first` | off | Put REASONING before VERDICT in the scoring prompt |
| `--no-analysis` | off | Skip the final reasoning analysis stage |
| `--limit N` | off | Cap training items to the first N |

---

## ICR_select

The full quality-gated pipeline. Every candidate case study must pass four gates:

1. **Candidate competition** — generate N candidates, pick the best
2. **Fix-rate gate** — must fix ≥ `fix-rate-threshold` of the failure batch
3. **Regression gate** — must not break > `regress-threshold` of previously-correct items
4. **Similarity gate** — LLM dedup: skip if duplicate, merge if overlapping, add if genuinely new

Periodic maintenance: **ablation pruning** (remove zero-contribution case studies) and **condensation** (rewrite when cheatsheet grows too large).

```bash
python -m ICR_select.pipeline \
    --dataset path/to/dataset.jsonl \
    --prior-knowledge path/to/prior_knowledge.txt \
    --model-score openai/gpt-oss-120b \
    --model-casestudy openai/gpt-4o \
    --bin-threshold 3 --batch-size 5 \
    --n-candidates 3 \
    --output-dir runs/select_run \
    --cheatsheet-out path/to/output_cheatsheet.txt
```

### All options

**Quality gates**

| Flag | Default | Description |
|---|---|---|
| `--n-candidates N` | `3` | Candidates generated per bin flush |
| `--flush-strategy` | `default` | `default`: discard on gate failure. `retry`: retry up to `--candidate-rounds` times with context from the previous attempt |
| `--candidate-rounds N` | `3` | Max retry rounds when `--flush-strategy retry` |
| `--fix-rate-threshold F` | `0.30` | Min fraction of failures a candidate must fix |
| `--regress-threshold F` | `0.15` | Max fraction of correct-pool items a candidate may break |
| `--min-pool-for-regression N` | `10` | Skip regression gate when the correct pool has fewer than N items (avoids false rejections early in training when the pool is too small) |
| `--no-similarity-gate` | off | Skip LLM dedup (faster, less selective) |
| `--validate-merge` | off | Only commit a merge if the merged entry is at least as good as the original |

**Oracle / prescore**

| Flag | Default | Description |
|---|---|---|
| `--oracle-csv FILE` | off | Oracle CSV — appends correct reasoning as a contrast signal in case study generation |
| `--prescore-file FILE` | off | Pre-computed score map `{id: {predicted, correct, ...}}` — skips the initial scoring pass |

**Maintenance**

| Flag | Default | Description |
|---|---|---|
| `--ablation-every N` | `5` | Run ablation pruning every N flushes |
| `--condense-at N` | `6` | Run condensation when case study count reaches this |

**Shared**

| Flag | Default | Description |
|---|---|---|
| `--bin-threshold N` | `5` | Failures needed to trigger a flush |
| `--batch-size N` | `10` | Items per mini-batch |
| `--concurrency N` | `10` | Parallel API requests |
| `--limit N` | off | Cap training items to the first N |
| `--reasoning-effort` | `low` | `low` / `medium` / `high` / `none` |
| `--cot-first` | off | REASONING before VERDICT in scoring prompt |

---

## Specifying Models

All pipelines accept per-stage model overrides:

| Flag | Stage |
|---|---|
| `--model-score MODEL_ID` | Scoring items during training |
| `--model-casestudy MODEL_ID` | Case study generation |
| `--model-init MODEL_ID` | Initial cheatsheet generation (if not using `--prior-knowledge`) |
| `--model MODEL_ID` | Default for all stages |

---

## Output

```
runs/select_run/
├── cheatsheet_init.txt        # starting cheatsheet (plain text)
├── cheatsheet_update_01.txt   # checkpoint after each case study addition
├── cheatsheet_final.txt       # final cheatsheet (plain text — what goes into prompts)
├── cheatsheet_final.json      # final cheatsheet (structured JSON — use with --init-cheatsheet)
└── update_log.json            # event log (adds, merges, discards, ablations, condensations)
```

The `.json` sidecar stores each case study as a **structured record** — not a flat string. Each entry contains all parsed fields (`activate_if`, `action`, `why_this_check_works`, `support_examples`, `feature_signature`, etc.) alongside running statistics (`creation_fix_rate`, `historical_fix_rate`, `n_activations`, `n_fixes`). The `.txt` is the human-readable render used in prompts.

---

## Extending the Pipeline

### Adding a new quality gate to ICR_select

Quality gates are applied inside `ICR_select/training/loop.py` in `_process_flush()` / `_process_flush_retry()`. To add a new gate, insert a check after the existing ones and return `None` (discard) or the candidate to pass it through:

```python
# ICR_select/training/loop.py — inside _process_flush()

# After the regression gate...

# --- Custom gate: reject if the case study is longer than 300 words ---
if len(best_candidate.split()) > 300:
    print("[gate:length] candidate too long — discarding.", flush=True)
    return None   # discard the bin; return candidate to accept
```

Expose it as a CLI flag by adding an argument in `ICR_select/pipeline.py` and threading it through `inner_kwargs` into `run_training_loop()`.

---

### Adding a new refinement mode

Each mode is a self-contained Python package. To add a new one (e.g. `ICR_mymode`):

**1.** Create the package directory:
```
ICRefine/
└── ICR_mymode/
    ├── __init__.py
    └── pipeline.py    # entry point
```

**2.** Implement `pipeline.py`. The minimal structure mirrors `ICR_naive/pipeline.py`:

```python
# ICR_mymode/pipeline.py
import argparse
from pathlib import Path
from utils.cheatsheet import Cheatsheet
from utils.data import load_jsonl
from utils.llm_client import get_api_key

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     required=True)
    p.add_argument("--output-dir",  default="runs/mymode")
    p.add_argument("--model-score", default="openai/gpt-oss-120b")
    # ... add your flags
    args = p.parse_args()

    api_key = get_api_key()
    items   = load_jsonl(Path(args.dataset))
    cs      = Cheatsheet()

    # --- Your refinement logic here ---
    # Score items, collect failures, update cs, repeat.

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    cs.save(Path(args.output_dir) / "cheatsheet_final")
    print(cs.render())

if __name__ == "__main__":
    main()
```

**3.** Run it:
```bash
python -m ICR_mymode.pipeline --dataset path/to/dataset.jsonl
```

Key building blocks already available to reuse:

| Import | What it provides |
|---|---|
| `utils.cheatsheet.Cheatsheet` | Cheatsheet dataclass with `render()`, `save()`, `load()`, `prior_knowledge` |
| `utils.case_study.CaseStudy` | Structured case study record — `from_text()`, `from_dict()`, `to_dict()`, `render()`, `record_activation()` |
| `utils.data.load_jsonl` | Load a `.jsonl` dataset |
| `utils.llm_client.call_llm` | Single synchronous LLM call (routes to vLLM or OpenRouter automatically) |
| `utils.scorer.score_batch` | Score a batch of items against a cheatsheet, returns per-item verdicts + post-think |
| `ICR_select.generators.case_study.generate_candidates` | Generate N candidate `CaseStudy` objects from a failure bin |
| `ICR_select.training.loop.run_training_loop` | Full inner CS loop with all four gates — use this to avoid reimplementing gating logic |

---

## Comparing Modes

`compare_modes.sh` trains all three modes on the same dataset for a side-by-side comparison:

```bash
# Edit DATASET and BASE_CHEATSHEET at the top of compare_modes.sh first
bash compare_modes.sh         # full comparison (100 training items)
bash compare_modes.sh smoke   # quick smoke test — verifies all gates fire without errors
```

---

## Project Structure

```
ICRefine/
├── utils/               # Shared utilities (used by all three ICR packages)
│   ├── case_study.py    # CaseStudy dataclass — structured record with routing metadata
│   ├── cheatsheet.py    # Cheatsheet dataclass — render, save, load (case_studies: list[CaseStudy])
│   ├── data.py          # Dataset loading, splitting, FailureBin, is_true
│   ├── parser.py        # Parse VERDICT / REASONING / PROOF / COUNTEREXAMPLE
│   ├── llm_client.py    # Unified LLM client — vLLM / OpenAI / OpenRouter routing
│   └── scorer.py        # score_batch, test_cheatsheet, TestResult
├── ICR_naive/           # Basic HypoGenic-style loop
├── ICR_reasoning/       # Post-think aware loop
├── ICR_select/          # Selective quality-gated loop (recommended)
│   └── training/
│       ├── loop.py                # Inner CS loop (4 gates + pruning + condensation)
│       └── roadmap_synthesizer.py # Absorbs case studies into a structured reasoning roadmap
├── smoke_test_gates.py     # Gate threshold smoke tests (no live LLM required)
├── eval_oracle_quality.py  # Compare case study quality with vs without oracle injection
├── compare_modes.sh
├── pyproject.toml          # Dependencies (managed by uv)
└── .env.example
```

### CaseStudy structured fields

Each case study stored in the JSON sidecar has these fields:

| Field | Purpose |
|---|---|
| `title` | Short descriptive label |
| `activate_if` | Parsed IDENTIFY conditions — list of strings that must ALL be true |
| `do_not_activate_if` | Boundary conditions — when NOT to fire |
| `action` | Conclusion when activated |
| `next_check` | Routing: `"DONE: TRUE"`, `"DONE: FALSE"`, or `"PROCEED TO: STEP N"` |
| `common_wrong_move` | What the model typically does wrong in these cases |
| `why_this_check_works` | Mathematical justification (WHY field) |
| `support_examples` | List of `{e1, e2, answer, note}` dicts |
| `feature_signature` | Compact structural tag, e.g. `"absorbing→general_L0"` |
| `target_roadmap_aspect` | DT step this case study corrects |
| `creation_fix_rate` | Fix rate on the flush bin that produced this entry |
| `historical_fix_rate` | Running average updated by ablation / eval passes |
| `n_activations` / `n_fixes` | Activation and precision counters |
