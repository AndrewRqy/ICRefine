# ICRefine — Iterative Cheatsheet Refinement

Automatically improves a cheatsheet used to prompt LLMs on **magma equation implication** tasks.

ICRefine works alongside `SAIR_eval_pipeline/` — both repos must be set up. They are expected to sit next to each other:

```
CHAI Project/
├── SAIR_eval_pipeline/
└── ICRefine/
```

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

**2. Set up SAIR_eval_pipeline first** (if you haven't already)

```bash
cd ../SAIR_eval_pipeline
uv sync
cp .env.example .env   # then add your OPENROUTER_API_KEY
cd ../ICRefine
```

**3. Install ICRefine dependencies and activate the environment**

```bash
uv sync
source .venv/bin/activate
```

All subsequent `python` commands assume the venv is active. Alternatively, prefix any command with `uv run` (e.g. `uv run python -m ICR_select.pipeline ...`).

**4. Configure your backend**

```bash
cp .env.example .env
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

When both `VLLM_BASE_URL` and `VLLM_MODEL` are set and `--model-score` matches `VLLM_MODEL`, scoring calls route to the local vLLM server automatically. See `SAIR_eval_pipeline/README.md` for cluster setup instructions.

**5. Run a smoke test**

*OpenRouter only:*
```bash
python -m ICR_select.pipeline \
    --dataset ../SAIR_eval_pipeline/datasets/normal.jsonl \
    --prior-knowledge ../SAIR_eval_pipeline/prompts/NeuriCo_cheatsheet.txt \
    --model-score openai/gpt-oss-120b \
    --model-casestudy openai/gpt-4o \
    --limit 20 \
    --output-dir runs/smoke
```

*vLLM for scoring + OpenRouter for case studies (recommended on cluster):*
```bash
python -m ICR_select.pipeline \
    --dataset ../SAIR_eval_pipeline/datasets/normal.jsonl \
    --prior-knowledge ../SAIR_eval_pipeline/prompts/NeuriCo_cheatsheet.txt \
    --model-score deepseek-r1-14b \
    --model-casestudy openai/gpt-4o \
    --limit 20 \
    --output-dir runs/smoke
```

This runs 20 items and exits. Check `runs/smoke/` for output artifacts.

**6. Run the full pipeline**

```bash
python -m ICR_select.pipeline \
    --dataset ../SAIR_eval_pipeline/datasets/normal.jsonl \
    --prior-knowledge ../SAIR_eval_pipeline/prompts/NeuriCo_cheatsheet.txt \
    --model-score deepseek-r1-14b \
    --model-casestudy openai/gpt-4o \
    --bin-threshold 5 --batch-size 10 \
    --n-candidates 3 \
    --output-dir runs/select_normal \
    --cheatsheet-out ../SAIR_eval_pipeline/prompts/NeuriCo_cheatsheet_refined.txt
```

The refined cheatsheet is written to `--cheatsheet-out` and can be passed directly to `SAIR_eval_pipeline/run_evaluation.py --cheatsheet`.

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
| _(none)_ | LLM generates a decision tree from seed examples |
| `--init-roadmap FILE` | Load a plain `.txt` as the trainable roadmap; case studies start empty |
| `--init-cheatsheet PATH` | Load a previously saved cheatsheet (`.json` sidecar); full state restored |
| `--prior-knowledge FILE` | Load a frozen plain-text file (e.g. NeuriCo prompt) into the `prior_knowledge` field — can be combined with any option above, or used alone to start with an empty trainable roadmap |

The recommended starting point is `--prior-knowledge ../SAIR_eval_pipeline/prompts/NeuriCo_cheatsheet.txt`.

---

## ICR_naive

```bash
python -m ICR_naive.pipeline \
    --dataset ../SAIR_eval_pipeline/datasets/normal.jsonl \
    --init-txt ../SAIR_eval_pipeline/prompts/NeuriCo_cheatsheet.txt \
    --model-score openai/gpt-oss-120b \
    --model-casestudy openai/gpt-4o \
    --bin-threshold 5 --batch-size 20 \
    --output-dir runs/naive_normal \
    --cheatsheet-out ../SAIR_eval_pipeline/prompts/NeuriCo_cheatsheet_naive.txt
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
    --dataset ../SAIR_eval_pipeline/datasets/hard1.jsonl \
    --init-txt ../SAIR_eval_pipeline/prompts/NeuriCo_cheatsheet.txt \
    --model-score openai/gpt-oss-120b \
    --model-casestudy openai/gpt-4o \
    --bin-threshold 3 --batch-size 10 \
    --output-dir runs/reasoning_hard1 \
    --cheatsheet-out ../SAIR_eval_pipeline/prompts/NeuriCo_cheatsheet_reasoning.txt
```

Additional flags beyond ICR_naive:

| Flag | Default | Description |
|---|---|---|
| `--cot-first` | off | Put REASONING before VERDICT in the scoring prompt |
| `--no-dt-patch` | off | Skip decision tree patching; only add case studies |
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

### CS-only run (no DT revision)

```bash
python -m ICR_select.pipeline \
    --dataset ../SAIR_eval_pipeline/datasets/normal.jsonl \
    --prior-knowledge ../SAIR_eval_pipeline/prompts/NeuriCo_cheatsheet.txt \
    --model-score openai/gpt-oss-120b \
    --model-casestudy openai/gpt-4o \
    --bin-threshold 3 --batch-size 5 \
    --n-candidates 3 \
    --output-dir runs/select_normal \
    --cheatsheet-out ../SAIR_eval_pipeline/prompts/NeuriCo_cheatsheet_select.txt
```

### With DT revision (harder datasets)

```bash
python -m ICR_select.pipeline \
    --dataset ../SAIR_eval_pipeline/datasets/hard1.jsonl \
    --prior-knowledge ../SAIR_eval_pipeline/prompts/NeuriCo_cheatsheet.txt \
    --model-score openai/gpt-oss-120b \
    --model-casestudy openai/gpt-4o \
    --bin-threshold 3 --batch-size 5 \
    --n-candidates 3 \
    --dt-rounds 3 \
    --output-dir runs/select_dt_hard1 \
    --cheatsheet-out ../SAIR_eval_pipeline/prompts/NeuriCo_cheatsheet_select_dt.txt
```

### All options

**Quality gates**

| Flag | Default | Description |
|---|---|---|
| `--n-candidates N` | `3` | Candidates generated per bin flush |
| `--flush-strategy` | `default` | `default`: discard on gate failure. `retry`: retry up to `--candidate-rounds` times with context from the previous attempt |
| `--candidate-rounds N` | `3` | Max retry rounds when `--flush-strategy retry` |
| `--fix-rate-threshold F` | `0.5` | Min fraction of failures a candidate must fix |
| `--regress-threshold F` | `0.15` | Max fraction of correct-pool items a candidate may break |
| `--no-similarity-gate` | off | Skip LLM dedup (faster, less selective) |
| `--validate-merge` | off | Only commit a merge if the merged entry is at least as good as the original |

**Oracle / prescore**

| Flag | Default | Description |
|---|---|---|
| `--oracle-csv FILE` | off | GPT-5.4 oracle CSV (`gpt5.4_normal_default.csv`) — appends correct reasoning as a contrast signal in case study generation |
| `--prescore-file FILE` | off | Pre-computed score map `{id: {predicted, correct, ...}}` — skips the initial scoring pass. Used automatically by the SAIR refinement pipeline |

**Maintenance**

| Flag | Default | Description |
|---|---|---|
| `--ablation-every N` | `5` | Run ablation pruning every N flushes |
| `--condense-at N` | `6` | Run condensation when case study count reaches this |

**DT revision outer loop**

| Flag | Default | Description |
|---|---|---|
| `--dt-rounds N` | `1` | Outer DT revision rounds (1 = no DT revision) |
| `--plateau-threshold F` | `0.02` | Stop outer loop if round-on-round improvement < F |
| `--keep-case-studies` | off | Carry case studies between DT revision rounds |
| `--min-failures-for-dt N` | `5` | Min failures needed to attempt DT revision |

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
| `--model-casestudy MODEL_ID` | Case study and DT revision generation |
| `--model-init MODEL_ID` | Initial cheatsheet generation (if not using `--prior-knowledge`) |
| `--model MODEL_ID` | Default for all stages |

---

## Output

```
runs/select_normal/
├── cheatsheet_init.txt        # starting cheatsheet
├── cheatsheet_update_01.txt   # checkpoint after each addition
├── cheatsheet_final.txt       # final cheatsheet (plain text)
├── cheatsheet_final.json      # final cheatsheet (structured, for --init-cheatsheet)
└── update_log.json            # event log (adds, merges, discards, ablations, condensations)
```

For DT revision runs, each round gets a subfolder with `cheatsheet_end_of_round.txt` and `dt_revision.json` (accepted/rejected, accuracy before/after).

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
from ICR_naive.core.cheatsheet import Cheatsheet
from ICR_naive.core.data import load_jsonl
from ICR_reasoning.core.llm_client import get_api_key

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
python -m ICR_mymode.pipeline --dataset ../SAIR_eval_pipeline/datasets/normal.jsonl
```

Key building blocks already available to reuse:

| Import | What it provides |
|---|---|
| `ICR_naive.core.cheatsheet.Cheatsheet` | Cheatsheet dataclass with `render()`, `save()`, `load()`, `prior_knowledge` |
| `ICR_naive.core.data.load_jsonl` | Load a `.jsonl` dataset |
| `ICR_reasoning.core.llm_client.call_llm` | Single synchronous LLM call (routes to vLLM or OpenRouter automatically) |
| `ICR_reasoning.training.scorer.score_batch` | Score a batch of items against a cheatsheet, returns per-item verdicts + post-think |
| `ICR_select.generators.case_study.generate_candidates` | Generate N candidate case studies from a failure bin |
| `ICR_select.training.loop.run_training_loop` | Full inner CS loop with all four gates — use this to avoid reimplementing gating logic |

---

## Comparing Modes

`compare_modes.sh` runs baseline → train → eval for all three modes:

```bash
bash compare_modes.sh         # full comparison (100 training items, 150 eval items)
bash compare_modes.sh smoke   # quick smoke test — verifies all gates fire without errors
bash compare_modes.sh eval    # eval only — re-evaluate already-generated cheatsheets
```

---

## Project Structure

```
ICRefine/
├── ICR_naive/           # Basic HypoGenic-style loop
├── ICR_reasoning/       # Post-think aware loop
├── ICR_select/          # Selective quality-gated loop (recommended)
│   └── training/
│       ├── loop.py                # Inner CS loop (4 gates + pruning + condensation)
│       ├── outer_loop.py          # Outer DT revision loop
│       ├── dt_reviser.py          # Validated decision tree revision
│       └── roadmap_synthesizer.py # Absorbs case studies into a structured reasoning roadmap
├── eval_oracle_quality.py  # Compare case study quality with vs without oracle injection
├── compare_modes.sh
├── pyproject.toml          # Dependencies (managed by uv)
└── .env.example
```
