# ICRefine — Iterative Cheatsheet Refinement

Automatically improves a cheatsheet used to prompt LLMs on **magma equation implication** tasks. Three refinement modes are provided, each building on the last.

All commands are run from inside the `ICRefine/` directory. The repo is designed to be used alongside `SAIR_evaluation_pipeline/`, which sits one level up.

---

## Setup

### 1. Install dependencies

```bash
pip3 install -r requirements.txt
```

### 2. Set your API key

Create a `.env` file (or copy `.env.example`):

```
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxx
```

The pipeline loads this automatically — no `export` needed.

---

## Modes

There are three refinement modes. Each is a self-contained Python package with its own `pipeline.py` entry point.

| Mode | Module | Description |
|---|---|---|
| Naive | `ICR_naive` | Basic HypoGenic-style loop — collect failures, generate a case study, append it |
| Reasoning | `ICR_reasoning` | Same loop, but passes the model's internal chain-of-thought to the case study generator for richer signal |
| Select | `ICR_select` | Full quality-gated loop with candidate competition, fix-rate/regression/similarity gates, ablation pruning, condensation, and optional DT revision |

**ICR_select is the recommended mode** for production runs.

---

## Cheatsheet Init Modes

All three pipelines support the same three ways to initialise the cheatsheet:

| Flag | Behaviour |
|---|---|
| _(none)_ | LLM generates a decision tree from seed examples |
| `--init-txt FILE` | Load a plain `.txt` file as the decision tree; case studies start empty |
| `--init-cheatsheet PATH` | Load a previously saved cheatsheet (`.json` sidecar) |

The most common starting point is `--init-txt` with `NeuriCo_cheatsheet.txt`.

---

## ICR_naive

Basic loop: score batches, collect failures in a bin, flush when full → generate one case study → append.

```bash
python -m ICR_naive.pipeline \
    --dataset ../SAIR_evaluation_pipeline/datasets/normal.jsonl \
    --init-txt ../SAIR_evaluation_pipeline/prompts/NeuriCo_cheatsheet.txt \
    --model-score openai/gpt-oss-120b \
    --model-casestudy openai/gpt-4o \
    --bin-threshold 5 --batch-size 20 \
    --output-dir runs/naive_normal \
    --cheatsheet-out ../SAIR_evaluation_pipeline/prompts/NeuriCo_cheatsheet_naive.txt
```

### Key options

| Flag | Default | Description |
|---|---|---|
| `--bin-threshold N` | `5` | Failures needed to trigger case study generation |
| `--batch-size N` | `20` | Items scored per mini-batch |
| `--val-split FRAC` | `0.2` | Fraction held out for validation |
| `--reasoning-effort` | `low` | `low` / `medium` / `high` / `none` |
| `--no-flush-remainder` | off | Skip the final flush of leftover failures |

---

## ICR_reasoning

Same flow as ICR_naive, but captures the scoring model's chain-of-thought and includes it in the case study generation prompt. Also runs a final reasoning analysis report.

```bash
python -m ICR_reasoning.pipeline \
    --dataset ../SAIR_evaluation_pipeline/datasets/hard1.jsonl \
    --init-txt ../SAIR_evaluation_pipeline/prompts/NeuriCo_cheatsheet.txt \
    --model-score openai/gpt-oss-120b \
    --model-casestudy openai/gpt-4o \
    --bin-threshold 3 --batch-size 10 \
    --output-dir runs/reasoning_hard1 \
    --cheatsheet-out ../SAIR_evaluation_pipeline/prompts/NeuriCo_cheatsheet_reasoning.txt
```

### Additional options (beyond ICR_naive)

| Flag | Default | Description |
|---|---|---|
| `--no-dt-patch` | off | Skip decision tree patching; only add case studies |
| `--cot-first` | off | Put REASONING before VERDICT in the scoring prompt |
| `--no-analysis` | off | Skip the final reasoning analysis stage |
| `--limit N` | off | Cap training items to the first N |

---

## ICR_select

The full quality-gated pipeline. Every candidate case study must pass four gates before entering the cheatsheet:

1. **Candidate competition** — generate N candidates, pick the best
2. **Fix-rate gate** — best candidate must fix ≥ `fix-rate-threshold` of the failure batch
3. **Regression gate** — must not break > `regress-threshold` of previously-correct items
4. **Similarity gate** — LLM dedup: skip if duplicate, merge if overlapping, add if genuinely new. With `--validate-merge`, a merge is only committed if the merged entry fixes at least as many failures as the existing one — otherwise the candidate is added as a new entry instead

Periodic maintenance:
- **Ablation pruning** — every N flushes, remove case studies with zero marginal contribution
- **Condensation** — when the cheatsheet grows too large, rewrite to a denser set and validate before swapping

### CS-only (no DT revision)

```bash
python -m ICR_select.pipeline \
    --dataset ../SAIR_evaluation_pipeline/datasets/normal.jsonl \
    --init-txt ../SAIR_evaluation_pipeline/prompts/NeuriCo_cheatsheet.txt \
    --model-score openai/gpt-oss-120b \
    --model-casestudy openai/gpt-4o \
    --bin-threshold 3 --batch-size 5 \
    --n-candidates 3 \
    --dt-rounds 1 \
    --output-dir runs/select_normal \
    --cheatsheet-out ../SAIR_evaluation_pipeline/prompts/NeuriCo_cheatsheet_select.txt
```

### With DT revision (recommended for harder datasets)

Wraps the CS loop in an outer round structure. After each round, the pipeline analyses which decision tree steps are most commonly misapplied, rewrites those steps, and validates the revision before accepting it. Starts the next round from the improved DT with a fresh set of case studies.

```bash
python -m ICR_select.pipeline \
    --dataset ../SAIR_evaluation_pipeline/datasets/hard1.jsonl \
    --init-txt ../SAIR_evaluation_pipeline/prompts/NeuriCo_cheatsheet.txt \
    --model-score openai/gpt-oss-120b \
    --model-casestudy openai/gpt-4o \
    --bin-threshold 3 --batch-size 5 \
    --n-candidates 3 \
    --dt-rounds 3 \
    --plateau-threshold 0.02 \
    --output-dir runs/select_dt_hard1 \
    --cheatsheet-out ../SAIR_evaluation_pipeline/prompts/NeuriCo_cheatsheet_select_dt.txt
```

### All options

**Quality gates**

| Flag | Default | Description |
|---|---|---|
| `--n-candidates N` | `3` | Candidates generated per bin flush |
| `--fix-rate-threshold F` | `0.5` | Min fraction of failures a candidate must fix |
| `--regress-threshold F` | `0.1` | Max fraction of correct-pool items a candidate may break |
| `--no-similarity-gate` | off | Skip LLM dedup check (faster, less selective) |
| `--validate-merge` | off | Before committing a merge, verify the merged entry fixes at least as many failures as the existing one. If not, add the candidate as a new entry instead |

**Maintenance**

| Flag | Default | Description |
|---|---|---|
| `--ablation-every N` | `5` | Run ablation pruning every N flushes |
| `--condense-at N` | `6` | Run condensation when case studies reaches this count |

**DT revision outer loop**

| Flag | Default | Description |
|---|---|---|
| `--dt-rounds N` | `1` | Outer DT revision rounds (1 = no DT revision) |
| `--plateau-threshold F` | `0.02` | Stop outer loop if round-on-round improvement < F |
| `--keep-case-studies` | off | Carry case studies over between DT revision rounds |
| `--min-failures-for-dt N` | `5` | Min failures needed to attempt DT revision |

**Shared with other modes**

| Flag | Default | Description |
|---|---|---|
| `--bin-threshold N` | `5` | Failures needed to trigger a flush |
| `--batch-size N` | `10` | Items scored per mini-batch |
| `--concurrency N` | `10` | Parallel API requests |
| `--val-split FRAC` | `0.0` | Fraction held out for validation |
| `--limit N` | off | Cap training items to the first N |
| `--reasoning-effort` | `low` | `low` / `medium` / `high` / `none` |
| `--cot-first` | off | REASONING before VERDICT in scoring prompt |

---

## Specifying Models

All pipelines accept `--model` as a default for all stages, plus per-stage overrides:

| Flag | Stage |
|---|---|
| `--model-init` | Initial cheatsheet generation |
| `--model-score` | Scoring items during training |
| `--model-casestudy` | Case study / DT revision generation |

```bash
# Use gpt-oss-120b for scoring, gpt-4o for writing case studies
--model-score openai/gpt-oss-120b --model-casestudy openai/gpt-4o
```

---

## Output

Each run saves artifacts to `--output-dir`:

```
runs/select_normal/
├── cheatsheet_init.txt        # starting cheatsheet
├── cheatsheet_update_01.txt   # checkpoint after each addition
├── cheatsheet_final.txt       # final cheatsheet (plain text)
├── cheatsheet_final.json      # final cheatsheet (structured, for --init-cheatsheet)
└── update_log.json            # event log (adds, merges, discards, ablations, condensations)
```

For DT revision runs, each round gets its own subfolder:

```
runs/select_dt_hard1/
├── round_01/
│   ├── cheatsheet_end_of_round.txt
│   └── dt_revision.json       # accepted/rejected, accuracy before/after
├── round_02/
│   └── ...
├── cheatsheet_final.txt
└── outer_loop_summary.json    # accuracy history across all rounds
```

Pass `--cheatsheet-out FILE` to also write the final cheatsheet directly into `SAIR_evaluation_pipeline/prompts/` for immediate use in evaluation.

---

## Comparing Modes

`compare_modes.sh` runs a full baseline → train → eval comparison of all three modes on the normal dataset.

```bash
# Full comparison (100 training items, 150 held-out eval items)
bash compare_modes.sh

# Quick smoke test — verifies all gates and DT revision fire without errors
bash compare_modes.sh smoke

# Eval only — re-evaluate already-generated cheatsheets
bash compare_modes.sh eval
```

---

## Project Structure

```
ICRefine/
├── ICR_naive/           # Basic HypoGenic-style loop
│   ├── pipeline.py      # CLI entry point
│   ├── core/            # Cheatsheet, data loading, LLM client
│   ├── generators/      # Initial cheatsheet + case study generation
│   ├── prompts/         # Prompt templates
│   └── training/        # Training loop + scorer
│
├── ICR_reasoning/       # Post-think aware loop
│   ├── pipeline.py      # CLI entry point
│   ├── analysis/        # Reasoning analysis report
│   ├── core/            # LLM client (reasoning-aware)
│   ├── generators/      # Case study generation with COT context
│   ├── prompts/
│   └── training/        # Training loop + scorer
│
├── ICR_select/          # Selective quality-gated loop (recommended)
│   ├── pipeline.py      # CLI entry point
│   ├── analysis/        # DT step misapplication profiler
│   ├── generators/      # Candidate case study generation
│   ├── prompts/
│   └── training/
│       ├── loop.py      # Inner CS loop (4 gates + pruning + condensation)
│       ├── outer_loop.py # Outer DT revision loop
│       └── dt_reviser.py # Validated decision tree revision
│
├── compare_modes.sh     # Baseline → train → eval comparison script
├── requirements.txt
└── .env.example
```
