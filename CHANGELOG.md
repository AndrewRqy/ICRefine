# ICRefine Changelog

---

## 2026-05-02

### Paper preparation, EA Phase 1, reasoning-first evaluation, and eval script suite

---

#### ICR_paper_prep/: full NeurIPS 2026 submission

A complete paper write-up of the ICRefine evaluation results.

**Paper:** *ICRefine: Diagnosing the Limits of Iterative Cheatsheet Transfer Across Model Families*

Key contents:

| File | Description |
|---|---|
| `icrefine_paper.tex` | Main LaTeX source (NeurIPS 2026 format, 22 pages) |
| `references.bib` | 21-entry bibliography |
| `neurips_2026.sty` | NeurIPS style file |
| `figures/fig1–fig9.png` | All figures (phase contribution, transfer gap, delta_cs heatmap, oracle gradient, variance, etc.) |
| `findings_draft.tex` | Earlier findings-only draft (preserved) |
| `overleaf_upload/` | Ready-to-upload Overleaf bundle (flat structure, `main.tex`) |
| `overleaf_upload.zip` | Zipped version of the Overleaf bundle |

**Paper contributions:**
- Three-phase ICRefine pipeline (Phase 0 bootstrap, Phase 1 PK patching, Phase 2 case studies)
- Cross-model transfer evaluation: 5 BBH tasks, 4 non-train evaluation model families (GPT-4.1, Claude-3.7, Gemini-2.0, Llama-3.3)
- Three identified failure modes: oracle contamination, partition overfitting, bootstrap convergence limits
- EA Phase 1 intervention (+6.0 pp GS PK accuracy) and oracle-free Phase 2 (+1.9–3.8 pp CJ accuracy)
- Task-level bootstrap CIs (10,000-iteration resampling) for all main results
- 3-seed means throughout; CS-ICL re-scored 3 times with RF prompting for apples-to-apples comparison

---

#### ICR_hybrid/training/ea_phase1.py: Evolutionary Algorithm Phase 1

Population-based PK search replacing single-candidate patching when the bootstrap PK already satisfies the acceptance gate.

**Design:**

- Population of 3 members with distinct roles: `conservative` (temp 0.2, targeted edits), `generative` (temp 0.7, unconstrained rewrites), `structural` (temp 0.3, explicit TIGHTEN/SPLIT/ADD_GUARD ops)
- 80/20 train/validation split; validation accuracy drives selection
- Curriculum net-gain gate: `n_fixed >= λ(g) * n_regressed` where `λ(g)` ramps 1.0 → 2.0 over generations
- Top-2 survivor selection + crossover into child PK each generation
- Elite tracking (best-ever PK replaces weakest member each generation)
- Exits on 2 consecutive static generations or 4-generation cap

**Results:** GS PK-only accuracy +6.0 pp (72.7% → 78.7%, 3-seed mean); GS case studies eliminated entirely (Phase 2 acceptance gate rejects all candidates); Llama seed variance reduced σ=10.4% → 4.7%.

---

#### scripts/: evaluation and plotting script suite

New directory with reproducible eval and figure-generation scripts used for the paper.

**Eval scripts (`scripts/eval/`):**

| Script | Purpose |
|---|---|
| `eval_cs_ablation.py` | RF scoring for all pipeline conditions (v3, E3, EA, EA+no-oracle) |
| `eval_csicl_rescore.py` | Standalone CS-ICL rescoring with RF prompts (used for 3-seed CS-ICL means) |
| `bootstrap_ci.py` | Task-resampling bootstrap CI (10,000 iterations) over 5-task averages |

**Plot scripts (`scripts/plots/`):**

| Script | Figure |
|---|---|
| `plot_transfer_fig2.py` | Fig 2: non-train transfer bar chart |
| `plot_delta_cs_fig7.py` | Fig 7: Δ_cs heatmap (5 tasks × 5 models) |
| `plot_oracle_gradient_fig8.py` | Fig 8: oracle contamination gradient (CJ + GS) |
| `plot_variance_fig6.py` | Fig 6: per-seed variance for EA-GS and E3-CJ |
| `plot_gs_oracle_variance_fig9.py` | Fig 9: GS cross-oracle per-seed variance |

---

#### Reasoning-first (RF) scoring

All task modules updated with `_rf` scoring variants (`build_scoring_prompt` + `parse_verdict` overrides). RF scoring prompts the model to reason before emitting `VERDICT: (X)`, eliminating format-sensitivity artifacts (web_of_lies: 42–66% verdict-first → 99–100% RF).

Affected files: `tasks/causal_judgement.py`, `tasks/geometric_shapes.py`, `tasks/formal_fallacies.py`, `tasks/disambiguation_qa.py`, `tasks/snarks.py`, `utils/scorer.py`.

`utils/scorer.py` updated: new `cot_first` parameter passed through `score_batch`; `reasoning_effort` parameter added for o-series models.

---

#### runs/variance/: 3-seed evaluation results

Seed 2 and seed 3 run outputs for v3, E3, and EA conditions used to compute 3-seed means throughout the paper:

| File | Contents |
|---|---|
| `runs/variance/eval_results/v3_seed2_rf.json` | v3 seed 2, RF scoring, all 5 tasks × 5 models |
| `runs/variance/eval_results/v3_seed3_rf.json` | v3 seed 3 |
| `runs/variance/eval_results/ea_seed2_rf.json` | EA seed 2 |
| `runs/variance/eval_results/ea_seed3_rf.json` | EA seed 3 |
| `runs/variance/eval_results/e3_seed2_rf.json` | E3 seed 2 |
| `runs/variance/eval_results/e3_seed3_rf.json` | E3 seed 3 |
| `runs/variance/v3_3seed_mean.json` | Averaged v3 results |
| `runs/csicl_seed2_rf.json` | CS-ICL rescore seed 2 |
| `runs/csicl_seed3_rf.json` | CS-ICL rescore seed 3 |
| `runs/bbh_v3_rf_canonical_5models.json` | Canonical v3 seed 1 (patched: Claude DQ cs_icl 0.62 → 0.93) |

---

#### Cleanup: removed obsolete scripts

Deleted one-off ablation and run scripts that were superseded by the `scripts/` suite:
`ablation_narrative.py`, `ablation_prompts.py`, `compare_modes.sh`, `download_bbh_tasks.py`, `eval_bbh_comparison.py`, `eval_oracle_quality.py`, `gen_csicl_*.py`, `gen_icr_bootstrap.py`, `oracle_generate.py`, `run_ablation.py`, `run_bbh_*.sh` (7 scripts), `score_eval.py`, `smoke_*.py`, `test_*.py`.

---

## 2026-04-30

### Codebase reconstruction: per-task files, shared prompt modules, eval instrumentation

A broad structural pass that splits monolithic task files, consolidates prompt templates into canonical locations, instruments all eval scripts with RunLogger, and fixes several import bugs found during the audit.

---

#### tasks/: one file per task

**Motivation:** All 13 task specifications (scoring prompts, answer parsing, bootstrap helpers) were packed into two files — `bbh_tasks.py` and `bbh_tasks_ext.py`. Adding, editing, or reviewing one task required navigating the whole file. Imports were also misleading: `ICR_select` imported from `ICR_naive.core` indirectly.

**New layout:**

Each task is now its own module under `tasks/`:

| File | Export |
|---|---|
| `tasks/causal_judgement.py` | `CAUSAL_JUDGEMENT_TASK`, `_causal_bootstrap`, `_CAUSAL_GEN_PROMPT` |
| `tasks/sports_understanding.py` | `SPORTS_TASK`, `_SPORTS_GEN_PROMPT` |
| `tasks/disambiguation_qa.py` | `DISAMBIGUATION_TASK` |
| `tasks/movie_recommendation.py` | `MOVIE_TASK` |
| `tasks/geometric_shapes.py` | `GEOMETRIC_TASK`, `GEO_PROMPTS`, `_GEO_GEN_PROMPT` |
| `tasks/formal_fallacies.py` | `FORMAL_FALLACIES_TASK` |
| `tasks/logical_deduction.py` | `LOGICAL_DEDUCTION_3_TASK` |
| `tasks/web_of_lies.py` | `WEB_OF_LIES_TASK`, `WOL_PROMPTS` |
| `tasks/date_understanding.py` | `DATE_UNDERSTANDING_TASK`, `DU_PROMPTS` |
| `tasks/navigate.py` | `NAVIGATE_TASK` |
| `tasks/snarks.py` | `SNARKS_TASK` |
| `tasks/boolean_expressions.py` | `BBH_BOOLEAN_TASK` |

**`tasks/utils/__init__.py` (new):**

18 shared helper functions extracted from both old task files into a single shared module — `_make_eval_prompt`, `_parse_yesno`, `_parse_mc`, `_extract_reasoning`, `_yesno_correct`, `_yesno_label`, `_mc_correct`, `_mc_label`, `_rule_score_prompt`, `_gen_prompt`, `_bootstrap_ruleset`, `_format_failure`, `_parse_valid_invalid`, `_valid_invalid_correct`, `_valid_invalid_label`, `_trivial_key`, `_trivial_conds`, `_generic_polarity`. Duplicates between the two old files were deduplicated here.

**`tasks/registry.py` updated:**

All 12 non-magma entries now point to the new individual module paths (e.g. `"causal_judgement": ("tasks.causal_judgement", "CAUSAL_JUDGEMENT_TASK")`).

**Backward-compat shims kept:**

`bbh_tasks.py`, `bbh_tasks_ext.py`, and `bbh_boolean.py` are now pure re-export shims. All existing `from tasks.bbh_tasks import …` and `from tasks.bbh_tasks_ext import …` callers — including pipeline scripts, dev ablation scripts, and shell launchers — continue to work without modification.

---

#### prompts/: canonical prompt modules

**Motivation:** Core prompt templates were defined separately inside `ICR_naive/prompts/templates.py`, `ICR_reasoning/prompts/templates.py`, and `ICR_select/prompts/templates.py`, causing drift between copies. Mechanism prompts (similarity gate, merge, prune, crossover) were mixed into the per-pipeline templates files.

**New canonical files:**

- `prompts/templates.py` — all core templates: `ROADMAP_PROMPT`, `DECISION_TREE_PROMPT`, `CASE_STUDIES_PROMPT`, `CASE_STUDY_PROMPT`, `SCORING_PROMPT`, `SCORING_PROMPT_COT_FIRST`, `CASE_STUDY_WITH_REASONING_PROMPT`, plus token-budget constants (`DT_MAX_TOKENS`, `CS_MAX_TOKENS`, `SCORING_MAX_TOKENS`, `FLUSH_MAX_TOKENS`).
- `prompts/mechanisms.py` — ICR_select mechanism prompts: `SIMILARITY_CHECK_PROMPT`, `MERGE_PROMPT`, `PRUNE_PROMPT`, `CONDENSATION_PROMPT`, `DT_STEP_ANALYSIS_PROMPT`, `DT_REVISION_PROMPT`, `ROADMAP_SYNTHESIS_PROMPT`, plus their `_MAX_TOKENS` constants and `N_CANDIDATES`, `CANDIDATE_TEMPS`, `CORRECT_POOL_MAX`, `RETRY_CONTEXT_TEMPLATE`.

Per-pipeline `prompts/templates.py` files are now shims that import from the canonical modules. `ICR_select/prompts/templates.py` additionally imports mechanism prompts from `prompts/mechanisms.py`.

**Versioned generation prompts moved into task files:**

`WOL_PROMPTS`, `DU_PROMPTS`, and `GEO_PROMPTS` (the v3/v4 versioned case-study generation prompts) now live inside their respective task files (`tasks/web_of_lies.py`, `tasks/date_understanding.py`, `tasks/geometric_shapes.py`). Each file selects the active version at import time via the `ICR_GEN_PROMPT_VERSION` env var.

`prompts/generation.py` converted to a facade: imports the dicts from task files, keeps the `get_prompt(task, version)` API for ablation script access, adds `VERSIONS` dict.

---

#### utils/cheatsheet.py: `extract_query_features` null sentinel for non-magma tasks

**Problem:** When `extract_query_features` was called on a BBH item (no `equation1` field), it fell back to `_features_from_pair("", "")` which returned TRIVIAL forms. These generated non-empty `tokens()` frozensets that produced spurious Jaccard hits in `render_for_query`, surfacing structurally-unrelated magma case studies for BBH queries.

**Fix:**

- Added `_NULL_FORM = "NONE"` sentinel constant.
- `extract_query_features(item)` returns a null `QueryFeatures` (all fields set to `_NULL_FORM` / zero) when `"equation1" not in item`.
- `QueryFeatures.tokens()` returns `frozenset()` when `form_e1 == _NULL_FORM`, so relevance scoring falls back to `historical_fix_rate` only (no structural component). This is correct: a BBH case study has no equation pair to match against.

---

#### Eval scripts: RunLogger wired to all five remaining scripts

`RunLogger` (from `utils/run_logger`) was previously wired to the pipeline entry points but missing from eval scripts. Added to:

- `scripts/eval/eval_transferability.py` — label `"eval_transfer"`
- `scripts/eval/eval_cot_mode.py` — label `"eval_cot"`
- `scripts/eval/eval_bbh_comparison.py` — label `"eval_bbh_compare"`
- `scripts/eval/eval_oracle_quality.py` — label `"eval_oracle"`
- `scripts/eval/score_eval.py` — label `"score_eval"`

Each script creates a `RunLogger` after `get_api_key()`, calls `set_logger()`, and prints the log directory to stderr.

---

#### Dev script and import bug fixes

Three bugs found during the shell-script / import path audit:

**`scripts/dev/ablation_narrative.py`:**
- `os.chdir(Path(__file__).parent)` pointed to `scripts/dev/`, making `DATA_DIR = Path("datasets/bbh")` resolve to a nonexistent path.
- Fixed: `_ROOT = Path(__file__).resolve().parent.parent.parent`; `os.chdir(_ROOT)`.
- Added missing `sys.path.insert(0, str(_ROOT))` before project-root imports.

**`scripts/dev/ablation_prompts.py`:**
- No `sys.path` anchor — project-root imports failed when run from any directory other than the root.
- Fixed: same `_ROOT` + `sys.path.insert` pattern.

**`scripts/dev/smoke_test_pk_patch.py`:**
- `str(Path(__file__).resolve()…)` appeared on line 21 before `from pathlib import Path` was imported.
- Fixed: moved `pathlib` import to the top. Removed a spurious `sys.path.insert(0, os.path.dirname(…))` that added `scripts/dev/` to the path.

**`tasks/bbh_tasks.py` shim extended:**
- The initial shim was missing `_CAUSAL_GEN_PROMPT`, `_SPORTS_GEN_PROMPT`, and `_GEO_GEN_PROMPT` re-exports needed by `scripts/dev/ablation_prompts.py`. Added explicit re-exports from the individual task files.

**`ICR_naive/pipeline.py`:**
- Three remaining `.core.*` shim imports (`from .core.cheatsheet`, `.core.data`, `.core.llm_client`) were still present after the `utils/` migration. Updated to `from utils.cheatsheet`, `from utils.data`, `from utils.llm_client`.

---

#### Smoke test: 64/64 checks passed

All existing smoke tests plus new import-path checks passed after the reconstruction:
- Pipelines: `ICR_naive`, `ICR_reasoning`, `ICR_select`, `ICR_partition`, `ICR_hybrid` entry points importable
- Eval scripts: all 5 new RunLogger-wired scripts importable
- Task files: all 13 individual task modules importable via new paths and via backward-compat shims
- Registry: all 13 `tasks.registry.load_task(name)` lookups return the correct task spec
- Prompt modules: `prompts/templates.py`, `prompts/mechanisms.py`, `prompts/generation.py` + 3 per-pipeline shims importable
- Versioned prompts: `WOL_PROMPTS`, `DU_PROMPTS`, `GEO_PROMPTS` accessible from task files and via `get_prompt()`
- Dev scripts: 7 scripts in `scripts/dev/` importable without errors

---

## 2026-04-28

### ICR_hybrid: CS-ICL auto-bootstrap (default on)

**Motivation:** BBH tasks with no `--init-cheatsheet` and no `--prior-knowledge` previously entered Phase 2 completely blind — the model had to rediscover general task principles from failure patterns alone. This caused overfitting on tasks like `geometric_shapes` (43% test accuracy) and `sports_understanding` (82% vs CS-ICL 96%).

**Changes (`ICR_hybrid/training/loop.py`):**

New parameters on `run_hybrid_loop`:
- `auto_bootstrap: bool = True` — when `True` and no prior knowledge exists after Phase 1, fires the auto-bootstrap block before Phase 2.
- `bootstrap_n_items: int = 75` — how many training items to sample.

Two-step bootstrap (mirrors CS-ICL's `generate_reason_api.py` + `cheatsheet_api.py`):

1. **Reasoning enrichment** — checks each sampled item for a `reason` or `gold_reason` field. Any items that lack one get a concurrent LLM call (`ThreadPoolExecutor`, up to 20 workers) using a zero-shot explanation prompt: *"Given the question and its correct answer, provide a clear step-by-step explanation…"*. This is a no-op for BBH tasks whose train JSONLs already carry a `reason` field.
2. **Knowledge summary** — formats all items as `Question / Reasoning / Answer` triples (or `Question / Answer` for items still without reasoning) and calls the LLM with the original CS-ICL prompt: *"Create a cheat sheet based on the examples below… identify which ones you find most difficult… only include specific, detailed points."* The response becomes `cheatsheet.prior_knowledge`, seeding `_pk_patch_phase` so Phase 2 always starts with non-trivial background knowledge.

The block is a no-op when any of these are true:
- `auto_bootstrap=False`
- A `--rule-set` was provided
- `prior_knowledge` is already non-empty (from `--init-cheatsheet`, `--prior-knowledge`, `auto_rule_init`, or `_pk_patch_phase`)

**Changes (`ICR_hybrid/pipeline.py`):**

Two new CLI flags:
- `--no-auto-bootstrap` — disables auto-bootstrap (sets `auto_bootstrap=False`). Default: bootstrap on.
- `--bootstrap-n-items N` — override sample size (default: 75).

Header banner updated to show `auto-bootstrap: yes (75 items)` or `no`.

**Early results on v3 runs (4 tasks, bootstrap enabled):**

| Task | v2 train acc | v3 train acc | v2 test acc |
|---|---|---|---|
| geometric_shapes | ~43% | 75%+ (iter 4) | 43% |
| sports_understanding | ~82% | 95%+ (iter 5) | 82% |
| formal_fallacies | ~65% | 74% (iter 4, 0 CS — bootstrap carrying it) | 65% |
| causal_judgement | ~65% | 66% (iter 3) | 65.5% |

---

### ICR_partition: Duplicate title check on CS acceptance

**Motivation:** The similarity gate uses an LLM call to detect near-duplicate case studies by content, but identical-title regenerations slipped through when the same partition was re-evaluated with a shifted failure set.

**Changes (`ICR_partition/training/loop.py`):**

Added a cheap exact title check (case-insensitive) at both CS acceptance points inside `_solve_bin`, under the `cs_lock`:
- **Archive path** — before accepting an archived candidate from a previous iteration.
- **Main ADD path** — before accepting any new candidate that passed all gates.

When a title match is found, the bin is skipped with `event: bin_skipped, reason: title_duplicate` logged to the update log. No LLM calls are saved or wasted — the check runs before the candidate is staged.

---

### ICR_partition / ICR_select: Periodic CS bank prune

**Motivation:** Long runs accumulated near-duplicate case studies with slightly different wordings that passed the title check and the LLM similarity gate individually but represented the same root failure pattern. The boolean expressions v2 run ended with 4 of 6 case studies being near-identical "Double Negation Cancellation" entries.

**New: `_prune_cs_bank()` (`ICR_select/training/gates.py`):**

Sends the full numbered case study bank to the LLM with `PRUNE_PROMPT` — instructs the model to output `REMOVE: <exact title>` for genuine duplicates or `KEEP ALL` if none. Filters `cheatsheet.case_studies` in-place and returns the count removed. Runs at temperature 0, `max_tokens=400`.

**New: `PRUNE_PROMPT` / `PRUNE_MAX_TOKENS` (`ICR_select/prompts/templates.py`):**

Single-shot prune prompt. Rules enforce that only genuine root-pattern duplicates are removed; cosmetically similar but mechanically distinct case studies must be kept.

**Changes (`ICR_partition/training/loop.py`):**

New parameter `prune_every_n: int = 5` on `run_partition_loop`. After each outer iteration's executor block, checks whether `len(cheatsheet.case_studies) // prune_every_n` crossed a new integer boundary since the start of the iteration — if so, fires `_prune_cs_bank`. Logged as `event: cs_pruned` in the update log.

---

### Remove: REASONING ROADMAP section from cheatsheet generation

**Motivation:** The REASONING ROADMAP field was a leftover from the original ICR_naive design where a structured decision tree guided scoring. With BBH tasks this field was always empty (no rule set → no roadmap patch), so it added noise to generation prompts and wasted token budget. The field is still stored in the `Cheatsheet` dataclass for backward compatibility.

**Changes:**

`utils/cheatsheet.py` — `_render_with_selection()` no longer writes the `=== REASONING ROADMAP ===` section. The budget calculation that previously subtracted the roadmap's character count was updated accordingly.

`tasks/bbh_tasks.py` and `tasks/bbh_tasks_ext.py` — removed from all 11 task generation prompts:
- Input section: `=== REASONING ROADMAP ===\n{roadmap}\n=== END ROADMAP ===`
- Output field: `TARGET_STEP: [roadmap step this corrects]`
- Output section: `OUTPUT 2 — ROADMAP PATCH / === ROADMAP PATCH === … === END ROADMAP PATCH ===`

`tasks/bbh_boolean.py` — same removals from `BBH_BOOLEAN_GENERATION_PROMPT`. Also removed the preamble paragraph "The cheatsheet the model uses has two parts:…" and added a TRANSFERABILITY REQUIREMENT block before `OUTPUT 1 — CASE STUDY`.

**Smoke tests updated:** `smoke_bbh_boolean.py` Test 6 and `smoke_bbh_tasks.py` Test 1 now assert that `{roadmap}` is **absent** from the generation prompt template (was previously required).

---

### ICR_hybrid: Fix misleading Phase 1 log banner

**Problem:** The training loop banner always printed `Phase 1: SKIPPED` when no `--rule-set` was provided, even when `_pk_patch_phase` (text-mode prior knowledge patching) was about to run.

**Fix (`ICR_hybrid/training/loop.py`):** Banner now shows one of three states:
- `rule-patch (N iters, goal=X%)` — when a `RuleSet` is provided.
- `pk-text-patch (N iters)` — when `prior_knowledge` is non-empty or `auto_rule_init` will produce it.
- `SKIPPED (no rule set or prior knowledge)` — truly skipped.

---

### New scripts

**`run_bbh_v3.sh`** — parallel runner for the 4 most underperforming BBH tasks (`geometric_shapes`, `causal_judgement`, `formal_fallacies`, `sports_understanding`) with two monitoring modes:
- Default: status board that refreshes every 10s showing state, last log line, current iter, accuracy, and CS count per task.
- `--tail`: raw interleaved log stream across all 4 logs.

Written for bash 3.2 (macOS default) — no associative arrays.

**`test_overfit_boolean.py`** — evaluates the boolean expressions cheatsheet from `bbh_overnight_v2` on both train and test splits, applies `_prune_cs_bank`, evaluates again, and prints a before/after table comparing train accuracy, test accuracy, and train-test gap.

---

## 2026-04-26

### ICR_hybrid: New hybrid rule-patch → case-study pipeline

A new `ICR_hybrid` pipeline module was added. It supersedes the task-specific magma pipeline and generalises to any task registered in `_TASK_MAP`.

**Design: two-phase refinement**

- **Phase 1 (rule patching)** — optional. Either provide `--rule-set` (a Jinja2 rule file) or use `--auto-rule-init` to bootstrap an initial rule set from scored failures. The phase iterates patch→score→patch until `--rule-acc-goal` is reached or `--max-rule-iters` is exhausted.
- **Phase 2 (case study generation)** — always runs. Uses the same partition-parallel quality-gated loop as ICR_partition on top of whatever Phase 1 established.

**PK regression guard** — `--pk-regression-guard` measures accuracy with only `prior_knowledge` (no case studies), then reverts Phase 2 additions if they fall more than `--pk-regression-tolerance` below that baseline. Prevents case studies from causing net harm when `prior_knowledge` is already strong (e.g. from a CS-ICL bootstrap cheat sheet).

**Entry point:** `python -m ICR_hybrid.pipeline`

---

### BBH extended tasks: 6 new tasks in `_TASK_MAP`

Six new BBH tasks were registered in `ICR_hybrid/pipeline.py` via `tasks/bbh_tasks_ext.py`:

| Task | Train items |
|---|---|
| `formal_fallacies` | 150 |
| `logical_deduction_three` | 150 |
| `web_of_lies` | 150 |
| `date_understanding` | 150 |
| `navigate` | 150 |
| `snarks` | 100 |

Training data added to `datasets/bbh/` as `{task}_train.jsonl`.

---

### CS-ICL baseline comparison framework

Two new scripts enable systematic comparison against CS-ICL (Cheat Sheet In-Context Learning):

**`gen_csicl_ext.py`** — generates CS-ICL cheat sheets for each BBH task using the original CS-ICL paper prompt: "Create a cheat sheet based on the examples below… only include specific, detailed points to address the challenging ones." Saved to `../cheat-sheet-icl/data/cheat_prompt/{task}/cheat_prompt.txt`.

**`eval_bbh_comparison.py`** — evaluates both CS-ICL and ICR on the test split. Reads ICR final cheatsheets from `--run-dir/{task}/cheatsheet_final.json`, reads CS-ICL cheat sheets from `--cs-icl-dir/{task}/cheat_prompt.txt`, scores both with the same model against the test split, writes combined results to `{run-dir}/comparison_results.json`.

**BBH ext results (test split, `openai/gpt-4.1-mini`):**

| Task | N_test | CS-ICL | ICR | Δ |
|---|---|---|---|---|
| formal_fallacies | ~50 | 60.0% | 65.0% | +5.0% |
| logical_deduction_three | ~50 | ~94% | ~94% | ≈0 |
| web_of_lies | ~50 | 55.0% | 57.0% | +2.0% |
| date_understanding | ~50 | ~72% | ~72% | ≈0 |
| navigate | ~50 | 78.0% | 76.0% | −2.0% |
| snarks | ~40 | ~90.1% | ~90.1% | ≈0 |

ICR matches or exceeds CS-ICL on 5 of 6 tasks; navigate is the only exception.

---

### MAGMA comparison: CS-ICL 88.1% vs ICR 80.95%

**New scripts:** `gen_csicl_magma.py` (generate CS-ICL cheat sheet from 100 magma training items) and `run_magma_comparison.sh` (run both pipelines and evaluate on 42 hard3 test items).

**Results (`runs/magma_comparison/comparison_results.json`, `openai/gpt-4.1-mini` scoring):**
- CS-ICL: **88.1%** (37/42)
- ICR: **80.95%** (34/42)
- Delta: **−7.1%** (ICR underperforms CS-ICL on MAGMA)

**Root cause analysis:**

CS-ICL generates mathematical pedagogy — translations (`L_x = left multiplication by x`), mathematical identities (idempotence, projection laws), a 24-row lookup table of concrete example pairs. This content is immediately applicable at inference time.

ICR generates structural patch descriptions in pipeline-internal terms (`topShape=v-m`, `xTop=both`, `ASPECT EXCEPTION`) — meaningful for the pipeline's iteration logic but not useful as mathematical guidance for the scoring model.

MAGMA's algebraic interconnectedness makes case-by-case refinement self-defeating: the `TRUE_left_proj` partition exhibited 43–86% regression on other TRUE items every time a targeted case study was added. Any fix to one TRUE-projection case broke algebraically related cases. CS-ICL avoids this by teaching the underlying mathematical structure rather than patching individual failure modes.

**Conclusion:** For algebraically-interconnected domains with large shared mathematical structure, pedagogical cheat sheets (CS-ICL style) beat patch-based refinement (ICR). The bootstrap+ICR hybrid below is designed to combine both.

---

### Bootstrap+ICR: CS-ICL seeded initial cheatsheet + ICR gap-filling

**Motivation:** ICR starts from a blank roadmap on BBH tasks and must rediscover general principles from failures. CS-ICL generates a complete pedagogical summary in one pass but cannot adapt to specific failure modes. The bootstrap approach seeds ICR with a CS-ICL cheat sheet, then uses Phase 1 (rule patching) to address gaps and Phase 2 to add case studies for residual hard failures.

**New file `gen_icr_bootstrap.py`:**

Generates a CS-ICL style cheat sheet for each task from the first N training items (75 per task, 50 for snarks) and stores it as a `Cheatsheet` JSON with the CS-ICL text in `prior_knowledge` — **not** in `roadmap`.

Key constraint: `ROADMAP_MAX_CHARS = 2500` means a 6k–8k char CS-ICL cheat sheet placed in `roadmap` would be silently truncated. Placing it in `prior_knowledge` avoids the cap and leaves the empty `roadmap` slot available for Phase 1 gap-filling patches.

```bash
python3 gen_icr_bootstrap.py [--tasks formal_fallacies snarks] [--model openai/gpt-4.1-mini]
# Output: runs/bbh_bootstrap/bootstrap_cheatsheets/{task}/bootstrap_cs.json
```

**New script `run_bbh_bootstrap.sh`:**

Orchestrates the full bootstrap+ICR run for all 6 BBH tasks:
1. Generate CS-ICL bootstrap cheat sheets (`gen_icr_bootstrap.py`)
2. Run ICR pipeline per task with `--init-cheatsheet`, `--auto-rule-init`, `--rule-acc-goal 0.95`, `--max-rule-iters 3`, `--max-cs-iters 5`, `--pk-regression-guard`
3. Run `eval_bbh_comparison.py` per task and accumulate results
4. Print final summary table comparing Bootstrap+ICR vs CS-ICL vs ICR-base

**New script `run_formal_fallacies_bootstrap.sh`:**

Single-task version for `formal_fallacies` (CS-ICL baseline: 60%, ICR-only: 65%). All three steps include skip logic — if output already exists, the step is skipped, so re-runs are safe. Uses `openai/gpt-4.1-mini` for all roles.

---

### Cheatsheet: `prior_knowledge_segments` field for per-segment ablation

**Motivation:** CS-ICL cheat sheets are structured documents with multiple sections (preamble, core rules, reference tables, pattern library, algorithm, tips). To identify which sections help and which hurt — without regenerating — each section should be independently disableable.

**Changes (`utils/cheatsheet.py`):**

New field `prior_knowledge_segments: list[dict] = field(default_factory=list)`.

Each segment dict has four keys:
- `id` — `"seg_0"`, `"seg_1"`, … (stable; set at generation time)
- `title` — first `###`/`##`/`####` heading in the block, or `"(preamble)"`
- `content` — full block text (heading included)
- `enabled` — `True` by default; set `False` to exclude from rendering

New method `_render_prior_knowledge()` — if `prior_knowledge_segments` is non-empty, joins all enabled segments with `"\n\n---\n\n"`; otherwise falls back to raw `self.prior_knowledge` (backward compatible). `_render_with_selection()` now calls this method instead of `self.prior_knowledge.strip()`.

New helper methods:
- `disable_prior_segment(segment_id) -> bool`
- `enable_prior_segment(segment_id) -> bool`
- `prior_segment_ids() -> list[str]`

`save()` writes `prior_knowledge_segments` conditionally and derives the flat `prior_knowledge` key from `_render_prior_knowledge()` for backward compatibility. `load()` reads `prior_knowledge_segments=data.get("prior_knowledge_segments", [])`.

**Segment generation (`gen_icr_bootstrap.py`):**

`_parse_segments(text)` splits the CS-ICL output by `\n---\n` delimiter. Each block becomes a segment dict. The `---` divider naturally separates the 5-layer CS-ICL structure so segments align with logical sections without any prompt changes.

---

### Pipeline: `--ablate-prior-segments` flag

**Changes (`ICR_hybrid/pipeline.py`):**

New flag `--ablate-prior-segments IDS` (comma-separated segment IDs, e.g. `seg_0,seg_3`). When specified, the pipeline loads `--init-cheatsheet`, calls `cs.disable_prior_segment(seg_id)` for each listed ID, logs the result, then proceeds normally.

This enables ablation studies without re-generating the bootstrap cheat sheet — pass different `--ablate-prior-segments` values to measure each section's contribution in isolation.

---

## 2026-04-10

### ICR_select: Skip redundant val scoring when prescore is provided

**Problem:** Inside the recursive-refine pipeline, ICR_select ran a full val-split
re-score at the end of each iteration (using vLLM) even though the very next SAIR
eval step would re-evaluate all items — including the val split — with the refined
cheatsheet. This added an unnecessary vLLM call during the inter-phase gap.

**Changes:**

`ICR_select/pipeline.py` — when `--prescore-file` is supplied (i.e., we are inside
the recursive-refine loop), `skip_final_val=True` is passed to `run_training_loop()`.

`ICR_select/training/loop.py` — `run_training_loop()` accepts a new
`skip_final_val: bool = False` parameter. The final `test_cheatsheet()` call on val
items is skipped when `True`.

No change to standalone ICR_select runs (val scoring still runs when prescore is absent).

---

## 2026-04-08

### Structural: CaseStudy dataclass replaces flat strings

**Motivation:** Case studies were stored as opaque strings. This blocked:
- Inference-time routing (can't match IDENTIFY conditions without parsing text each time)
- Per-case precision tracking (can't count activations vs fixes on a string)
- Machine-readable dedup in the similarity gate (compares raw text instead of structured fields)
- Budget-aware rendering (must include all cases; can't rank by relevance)

**What changed:**

New file `utils/case_study.py` — `CaseStudy` dataclass with fields:
- `title`, `activate_if`, `do_not_activate_if`, `action` — core gate fields parsed from IDENTIFY / ACTION / DOES NOT APPLY TO
- `next_check` — routing: "DONE: TRUE/FALSE" or "PROCEED TO: STEP N"
- `common_wrong_move` — what the model typically does wrong (machine-readable)
- `why_this_check_works` — WHY field
- `support_examples` — EXAMPLES parsed into `list[{e1, e2, answer, note}]`
- `feature_signature` — compact one-line structural tag
- `target_roadmap_aspect` — which DT step this corrects
- `creation_fix_rate`, `historical_fix_rate`, `n_activations`, `n_fixes` — running stats
- `raw_text` — original LLM output preserved for debugging

`CaseStudy.render()` produces identical human-readable text to the old format — scoring
prompt is **unchanged**.  `CaseStudy.from_text(s)` parses old-format strings (backward
compat); `CaseStudy.to_dict()` / `from_dict()` for the JSON sidecar.

`utils/cheatsheet.py` — `case_studies: list[CaseStudy]` (was `list[str]`).  `__post_init__`
auto-wraps any plain strings.  `save()` writes structured dicts; `load()` is backward
compatible with old plain-string JSON.

`ICR_reasoning/prompts/templates.py` — generation prompt extended with four new output
fields: `FEATURE_SIGNATURE`, `COMMON_WRONG_MOVE`, `TARGET_STEP`, `NEXT_CHECK`.
`FLUSH_MAX_TOKENS` raised 600 → 900 to accommodate them.

`ICR_reasoning/generators/case_study.py` — `_parse_response()` now returns a `CaseStudy`
(not a plain string). `_render_case_studies_text()` calls `cs.render()`.

`ICR_select/generators/case_study.py` — `generate_candidates()` returns `list[CaseStudy]`.
`prev_attempt["candidate"]` uses `.render()` in retry context template.

`ICR_select/training/gates.py` — all helpers (`_mini_eval`, `_mini_eval_full`,
`_replace_eval`, `_regression_check`, `_similarity_gate`, `_merge_case_studies`,
`_format_existing`) typed to `CaseStudy`. `_merge_case_studies` returns `CaseStudy`
and carries forward `max(cs_a.creation_fix_rate, cs_b.creation_fix_rate)`.

`ICR_select/training/loop.py` — sets `creation_fix_rate` and `historical_fix_rate` on
the winning candidate before `add_case_study()`. Logs `title` in added/merged events.

`ICR_select/training/maintenance.py` — condensation wraps `split_case_studies()` output
in `CaseStudy.from_text()`.

`ICR_naive/generators/initial.py` — seed case studies wrapped in `CaseStudy.from_text()`.

`ICR_naive/core/cheatsheet.py` — re-exports `_extract_title_from_text` as `_extract_title`
for any external callers.

All existing smoke tests pass unchanged.

---

### ICR_select: Loosen fix-rate gate + min-pool guard on regression gate

**Problem:** Last night's run (`refine_20260408_071258`, 6 iterations) showed flat
accuracy at 71.72% across all iterations. All 4 candidate batches exhausted all 3
retry rounds and were discarded with `last_reason: "fix_rate"`. No cheatsheet update
ever passed the gates, so accuracy never moved.

Two root causes identified:
1. **fix_rate_threshold=0.50** too strict — failure bins are hard clusters the model
   consistently botches; no candidate can fix half of them in one shot.
2. **regression gate with tiny correct_pool** — early in the run the correct pool has
   only 5–9 items, so a single regression = 10–20% rate, falsely rejecting good candidates.

**Changes (`ICR_select/training/loop.py`, `ICR_select/pipeline.py`):**
- `fix_rate_threshold` default lowered **0.50 → 0.30**.
- New parameter `min_pool_for_regression: int = 10` on `run_training_loop()`.
  Regression gate is now skipped entirely when `len(correct_pool) < min_pool_for_regression`.
  Logged with a `[gate:regression] skipped — pool too small` message when triggered.
- Applied to both the `_process_flush` (default strategy) and `_process_flush_retry`
  (retry strategy) paths.
- **Diagnostic improvement:** `best_fix_rate` is now recorded in the `update_log`
  discard entries on the retry path (was missing — made post-mortems hard to read).
- New CLI flag: `--min-pool-for-regression N` (default: 10).
- `--fix-rate-threshold` default updated to 0.30 in `--help`.

**Changes (`SAIR_eval_pipeline/recursive_refine/config.py`, `updater.py`, `run_recursive_refine.py`):**
- `RecursiveConfig` gains `icr_min_pool_for_regression: int`.
- New CLI flag `--icr-min-pool-for-regression N` (default: 10).
- Env var `ICR_SELECT_MIN_POOL_FOR_REGRESSION` read by updater and forwarded to the
  ICR_select subprocess as `--min-pool-for-regression`.
- Startup banner now prints `min_pool` alongside the regression threshold.

**Smoke test (`smoke_test_gates.py`):** 9 checks, all pass:
- Correct defaults for both new params
- Regression gate skipped when pool < min_pool (patched `_regression_check` never called)
- Regression gate runs when pool ≥ min_pool
- fix_rate=0.30 accepts a 40%-fixing candidate; fix_rate=0.50 blocks the same candidate
- `best_fix_rate` present in retry-path discard log
- CLI `--min-pool-for-regression` and updated default present in `--help`

Both options remain fully tunable via CLI/env — they are not hardcoded.

---

## 2026-04-08 (continued)

### Rename: decision_tree → roadmap throughout the codebase

**Motivation:** The internal field and all user-facing strings still called the structured
reasoning guide a "decision tree", which conflicted with the `roadmap_synthesizer.py`
vocabulary and with the new ASPECT-based format. Unified everything under "roadmap".

**Scope:**

`utils/cheatsheet.py`:
- Field `decision_tree` → `roadmap` on the `Cheatsheet` dataclass.
- `DECISION_TREE_MAX_CHARS` → `ROADMAP_MAX_CHARS` (old name kept as backward-compat alias).
- `DECISION_TREE_HEADER` removed; render always uses `ROADMAP_HEADER = "=== REASONING ROADMAP ==="`.
  The fragile `if lstrip().startswith("ASPECT")` heuristic is gone.
- `patch_decision_tree` → `patch_roadmap` (old name kept as alias).
- JSON sidecar key `"decision_tree"` → `"roadmap"`; `load()` falls back to `"decision_tree"`
  for backward compatibility with old saved checkpoints.

`ICR_naive/prompts/templates.py`:
- `DECISION_TREE_PROMPT` renamed to `ROADMAP_PROMPT`; old name kept as alias.
- Prompt text changed from "Design a DECISION TREE … STEP 1, STEP 2" to
  "Design a REASONING ROADMAP … ASPECT N: / CHECK: / IF YES: / IF NO:" format.
- `{decision_tree}` template variable → `{roadmap}` in `CASE_STUDIES_PROMPT`.

`ICR_reasoning/prompts/templates.py`:
- `=== DECISION TREE ===` → `=== REASONING ROADMAP ===` in `CASE_STUDY_WITH_REASONING_PROMPT`.
- OUTPUT 2 renamed from "DECISION TREE PATCH" to "ROADMAP PATCH".
- `TARGET_STEP` description updated to reference "roadmap aspect" instead of "decision tree step".
- Patch block header regex updated to accept both old and new header names.

`ICR_select/prompts/templates.py`:
- All `{decision_tree}` template vars → `{roadmap}` in `CONDENSATION_PROMPT`,
  `DT_STEP_ANALYSIS_PROMPT`, `DT_REVISION_PROMPT`.
- Section headers updated to "REASONING ROADMAP".

All pipeline, generator, and training files: `Cheatsheet(decision_tree=...)` →
`Cheatsheet(roadmap=...)`, `cheatsheet.decision_tree` → `cheatsheet.roadmap`,
`.format(decision_tree=...)` → `.format(roadmap=...)`.

---

### Remove: DT update loop; keep roadmap construction loop

**Motivation:** The codebase had two separate mechanisms for updating the roadmap:
1. **DT revision outer loop** (`outer_loop.py` + `dt_reviser.py`) — a multi-round wrapper
   that re-scored the full training set after each inner loop run, analysed which roadmap
   steps were broken, and rewrote them via `DT_STEP_ANALYSIS_PROMPT` + `DT_REVISION_PROMPT`.
2. **Roadmap construction** (`roadmap_synthesizer.py`) — synthesises accumulated case
   studies into a new structured reasoning roadmap via `ROADMAP_SYNTHESIS_PROMPT`.

The DT revision loop added complexity and a separate update path that diverged from the
roadmap vocabulary. The roadmap synthesizer is the correct abstraction going forward.

**Deleted:**
- `ICR_select/training/dt_reviser.py` — DT step analysis + revision logic.
- `ICR_select/training/outer_loop.py` — multi-round DT revision outer loop.

**Removed from `ICR_select/pipeline.py`:**
- `from .training.outer_loop import run_outer_loop` import.
- CLI argument group "DT revision outer loop": `--dt-rounds`, `--plateau-threshold`,
  `--keep-case-studies`, `--min-failures-for-dt`.
- The `if args.dt_rounds > 1: … else: …` dispatch — Stage 2 is now always a single
  case study accumulation loop (`run_training_loop`).

**Removed from `ICR_reasoning/training/loop.py`:**
- `apply_dt_patch: bool = True` parameter.
- All `if apply_dt_patch and …` branches.
- Log messages cleaned up: "DT patch" → "roadmap patch".

**Kept:** `ICR_select/training/roadmap_synthesizer.py` (field access updated to `.roadmap`).

---

### Case study generation prompt reframed as reasoning-move teaching

**Motivation:** The old `CASE_STUDY_WITH_REASONING_PROMPT` asked the model to
"identify the structural feature" — a classification framing that produced verdict rules
("If E1 is absorbing → TRUE") without explaining *why the model gets it wrong* or
*what the correct mechanical move is*. Case studies gave answers, not teaching.

**Changes (`ICR_reasoning/prompts/templates.py`):**

- `CASE_STUDY_WITH_REASONING_PROMPT` completely rewritten with a **four-step generation
  scaffolding** for the producing model:
  1. **MISTAKEN SHORTCUT** — find the specific wrong reasoning move the model consistently
     makes (quote or paraphrase from the failure traces).
  2. **CORRECT MOVE** — the exact mechanical check that produces the right answer.
  3. **TRIGGER** — narrow structural conditions distinguishing these equations from
     ones where the shortcut is actually fine. Prompt explicitly warns: "A trigger that
     fires on too many cases causes regressions."
  4. **ANTI-TRIGGER** — 1–2 similar cases where this note should NOT fire.

- Output format uses the **teaching-move field order** (pedagogically motivated):
  `ACTIVATE IF` → `DO NOT ACTIVATE IF` → `COMMON WRONG MOVE` → `NEXT CHECK` →
  `WHY THIS WORKS` → `SUPPORT`

  `COMMON WRONG MOVE` is now the **third field** (immediately after the trigger), not an
  afterthought at the bottom. This is the most important signal for the weaker scoring
  model — it should appear before the correct answer.

- Task reframed in the preamble: *"tell the model exactly what shortcut it is tempted to
  take, why that shortcut is wrong here, and what it should do instead"* (was: *"identify
  the structural feature"*).

- Title instruction: *"name the mistaken shortcut or the structural trap, not just the
  equation type"* — prevents generic titles like "Absorbing Case" in favour of
  "Stops at Depth-1 When E1 Has a Fresh Variable".

- `FLUSH_MAX_TOKENS` raised **600 → 900** to accommodate the richer teaching-note format
  (all 8 fields + support examples).

`utils/case_study.py` — parser updated with the new field names as first-priority
alternatives: `ACTIVATE IF` (before `IDENTIFY`), `DO NOT ACTIVATE IF` (before
`DOES NOT APPLY TO`), `COMMON WRONG MOVE` / `NEXT CHECK` / `WHY THIS WORKS` / `SUPPORT`.
Both old and new format are accepted for backward compatibility.

---

### Cheatsheet: query-routed render via `render_for_query(item, top_k)`

**Motivation:** `render()` dumps all recent case studies globally into every prompt —
the same set regardless of what equation pair is being scored. A case study about
singleton absorbing patterns is noise when scoring a deep general equation. Budget is
wasted on irrelevant cases; the relevant ones may not even appear if the list is long.

**Changes (`utils/cheatsheet.py`):**

New method `render_for_query(item, top_k=3)`:
- Extracts structural features from `item["equation1"]` / `item["equation2"]` via pure
  string parsing (no LLM).
- Ranks every case study by a blended relevance score:
  `0.7 × structural_similarity + 0.3 × historical_fix_rate`
  - Structural similarity = token Jaccard on `feature_signature` strings + keyword bonus
    from `activate_if` text (caps at 0.4, prevents keyword-only cases from dominating).
- Renders only the top-k highest-scoring case studies within the standard character budget.
- Decision tree and prior knowledge sections are identical to `render()` — drop-in
  replacement for the scorer.

New supporting code (all in `utils/cheatsheet.py`):
- `QueryFeatures` NamedTuple — `form_e1/e2`, `l_e1/e2`, `vars_e1/e2`, `depth_e1/e2`.
  `.signature()` produces the same compact tag format as `CaseStudy.feature_signature`.
  `.tokens()` returns the frozenset used for Jaccard comparison.
- `extract_query_features(item)` — public entry point; calls `_features_from_pair`.
- `_features_from_pair(e1_raw, e2_raw)` — classifies each equation as TRIVIAL /
  SINGLETON / ABSORBING / STANDARD / GENERAL using the paren-depth-aware `_split_eq`
  helper. ABSORBING = bare-var side does **not** appear in the other side.
- `_sig_tokens(sig)` — tokenises `feature_signature` strings by splitting on
  non-alphanumeric characters.
- `_relevance_score(cs, qf)` — the scoring function.
- `_select_top_k(qf, top_k)` — returns sorted `(original_index, CaseStudy)` pairs;
  original index preserved so Case Study N display numbers stay stable.
- `_render_with_selection(selected)` — shared render core used by both `render()` and
  `render_for_query()`; `render()` refactored to call this.

**Smoke test (`smoke_test_gates.py`) — Test 9 added (7 checks):**
- `extract_query_features` correctly identifies ABSORBING / GENERAL for `x*y=z` / `x*y=z*w`.
  Key edge case surfaced: `x*y=x` is STANDARD (rhs var `x` appears in lhs), not ABSORBING;
  ABSORBING requires the bare-var side to be entirely absent from the other expression.
- top-2 routing includes the absorbing case study, excludes singleton and standard_trivial.
- Decision tree always present in routed output.
- top-4 returns all 4 cases.

---

### Cluster-aware failure bins keyed by E1 structural form

**Problem:** Both `disagree_bin` and `both_wrong_bin` were single FIFO queues. When
they filled, the batch sent to case study generation was a structurally mixed bag —
absorbing-form failures alongside trivial-form and standard-form failures. The LLM then
had to find one pattern that explained a heterogeneous set, producing overly broad case
studies with wide trigger conditions. Those wide triggers then regressed on structural
sub-cases the case study was never meant to cover.

**Fix:** Replace each single bin with a `dict[str, Bin]` keyed by the E1 structural
form of the failing item (`TRIVIAL`, `SINGLETON`, `ABSORBING`, `STANDARD`, `GENERAL`).
Each cluster flushes independently, so the generator always receives a homogeneous batch
and can write a narrow, precise case study.

**Changes (`ICR_select/training/loop.py`):**
- `extract_query_features` added to the `utils.cheatsheet` import.
- `_cluster_key(item)` helper: calls `extract_query_features(item).form_e1`; falls back
  to `"GENERAL"` on parse error.
- `disagree_bin = DisagreementBin(...)` → `disagree_bins: dict[str, DisagreementBin] = {}`.
- `both_wrong_bin = FailureBin(...)` → `both_wrong_bins: dict[str, FailureBin] = {}`.
- Routing loop: `disagree_bins.setdefault(key, DisagreementBin(threshold)).add(item)`
  and equivalent for `both_wrong_bins`. Cluster key computed once per item.
- Per-batch flush loop: iterates `sorted(disagree_bins)` first (all full clusters), then
  `sorted(both_wrong_bins)` — but only when no disagree cluster is full (preserving the
  existing priority invariant across clusters).
- Remainder flush: each non-empty cluster flushed as its own homogeneous batch, disagree
  clusters before both-wrong clusters. No cross-cluster combining.
- Batch log line updated to show total disagree/both-wrong item counts and active cluster
  counts (`clusters=Nd+Mbw`).
- Routing log line shows the sorted cluster keys for each bin type.

No new CLI flags — clustering is always active and uses zero additional API calls
(`extract_query_features` is pure structural string parsing).

---

### Step parser: literal regex → structured checkpoint ID matching

**Problem:** `ICR_select/analysis/step_parser.py` detected which roadmap aspects a
failing model applied by scanning free-form reasoning text for `STEP N` / `RULE N`
strings. Models almost never write these literally — they write "the absorbing check",
"following the second aspect", or nothing at all. The regex silently missed all
paraphrased references, producing empty misapplication profiles and making the revision
loop unable to identify which aspects to fix.

**Fix — two-part change:**

`ICR_naive/prompts/templates.py` — `SCORING_PROMPT` and `SCORING_PROMPT_COT_FIRST`
updated to require structured checkpoint tags in the model's REASONING output:
- When applying ASPECT 1 the model must write `[CK:A1]`, ASPECT 2 → `[CK:A2]`, etc.
- `SCORING_PROMPT_COT_FIRST` instruction: *"For each ASPECT you consult, begin that
  clause with its checkpoint tag … This tagging is required — it is used to track
  which aspects are applied correctly vs. incorrectly."*
- `SCORING_PROMPT` adds a lighter version of the same instruction in the REASONING
  field description.
- The parser now reads tags the model was explicitly asked to emit, not tags it might
  coincidentally write.

`ICR_select/analysis/step_parser.py` — rewritten around exact tag matching:
- `extract_checkpoint_ids(roadmap_text)` replaces `extract_step_names`: parses
  `ASPECT N:` headers from the roadmap and returns `["A1", "A2", …]`. The old
  `extract_step_names` is kept as an alias.
- `mentions_in_trace(trace, checkpoint_ids)` now does exact `[CK:AN]` substring
  search — no regex guessing on free-form text. A paraphrased reference that lacks a
  tag is correctly reported as 0 matches, not silently missed and not falsely matched.
- `_best_quote(trace, checkpoint_id)` extracts the clause around the `[CK:AN]` tag
  using `str.find` instead of a pattern search.
- `build_profile`: items whose traces contain no checkpoint tags contribute to
  `n_no_trace` (same as before for empty traces) rather than producing spurious counts.
- `format_profile`: empty-profile message updated to name the likely cause ("ensure
  the scoring prompt emits [CK:AN] tags and --cot-first is active").
- `StepMisapplication.step_name` is now the checkpoint ID (`"A2"`); display label
  `"ASPECT 2 [A2]"` is constructed at render time.

Old-format roadmaps (STEP N / RULE N headers) return an empty checkpoint list and an
empty profile — the correct result, since they are not compatible with tag-based tracking.

---

### Roadmap synthesizer: controller over the case bank, not a replacement for it

**Motivation:** The roadmap synthesizer was designed to "absorb case studies into the
roadmap" — knowledge flowed one-way from the case bank into a monolithic roadmap text,
and after synthesis the case studies were cleared. This inverted the intended architecture:
the roadmap should be a lightweight navigation layer, not a knowledge dump. Inlining
case-level reasoning into the roadmap overrides fine-grained case guidance on every
query and ignores the routing already done by `render_for_query`.

The correct design: **roadmap as controller, cases as tools.** The roadmap tells the
student which structural dimension to probe and when to consult the case bank; the case
studies carry the detailed reasoning for each structural sub-case.

**Changes:**

`ICR_select/prompts/templates.py` — `ROADMAP_SYNTHESIS_PROMPT` rewritten:
- Framing changed from "synthesise case studies into a roadmap" to "write a routing
  controller that works alongside the case bank".
- Explicit anti-pattern example added: IF YES/IF NO branches must NOT give verdicts
  (`"E1 is absorbing — therefore E1 implies E2"` is wrong); they name the structural
  signal and route to the case bank (`"E1 has a fresh rhs variable — consult CASE BANK
  for absorbing patterns"`).
- `GROUNDED IN` field removed from the ASPECT format — it encouraged inlining case study
  conclusions into the roadmap. Replaced by `WATCH OUT` (misclassification traps the
  case bank catches).
- `ROADMAP_SYNTHESIS_MAX_TOKENS` 1 500 → 1 600.

`ICR_select/training/roadmap_synthesizer.py`:
- Module docstring updated: "controller over the case bank" architecture explained;
  explicit note that callers must NOT clear `case_studies` after synthesis.
- Validation: `cs_after` now keeps `case_studies=cheatsheet.case_studies` (was `[]`).
  Before/after accuracy comparison now measures whether the new roadmap improves
  navigation *with the same case bank*, which is the correct null hypothesis.
  Previously it was testing the roadmap in isolation — a strictly harder and wrong bar.
- `Returns` docstring updated: "caller must NOT clear case_studies".

`README.md` — project structure comment updated from "Absorbs case studies into a
structured reasoning roadmap" to "Synthesises a routing controller roadmap over the
case bank".

---

### Disagreement bin mining — oracle nearest-neighbour pairing (ICR_select)

**Motivation:** The training loop previously accumulated all student-wrong items into a
single failure bin with no discrimination. Items where the teacher (oracle) is also wrong
have no distillable signal — forcing a case study from them produces low-quality output.

**Changes:**

`utils/oracle_index.py` (new):
- `OracleEntry` NamedTuple: `eq1, eq2, reasoning, features`.
- `OracleIndex`: pre-computes `QueryFeatures` for every oracle entry at construction.
  `find_nearest(item)` returns the highest-Jaccard oracle entry whose token set overlaps
  the item's tokens, excluding exact-key matches (those are handled by the oracle exact
  lookup in the generator). Returns `None` when no entry exceeds `min_similarity`.

`utils/data.py`:
- `DisagreementBin` dataclass (mirrors `FailureBin`).  `add()` asserts `oracle_nearest`
  is present on the item — structural guard.

`ICR_select/training/loop.py`:
- New parameter `oracle_min_similarity: float = 0.25`.
- Builds `OracleIndex` at loop start if an oracle dict is provided.
- Wrong items are routed: if a nearest oracle entry is found → `oracle_nearest` and
  `oracle_sim` are attached, item goes to `disagree_bin`. Otherwise → `both_wrong_bin`.
- `disagree_bin` flushes first (teacher-signal priority). `both_wrong_bin` only flushes
  when `disagree_bin` is below threshold.
- `TrainingResult` gains `n_disagree` and `n_both_wrong` counters.
- Batch log updated to show both bin sizes.

`ICR_select/pipeline.py`:
- `--oracle-min-similarity F` flag added (default: 0.25).
- Stage 3 report shows `disagree_items` and `both_wrong_items`.

`ICR_reasoning/generators/case_study.py` (`_format_failures_with_reasoning`):
- Priority 1: exact oracle match — shows `CORRECT reasoning (oracle — exact same pair)`.
- Priority 2: `oracle_nearest` annotation — shows `STRUCTURALLY SIMILAR correct oracle
  case (sim=X.XX)` with the neighbour's `E1'`, `E2'`, and correct reasoning.

**Smoke test — Test 10 added (8 checks):** OracleIndex construction, `find_nearest`
match/similarity/to_dict, exact-key exclusion, no-crash on any input, item annotation,
CLI flag.

---

### Utility gate — continuous candidate scoring replacing hard thresholds

**Motivation:** The fix-rate and regression gates are binary — they accept or reject a
candidate by comparing a single scalar to a fixed threshold. This produces high false-reject
rates early in training (when the correct pool is small and regression estimates are noisy)
and gives no signal about *how much* a candidate improves things across the full val set.

**Design:** Replace with a continuous utility score per candidate:

```
U(c) = ΔAccS(Vmatch; c) + λ · ΔAccS(Vgap; c) − μ · RegressS(Veasy; c) − ν · chars(c) / 1000
```

Slice definitions:
- **Vmatch** — val items whose structural feature tokens overlap any candidate's
  `feature_signature` (pure string matching, no API call).
- **Vgap** — rolling reservoir (max 40) of disagree_bin items that were NOT generated
  from the current failure batch. Populated via reservoir sampling every time an item is
  routed to `disagree_bin`.
- **Veasy** — items from `correct_pool`. Measures stability.

API cost: **N+1** calls per flush (1 shared baseline without any candidate, then N
parallel with-candidate calls). This halves cost vs. naïve per-candidate scoring.

**New file `ICR_select/training/utility_gate.py`:**
- `UtilityConfig` dataclass (λ=0.5, μ=1.0, ν=0.1, threshold=0.0, min_slice=5).
- `UtilityResult` dataclass with all score components plus `fell_back` flag.
- `VGAP_RESERVE_MAX = 40` constant.
- `build_vmatch(candidate, val_items)` — pure structural matching, no API.
- `score_baseline(vmatch, vgap, veasy, ...)` — one shared API call.
- `score_utility_one(candidate, cheatsheet, ..., baseline, ...)` — one with-candidate call.
- `score_utility_batch(candidates, ...)` — N+1 calls total; parallel with-calls via
  `ThreadPoolExecutor`; graceful fallback (`fell_back=True`) when slices < `min_slice`.

**Changes to `ICR_select/training/loop.py`:**
- New parameters: `utility_gate: bool = False`, `utility_config: UtilityConfig | None`.
- `vgap_reserve: list[dict]` rolling reservoir; populated with reservoir sampling when
  items are routed to `disagree_bin`.
- `_process_flush`: when `utility_gate=True`, replaces mini-eval + fix-rate + regression
  gates with `score_utility_batch`. Picks best candidate by U; discards if U ≤ threshold.
  Falls back to classic gates when `fell_back=True`.
- `_process_flush_retry`: same logic; retry loop continues on utility threshold failures.
  Similarity gate and add/merge steps are identical on both paths.
- `TrainingResult` gains `n_utility_accepted` and `n_utility_fallbacks`.

**Changes to `ICR_select/pipeline.py`:**
- New CLI flags: `--utility-gate`, `--utility-lambda F`, `--utility-mu F`, `--utility-nu F`,
  `--utility-threshold F`, `--utility-min-slice N`.
- Constructs `UtilityConfig` and passes it as `utility_config` to `run_training_loop`.
- Stage 3 report shows `utility_accepted` and `utility_fallbacks`.

**Smoke test — Test 11 added (12 checks):** UtilityConfig defaults, accept when U=0.30 >
threshold=0.0, discard when U=-0.05 ≤ threshold=0.0, fallback to classic gates when
`fell_back=True` and `n_utility_fallbacks` incremented, CLI flags in `--help`.

---

## 2026-04-04 (commit fe6480b — pulled)

### ICR_select: Merge validation gate (loop.py, pipeline.py)

**What changed:**
- Added `_replace_eval()` helper: scores the failure batch with the merged CS replacing the existing entry at `merge_idx`, returning its fix rate.
- Added `validate_merge: bool = False` parameter to `run_training_loop()`. When enabled, a merge is only applied if the merged CS achieves at least the same fix rate on the current failures as the existing CS did. If the merge would hurt, the candidate is added as a new entry instead of being merged.
- `pipeline.py` wired up a `--validate-merge` flag to expose this option.

**Why it matters:** Previously, a MERGE decision would blindly combine two case studies via LLM without checking whether the result was actually better. The merged version could be worse than the original. This gate prevents regressions from merge operations.

**Files changed:** `ICR_select/training/loop.py`, `ICR_select/pipeline.py`, `README.md` (new)

---

## 2026-04-04

### ICR_select: Parallelize gating and maintenance operations (loop.py)

**Problem:** Each bin flush was slow because candidate mini-evals, ablation scoring, and condensation validation all ran sequentially.

**Changes:**
- Candidate mini-evals: all K candidates now evaluated in parallel via `ThreadPoolExecutor`. With K=3 this cuts mini-eval time by ~3×.
- Ablation pruning: all N leave-one-out score_batch calls now run in parallel. With 4 case studies this is ~4× faster.
- Condensation validation: the new and old cheatsheet are scored simultaneously instead of back-to-back.

**Expected impact:** Flush time roughly halved for a typical run. No change to logic or outputs.

---

### ICR_select smoke tests completed (runs/smoke_cs, runs/smoke_dt)

**Setup:** 100 normal-dataset items, NeuriCo DT as prior, gpt-oss-120b scoring, gpt-4o case study generation.

**smoke_cs results (CS-only, no DT revision):**
- 4 case studies added, all on STANDARD/GENERAL misclassification pattern
- Similarity gate did not catch near-duplicate entries — known issue
- Final cheatsheet eval: not separately recorded

**smoke_dt results (CS loop + DT revision, 2 rounds):**
- Round 1: 4 case studies added, 1 bin discarded (fix-rate gate, 33%). DT revision accepted (92%→91%).
- Round 2: 2 case studies added, 1 bin discarded (regression gate, 10.5%).
- Eval on normal (100 items, seed=77): **75%** vs baseline **80%** — ICR slightly worse, likely due to small training set overfitting.
- Eval on hard1 (69 items): **42%** vs baseline **41%** — essentially tied.

**Takeaways:**
- 100 training items is too few to see real improvement on unseen data.
- Case studies generated are too similar to each other — similarity gate needs tighter prompting.
- DT revision adds minor clarifications but not substantive fixes.

---

### Eval results summary

| Cheatsheet | Dataset | Score |
|---|---|---|
| NeuriCo baseline | Normal (100, seed 77) | 80% |
| ICR-Select smoke_dt | Normal (100, seed 77) | 75% |
| NeuriCo baseline | Hard1 (69 items) | 41% |
| ICR-Select smoke_dt | Hard1 (69 items) | 42% |

---

### ICR_select: New module built

Full ICR-Select pipeline with four quality gates:
- Candidate competition (K=3 parallel candidates)
- Fix-rate gate (≥50%)
- Regression gate (≤10% correct-pool regression)
- Similarity gate (LLM dedup: ADD/SKIP/MERGE)

Periodic maintenance:
- Ablation pruning (every 5 flushes)
- Condensation (when ≥6 case studies)

DT revision outer loop:
- Step parser extracts misapplied steps from post-think traces
- LLM analysis + targeted rewrite of broken steps only
- Validation: accept if accuracy drops ≤1 item

Files added:
- `ICR_select/__init__.py`
- `ICR_select/pipeline.py`
- `ICR_select/prompts/templates.py`
- `ICR_select/generators/case_study.py`
- `ICR_select/training/loop.py`
- `ICR_select/training/dt_reviser.py`
- `ICR_select/training/outer_loop.py`
- `ICR_select/analysis/step_parser.py`
- `compare_modes.sh`

---

### ICR_reasoning: Fixes and alignment

- Scorer: replaced custom verdict regex with SAIR's `parse_response`. Added `_normalize()` to strip `**BOLD**` markdown from headers before parsing. Changed default temperature 1.0 → 0.0 to match SAIR eval.
- LLM client: changed timeout from flat 180s to (10, 90) tuple. Added retry logging.
- Pipeline: added `--no-dt-patch`, `--cot-first`, `--limit` flags.

---

### ICR_naive: Scorer alignment

- Same SAIR parser + markdown normalizer as ICR_reasoning.
- Temperature default changed to 0.0.

---

### Proposal: ICR_proposal_v5.tex

Standalone proposal (not a revision response). Key sections:
- Prior knowledge initialization (NeuriCo DT as D0)
- Post-think distillation instead of full CoT (citing Heddaya et al. ACL 2026)
- Four-gate selective update pipeline
- DT revision outer loop
- Outcome analysis section (accuracy vs preprocessing cost comparison)
- Honest implementation status — current code is a first pass with known gaps
- 7-week timeline

---

## 2026-04-04 (continued)

### Case study format: IDENTIFY + ACTION + DOES NOT APPLY TO

**Problem:** Generated case studies were writing broad rules like "IF E1 is GENERAL and E2 is GENERAL → lean FALSE" which fired on too many cases and caused heavy regressions (up to 57% of correct items broken). The format gave no guard condition — the model applied rules everywhere.

**Changes:**

`ICR_reasoning/prompts/templates.py` — rewrote `CASE_STUDY_WITH_REASONING_PROMPT`:
- Replaced `PATTERN / RULE / WHY / EXAMPLES / EXCEPTIONS` format with `IDENTIFY / ACTION / WHY / EXAMPLES / DOES NOT APPLY TO`
- `IDENTIFY` is now a precise checklist of structural conditions that must ALL be true before the rule fires (equation forms, variable counts, nesting depth, left/right structure)
- `DOES NOT APPLY TO` explicitly states boundary cases — similar-looking pairs where the rule should not fire
- Prompt now explicitly instructs: "a very narrow rule that fires on 2-3 cases correctly is better than a broad rule that fires on 10 and gets half wrong"

`ICR_select/prompts/templates.py` — updated `SIMILARITY_CHECK_PROMPT` and `MERGE_PROMPT`:
- Similarity gate now compares IDENTIFY conditions structurally, not surface text — better duplicate detection
- Merge prompt combines IDENTIFY checklists from both entries rather than merging prose rules

**Expected impact:** Case studies should be narrower and less likely to cause regressions on unseen data.

---

### Hard1 round 1 results (runs/hard1_select)

- Train accuracy: 59.4% (vs 41% baseline) — improvement on training data
- Case studies added: 0 — every bin discarded by regression gate (rates 23–57%)
- DT revision: accepted, 52.2% → 55.1% (+3 points)
- Root cause: 10% regression threshold too strict for hard1's diverse failure patterns; any case study that fixes some hard1 failures tends to confuse the model on others

---

## 2026-04-04 (continued)

### Oracle reasoning injection for case study generation

**Problem:** Case study generator only sees the wrong model reasoning (post-think) — no signal for what the correct approach looks like.

**Changes:**

`ICR_reasoning/core/oracle.py` (new):
- Loads GPT-5.4 correct reasoning traces from `gpt5.4_normal_default.csv`
- Keyed by `(equation1, equation2)` tuple — 200 correct entries, 13 incorrect skipped
- Exposes `load_oracle_csv(path) -> OracleDict`

`ICR_reasoning/generators/case_study.py`:
- `_format_failures_with_reasoning()` now accepts optional `oracle` dict; appends `CORRECT reasoning (oracle): ...` under each failure that has a match
- `generate_case_study_with_reasoning()` accepts and passes `oracle` through; logs how many failures have oracle contrast

`ICR_select/generators/case_study.py`:
- `generate_candidates()` accepts and passes `oracle` to the formatter

`ICR_select/training/loop.py`:
- `run_training_loop()` accepts `oracle: OracleDict | None` and passes it to `generate_candidates()`

`ICR_select/pipeline.py`:
- `--oracle-csv` flag added; oracle is loaded and forwarded through `inner_kwargs` to every training round

`ICR_reasoning/prompts/templates.py`:
- Updated case study prompt to explain the oracle contrast signal and instruct the generator to use correct reasoning rather than following the wrong reasoning

**Eval tool:** `eval_oracle_quality.py` (new) — generates one case study with and without oracle from a real failure bin, scores both against the failures, prints fix rates and winner. Saves state to `runs/oracle_eval/bin_state.json` for replay without regeneration (`--from-bin` flag).

**Level 1 result (3 failures, normal dataset, gpt-5.4 generator):**
- WITHOUT oracle: broad syntactic rule mixing two unrelated patterns (comb-tail + fake-substitution)
- WITH oracle: algebraic collapse mechanism identified correctly (`a*b=f(a)`, `f³=id`; x-irrelevance → constant), WHY grounded in actual algebra
- Qualitatively better; fix rate on failure bin: **TIE** — 3-failure bin too small to differentiate
- Next eval should use `--bin-size 8` or larger for a more meaningful signal

---

## 2026-04-05

### vLLM backend support for ICRefine pipelines

**What changed:**

`ICR_naive/core/llm_client.py` and `ICR_reasoning/core/llm_client.py`:
- Added vLLM routing via `VLLM_BASE_URL` and `VLLM_MODEL` env vars. When both are set and the requested model matches `VLLM_MODEL`, calls are routed to the local vLLM server instead of OpenRouter/OpenAI.
- `reasoning` payload parameter is skipped for vLLM (not supported).
- `Authorization` header is now only added when a key is present (vLLM often runs without auth).
- `ICR_reasoning` client: vLLM responses read `reasoning_content` for thinking tokens (vLLM's field name) instead of `reasoning` (OpenRouter's field name).
- Read timeout increased from 90s → 300s to accommodate reasoning model latency.

`ICR_naive/training/scorer.py` and `ICR_reasoning/training/scorer.py`:
- Fixed hardcoded `SAIR_evaluation_pipeline` path → `SAIR_eval_pipeline` to match actual directory name.

`ICR_select/pipeline.py`:
- Fixed hardcoded `SAIR_evaluation_pipeline` path in `load_dotenv` call → `SAIR_eval_pipeline`.

`ICRefine/.env`:
- Added `VLLM_BASE_URL`, `VLLM_API_KEY`, and `VLLM_MODEL` entries.

`SAIR_eval_pipeline/models/complete_model_list.csv` and `model_activated.csv`:
- Added `qwen/qwen2.5-7b-instruct` (local vLLM entry, later superseded by DeepSeek-R1-14B).

**Why it matters:** Enables using locally-hosted open-source models on the UChicago DSI cluster via vLLM as a drop-in replacement for OpenRouter, with per-model routing so cloud models (gpt-4o) and local models (deepseek-r1-14b) can be mixed in the same run.

**Cluster setup:**
- Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` downloaded to `/net/scratch/renqy/DeepSeek-R1-Distill-Qwen-14B`
- vLLM env: `/net/scratch/renqy/vllm-env` (Python 3.11, vLLM 0.19.0, CUDA 12.1)
- Served as `deepseek-r1-14b` on port 8000 via Slurm general partition (H100 node)
- Access via SSH tunnel: `ssh -L 8000:<node>:8000 dsi-cluster`

---

## 2026-04-06

### ICR_select: Prescore reuse, vLLM parse fix, scoring token cap, similarity gate guard

**What changed:**

`ICR_select/training/loop.py`:
- Added `_apply_prescore()` — splits a batch into correct/wrong using pre-computed SAIR eval scores instead of calling the model, avoiding a redundant scoring pass when running inside the recursive refinement pipeline.
- Added `prescore_map: dict | None = None` parameter to `run_training_loop()`. When provided, the first pass uses prescore results; subsequent passes (after cheatsheet updates) still call the model.
- Added `_MIN_CS_FOR_SIMILARITY = 3` guard — similarity gate now skips until at least 3 case studies exist. Previously it fired on iterations 1–2 when the cheatsheet had 0–1 case studies, wasting an API call.

`ICR_select/pipeline.py`:
- Added `--prescore-file FILE` CLI argument. Loads a JSON dict of `{id: {predicted, correct, post_think, thinking, raw_response}}` and passes it to `run_training_loop` as `prescore_map`.

`ICR_reasoning/core/llm_client.py`:
- Added vLLM fallback: if `content` is empty but `reasoning_content` has data, use reasoning_content as content. Prevents parse errors when DeepSeek-R1-14B exhausts its token budget in the thinking pass before writing the structured `VERDICT:` line.

`ICR_naive/prompts/templates.py`:
- Reduced `SCORING_MAX_TOKENS` from 16K → 8K. 16K was enough for DeepSeek-R1-14B to think exhaustively and then run out of budget before writing the answer. 8K is sufficient for both thinking and structured output.

`.gitignore`:
- Added `restart_vllm.sh` — local cluster script, not tracked.

**Why it matters:** When running ICR_select inside the SAIR recursive refinement loop, the initial SAIR eval already scores all 200 items. Feeding those results as a prescore map eliminates the duplicate scoring pass (~40 min saved per iteration at concurrency=8). The parse fix and token cap address DeepSeek-R1-14B's tendency to use all tokens for thinking.

---

## 2026-04-06 (continued)

### ICR_select: Retry flush strategy with previous candidate context

**What changed:**

`ICR_select/training/loop.py`:
- Added `_mini_eval_full()` — same as `_mini_eval` but returns `(fix_rate, still_wrong_items)`. Used by the retry path to get the still-wrong items for free without an extra API call.
- Added `_process_flush_retry()` — a new flush function that retries up to `candidate_rounds` times before discarding a bin. On each retry, the previous candidate text and its still-wrong items are passed to `generate_candidates` as context so the model knows what was tried and what it missed. SKIP from the similarity gate still causes an immediate discard (retrying a duplicate is pointless).
- Added `candidate_rounds: int = 3` and `flush_strategy: str = "default"` parameters to `run_training_loop()`. The existing `_process_flush` (default path) is completely untouched.

`ICR_select/generators/case_study.py`:
- Added `prev_attempt: dict | None = None` parameter to `generate_candidates()`. When provided, appends a `RETRY_CONTEXT_TEMPLATE` section to the prompt with the rejected candidate, why it failed, and the items still wrong after applying it.

`ICR_select/prompts/templates.py`:
- Added `RETRY_CONTEXT_TEMPLATE` — injected on retries to tell the model what was tried, why it failed, and what items remain wrong.

`ICR_select/pipeline.py`:
- Added `--flush-strategy {default,retry}` and `--candidate-rounds N` CLI flags, wired into `inner_kwargs`.

`SAIR_eval_pipeline/recursive_refine/config.py`, `run_recursive_refine.py`, `updater.py`:
- Added `--icr-flush-strategy` CLI arg that forwards to `ICR_SELECT_FLUSH_STRATEGY` env var, picked up by the updater and passed to ICR_select.

**Why it matters:** Previously, if all N candidates failed any gate, the entire failure bin was discarded. With `--flush-strategy retry`, the pipeline tries up to 3 rounds of fresh candidates, each time showing the model what the previous attempt got wrong — giving it a direct signal to write a narrower or different rule.

---

## 2026-04-07

### ICR_select: Reasoning roadmap synthesizer + prior_knowledge architecture

**What changed:**

`ICR_naive/core/cheatsheet.py`:
- Added `prior_knowledge: str = ""` field to `Cheatsheet`. Rendered before the DT/roadmap section under `=== PRIOR KNOWLEDGE ===` header.
- Added `PRIOR_KNOWLEDGE_HEADER` and `ROADMAP_HEADER` constants.
- `render()` detects roadmap vs DT header and prepends prior_knowledge section if non-empty.
- `save()` / `load()` persist `prior_knowledge` in the JSON sidecar.

`ICR_select/pipeline.py`:
- Renamed `--init-txt` → `--init-roadmap` (clearer intent: initializes the trainable roadmap/DT).
- Added `--prior-knowledge FILE` argument: loads frozen content (e.g. NeuriCo prompt) into `Cheatsheet.prior_knowledge`.
- Added `elif prior_knowledge:` init branch: when only `--prior-knowledge` is given (no `--init-roadmap`), starts with an empty trainable roadmap.

`ICR_select/prompts/templates.py`:
- Added `ROADMAP_SYNTHESIS_PROMPT` and `ROADMAP_SYNTHESIS_MAX_TOKENS = 1_500`.
- Synthesis prompt uses ASPECTS structure with mechanical checkpoints: each CHECK must be answerable by direct inspection (counting, string matching, set membership) — no reasoning or judgment allowed.

`ICR_select/training/roadmap_synthesizer.py` (new):
- `run_roadmap_synthesis()`: absorbs accumulated case studies into a structured REASONING ROADMAP.
- Validates by re-scoring `train_seen` before/after; accepts if `delta >= -regress_tolerance` (default 10%).
- Returns `RoadmapSynthesisResult(accepted, roadmap, accuracy_before, accuracy_after, n_case_studies_used)`.

`ICR_select/training/dt_reviser.py`:
- Changed fixed `-1 item` acceptance threshold to percentage-based: `delta >= -regress_tolerance` (default `regress_tolerance=0.10`). More consistent across different training set sizes.

`SAIR_eval_pipeline/recursive_refine/updater.py`:
- Added `@register("icr_roadmap")` updater: Phase 1 = ICR_select (collect case studies), Phase 2 = roadmap synthesis (absorb into roadmap, clear case studies).
- Fixed cheatsheet routing: **JSON cheatsheet** → `--init-cheatsheet`; **plain-text cheatsheet** (e.g. NeuriCo on iteration 0) → `--prior-knowledge` (frozen, trainable roadmap starts empty). Previously plain text was passed as `--init-txt` which put all NeuriCo content into the trainable `decision_tree`.
- Fixed truncation: reads `cheatsheet_final.json` instead of `cheatsheet_refined.txt` (render() is capped at 2,500 chars).
- Fixed prescore carry-forward: when eval is skipped (`eval_first_and_last`), loads prescore from `_last_eval_run_dir` in stats_doc to avoid falling back to the full 1,000-item dataset.
- Added `--limit` forwarding to ICR_select subprocess.
- Always passes `--flush-strategy` explicitly.

`SAIR_eval_pipeline/recursive_refine/runner.py`:
- Tracks `last_eval_run_dir` per iteration; injects `_last_eval_run_dir` into stats_doc when eval is skipped.
- Resume (`_resume()`) restores `last_eval_run_dir`; prefers `cheatsheet_final.json` over `cheatsheet_refined.txt` when loading the checkpoint cheatsheet.

`SAIR_eval_pipeline/recursive_refine/config.py`:
- `--icr-regress-threshold` default: 0.15 → 0.20 (less aggressive rejection when correct pool is small).
- `--icr-flush-strategy` default: `"default"` → `"retry"`.

`SAIR_eval_pipeline/run_recursive_refine.py`:
- Sets `ICR_SELECT_REGRESS_THRESHOLD`, `ICR_SELECT_MAX_ITERATIONS`, `ICR_SELECT_FLUSH_STRATEGY`, `ICR_SELECT_LIMIT` env vars from CLI args.

**Why it matters:** Addresses the three main ICR inefficiency problems identified from the cluster runs:
1. Truncation between iterations caused case studies to be silently dropped.
2. Skipped-eval iterations fell back to 1,000 items instead of the ~200-item prescore set.
3. The regression gate was rejecting nearly all candidates because 15% of a 40-item pool is only 6 items — one flip over the threshold killed the candidate.

The roadmap synthesizer provides an alternative to growing case study lists: after enough evidence accumulates, the pipeline distills it into a structured reasoning guide with mechanical, inspectable checkpoints.

---

## 2026-04-07 (continued)

### ICRefine: Standalone project — removed runtime dependency on SAIR_eval_pipeline

**What changed:**

`ICR_naive/core/parser.py` (new):
- Self-contained copy of the response parser previously imported from `SAIR_eval_pipeline/pipeline/parser.py`.
- Exports: `parse_response()`, `compute_correct()`, `normalize()`, `_extract_section()`.
- `normalize()` strips markdown bold/italic (`**text**`, `*text*`) from headers before parsing — previously `_normalize()` in each scorer.

`ICR_naive/training/scorer.py` and `ICR_reasoning/training/scorer.py`:
- Replaced `sys.path` hack that pointed at `SAIR_eval_pipeline/` with a direct import: `from ..core.parser import parse_response as _sair_parse, normalize as _normalize` (naive) and `from ICR_naive.core.parser import ...` (reasoning).
- Removed the `_MD_BOLD_RE` regex and `_normalize()` definition from both files (now in parser.py).

`ICR_naive/core/llm_client.py` and `ICR_reasoning/core/llm_client.py`:
- Removed `load_dotenv(... / "SAIR_evaluation_pipeline" / ".env")` fallback. Each client now loads only its own `.env` file.

`ICR_naive/pipeline.py`, `ICR_reasoning/pipeline.py`, `ICR_select/pipeline.py`, `ICR_naive/generators/initial.py`, `eval_oracle_quality.py`:
- Removed `load_dotenv` calls that referenced `SAIR_evaluation_pipeline/` paths.

`compare_modes.sh`:
- Removed `SAIR_DIR` variable and the `run_eval()` function (which called `SAIR_eval_pipeline/run_evaluation.py`).
- `DATASET` and `BASE_CHEATSHEET` are now top-level config variables with placeholder values.
- Output cheatsheets written to `runs/compare_*/cheatsheet_final.txt` instead of `SAIR_eval_pipeline/prompts/`.
- Removed `eval` mode (evaluation is external to ICRefine).

`README.md`:
- Removed "must sit next to SAIR_eval_pipeline" requirement from the intro.
- Removed "Set up SAIR_eval_pipeline first" from Quick Start.
- All dataset/cheatsheet path examples use generic `path/to/dataset.jsonl` and `path/to/prior_knowledge.txt`.
- Removed references to `SAIR_eval_pipeline/` throughout modes, examples, and Comparing Modes section.

All pipeline docstrings updated to use generic paths.

**Why it matters:** ICRefine can now be cloned and used independently. The only inputs are a dataset (`.jsonl`) and an optional prior-knowledge file; no sibling repository is required.

---

## 2026-04-07 (continued)

### SAIR_eval_pipeline: Dead-code cleanup

**What changed:**

`pipeline/config.py`:
- Removed `VLLM_DEFAULT_BASE_URL` constant — defined but never read (actual URL comes from `VLLM_BASE_URL` env var).
- Changed hardcoded `choices=["low", "medium", "high"]` in `build_parser()` → `choices=list(REASONING_EFFORT_LEVELS)` so the constant is the single source of truth.

`pipeline/results.py`:
- Removed `result_path()` and `is_done()` — both defined but never called anywhere in the project.
- Fixed return type annotation on `write_run_summary()`: `-> tuple[Path, Path]` → `-> tuple[Path, Path, dict]` (the function already returned 3 values).
- Removed stale `is_done(...)` line from module docstring.

`recursive_refine/updater.py`:
- Removed dead `from ICR_reasoning.core.llm_client import call_llm` import in `_icr_roadmap_updater` — imported but never called directly (used only inside `run_roadmap_synthesis`).
- Updated `ICR_SELECT_DATASET` docstring entry to accurately describe its fallback-only behavior.

---

## 2026-04-07 (continued)

### ICRefine: Shared modules moved to `utils/`

**What changed:**

`utils/` (new package):
- `utils/cheatsheet.py` — canonical home for the `Cheatsheet` dataclass and associated constants.
- `utils/data.py` — canonical home for `is_true`, `FailureBin`, `load_jsonl`, `sample_instances`, `split_dataset`.
- `utils/parser.py` — canonical home for `parse_response`, `split_case_studies`, `normalize`, `compute_correct`.
- `utils/llm_client.py` — canonical home for `LLMResponse`, `call_llm`, `call_llm_batch`, `get_api_key`. `load_dotenv` path adjusted for new depth (`parent.parent` instead of `parent.parent.parent`).
- `utils/scorer.py` — canonical home for `score_batch`, `test_cheatsheet`, `TestResult`. Internal imports converted to relative (`.data`, `.llm_client`, `.parser`).

Old locations (`ICR_naive/core/{cheatsheet,data,parser,llm_client}.py`, `ICR_naive/training/scorer.py`, `ICR_reasoning/core/llm_client.py`, `ICR_reasoning/training/scorer.py`) reduced to one-line shims that re-export from `utils.*` — backward compatibility for SAIR and any external callers is preserved.

All cross-package imports updated to use `utils.*` directly:
- `ICR_reasoning/generators/case_study.py`, `training/loop.py`, `pipeline.py`
- `ICR_select/pipeline.py`, `generators/case_study.py`, `training/loop.py`, `gates.py`, `maintenance.py`, `dt_reviser.py`, `outer_loop.py`, `roadmap_synthesizer.py`
- `eval_oracle_quality.py`

**Why it matters:** Five modules were shared across all three ICR packages but lived inside `ICR_naive/`, making cross-package imports confusing (`ICR_select` importing from `ICR_naive`). Moving them to a neutral `utils/` package makes the dependency direction explicit and eliminates the chain of shims from `ICR_reasoning` → `ICR_naive`.

---

## 2026-04-22

### ICR_adaptive: New domain-agnostic cheatsheet refinement mode

A new refinement mode `ICR_adaptive` was added alongside `ICR_naive`, `ICR_reasoning`, and `ICR_select`. Unlike the others, `ICR_adaptive` makes no assumptions about the case study format, protocol structure, or scoring model — everything is injected via two config objects.

**Design:**

`ICR_adaptive/config.py` — two dataclasses:
- `TaskConfig`: domain_description, input_fields, answer_field, verdict_pattern, answer_map, truncation_token_threshold. `base_partition_key(item)` and `query_features(item)` for bin keying and routing. `validate()` raises on blank description.
- `PipelineConfig`: scoring_models, generator_model, fix_rate_threshold, regress_threshold, regress_relative_factor, min_pool_for_regression, bin_threshold, n_candidates, utility_threshold, lam, mu, nu. `primary_model()` returns `scoring_models[0]`. `adaptive_regress_limit(pool_size, fix_count)` = `max(ceil(regress_threshold×pool_size), ceil(fix_count×relative_factor))`.

**Components** (`ICR_adaptive/components/`):

- `format_filter.py` — `FormatFilter.check()` classifies each response as empty/truncated/no_verdict/ok.
- `failure_classifier.py` — `FailureClassifier.classify()` returns `FailureType` (CORRECT / WRONG_ANSWER / ABANDONMENT / FORMAT).
- `execution_parser.py` — `ExecutionPathParser.parse()` finds the last `[STEP:]`/`[RULE:]` tag seen in the response; with oracle trace finds the first step present in oracle but absent in model response.
- `generator_router.py` — `GeneratorRouter.route()` finds the most relevant case bank entry via token Jaccard similarity.
- `multi_model_scorer.py` — `MultiModelScorer.score()` runs score_fn across all scoring models; `ScorerResult.all_pass_fix_rate()` and `.any_regresses()` for gate checks.

**Training** (`ICR_adaptive/training/`):

- `adaptive_gate.py` — `AdaptiveRegressionGate.evaluate()` runs three sequential gates: fix_rate → regression → utility. Utility = lam×Vgap − mu×regress_cost − nu×length_penalty. Returns `GateResult(passed, reason, utility, fix_rate, regress_count)`.
- `loop.py` — `AdaptiveTrainingLoop.run()`: score → bin failures (skip FORMAT) → pick largest eligible bin ≥ threshold → generate N candidates → gate each → accept best → repeat. `IterationResult` tracks acc_before/acc_after for the summary table. Bins are logged with sizes; every gate decision logged at INFO.

**Prompts** (`ICR_adaptive/prompts/`):

- `strategies.py` — `build_prompt(ctx, strategy)` dispatches to DIRECT_FIX / ORACLE_GUIDED / CONTRAST. All strategies include `_FORMAT_BLOCK` requiring plain text + `[STEP:]`/`[RULE:]` annotations + two format examples (TRUE motif case, FALSE counterexample case). Both examples show `[RULE:]` tags so the generator knows to annotate both proof and counterexample paths.

**Facade:**

- `pipeline.py` — `AdaptivePipeline(task_cfg, pipe_cfg, score_fn, generate_fn)` validates both configs then delegates to `AdaptiveTrainingLoop`.

**Tests:** 56 smoke tests across 10 test files (one per component), all passing.

**Demo run script:** `run_adaptive_hard1.py` (project root) — runs the adaptive pipeline on `hard1` (69 items) with GPT-OSS-120B only and a blank initial cheatsheet. Async concurrency=20 via `httpx.AsyncClient` + `asyncio.gather`; sync/async bridge via `ThreadPoolExecutor(max_workers=1) + asyncio.run()`. Intermediate accuracy printed after each scoring pass. Final clean scoring pass and per-iteration summary table printed on completion.

---

### ICR_adaptive: Bug fixes from first two demo runs

Three bugs were identified from demo runs on `hard1` and fixed:

**1. Equation hallucination in case study generation (`strategies.py`)**

Problem: The generator (GPT-OSS-120B) ignored the sample item's actual equations and substituted a different, easier problem it found familiar. The gate did not catch this because it only checks accuracy on the training set, not whether the case covers the correct equations. The hallucinated case study was accepted and added to the cheatsheet with wrong equations.

Fix: Added a `CRITICAL CONSTRAINT` block to `_direct_fix()` that explicitly names both equations and the expected answer at the top of the prompt, and mandates that the first step must be `[STEP: parse_equations]` listing those exact equations. The constraint appears before the failure details so the model sees it before writing anything.

**2. Duplicate bin targeting — same bin accepted multiple times (`loop.py`)**

Problem: The loop had no memory of which bins had already been addressed. After a candidate was accepted for a bin, subsequent iterations still re-targeted the same bin (same failure type + divergence step + structural key) once the bin still had failures, producing redundant and often near-duplicate case studies.

Fix: Added `completed_bins: set` in `AdaptiveTrainingLoop.run()`. When selecting the target bin, only bins not in `completed_bins` are eligible. When a candidate is accepted, `completed_bins.add(target_key)` marks that bin as done. The loop stops early when all bins are either addressed or below threshold, with a log message "All failure bins already addressed".

**3. Sparse `[RULE:]` annotations in FALSE / counterexample cases (`strategies.py`)**

Problem: The existing `_FORMAT_BLOCK` contained only a TRUE/motif-rule example. Generated case studies for FALSE items had zero `[RULE:]` markers because the model had no example showing how to annotate counterexample-construction steps. This made the execution parser unable to identify divergence points in FALSE failures.

Fix: Added a second format example (FORMAT EXAMPLE B) to `_FORMAT_BLOCK` showing a complete FALSE case with `[RULE:]` tags on every check: probe passes, refutation rules, and counterexample confirmation. Added an explicit note: "Every step that checks or applies a rule MUST emit at least one [RULE:] tag."

---

## Next Steps

- Evaluate `runs/bbh_v3/` results on test splits once `run_bbh_v3.sh` completes; compare to v2 and CS-ICL
- Run the full v3 pipeline on remaining tasks (`web_of_lies`, `navigate`, `date_understanding`, `logical_deduction_three`, `snarks`) to build a complete comparison table
- Re-run `boolean_expressions` with auto-bootstrap enabled to test whether seeded prior knowledge improves on both anti-overfit and v2 results
- Run per-segment ablations on formal_fallacies bootstrap cheat sheet to identify which sections drive accuracy
- Rerun MAGMA comparison with `openai/gpt-4.1` (gen/patch/casestudy) to measure model quality impact on the CS-ICL vs ICR gap
- Run ICR_adaptive on hard1 after the three bug fixes to measure impact
