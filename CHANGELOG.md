# ICRefine Changelog

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

## Next Steps

- Run full recursive refinement pipeline (5 iterations, 200 items, eval-first-and-last) with oracle CSV
- Fix similarity gate to catch semantic duplicates with different wording
- Consider tighter DT revision prompt that makes substantive changes rather than adding clarifying notes
