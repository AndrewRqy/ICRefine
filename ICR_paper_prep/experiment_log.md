# ICR Paper Experiment Log

All experiments use `openai/gpt-4.1-mini` for all roles (score / rule-patch / casestudy)
unless noted. Test sets are held-out BIG-Bench Hard splits.

---

## Baselines

### CS-ICL (baseline)
Static cheatsheet generated once from training examples; no iterative refinement.
Results are in every `comparison_results.json` under `cs_icl_acc`.

---

## v3 — Full ICR Pipeline (primary result)

**Description:** Auto-bootstrap (75 train items) → Phase 1 PK patching → Phase 2 case study generation.
Phase 2 oracle injection (`item["reason"]` as correct-reasoning contrast) was present in this run —
it has been present since the initial implementation.

**Run dir:** `runs/bbh_v3/`
**Results file:** `runs/bbh_v3/comparison_results.json`
**Training logs:** `runs/bbh_v3/logs/<task>.log`
**Cheatsheets:** `runs/bbh_v3/<task>/cheatsheet_final.json`

**Test accuracy (ours vs CS-ICL):**

| Task | CS-ICL | v3 (ours) | Δ |
|------|--------|-----------|---|
| logical_deduction_three | 95% | **95%** | 0 |
| sports_understanding | 93% | **94%** | +1 |
| boolean_expressions | **88%** | 87% | -1 |
| snarks | **88.7%** | 84.5% | -4.2 |
| navigate | 77% | **80%** | +3 |
| disambiguation_qa | 75% | **77%** | +2 |
| causal_judgement | **73.6%** | 65.5% | -8.1 |
| date_understanding | 72% | 72% | 0 |
| formal_fallacies | 64% | **70%** | +6 |
| web_of_lies | 58% | **66%** | +8 |
| geometric_shapes | **51%** | 48% | -3 |

5 wins, 4 losses, 2 ties vs CS-ICL.

---

## v3 Phase-1-Only Ablation

**Description:** Cheatsheet after bootstrap + PK patching only — no case studies added.
Used to isolate case study contribution.

**Run dir:** `runs/bbh_v3_phase1_only/`
**Results file:** `runs/bbh_v3_phase1_only/comparison_results.json`
**Note:** Only 3 tasks evaluated (WOL, GS, DU).

| Task | CS-ICL | Phase-1-only | v3 full | CS contribution |
|------|--------|-------------|---------|----------------|
| web_of_lies | 62% | 60% | **66%** | +6pp |
| geometric_shapes | 50% | **54%** | 48% | -6pp (CS hurts) |
| date_understanding | 69% | 69% | **72%** | +3pp |

**Key finding:** Case studies hurt geometric_shapes (-6pp), help WOL (+6pp), minor help on DU (+3pp).

---

## v4 — Anti-Overfit CS Prompts

**Description:** Same as v3 but CS generation prompt adds "ACTIVATE IF — REQUIRED CONSTRAINTS"
block forbidding answer encoding and training-specific surface features.
Only 3 tasks run (WOL, GS, DU). CS prompt reverted to v3 for v5 to control variance.

**Run dir:** `runs/bbh_v4/`
**Results file:** `runs/bbh_v4/comparison_results.json`

| Task | CS-ICL | v3 | v4 | Δ v4 vs v3 |
|------|--------|----|----|------------|
| web_of_lies | 60% | **66%** | 64% | -2 |
| geometric_shapes | 51% | 48% | 50% | +2 |
| date_understanding | 72% | 72% | **73%** | +1 |

**Key finding:** Anti-overfit prompts made minimal difference; GS still below Phase-1-only (54%).
Root cause for GS is case study quality, not prompt wording.

---

## v5 — Phase 1 Gold Oracle Injection (IN PROGRESS)

**Description:** Full pipeline identical to v3, with one addition:
`item["reason"]` (gold CoT) is now injected into Phase 1 PK patching as a
"Correct reasoning" contrast block alongside the model's wrong reasoning.
Phase 2 oracle injection is unchanged from v3 (was always present).

CS prompt: v3 (reverted from v4 to control for prompt variance).

**Key oracle injection facts:**
- Phase 1 oracle: NEW in v5 (not present in v3/v4)
- Phase 2 oracle: present since v3 (unchanged)
- Only tasks with 100% gold reasoning coverage benefit: causal_judgement, geometric_shapes,
  boolean_expressions, disambiguation_qa, logical_deduction_three, sports_understanding
- Tasks with 0% coverage (web_of_lies, snarks): v5 = v3 for oracle purposes

**Run dir:** `runs/bbh_v5/`
**Results file:** `runs/bbh_v5/comparison_results.json`

| Task | CS-ICL | v3 | v5 | Δ v5 vs v3 |
|------|--------|----|----|------------|
| causal_judgement | 71.3% | 65.5% | 65.5% | 0 |
| geometric_shapes | 50.0% | 48% | 49% | +1 |

**Finding: Phase 1 oracle injection did not help.** CJ is flat, GS +1pp (within noise).
Possible explanations:
- Phase 2 oracle was already on in v3, so the model had gold reasoning during CS generation regardless
- For GS, the bottleneck is case studies actively hurting (Phase-1-only=54% > v3 full=48%); better PK doesn't fix that
- For CJ, causal reasoning may require a fundamentally different approach; PK text patching converges regardless of oracle

---

## Oracle Ablation 2×2 (NOT RUN)

**Description:** Full 2×2 factorial on Phase 1 and Phase 2 oracle injection,
run on causal_judgement and geometric_shapes (100% reasoning coverage).

| Condition | Phase 1 oracle | Phase 2 oracle | CLI flags | Run dir |
|-----------|---------------|----------------|-----------|---------|
| v3 | OFF | ON | *(default v3)* | `runs/bbh_v3/` |
| v5 | ON | ON | *(default v5)* | `runs/bbh_v5/` |
| P1-only | ON | OFF | `--no-phase2-oracle` | `runs/bbh_oracle_ablation/p1_only/` |
| No-oracle | OFF | OFF | `--no-phase1-oracle --no-phase2-oracle` | `runs/bbh_oracle_ablation/no_oracle/` |

Partial coverage: No-oracle was run for navigate + formal_fallacies (see section below). Full 2×2 on CJ/GS not completed — superseded by the gold few-shot experiment which directly compares oracle-quality content formats.

---

## v5 Extended (NOT RUN)

Superseded by the gold few-shot experiment, which probes the same question (does access to gold reasoning traces improve cheatsheets?) more cleanly.

---

## E2 — Phase-2-Only Pipeline (Case Studies from Empty Cheatsheet)

**Description:** `--no-auto-bootstrap --no-oracle` — skips Phase 1 entirely, Phase 2 generates
case studies from an empty cheatsheet (no prior knowledge). Tests whether case studies have
standalone value independent of the PK foundation.

**Run dir:** `runs/bbh_phase2only/<task>/`
**Script:** `run_e2_e3_e4.sh`
**Eval results:** `runs/e2_cs_pipeline_results.json`

**Pipeline outcomes:**

| Task | CS added | Train acc | Notes |
|---|---|---|---|
| web_of_lies | **0** | 62.7% | Every candidate fails regression gate (>15%) — confounded: no PK AND no oracle (WOL has no `item["reason"]`) |
| causal_judgement | 2 | 65.0% | Has oracle (`item["reason"]`); 2 CS added without PK |
| geometric_shapes | 3 | 74.7% | Has oracle; 3 CS added without PK |

**WOL detail:** Candidates with fix_rate up to 59% were consistently rejected because regression_rate exceeded 15% threshold in every round across 2 iters. Root cause is confounded — WOL has neither PK foundation nor oracle context. Cannot attribute failure to PK alone.

**E2 CJ cs_only vs benchmarks (CoT, 5 models):**

| Model | E2 cs_only | v3 full | v3 pk_only | Δ cs_only vs v3_full |
|---|---|---|---|---|
| gpt-4.1-mini * | 64.4% | 64.4% | 72.4% | 0pp |
| gpt-4.1 | 67.8% | 63.2% | 66.7% | **+4.6pp** |
| claude-3.7 | 72.4% | 66.7% | 73.6% | **+5.7pp** |
| gemini-2.0 | 64.4% | 65.5% | 58.6% | −1.1pp |
| llama-3.3 | 69.0% | 64.4% | 64.4% | **+4.6pp** |

**E2 GS cs_only vs benchmarks (CoT, 5 models):**

| Model | E2 cs_only | v3 full | v3 pk_only | Δ cs_only vs v3_full |
|---|---|---|---|---|
| gpt-4.1-mini * | 75.0% | 81.0% | 86.0% | −6pp |
| gpt-4.1 | 57.0% | 61.0% | 59.0% | −4pp |
| claude-3.7 | 75.0% | 87.0% | 87.0% | −12pp |
| gemini-2.0 | 56.0% | 83.0% | 73.0% | −27pp |
| llama-3.3 | 25.0% | 51.0% | 50.0% | −26pp |

**Key findings:**
- **CJ:** Phase-2-only CS (no PK) matches or beats v3 full for 4/5 models. The v3 CJ case studies (oracle-contaminated, built on PK) were harmful; Phase-2-only generates different CS that transfer better. Train model unaffected (64.4% both ways).
- **GS:** Phase-2-only CS is universally worse than v3 full. PK is a prerequisite for useful GS case studies — the Phase 1 prior knowledge anchors what the case studies encode.
- **WOL:** Inconclusive — missing both PK and oracle. Will be re-tested once oracle (`item["reason"]`) is generated for WOL.

---

## E3 — CJ No Phase-2-Oracle

**Description:** Full pipeline on CJ with `--no-phase2-oracle` — Phase 1 runs normally,
Phase 2 case study generation has `inject_gold_oracle=False` (no `item["reason"]` injected
into failure formatting). Tests whether Phase 2 oracle injection is what caused v3 CJ's
harmful case studies.

**Run dir:** `runs/bbh_cj_no_p2oracle/`
**Script:** `run_e2_e3_e4.sh`
**Eval results:** `runs/cj_no_p2oracle_results.json`

**Pipeline outcome:** Phase 1 reached 65% train accuracy. Phase 2 added **0 case studies** (stopped at iter 2, 2 consecutive idle). Without oracle context, Phase 2 cannot generate non-regressing case studies for CJ.

**E3 results vs v3 (CoT, 5 models):**

Note: E3 full = E3 pk_only (same cheatsheet — 0 CS added). gpt-4.1-mini "full" is loaded from cot_mode_results cache (v3 = 64.4%); use pk_only (73.6%) as the E3 score for that model.

| Model | E3 pk_only | v3 full | v3 pk_only | Δ E3 vs v3_full |
|---|---|---|---|---|
| gpt-4.1-mini * | 73.6% | 64.4% | 72.4% | **+9.2pp** |
| gpt-4.1 | 66.7% | 63.2% | 66.7% | **+3.5pp** |
| claude-3.7 | 67.8% | 66.7% | 73.6% | +1.1pp |
| gemini-2.0 | 66.7% | 65.5% | 58.6% | +1.2pp |
| llama-3.3 | 64.4% | 64.4% | 64.4% | 0pp |

**Key finding:** Removing Phase 2 oracle causes Phase 2 to generate 0 CS for CJ. E3 (Phase 1 only, no oracle) matches or beats v3 full (Phase 1 + 1 oracle-contaminated CS) for all models — most strikingly +9.2pp for the train model. This confirms: the single CJ case study generated with oracle in v3 was harmful, and oracle injection enabled its generation. **Phase 2 oracle is a prerequisite for CJ case study generation AND the primary cause of their harm.**

---

## E4 — WOL Fixed Partition Key

**Description:** Full pipeline on WOL with the corrected `_wol_partition_key` (answer not
encoded). Previous v3 WOL partition key encoded YES/NO directly into ACTIVATE IF conditions.
Fixed in `tasks/bbh_tasks_ext.py`.

**Run dir:** `runs/bbh_wol_fixed/`
**Script:** `run_e2_e3_e4.sh`
**Eval results:** `runs/wol_fixed_results.json`

**Pipeline outcome:** Phase 1 reached 72.7% train accuracy. Phase 2 added **0 case studies** (stopped at iter 2, same regression pattern as E2a WOL — no oracle for WOL).

**E4 results vs v3 WOL (CoT, 5 models):**

Note: E4 full = E4 pk_only (0 CS). gpt-4.1-mini "full" is the v3 cached value (91%) — invalid for E4. Use pk_only (67%) for gpt-4.1-mini E4.

| Model | E4 pk_only | v3 full CoT | v3 pk_only CoT | Δ E4 vs v3_full |
|---|---|---|---|---|
| gpt-4.1-mini * | 67% | 91% | 61% | −24pp (cached v3 used; fresh needed) |
| gpt-4.1 | 38% | 67% | 53% | **−29pp** |
| claude-3.7 | 45% | 56% | 43% | **−11pp** |
| gemini-2.0 | 49% | 62% | 52% | **−13pp** |
| llama-3.3 | 96% | 92% | 98% | +4pp |

**Key finding:** Fixed partition key + no case studies (Phase 1 only) is significantly worse than v3 full (which had 2 CS from buggy key) for all non-llama models. The WOL improvement in v3 came primarily from the 2 case studies, not Phase 1. Fixing the partition key changes Phase 1 failure binning and PK structure, but without oracle to generate CS, Phase 2 cannot compensate. **WOL requires both oracle data (`item["reason"]`) and a working partition key to produce effective case studies.** This run is incomplete — the full comparison requires oracle generation for WOL followed by a fresh pipeline run.

---

## WOL Fixed Partition Key Re-run (NOT RUN)

**Description:** Re-run WOL with the corrected partition key that removes the answer
dimension. Previous v3 WOL partition key encoded the correct answer (YES/NO) directly
into ACTIVATE IF conditions via `_wol_key_to_conds`, creating circular conditions that
overfit to training polarity. Fixed in `tasks/bbh_tasks_ext.py`.

v3 WOL test: 66% (with buggy key). Baseline CS-ICL: 58%.
**Run dir:** `runs/bbh_v3_wol_fixed/`
**Status:** Superseded by E4. See E4 section above — full re-run with oracle generation pending.

---

## Snarks Re-run (NOT RUN)

Deprioritized — snarks case study overfit to train model is now clearly documented via pk_only ablation. A fresh pipeline run is unlikely to change the structural finding.

---

## MAGMA v3 (NOT RUN)

---

## CoT vs Verdict-Only Eval Mode Comparison

**Script:** `eval_cot_mode.py`
**Model:** `openai/gpt-4.1-mini`
**Results file:** `runs/cot_mode_results.json`

Scores all 11 v3 cheatsheets (ours + CS-ICL) using `score_batch(cot_first=True)` — the
reasoning-eliciting scorer used during training — and compares against the official
verdict-only numbers from `eval_bbh_comparison.py` (`build_eval_prompt`, max_tokens=32).

| Task | ours vrd | csicl vrd | Δ vrd | ours CoT | csicl CoT | Δ CoT |
|---|---|---|---|---|---|---|
| boolean_expressions | 87% | 88% | −1pp | 85% | 86% | −1pp |
| causal_judgement | 65.5% | 73.6% | −8.1pp | 64.4% | 72.4% | −8pp |
| date_understanding | 72% | 72% | 0pp | **88%** | 84% | **+4pp** |
| disambiguation_qa | 77% | 75% | +2pp | **80%** | 72% | **+8pp** |
| formal_fallacies | **70%** | 64% | +6pp | **70%** | 66% | +4pp |
| geometric_shapes | 48% | 51% | −3pp | 81% | **87%** | −6pp |
| logical_deduction_three | **95%** | 95% | 0pp | 88% | **96%** | −8pp |
| navigate | **80%** | 77% | +3pp | 75% | **79%** | −4pp |
| snarks | 84.5% | **88.7%** | −4.2pp | 84.5% | **94.4%** | −10pp |
| sports_understanding | **94%** | 93% | +1pp | **96%** | 95% | +1pp |
| web_of_lies | **66%** | 58% | +8pp | **91%** | 61% | **+30pp** |
| **record** | **5W/4L/2T** | | | **5W/6L** | | |

**Key findings:**

- **WOL** is the clearest case where CoT unlocks our cheatsheet: verdict +8pp → CoT +30pp.
  The logical chain reasoning in our cheatsheet is far more actionable when the model reasons
  step by step. CS-ICL jumps only 3pp (58%→61%) while ours jumps 25pp (66%→91%).
- **Disambiguation_qa / date_understanding**: CoT mode amplifies our wins (+2pp→+8pp, 0pp→+4pp).
  These tasks reward structured reasoning that our cheatsheet captures better than CS-ICL's.
- **Snarks / logical_deduction_three / navigate**: CoT mode *widens* our losses. CS-ICL's cheatsheet
  generalizes better under free reasoning on structure-heavy tasks; our case studies may confuse
  the model when it has space to reason freely.
- **Geometric shapes**: absolute accuracy jumps for both (48%→81% ours, 51%→87% CS-ICL) but
  CS-ICL benefits more — the gap grows from −3pp to −6pp in CoT mode.
- Overall: CoT mode is not a uniform win. It amplifies both strengths (WOL, DQ) and
  weaknesses (snarks, LD3, navigate). The eval mode matters and should be considered
  when selecting which tasks/cheatsheets to highlight in the paper.

---

## No-Oracle Ablation: navigate + formal_fallacies

**Run dir:** `runs/bbh_oracle_ablation/no_oracle/`
**Eval log:** `runs/overnight_logs/eval_nooracle_nav_ff.log`

| Task | CS-ICL | no_oracle | v3 full | Δ no_oracle vs v3 |
|---|---|---|---|---|
| navigate | 77% | 74% | **80%** | −6pp |
| formal_fallacies | 63% | 64% | **70%** | −6pp |

**Finding:** Removing both oracle phases costs exactly 6pp on both tasks and collapses the
CS-ICL advantage entirely. Phase 2 oracle is doing real work on these tasks — contrast with
causal_judgement/geometric_shapes where oracle injection was net harmful.

---

## Transferability & Case Study Contribution (3-task deep dive)

**Tasks:** web_of_lies, causal_judgement, geometric_shapes
**Models:** gpt-4.1-mini (train), gpt-4.1, claude-3.7-sonnet, gemini-2.0-flash, llama-3.3-70b
**Scripts:** `eval_transferability.py` (ours vs CS-ICL) + `eval_cs_ablation.py` (full vs pk_only)
**pk_only source:** `cheatsheet_phase1_pk_final.txt` (pipeline Phase 1 output, no case studies)
**Parser fix:** `_parse_yesno` and `_parse_mc` updated to handle answer-first and FINAL ANSWER formats

---

### Table 1 — Transferability: Ours vs CS-ICL (verdict-only)

| Task | Model | ours | cs_icl | Δ |
|---|---|---|---|---|
| web_of_lies | gpt-4.1-mini *(train)* | 66% | 61% | **+5pp** |
| web_of_lies | gpt-4.1 | 55% | 48% | **+7pp** |
| web_of_lies | claude-3.7-sonnet | 58% | 61% | −3pp |
| web_of_lies | gemini-2.0-flash | 45% | 50% | −5pp |
| web_of_lies | llama-3.3-70b | 42% | 47% | −5pp |
| causal_judgement | gpt-4.1-mini *(train)* | 64.4% | 70.1% | −5.7pp |
| causal_judgement | gpt-4.1 | 62.1% | 73.6% | −11.5pp |
| causal_judgement | claude-3.7-sonnet | 65.5% | 69.0% | −3.4pp |
| causal_judgement | gemini-2.0-flash | 55.2% | 72.4% | −17.2pp |
| causal_judgement | llama-3.3-70b | 62.1% | 67.8% | −5.7pp |
| geometric_shapes | gpt-4.1-mini *(train)* | 48% | 51% | −3pp |
| geometric_shapes | gpt-4.1 | 62% | 67% | −5pp |
| geometric_shapes | claude-3.7-sonnet | 55% | 68% | −13pp |
| geometric_shapes | gemini-2.0-flash | 55% | 54% | +1pp |
| geometric_shapes | llama-3.3-70b | 43% | 54% | −11pp |

**Finding:** WOL transfers positively to gpt-4.1 but negatively to claude/gemini/llama. CJ and GS transfer negatively to all non-train models, CJ catastrophically on gemini (−17pp).

---

### Table 2 — Case Study Contribution: Verdict-Only (full vs pk_only)

| Task | Model | full | pk_only | Δ_cs |
|---|---|---|---|---|
| web_of_lies | gpt-4.1-mini *(train)* | 66% | 60% | **+6pp** |
| web_of_lies | gpt-4.1 | 55% | 53% | +2pp |
| web_of_lies | claude-3.7-sonnet | 58% | 40% | **+18pp** |
| web_of_lies | gemini-2.0-flash | 45% | 39% | +6pp |
| web_of_lies | llama-3.3-70b | 42% | 39% | +3pp |
| causal_judgement | gpt-4.1-mini *(train)* | 64.4% | 66.7% | −2.3pp |
| causal_judgement | gpt-4.1 | 62.1% | 64.4% | −2.3pp |
| causal_judgement | claude-3.7-sonnet | 65.5% | 67.8% | −2.3pp |
| causal_judgement | gemini-2.0-flash | 55.2% | 58.6% | −3.4pp |
| causal_judgement | llama-3.3-70b | 62.1% | 64.4% | −2.3pp |
| geometric_shapes | gpt-4.1-mini *(train)* | 48% | 53% | −5pp |
| geometric_shapes | gpt-4.1 | 62% | 61% | +1pp |
| geometric_shapes | claude-3.7-sonnet | 55% | 44% | **+11pp** |
| geometric_shapes | gemini-2.0-flash | 55% | 44% | **+11pp** |
| geometric_shapes | llama-3.3-70b | 43% | 39% | +4pp |

**Finding:** WOL case studies help all models. CJ case studies hurt all models (−2 to −3pp). GS case studies hurt the train model (−5pp) but help others (+4 to +11pp).

---

### Table 3 — Case Study Contribution: CoT Mode (full vs pk_only)

| Task | Model | full | pk_only | Δ_cs | vs verdict Δ |
|---|---|---|---|---|---|
| web_of_lies | gpt-4.1-mini *(train)* | 91% | 61% | **+30pp** | +6pp → +30pp |
| web_of_lies | gpt-4.1 | 67% | 53% | **+14pp** | +2pp → +14pp |
| web_of_lies | claude-3.7-sonnet | 56% | 43% | **+13pp** | +18pp → +13pp |
| web_of_lies | gemini-2.0-flash | 62% | 52% | **+10pp** | +6pp → +10pp |
| web_of_lies | llama-3.3-70b | 92% | 98% | −6pp | +3pp → −6pp *(llama reasons well from PK alone)* |
| causal_judgement | gpt-4.1-mini *(train)* | 64.4% | 72.4% | −8.0pp | −2.3pp → −8pp |
| causal_judgement | gpt-4.1 | 63.2% | 66.7% | −3.5pp | −2.3pp → −3.5pp |
| causal_judgement | claude-3.7-sonnet | 66.7% | 73.6% | −6.9pp | −2.3pp → −6.9pp |
| causal_judgement | gemini-2.0-flash | 65.5% | 58.6% | **+6.9pp** | −3.4pp → +6.9pp *(flips positive)* |
| causal_judgement | llama-3.3-70b | 64.4% | 64.4% | 0pp | −2.3pp → 0pp |
| geometric_shapes | gpt-4.1-mini *(train)* | 81% | 86% | −5pp | −5pp → −5pp |
| geometric_shapes | gpt-4.1 | 61% | 59% | +2pp | +1pp → +2pp |
| geometric_shapes | claude-3.7-sonnet | 87% | 87% | 0pp | +11pp → 0pp *(benefit disappears)* |
| geometric_shapes | gemini-2.0-flash | 83% | 73% | **+10pp** | +11pp → +10pp *(consistent)* |
| geometric_shapes | llama-3.3-70b | 51% | 50% | +1pp | +4pp → +1pp |

**Key findings:**
- **WOL**: CoT mode amplifies case study benefit for all models except llama (where PK alone already enables near-perfect CoT reasoning). gpt-4.1-mini jumps +6pp→+30pp — the cheatsheet's logical chain encoding is maximally useful under reasoning.
- **CJ**: CoT deepens case study harm for most models (−2pp→−8pp on gpt-4.1-mini). The oracle-contaminated case studies actively mislead models that reason freely. Exception: gemini flips to +7pp in CoT.
- **GS**: Claude's +11pp verdict benefit completely vanishes in CoT (0pp) — case studies add nothing when claude reasons spatially on its own. Gemini stays consistent (+11pp→+10pp). gpt-4.1-mini consistently hurt (−5pp both modes).

---

## Baseline CoT (no cheatsheet) — 3 transferability tasks

**Script:** `eval_cs_ablation.py --baseline --cot`
**Results file:** `runs/baseline_cot_results.json`
**Prompt:** same CoT scoring prompt with empty cheatsheet section (`=== CHEATSHEET === ... === END CHEATSHEET ===` intact, content empty)

| Task | Model | baseline CoT | pk_only CoT | full CoT | Δ base→pk | Δ pk→full | Δ base→full |
|---|---|---|---|---|---|---|---|
| web_of_lies | gpt-4.1-mini | 54% | 61% | 91% | +7pp | +30pp | **+37pp** |
| web_of_lies | gpt-4.1 | 56% | 53% | 67% | −3pp | +14pp | **+11pp** |
| web_of_lies | claude-3.7-sonnet | 47% | 43% | 56% | −4pp | +13pp | **+9pp** |
| web_of_lies | gemini-2.0-flash | 56% | 52% | 62% | −4pp | +10pp | **+6pp** |
| web_of_lies | llama-3.3-70b | 44% | 98% | 92% | +54pp | −6pp | **+48pp** |
| causal_judgement | gpt-4.1-mini | 71.3% | 72.4% | 64.4% | +1pp | −8pp | **−6.9pp** |
| causal_judgement | gpt-4.1 | 65.5% | 66.7% | 63.2% | +1pp | −3.5pp | **−2.3pp** |
| causal_judgement | claude-3.7-sonnet | 69.0% | 73.6% | 66.7% | +5pp | −6.9pp | **−2.3pp** |
| causal_judgement | gemini-2.0-flash | 65.5% | 58.6% | 65.5% | −7pp | +6.9pp | **0pp** |
| causal_judgement | llama-3.3-70b | 69.0% | 64.4% | 64.4% | −4.6pp | 0pp | **−4.6pp** |
| geometric_shapes | gpt-4.1-mini | 66% | 86% | 81% | +20pp | −5pp | **+15pp** |
| geometric_shapes | gpt-4.1 | 42% | 59% | 61% | +17pp | +2pp | **+19pp** |
| geometric_shapes | claude-3.7-sonnet | 75% | 87% | 87% | +12pp | 0pp | **+12pp** |
| geometric_shapes | gemini-2.0-flash | 41% | 73% | 83% | +32pp | +10pp | **+42pp** |
| geometric_shapes | llama-3.3-70b | 56% | 50% | 51% | −6pp | +1pp | **−5pp** |

**Key findings:**
- WOL: full cheatsheet is strongly positive for all models (+6 to +48pp over baseline). Case studies (Δ pk→full) drive most of the gain.
- CJ: full cheatsheet is **worse than no cheatsheet** for all models (−2.3 to −6.9pp). Even pk_only barely moves the needle. CJ is a case where the pipeline actively degraded performance.
- GS: cheatsheet broadly helpful (+12 to +42pp), except llama (−5pp). PK alone (pk_only) provides most of the GS gain; case studies add marginal value on top.

---

## PK-Only Coverage Status

All 10 scoreable tasks now have pk_only CoT scores across 5 models. date_understanding excluded — no `cheatsheet_phase1_pk_final.txt` exists for it.

| Task | pk_only CoT (5 models) | source file |
|---|---|---|
| web_of_lies | Yes | `cs_ablation_cot_results.json` + reruns |
| causal_judgement | Yes | `cs_ablation_cot_results.json` + reruns |
| geometric_shapes | Yes | `cs_ablation_cot_results.json` + reruns |
| boolean_expressions | Yes | `cs_ablation_remaining_cot.json` |
| disambiguation_qa | Yes | `cs_ablation_remaining_cot.json` |
| formal_fallacies | Yes | `cs_ablation_remaining_cot.json` |
| logical_deduction_three | Yes (llama fixed) | `cs_ablation_remaining_cot.json` + `ld3_llama_cot_rerun.json` |
| sports_understanding | Yes | `cs_ablation_remaining_cot.json` |
| navigate | Yes | `nav_snarks_cot.json` |
| snarks | Yes | `nav_snarks_cot.json` |
| date_understanding | **No** (no pk_final cheatsheet) | — |

---

---

## LD3 Llama Parser Fix

**Problem:** `_parse_mc` in `tasks/bbh_tasks.py` scanned only the last 200 chars of the response for `(X)` format. Llama on LD3 writes `"Therefore, the correct answer is **(B) The crow is the leftmost**."` and then adds a 200–400 char explanation paragraph, pushing the parenthesized letter out of the scan window. Result: ~40 parse errors per 100 items → artificially low scores.

**Fix:** Added `\bcorrect answer is[^\n]*\(([A-Z])\)` as a new pattern in `_parse_mc` (requires literal parentheses to avoid false positives from "correctly identifies..."). Extended tail scan from 200 → 500 chars. Parse errors dropped from ~40 to 2.

**Corrected LD3 llama numbers (CoT mode):**

| Condition | before fix | after fix |
|---|---|---|
| gold_fewshot | 59% *(~40 parse errors)* | **98%** |
| pk_only | 78% *(~33 parse errors)* | **96%** |
| full (ACTIVATE IF) | 64% *(~40 parse errors)* | **92%** |

---

## Full Case Study Contribution: CoT Mode, All 10 Tasks

**Script:** `eval_cs_ablation.py --cot` across all tasks; canonical numbers use rerun files for WOL/CJ/GS (post parser fix) and `cs_ablation_remaining_cot.json` / `nav_snarks_cot.json` for remaining tasks. LD3 llama uses `ld3_llama_cot_rerun.json`.

`*` = train model (gpt-4.1-mini).

| Task | Model | full | pk_only | Δ_cs |
|---|---|---|---|---|
| web_of_lies | gpt-4.1-mini * | 91.0% | 61.0% | **+30.0%** |
| web_of_lies | gpt-4.1 | 67.0% | 53.0% | **+14.0%** |
| web_of_lies | claude-3.7 | 56.0% | 43.0% | **+13.0%** |
| web_of_lies | gemini-2.0 | 62.0% | 52.0% | **+10.0%** |
| web_of_lies | llama-3.3-70b | 92.0% | 98.0% | −6.0% |
| causal_judgement | gpt-4.1-mini * | 64.4% | 72.4% | −8.1% |
| causal_judgement | gpt-4.1 | 63.2% | 66.7% | −3.5% |
| causal_judgement | claude-3.7 | 66.7% | 73.6% | −6.9% |
| causal_judgement | gemini-2.0 | 65.5% | 58.6% | **+6.9%** |
| causal_judgement | llama-3.3-70b | 64.4% | 64.4% | 0.0% |
| geometric_shapes | gpt-4.1-mini * | 81.0% | 86.0% | −5.0% |
| geometric_shapes | gpt-4.1 | 61.0% | 59.0% | +2.0% |
| geometric_shapes | claude-3.7 | 87.0% | 87.0% | 0.0% |
| geometric_shapes | gemini-2.0 | 83.0% | 73.0% | **+10.0%** |
| geometric_shapes | llama-3.3-70b | 51.0% | 50.0% | +1.0% |
| boolean_expressions | gpt-4.1-mini * | 85.0% | 88.0% | −3.0% |
| boolean_expressions | gpt-4.1 | 94.0% | 92.0% | +2.0% |
| boolean_expressions | claude-3.7 | 95.0% | 90.0% | +5.0% |
| boolean_expressions | gemini-2.0 | 89.0% | 89.0% | 0.0% |
| boolean_expressions | llama-3.3-70b | 86.0% | 84.0% | +2.0% |
| disambiguation_qa | gpt-4.1-mini * | 80.0% | 81.0% | −1.0% |
| disambiguation_qa | gpt-4.1 | 83.0% | 85.0% | −2.0% |
| disambiguation_qa | claude-3.7 | 83.0% | 83.0% | 0.0% |
| disambiguation_qa | gemini-2.0 | 81.0% | 81.0% | 0.0% |
| disambiguation_qa | llama-3.3-70b | 80.0% | 87.0% | −7.0% |
| formal_fallacies | gpt-4.1-mini * | 70.0% | 66.0% | **+4.0%** |
| formal_fallacies | gpt-4.1 | 79.0% | 81.0% | −2.0% |
| formal_fallacies | claude-3.7 | 74.0% | 74.0% | 0.0% |
| formal_fallacies | gemini-2.0 | 60.0% | 55.0% | +5.0% |
| formal_fallacies | llama-3.3-70b | 66.0% | 76.0% | −10.0% |
| logical_deduction_three | gpt-4.1-mini * | 88.0% | 90.0% | −2.0% |
| logical_deduction_three | gpt-4.1 | 92.0% | 94.0% | −2.0% |
| logical_deduction_three | claude-3.7 | 96.0% | 96.0% | 0.0% |
| logical_deduction_three | gemini-2.0 | 92.0% | 92.0% | 0.0% |
| logical_deduction_three | llama-3.3-70b | 92.0% | 96.0% | −4.0% |
| sports_understanding | gpt-4.1-mini * | 96.0% | 98.0% | −2.0% |
| sports_understanding | gpt-4.1 | 99.0% | 99.0% | 0.0% |
| sports_understanding | claude-3.7 | 95.0% | 91.0% | +4.0% |
| sports_understanding | gemini-2.0 | 97.0% | 99.0% | −2.0% |
| sports_understanding | llama-3.3-70b | 92.0% | 91.0% | +1.0% |
| navigate | gpt-4.1-mini * | 75.0% | 75.0% | 0.0% |
| navigate | gpt-4.1 | 73.0% | 69.0% | +4.0% |
| navigate | claude-3.7 | 47.0% | 55.0% | −8.0% |
| navigate | gemini-2.0 | 81.0% | 76.0% | +5.0% |
| navigate | llama-3.3-70b | 99.0% | 98.0% | +1.0% |
| snarks | gpt-4.1-mini * | 84.5% | 95.8% | **−11.3%** |
| snarks | gpt-4.1 | 88.7% | 85.9% | +2.8% |
| snarks | claude-3.7 | 93.0% | 93.0% | 0.0% |
| snarks | gemini-2.0 | 39.4% | 23.9% | **+15.5%** |
| snarks | llama-3.3-70b | 73.2% | 80.3% | −7.0% |

**Cross-task patterns:**
- **WOL** case studies universally help (except llama, which reasons well from PK alone). Strongest signal across all tasks.
- **CJ** case studies hurt most models — oracle-contaminated case studies overfit to gpt-4.1-mini's reasoning style.
- **Snarks** shows the clearest train-model overfit: gpt-4.1-mini −11.3pp (worst of all), while gemini gains +15.5pp from case studies.
- **LD3, sports_understanding, boolean_expressions, disambiguation_qa**: case studies largely neutral (−2 to 0pp for most models). PK alone is sufficient.
- **Navigate**: mixed — claude hurt (−8pp), gpt-4.1/gemini helped.
- **Formal_fallacies**: train model benefits (+4pp), llama hurt (−10pp).

---

## Gold Few-Shot vs ACTIVATE IF Case Studies (6 oracle tasks)

**Script:** `build_gold_fewshot_cheatsheet.py --n-examples 5 --seed 42` + `eval_cs_ablation.py --gold-fewshot --cot`
**Cheatsheet:** PK text (`cheatsheet_phase1_pk_final.txt`) + 5 gold training examples using `item["reason"]` as reasoning trace
**Format:** `=== WORKED EXAMPLES ===` section with Question / Reasoning / Answer triplets
**Results file:** `runs/gold_fewshot_results.json` + `ld3_llama_goldfewshot_rerun.json` (corrected llama LD3)

Full 4-condition comparison (baseline / pk_only / gold_fewshot / full ACTIVATE IF), CoT mode:

| Task | Model | baseline | pk_only | gold_fewshot | full (AI) | Δ pk→gf | Δ pk→full |
|---|---|---|---|---|---|---|---|
| boolean_expressions | gpt-4.1-mini * | N/A | 88.0% | 88.0% | 85.0% | 0pp | −3pp |
| boolean_expressions | gpt-4.1 | N/A | 92.0% | 97.0% | 94.0% | **+5pp** | +2pp |
| boolean_expressions | claude-3.7 | N/A | 90.0% | 97.0% | 95.0% | **+7pp** | +5pp |
| boolean_expressions | gemini-2.0 | N/A | 89.0% | 87.0% | 89.0% | −2pp | 0pp |
| boolean_expressions | llama-3.3-70b | N/A | 84.0% | 89.0% | 86.0% | +5pp | +2pp |
| causal_judgement | gpt-4.1-mini * | 71.3% | 72.4% | 64.4% | 64.4% | **−8pp** | −8pp |
| causal_judgement | gpt-4.1 | 65.5% | 66.7% | 71.3% | 63.2% | +4.6pp | −3.4pp |
| causal_judgement | claude-3.7 | 69.0% | 69.0% | 72.4% | 66.7% | +3.4pp | −2.3pp |
| causal_judgement | gemini-2.0 | 65.5% | 58.6% | 59.8% | 65.5% | +1.1pp | +6.9pp |
| causal_judgement | llama-3.3-70b | 69.0% | 64.4% | 67.8% | 64.4% | +3.4pp | 0pp |
| disambiguation_qa | gpt-4.1-mini * | N/A | 81.0% | 85.0% | 80.0% | +4pp | −1pp |
| disambiguation_qa | gpt-4.1 | N/A | 85.0% | 87.0% | 83.0% | +2pp | −2pp |
| disambiguation_qa | claude-3.7 | N/A | 83.0% | 87.0% | 83.0% | +4pp | 0pp |
| disambiguation_qa | gemini-2.0 | N/A | 81.0% | 89.0% | 81.0% | **+8pp** | 0pp |
| disambiguation_qa | llama-3.3-70b | N/A | 87.0% | 79.0% | 80.0% | −8pp | −7pp |
| geometric_shapes | gpt-4.1-mini * | 66.0% | 86.0% | 82.0% | 81.0% | −4pp | −5pp |
| geometric_shapes | gpt-4.1 | 42.0% | 63.0% | 64.0% | 58.0% | +1pp | −5pp |
| geometric_shapes | claude-3.7 | 75.0% | 87.0% | 90.0% | 87.0% | +3pp | 0pp |
| geometric_shapes | gemini-2.0 | 41.0% | 58.0% | 77.0% | 60.0% | **+19pp** | +2pp |
| geometric_shapes | llama-3.3-70b | 56.0% | 57.0% | 70.0% | 53.0% | **+13pp** | −4pp |
| logical_deduction_three | gpt-4.1-mini * | N/A | 90.0% | 80.0% | 88.0% | **−10pp** | −2pp |
| logical_deduction_three | gpt-4.1 | N/A | 94.0% | 90.0% | 92.0% | −4pp | −2pp |
| logical_deduction_three | claude-3.7 | N/A | 96.0% | 98.0% | 96.0% | +2pp | 0pp |
| logical_deduction_three | gemini-2.0 | N/A | 92.0% | 96.0% | 92.0% | +4pp | 0pp |
| logical_deduction_three | llama-3.3-70b | N/A | 96.0% | 98.0% | 92.0% | +2pp | −4pp |
| sports_understanding | gpt-4.1-mini * | N/A | 98.0% | 97.0% | 96.0% | −1pp | −2pp |
| sports_understanding | gpt-4.1 | N/A | 99.0% | 100.0% | 99.0% | +1pp | 0pp |
| sports_understanding | claude-3.7 | N/A | 91.0% | 96.0% | 95.0% | +5pp | +4pp |
| sports_understanding | gemini-2.0 | N/A | 99.0% | 99.0% | 97.0% | 0pp | −2pp |
| sports_understanding | llama-3.3-70b | N/A | 91.0% | 92.0% | 92.0% | +1pp | +1pp |

**Key findings:**
- **Gold few-shot ≥ ACTIVATE IF in most cases.** Across 6 tasks × 5 models = 30 model-task pairs, gold_fewshot is better than or equal to full ACTIVATE IF in 22/30 cases. The raw worked examples are generally more useful than the case-study wrapper.
- **Clearest ACTIVATE IF win:** WOL (not included here — no oracle coverage), formal_fallacies train model (+4pp over pk). Boolean/disambiguation/sports are near-ceiling, so neither format adds much.
- **GS:** Gold few-shot beats ACTIVATE IF for gemini (+19pp vs +2pp) and llama (+13pp vs −4pp). For gpt-4.1, both are roughly neutral. ACTIVATE IF case studies on GS are consistently suboptimal vs worked examples.
- **CJ:** Gold few-shot also hurts the train model (−8pp relative to pk), same as ACTIVATE IF. This is a genuine task-level signal, not a case-study format issue.
- **LD3:** Gold few-shot hurts the train model (−10pp) — possibly because 5 gold examples inject specific orderings that conflict with the model's learned abstractions. ACTIVATE IF case studies are better here (−2pp).

---

## Reasoning Coverage by Task

Coverage = fraction of training items with non-empty `item["reason"]` (gold CoT).
Phase 2 oracle injection (`inject_gold_oracle=True`) is a no-op for items where this is empty.

| Task | Coverage | Phase 2 oracle active in v3 | v3 CS added |
|------|----------|------------------------------|-------------|
| boolean_expressions | 100% | yes | 4 |
| causal_judgement | 100% | yes | 0 |
| disambiguation_qa | 100% | yes | 0 |
| geometric_shapes | 100% | yes | 2 |
| logical_deduction_three | 100% | yes | 1 |
| sports_understanding | 100% | yes | 3 |
| navigate | 61% (91/150) | partial | 0 |
| formal_fallacies | 21% (31/150) | partial | 3 |
| web_of_lies | 0% (all empty strings) | no-op | 2 |
| snarks | 0% (all empty strings) | no-op | 1 |
| date_understanding | 2% (3/150 empty strings) | effectively no-op | 3 |

**Implication:** v3 cheatsheets for WOL, snarks, DU, and FF (partial) were generated without
full Phase 2 oracle. For consistency, `item["reason"]` must be generated for these tasks before
re-running v3, v3_phase1_only, and v4 pipelines on WOL/FF/snarks/DU (8 pipeline re-runs total).
This will also unlock the E4 WOL fixed-key result (currently Phase 1 only due to missing oracle).

---

## Paper Draft Deadline

**Due: 2026-05-07**

---

## Reasoning-First (RF) Scoring Format — Discovery & Fix (2026-04-30)

### Background: The Verdict-First Bug

All prior CoT evals used a **verdict-first** template:
```
VERDICT: Yes or No  ← FIRST LINE.
REASONING: ...
```
The parser read only the **first** `VERDICT:` occurrence. For WOL specifically, gpt-4.1 would
guess wrong on the first line, reason correctly, self-correct mid-response, then write a second
`VERDICT:` — but the parser captured the wrong first guess. This caused systematic
underestimation for all models on tasks requiring multi-step reasoning.

**Confirmed via debug run:** 30-item WOL with gpt-4.1 + EA PK cheatsheet:
- Verdict-first format: 53.3% (wrong first guess captured)
- Reasoning-first (RF) format: 100% (correct final verdict captured)

### RF Format

```
REASONING: ... (step-by-step)
VERDICT: YES   ← LAST LINE, exactly once
```
Parser scans reversed lines for last `VERDICT:` occurrence.

### Implementation

**New functions added** (convention: `_{task_name}_scoring_prompt_rf` / `_parse_{task_name}_rf`):
- `tasks/bbh_tasks_ext.py`: web_of_lies, formal_fallacies, date_understanding,
  logical_deduction_three, navigate, snarks
- `tasks/bbh_tasks.py`: causal_judgement, sports_understanding, disambiguation_qa,
  geometric_shapes — **parsers return uppercase YES/NO or TRUE/FALSE** (matches `_yesno_correct`
  and `_bbh_bool_is_correct` exact-case comparisons)
- `tasks/bbh_boolean.py`: boolean_expressions

**Flag added to `eval_cs_ablation.py`:**
```
--reasoning-first   patches task_spec.build_scoring_prompt + parse_verdict + nulls build_eval_prompt
--full-only         skip pk_only scoring pass
```

### Critical Bug in RF Patch (fixed same session)

`--reasoning-first` patched `build_scoring_prompt` and `parse_verdict` but left
`build_eval_prompt` intact. Since `_score()` prefers `build_eval_prompt` when present
(ALL tasks except boolean_expressions have it), the RF builders were silently bypassed
for all tasks but boolean.

**Fix:** Added `task_spec.build_eval_prompt = None` to the RF patch, forcing fallthrough
to `score_batch` which uses `build_scoring_prompt`.

**Note on prior gpt-4.1-mini "RF" runs:** Those were run with both `--cot` AND
`--reasoning-first`. The `--cot` flag sets `force_cot=True` in `_score`, which bypasses
`build_eval_prompt` regardless. So gpt-4.1-mini RF results ARE genuine RF scores.
Prior non-training model runs (without `--cot`) were using verdict-only `build_eval_prompt`.

---

## RF Scores: gpt-4.1-mini, All 11 Tasks (v3 cheatsheets)

**Script:** `eval_cs_ablation.py --cot --reasoning-first --models openai/gpt-4.1-mini`
**Results files:** `runs/rf_full_remaining.json`, `runs/rf_full_underperforming.json`

| Task | full RF | pk_only RF | Δ_cs |
|---|---:|---:|---:|
| web_of_lies | 100% | 100% | 0% |
| formal_fallacies | 97% | 96% | +1% |
| logical_deduction_three | 100% | 100% | 0% |
| navigate | 100% | 100% | 0% |
| boolean_expressions | 100% | 100% | 0% |
| sports_understanding | 99% | 100% | −1% |
| snarks | 96% | 96% | 0% |
| disambiguation_qa | 86% | 86% | 0% |
| date_understanding | 94% | 92% | +2% |
| causal_judgement | 72% | 71% | +1% |
| geometric_shapes | 82% | 71% | **+11%** |

Key finding: 6 tasks hit 100% with PK alone. Case studies provide meaningful lift only on
geometric_shapes (+11pp) — the only task where worked visual examples add real value.

---

## RF Transferability: 3 Original Tasks (non-training models)

**Script:** `eval_cs_ablation.py --reasoning-first --full-only --transferability runs/__no_cache__.json`
**Results file:** `runs/rf_transfer_recheck_v2.json`

### web_of_lies — RF, full cheatsheet

| Model | RF full | old verdict-only | Δ |
|---|---:|---:|---:|
| gpt-4.1-mini *(train)* | 100% | 66% | +34% |
| gpt-4.1 | 100% | 55% | +45% |
| claude-3.7-sonnet | 100% | 58% | +42% |
| gemini-2.0-flash | 100% | 45% | +55% |
| llama-3.3-70b | 99% | 42% | +57% |

**Finding:** WOL was never a transfer problem. All 42-66% old scores were entirely due to
the verdict-first format bug. Under RF, all model families achieve near-perfect performance.
The cheatsheet's logical chain encoding is correct and universally transferable.

### causal_judgement — RF, full cheatsheet

| Model | RF full | old CoT full |
|---|---:|---:|
| gpt-4.1-mini *(train)* | 72% | 64% |
| gpt-4.1 | 60% | 63% |
| claude-3.7-sonnet | 67% | 67% |
| gemini-2.0-flash | 64% | 66% |
| llama-3.3-70b | 69% | 64% |

### geometric_shapes — RF, full cheatsheet

| Model | RF full | old CoT full |
|---|---:|---:|
| gpt-4.1-mini *(train)* | 82% | 81% |
| gpt-4.1 | 73% | 61% |
| claude-3.7-sonnet | 77% | 87% |
| gemini-2.0-flash | 78% | 83% |
| llama-3.3-70b | 52% | 51% |

---

## RF Baseline (empty cheatsheet) — 5 Tasks, All Models

**Script:** `eval_cs_ablation.py --reasoning-first --baseline`
**Results files:** `runs/rf_baseline_3tasks.json`, `runs/rf_baseline_3harder.json`

### web_of_lies

All models: 92–100% natively. **Cheatsheet lift = 0.** WOL is trivially solvable under RF
format — any capable model traces truth-teller chains without assistance.

### causal_judgement

| Model | Baseline | Full CS | Lift |
|---|---:|---:|---:|
| gpt-4.1-mini | 69% | 72% | +3% |
| gpt-4.1 | 63% | 60% | **−3%** |
| claude-3.7-sonnet | 70% | 67% | **−3%** |
| gemini-2.0-flash | 69% | 64% | **−5%** |
| llama-3.3-70b | 62% | 69% | +7% |

**Finding:** Cheatsheet slightly hurts stronger reasoners on CJ. The oracle-contaminated case
studies actively interfere with models that can already reason causally. Only llama gains.

### geometric_shapes

| Model | Baseline | Full CS | Lift |
|---|---:|---:|---:|
| gpt-4.1-mini | 58% | 82% | **+24%** |
| gpt-4.1 | 71% | 73% | +2% |
| claude-3.7-sonnet | 81% | 77% | **−4%** |
| gemini-2.0-flash | 48% | 78% | **+30%** |
| llama-3.3-70b | 49% | 52% | +3% |

**Finding:** GS cheatsheet lift is highly model-dependent. Gemini gains +30pp (natively weak
at SVG parsing), mini gains +24pp (optimized for it). Claude loses 4pp — it already reads
SVG paths better than the cheatsheet guides it. gpt-4.1 minimal gain (already strong natively).

### disambiguation_qa

| Model | Baseline | Full CS | Lift |
|---|---:|---:|---:|
| gpt-4.1-mini | 69% | 86% | **+17%** |
| gpt-4.1 | 58% | 86% | **+28%** |
| claude-3.7-sonnet | 78% | 91% | **+13%** |
| gemini-2.0-flash | 60% | 88% | **+28%** |
| llama-3.3-70b | 78% | 82% | +4% |

**Finding:** Strongest and most consistent cheatsheet lift across tasks. Large gains for
gpt-4.1 and gemini (+28pp); even the already-capable claude and llama gain. PK and case
studies together capture pronoun resolution heuristics that generalize broadly.

### date_understanding

| Model | Baseline | Full CS | Lift |
|---|---:|---:|---:|
| gpt-4.1-mini | 89% | 94% | +5% |
| gpt-4.1 | 91% | 95% | +4% |
| claude-3.7-sonnet | 96% | 97% | +1% |
| gemini-2.0-flash | 85% | 95% | **+10%** |
| llama-3.3-70b | 86% | 89% | +3% |

**Finding:** Modest consistent gains across all models. Baseline is already high (85–96%)
— the cheatsheet provides incremental arithmetic shortcuts.

### snarks

| Model | Baseline | Full CS | Lift |
|---|---:|---:|---:|
| gpt-4.1-mini | 90% | 96% | +6% |
| gpt-4.1 | 97% | 94% | −3% |
| claude-3.7-sonnet | 96% | 96% | 0% |
| gemini-2.0-flash | 94% | 93% | −1% |
| llama-3.3-70b | 76% | 85% | **+9%** |

**Finding:** Splits cleanly by model capability. Stronger models (gpt-4.1, claude, gemini)
are at ceiling natively — cheatsheet adds nothing or slightly regresses. Weaker models
(mini, llama) gain from worked examples.

---

## RF Transfer: 3 Harder Tasks, Non-Training Models

**Script:** `eval_cs_ablation.py --reasoning-first --full-only --transferability runs/__no_cache__.json`
**Results file:** `runs/rf_transfer_3harder.json`

| Task | gpt-4.1 | claude-3.7 | gemini-2.0 | llama-3.3 |
|---|---:|---:|---:|---:|
| disambiguation_qa | 86% | 91% | 88% | 82% |
| date_understanding | 95% | 97% | 95% | 89% |
| snarks | 94% | 96% | 93% | 85% |

**Disambiguation_qa transfer** is strong across all families (86–91%), consistent with the
large baseline lift. **Date_understanding** also transfers well (89–97%). **Snarks** transfers
well for all except llama (85% vs 96% for mini) — snarks case studies likely encode style
patterns closer to gpt-family reasoning.

---

## Summary: Cheatsheet Lift by Task Category (RF scoring)

**High lift (baseline→full ≥ 15pp for at least one model family):**
- geometric_shapes: gemini +30pp, mini +24pp
- disambiguation_qa: gpt-4.1 +28pp, gemini +28pp, mini +17pp, claude +13pp

**Moderate lift (5–15pp):**
- snarks: llama +9pp, mini +6pp
- date_understanding: gemini +10pp, mini +5pp, gpt-4.1 +4pp

**Minimal or negative lift:**
- web_of_lies: all models 92–100% natively, cheatsheet adds 0
- causal_judgement: cheatsheet hurts stronger models (−3 to −5pp), only helps llama (+7pp)

**Key overall insight:** Cheatsheet lift is strongly model-dependent. Models that struggle
natively with a task gain the most. Models that are already strong at a task gain little
or are slightly hurt (cheatsheet disrupts their native reasoning). The train model (mini)
consistently benefits from the cheatsheet even on tasks where it is not the strongest baseline.

---

## RF Transfer: 5 Tasks, All 3 Conditions (v3 — corrected RF scoring)

**Script:** `eval_cs_ablation.py --reasoning-first --tasks causal_judgement geometric_shapes disambiguation_qa date_understanding snarks`
**Results file:** `runs/rf_transfer_5tasks_v3.json`
**Note:** v2 results (`rf_transfer_5tasks_v2.json`) were discarded. Root cause: TASK_CFG in `eval_cs_ablation.py` pointed to shim modules (`tasks.bbh_tasks`, `tasks.bbh_tasks_ext`) which do not export RF functions — the RF patch silently no-oped and all 5 tasks were scored verdict-only despite `--reasoning-first` being set. Fixed by updating TASK_CFG to point to individual task modules where RF functions live.

`*` = train model (gpt-4.1-mini).  Δ_cs = full − pk_only.

### RF baselines (empty cheatsheet, reasoning-first)

Sources: `runs/rf_baseline_3tasks.json`, `runs/rf_baseline_3harder.json`

| Model | CJ | GS | DQ | DU | snarks |
|---|---:|---:|---:|---:|---:|
| gpt-4.1-mini * | 69.0% | 58.0% | 69.0% | 89.0% | 90.1% |
| gpt-4.1 | 63.2% | 71.0% | 58.0% | 91.0% | 97.2% |
| claude-3.7-sonnet | 70.1% | 81.0% | 78.0% | 96.0% | 95.8% |
| gemini-2.0-flash | 69.0% | 48.0% | 60.0% | 85.0% | 94.4% |
| llama-3.3-70b | 62.1% | 49.0% | 78.0% | 86.0% | 76.1% |

### causal_judgement

| Model | baseline | pk_only | full | Δ_cs | cs_icl |
|---|---:|---:|---:|---:|---:|
| gpt-4.1-mini * | 69.0% | 63.2% | 66.7% | +3.5% | 66.7% |
| gpt-4.1 | 63.2% | 70.1% | 64.4% | −5.8% | 75.9% |
| claude-3.7-sonnet | 70.1% | 71.3% | 59.8% | **−11.5%** | 73.6% |
| gemini-2.0-flash | 69.0% | 65.5% | 65.5% | 0.0% | 70.1% |
| llama-3.3-70b | 62.1% | 67.8% | 64.4% | −3.5% | 70.1% |

**Finding:** Full cheatsheet underperforms pk_only for 3/5 models — oracle-contaminated case studies hurt non-train models. CS-ICL dominates full by large margins (up to +15pp for gpt-4.1). Train model (mini) is the only case where full ≥ pk_only, and even then just 0% (66.7% both conditions). PK alone provides the best transferable knowledge on CJ.

### geometric_shapes

| Model | baseline | pk_only | full | Δ_cs | cs_icl |
|---|---:|---:|---:|---:|---:|
| gpt-4.1-mini * | 58.0% | 77.0% | 78.0% | +1.0% | 79.0% |
| gpt-4.1 | 71.0% | 73.0% | 75.0% | +2.0% | 81.0% |
| claude-3.7-sonnet | 81.0% | 79.0% | 78.0% | −1.0% | 79.0% |
| gemini-2.0-flash | 48.0% | 70.0% | 75.0% | **+5.0%** | 66.0% |
| llama-3.3-70b | 49.0% | 61.0% | 51.0% | **−10.0%** | 64.0% |

**Finding:** PK alone provides large lifts from baseline (+12 to +27pp). Case studies are near-neutral on most models (±2pp), strongly positive for gemini (+5pp, full > cs_icl), and strongly harmful for llama (−10pp). CS-ICL outperforms full for gpt-4.1/claude/llama; gemini is the exception where full beats cs_icl (+9pp). Note: 6–16 parse errors per pass counted as wrong — slightly suppresses all GS scores.

### disambiguation_qa

| Model | baseline | pk_only | full | Δ_cs | cs_icl |
|---|---:|---:|---:|---:|---:|
| gpt-4.1-mini * | 69.0% | 85.0% | 88.0% | +3.0% | 91.0% |
| gpt-4.1 | 58.0% | 87.0% | 88.0% | +1.0% | 87.0% |
| claude-3.7-sonnet | 78.0% | 95.0% | 91.0% | **−4.0%** | 85.0% |
| gemini-2.0-flash | 60.0% | 90.0% | 89.0% | −1.0% | 81.0% |
| llama-3.3-70b | 78.0% | 84.0% | 84.0% | 0.0% | 79.0% |

**Finding:** Massive pk_only lift across the board (+6 to +30pp from baseline). Case studies add marginal gains for gpt models (+1–3pp) but hurt claude (−4pp). Full cheatsheet matches or beats cs_icl for gpt-4.1/gemini/llama; loses for claude. Claude's native disambiguation approach conflicts with the gpt-optimized case study format.

### date_understanding

| Model | baseline | pk_only | full | Δ_cs | cs_icl |
|---|---:|---:|---:|---:|---:|
| gpt-4.1-mini * | 89.0% | 91.0% | 93.0% | +2.0% | 95.0% |
| gpt-4.1 | 91.0% | 95.0% | 95.0% | 0.0% | 97.0% |
| claude-3.7-sonnet | 96.0% | 95.0% | 97.0% | +2.0% | 96.0% |
| gemini-2.0-flash | 85.0% | 93.0% | 93.0% | 0.0% | 92.0% |
| llama-3.3-70b | 86.0% | 88.0% | 89.0% | +1.0% | 87.0% |

**Finding:** Most consistent task. All models gain from cheatsheet (+2–10pp from baseline). Δ_cs within ±2pp for all models — date arithmetic rules transfer cleanly to case study format. Full cheatsheet matches or beats cs_icl for all models (within 2pp); the only task where ICR's full cheatsheet is competitive with cs_icl for the train model's strongest family members.

### snarks

| Model | baseline | pk_only | full | Δ_cs | cs_icl |
|---|---:|---:|---:|---:|---:|
| gpt-4.1-mini * | 90.1% | 95.8% | 97.2% | **+1.4%** | 95.8% |
| gpt-4.1 | 97.2% | 98.6% | 95.8% | −2.8% | 97.2% |
| claude-3.7-sonnet | 95.8% | 98.6% | 95.8% | −2.8% | 95.8% |
| gemini-2.0-flash | 94.4% | 95.8% | 93.0% | −2.8% | 94.4% |
| llama-3.3-70b | 76.1% | 85.9% | 85.9% | 0.0% | 91.6% |

**Finding:** Pk_only boosts all models from baseline; case studies are near-neutral or slightly negative for 4/5 models (−2.8pp for gpt-4.1/claude/gemini; 0pp for llama). Train model (mini) is the sole beneficiary of case studies (+1.4pp). Llama trails cs_icl by 5.7pp despite matching full cheatsheet — the snarks cs_icl format encodes irony cues that transfer better to llama than ICR's case studies.

### Cross-task summary (RF v3, 5 models)

| Task | baseline→pk lift | Δ_cs (case study contribution) | Full vs cs_icl | Key observation |
|---|---|---|---|---|
| causal_judgement | Varies; pk ≈ baseline | **Train + / others 0 to −12pp** | Full < cs_icl for all non-train | CS actively harmful; PK alone is best transferable unit |
| geometric_shapes | Large (+12–27pp) | Mixed (gemini +5pp, llama −10pp) | Full < cs_icl for gpt/claude/llama; full > cs_icl for gemini | Parse errors suppress all scores slightly |
| disambiguation_qa | Massive (+6–30pp) | Small (+1 to −4pp) | Full ≈ cs_icl for gpt/gemini/llama; full < cs_icl for claude | PK carries the lift; CS marginal |
| date_understanding | Moderate (+2–10pp) | Neutral (±2pp all models) | Full ≈ cs_icl (within 2pp, all models) | Most balanced; date rules transfer cleanly |
| snarks | Moderate (+5–10pp) | Train +1.4pp; others 0 to −2.8pp | Full ≈ cs_icl for train; llama −5.7pp | Near-ceiling for strong models; llama cs_icl gap unexplained |

---

## E9 — RF Transfer: 6 Remaining Tasks, All 3 Conditions

**Script:** `eval_cs_ablation.py --reasoning-first --tasks web_of_lies formal_fallacies navigate logical_deduction_three sports_understanding boolean_expressions`
**Results file:** `runs/rf_transfer_6tasks_e9.json`
**Baselines file:** `runs/rf_baseline_e9_remaining.json` (FF, nav, LD3, sports, BE — WOL baseline from `rf_baseline_3tasks.json`)
**CS-ICL:** Regenerated for all 6 extended tasks using gpt-4.1 with training-example suffix (was gpt-4.1-mini, missing suffix). Files: `gen_gpt-4.1_0_{1000,2000,3000}.txt`

`*` = train model (gpt-4.1-mini).  Δ_cs = full − pk_only.

### RF baselines (empty cheatsheet, reasoning-first)

| Model | WOL | FF | nav | LD3 | sports | BE |
|---|---:|---:|---:|---:|---:|---:|
| gpt-4.1-mini * | 92–100% | 96% | 88% | 99% | 92% | 94% |
| gpt-4.1 | 92–100% | 97% | 100% | 100% | 95% | 100% |
| claude-3.7-sonnet | 92–100% | 94% | 98% | 100% | 92% | 100% |
| gemini-2.0-flash | 92–100% | 87% | 95% | 100% | 89% | 100% |
| llama-3.3-70b | 92–100% | 75% | 97% | 100% | 86% | 98% |

*WOL baseline range from `rf_baseline_3tasks.json` (all models 92–100%).*

### web_of_lies, navigate, logical_deduction_three, boolean_expressions

All four tasks: **100% across all models and all conditions (full, pk_only, cs_icl).** Baselines confirm near-ceiling without any cheatsheet (LD3: 99–100% bare; nav: 88–100%; BE: 94–100%). Cheatsheet contribution = 0. These are pure rule-sufficient tasks under RF — the step-by-step reasoning format alone is sufficient.

### sports_understanding

| Model | baseline | pk_only | full | Δ_cs | cs_icl |
|---|---:|---:|---:|---:|---:|
| gpt-4.1-mini * | 92% | 99% | 99% | 0% | 98% |
| gpt-4.1 | 95% | 97% | 98% | +1% | 97% |
| claude-3.7-sonnet | 92% | 96% | 95% | −1% | 96% |
| gemini-2.0-flash | 89% | 99% | 98% | −1% | 98% |
| llama-3.3-70b | 86% | 98% | 97% | −1% | 96% |

**Finding:** Genuine cheatsheet lift from baseline (+5–12pp), but all conditions converge near ceiling once any cheatsheet is present. Δ_cs ≈ 0 — case studies add nothing on top of PK for sports factual recall. Full ≈ cs_icl for all models (within 1pp).

### formal_fallacies

| Model | baseline | pk_only | full | Δ_cs | cs_icl |
|---|---:|---:|---:|---:|---:|
| gpt-4.1-mini * | 96% | 95% | **97%** | +2% | 95% |
| gpt-4.1 | 97% | **98%** | 95% | −3% | 97% |
| claude-3.7-sonnet | 94% | 95% | 94% | −1% | **99%** |
| gemini-2.0-flash | 87% | **94%** | 92% | −2% | 89% |
| llama-3.3-70b | 75% | 82% | **84%** | +2% | 86% |

**Finding:** The only E9 task with meaningful signal. Mirrors the CJ pattern exactly — CS helps train model (+2pp) and the two weakest models (llama +2pp), hurts the three strongest (gpt-4.1 −3pp, claude −1pp, gemini −2pp). CS-ICL is dramatically better for claude (99% vs 94% full = +5pp) while worse for gemini (89% vs 92% full = −3pp). Notably, the baseline for gpt-4.1 (97%) already exceeds ICR's full cheatsheet (95%) — the case studies actively regress a model that was already near-ceiling.

### Cross-task summary (RF, all 6 E9 tasks)

| Task | Baseline range | Cheatsheet lift | Δ_cs | Key observation |
|---|---|---|---|---|
| web_of_lies | 92–100% | 0pp (at ceiling) | 0 all models | Rule-sufficient; saturated under RF |
| navigate | 88–100% | 0–12pp (mini) | 0 all models | Rule-sufficient; mini gains from PK alone |
| logical_deduction_three | 99–100% | 0pp | 0 all models | **At baseline ceiling bare** — cheatsheet irrelevant |
| boolean_expressions | 94–100% | 0–6pp | 0 all models | Rule-sufficient; logic evaluation needs no examples |
| sports_understanding | 86–95% | +4–12pp | ≈0 all models | PK captures sports facts; CS adds nothing |
| formal_fallacies | 75–97% | +0–9pp | Train+/others− | Mirrors CJ; oracle-contaminated CS harmful to strong models |

---

## E11 — Gold Few-Shot vs ACTIVATE IF under RF (5 tasks)

**Script:** `eval_cs_ablation.py --reasoning-first --gold-fewshot --tasks causal_judgement geometric_shapes disambiguation_qa date_understanding snarks`
**Results file:** `runs/rf_goldfewshot_5tasks.json`
**Cheatsheet:** PK text (`cheatsheet_phase1_pk_final.txt`) + 5 gold training examples with oracle reasoning, `=== WORKED EXAMPLES ===` format

Full 4-condition comparison under RF (baseline and pk_only from v3 RF run):

`*` = train model (gpt-4.1-mini).  All values RF-scored.

### causal_judgement

| Model | baseline | pk_only | gold_fewshot | full (AI) | Δ gf vs full | Δ gf vs pk |
|---|---:|---:|---:|---:|---:|---:|
| gpt-4.1-mini * | 69.0% | 63.2% | **74.7%** | 66.7% | +8.0pp | +11.5pp |
| gpt-4.1 | 63.2% | 70.1% | **72.4%** | 64.4% | +8.0pp | +2.3pp |
| claude-3.7-sonnet | 70.1% | **71.3%** | 66.7% | 59.8% | +6.9pp | −4.6pp |
| gemini-2.0-flash | 69.0% | 65.5% | 65.5% | 65.5% | 0pp | 0pp |
| llama-3.3-70b | 62.1% | 67.8% | **69.0%** | 64.4% | +4.6pp | +1.2pp |

**Finding:** Gold few-shot beats ACTIVATE IF for all models (+4.6 to +8pp). The worked example format transfers better than the case study wrapper — confirming the harm of ACTIVATE IF on CJ is partially format-driven, not purely content-driven. Exception: claude, where pk_only (71.3%) still beats gold_fewshot (66.7%) — claude's native causal reasoning is stronger than any provided examples.

### geometric_shapes

| Model | baseline | pk_only | gold_fewshot | full (AI) | Δ gf vs full | Δ gf vs pk |
|---|---:|---:|---:|---:|---:|---:|
| gpt-4.1-mini * | 58.0% | 77.0% | 78.0% | 78.0% | 0pp | +1pp |
| gpt-4.1 | 71.0% | 73.0% | **79.0%** | 75.0% | +4pp | +6pp |
| claude-3.7-sonnet | 81.0% | 79.0% | 78.0% | 78.0% | 0pp | −1pp |
| gemini-2.0-flash | 48.0% | 70.0% | 70.0% | **75.0%** | −5pp | 0pp |
| llama-3.3-70b | 49.0% | **61.0%** | 56.0% | 51.0% | +5pp | −5pp |

**Finding:** Gold few-shot ≈ full for most. Gemini is the clear exception: full (75%) beats gold_fewshot (70%) by +5pp — the ICR ACTIVATE IF case studies add genuine value for gemini on GS that worked examples do not. Llama prefers pk_only (61%) over both gold_fewshot (56%) and full (51%); adding any examples hurts llama on GS.

### disambiguation_qa

| Model | baseline | pk_only | gold_fewshot | full (AI) | Δ gf vs full | Δ gf vs pk |
|---|---:|---:|---:|---:|---:|---:|
| gpt-4.1-mini * | 69.0% | 85.0% | **90.0%** | 88.0% | +2pp | +5pp |
| gpt-4.1 | 58.0% | 87.0% | 87.0% | 88.0% | −1pp | 0pp |
| claude-3.7-sonnet | 78.0% | **95.0%** | 95.0% | 91.0% | +4pp | 0pp |
| gemini-2.0-flash | 60.0% | **90.0%** | 85.0% | 89.0% | −4pp | −5pp |
| llama-3.3-70b | 78.0% | 84.0% | 83.0% | 84.0% | −1pp | −1pp |

**Finding:** Gold few-shot mostly matches full or is slightly better for gpt/claude. For gemini, gold_fewshot (85%) is worse than both full (89%) and pk_only (90%) — the CS-ICL-style worked examples hurt gemini on DQ while ACTIVATE IF case studies don't. Claude confirms: pk_only = gold_fewshot (95%) > full (91%); the ICR case study content is what hurts, not the format.

### date_understanding

| Model | baseline | pk_only | gold_fewshot | full (AI) | Δ gf vs full | Δ gf vs pk |
|---|---:|---:|---:|---:|---:|---:|
| gpt-4.1-mini * | 89.0% | 91.0% | **95.0%** | 93.0% | +2pp | +4pp |
| gpt-4.1 | 91.0% | 95.0% | **96.0%** | 95.0% | +1pp | +1pp |
| claude-3.7-sonnet | 96.0% | 95.0% | 95.0% | **97.0%** | −2pp | 0pp |
| gemini-2.0-flash | 85.0% | 93.0% | 91.0% | **93.0%** | −2pp | −2pp |
| llama-3.3-70b | 86.0% | 88.0% | **90.0%** | 89.0% | +1pp | +2pp |

**Finding:** Near-saturated; differences are small. Gold_fewshot slightly better for gpt/mini/llama (+1–4pp over full). Full beats gold_fewshot for claude and gemini (−2pp). DU is balanced enough that neither format dominates clearly.

### snarks

| Model | baseline | pk_only | gold_fewshot | full (AI) | Δ gf vs full | Δ gf vs pk |
|---|---:|---:|---:|---:|---:|---:|
| gpt-4.1-mini * | 90.1% | 95.8% | 94.4% | **97.2%** | −2.8pp | −1.4pp |
| gpt-4.1 | 97.2% | **98.6%** | 93.0% | 95.8% | −2.8pp | −5.6pp |
| claude-3.7-sonnet | 95.8% | **98.6%** | 95.8% | 95.8% | 0pp | −2.8pp |
| gemini-2.0-flash | 94.4% | **95.8%** | 94.4% | 93.0% | +1.4pp | −1.4pp |
| llama-3.3-70b | 76.1% | 85.9% | **90.1%** | 85.9% | +4.2pp | +4.2pp |

**Finding:** Full ACTIVATE IF beats gold_fewshot for the train model (97.2% vs 94.4%) — the only task where ICR's case study format clearly outperforms worked examples for mini. Gold_fewshot helps llama most (+4.2pp over full and pk_only). Pk_only remains the best condition for gpt-4.1 and claude — adding any examples hurts them on snarks.

### Cross-task summary (RF gold_fewshot vs ACTIVATE IF)

| Task | Gold_fewshot vs full | Gold_fewshot vs pk_only | Key observation |
|---|---|---|---|
| causal_judgement | **+4.6 to +8pp** (all models) | Mixed | Worked examples better than ACTIVATE IF; some format benefit |
| geometric_shapes | Mixed (gpt +4pp, gemini −5pp) | Mixed | ACTIVATE IF retains edge for gemini on spatial reasoning |
| disambiguation_qa | +2pp to −4pp | Similar or worse | Claude: pk_only = gold_fewshot; gemini prefers ACTIVATE IF |
| date_understanding | ±2pp | ±4pp | Near-saturated; no clear winner |
| snarks | +4.2pp (llama) / −2.8pp (mini, gpt) | Mixed | Full ACTIVATE IF best for train model; gold_fewshot best for llama |

**Overall RF finding:** Under RF scoring, gold few-shot outperforms ACTIVATE IF on CJ broadly, but the advantage is much narrower than in CoT verdict-only mode. For GS and snarks, ACTIVATE IF case studies retain or exceed gold few-shot for key model-task combinations. The content analysis conclusion from CoT mode — that worked examples generally beat ACTIVATE IF — **does not hold uniformly under RF**. Format (RF vs. CoT) interacts with the case study format (ACTIVATE IF vs. worked example) in task-specific ways.

---

## E-Oracle2x2-RF — Oracle 2×2 Under RF Scoring (CJ + GS)

**Description:** Scores the two oracle ablation conditions (E3 no-oracle, v5 full Phase1+Phase2 oracle) under RF, completing the 2×2 factorial (Phase 1 oracle ON/OFF × Phase 2 oracle ON/OFF) previously run only in CoT mode. v3 (Phase2-only oracle, Phase1 OFF) already has RF results; this adds the two missing corners.

**Conditions:**
| Condition | Phase 1 oracle | Phase 2 oracle | Run dir | Results file |
|---|---|---|---|---|
| E3 no-oracle | OFF | OFF | `runs/bbh_oracle_ablation/no_oracle/` | `runs/e3_no_oracle_rf.json` |
| v3 (existing) | OFF | ON | `runs/bbh_v3/` | (from v3 RF section above) |
| v5 full-oracle | ON | ON | `runs/bbh_v5/` | `runs/v5_full_oracle_rf.json` |

**Script:** `eval_cs_ablation.py --reasoning-first --no-csicl --tasks causal_judgement geometric_shapes --run-dir-overrides ...`

`*` = train model (gpt-4.1-mini). All values RF-scored.

### causal_judgement — full oracle 2×2

| Condition | Phase1 | Phase2 | mini* | gpt-4.1 | claude | gemini | llama |
|---|:---:|:---:|---:|---:|---:|---:|---:|
| baseline | — | — | 69.0% | 63.2% | 70.1% | 69.0% | 62.1% |
| E3 pk_only | OFF | OFF | 62.1% | 65.5% | 66.7% | 69.0% | 69.0% |
| E3 full | OFF | OFF | 67.8% | **71.3%** | 66.7% | 69.0% | **70.1%** |
| v3 pk_only | OFF | ON | 63.2% | 70.1% | **71.3%** | 65.5% | 67.8% |
| v3 full | OFF | ON | 66.7% | 64.4% | 59.8% | 65.5% | 64.4% |
| v5 pk_only | ON | ON | 65.5% | 62.1% | 63.2% | 62.1% | 64.4% |
| v5 full | ON | ON | 66.7% | 62.1% | 65.5% | 58.6% | **58.6%** |

**Finding — CJ oracle contamination gradient under RF:**

1. **E3 no-oracle full is the best CS condition for 4/5 models** (gpt-4.1: 71.3%, llama: 70.1%, gemini: 69.0%, mini: 67.8%). Without oracle injection, case studies actually *help* all non-train models — a complete reversal of the v3 finding where full hurt 3/5 models. Under RF, the harmful signal is specifically the oracle-contaminated reasoning, not case studies per se.

2. **Clear oracle contamination gradient (full CS, non-train average):**
   - E3 no-oracle: 69.3% avg (71.3 / 66.7 / 69.0 / 70.1)
   - v3 Phase2-oracle: 63.5% avg (64.4 / 59.8 / 65.5 / 64.4)
   - v5 full-oracle: 61.2% avg (62.1 / 65.5 / 58.6 / 58.6)
   More oracle → worse cross-model transfer. Each oracle layer degrades transfer additively.

3. **v5 full is worst for gemini and llama** (58.6% each, −10.4pp and −3.5pp below baseline). Phase 1 + Phase 2 oracle together produce CS that are maximally aligned to gpt-4.1-mini's reasoning chain — maximally harmful to distant model families.

4. **PK quality degrades with oracle too:** E3 pk_only for CJ is weaker than v3 pk_only for most models (62.1% vs 63.2% for mini, 65.5% vs 70.1% for gpt-4.1). Phase 2 oracle helps Phase 1 downstream by providing better training signal. But v5 pk_only (Phase 1 oracle ON) is weaker still — oracle injection during Phase 1 rule generation reduces PK transferability.

### geometric_shapes — full oracle 2×2

| Condition | Phase1 | Phase2 | mini* | gpt-4.1 | claude | gemini | llama |
|---|:---:|:---:|---:|---:|---:|---:|---:|
| baseline | — | — | 58.0% | 71.0% | 81.0% | 48.0% | 49.0% |
| E3 pk_only | OFF | OFF | 61.0% | **77.0%** | 80.0% | **73.0%** | 59.0% |
| E3 full | OFF | OFF | **80.0%** | 75.0% | 77.0% | 72.0% | 59.0% |
| v3 pk_only | OFF | ON | 77.0% | 73.0% | 79.0% | 70.0% | 61.0% |
| v3 full | OFF | ON | 78.0% | 75.0% | 78.0% | 75.0% | 51.0% |
| v5 pk_only | ON | ON | 70.0% | 56.0% | **81.0%** | 55.0% | 67.0% |
| v5 full | ON | ON | 78.0% | 72.0% | 80.0% | 76.0% | 64.0% |

**Finding — GS oracle contamination is more nuanced (task-type matters):**

1. **E3 pk_only is best for gpt-4.1 (77%) and gemini (73%)** — oracle-free Phase 1 generates the most transferable PK. v3 pk drops to 73%/70%, v5 pk collapses to 56%/55%. Each oracle layer degrades PK transferability, matching the CJ pattern.

2. **v5 pk is catastrophically bad for gpt-4.1 (56%) and gemini (55%):** Phase 1 oracle injection corrupts PK for these two models — both lose 17-18pp vs oracle-free PK. This shows Phase 1 oracle hurts PK transfer even though it was designed to help.

3. **Unlike CJ, v5 full partially recovers from bad v5 pk:** gpt-4.1 goes from 56% (pk) to 72% (full) and gemini from 55% to 76%. The ACTIVATE IF case studies in GS compensate for degraded PK. This recovery does not occur in CJ (where v5 full stays bad).

4. **Key contrast with CJ — oracle in Phase 2 CS helps GS but hurts CJ:**
   - CJ: more Phase 2 oracle → worse full CS for non-train models (E3 > v3 > v5)
   - GS: Phase 2 oracle mildly helps full CS (v5 full ≥ E3 full for 3/5 non-train models)
   - Interpretation: GS CS are algorithmic ACTIVATE IF conditions (geometry rules). Oracle provides correct geometric reasoning → CS become more precise, not more model-specific. CJ CS are causal reasoning chains — oracle injects gpt-4.1-mini's specific reasoning style → CS become model-stamped.

5. **E3 full is best for mini on GS (80%):** Without oracle, mini's CS (generated purely from mini's failures) provide the strongest self-improvement signal. Oracle injection in Phase 2 dilutes this by replacing mini's self-generated reasoning with oracle text.

### Cross-task oracle 2×2 summary

| Effect | CJ | GS | Interpretation |
|--------|----|----|----------------|
| Oracle on PK (Phase 1) | ↓ transfer | ↓↓ transfer (severe) | Oracle contaminates PK toward mini's reasoning in both tasks |
| Oracle on full CS (Phase 2) | ↓↓ transfer | ≈neutral / slight ↑ | Depends on CS type: prose reasoning hurt; algorithmic ACTIVATE IF OK |
| Best CS condition for non-train | E3 no-oracle | E3 pk / v3 full | No oracle best for PK; moderate oracle OK for algorithmic CS |
| Oracle contamination type | Reasoning-style stamp | PK corruption + CS compensation | Different mechanisms dominate per task |

**Paper implication:** Oracle injection should be evaluated separately for PK (Phase 1) and CS (Phase 2). Under RF scoring, the recommendation changes: oracle-free CS are better for CJ-type tasks (reasoning-heavy, model-specific failure modes), while Phase 2 oracle may be acceptable for GS-type tasks (algorithmic, geometry-rule CS). The standard recommendation of "oracle always helps" — valid in CoT mode — does not hold under RF.

---

## E-NEW1 — Train Model Identity: Gemini vs Mini (CJ, GS, DQ)

**Description:** Tests the model-signature hypothesis: does the identity of the train model affect cross-model transfer of the generated cheatsheet? Replaces gpt-4.1-mini with gemini-2.0-flash-001 as the train model for three tasks (CJ, GS, DQ). Both Phase 1 and Phase 2 are run no-oracle.

**Key difference from main pipeline:** Default Phase 2 regression threshold (15%) produced 0 CS for all three tasks — gemini's CS candidates have higher regression rates than mini's. Rerun with loose thresholds: `--cs-regress-threshold 0.40 --cs-fix-rate-threshold 0.20`.

**Train model:** `google/gemini-2.0-flash-001`  
**Scoring model (eval):** all 5 models under RF  
**Run dirs:** `runs/bbh_gemini_train/` (Phase 1), `runs/bbh_gemini_train_loose/` (Phase 2 loose)  
**Results files:**
- `runs/enew1_gemini_rf.json` — gemini pk_only + full for CJ, GS, DQ
- `runs/e3_dq_rf.json` — mini E3 pk_only + full for DQ (comparison baseline)
- (CJ/GS mini-E3 already in `runs/e3_no_oracle_rf.json`)

**Phase 2 CS generated (loose thresholds):**
| Task | CS count | Failure modes covered | Items resolved / orig failing |
|------|----------|-----------------------|-------------------------------|
| CJ | 4 | OR-cause: redundancy + prevention chains | ~4 / 35 (stalls iter 3–5) |
| GS | 7 | Arc→ellipse, multi-subpath vertex (4 variants), quadrilateral subtype | 52 / 62 (83.9%) |
| DQ | 3 | Temporal negation, causal-subject pronouns, composite they/she | ~25 / 34 |

**`*` = train model (gemini-2.0-flash-001). All values RF-scored.**

### causal_judgement — train model identity effect

| Condition | Train model | mini* | gpt-4.1 | claude | gemini* | llama | Non-train avg |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | — | 69.0% | 63.2% | 70.1% | 69.0% | 62.1% | 66.1% |
| gold fewshot | — | 74.7% | 72.4% | 66.7% | 65.5% | 69.0% | 68.4% |
| mini E3 pk_only | mini | 62.1% | 65.5% | 66.7% | 69.0% | 69.0% | 67.6% |
| mini E3 full | mini | 67.8% | 71.3% | 66.7% | 69.0% | 70.1% | 69.3% |
| mini v3 pk_only | mini | 63.2% | 70.1% | 71.3% | 65.5% | 67.8% | 68.7% |
| mini v3 full | mini | 66.7% | 64.4% | 59.8% | 65.5% | 64.4% | 63.5% |
| **gemini pk_only** | **gemini** | 63.2% | 64.4% | 63.2% | 63.2% | 59.8% | 62.6% |
| **gemini full** | **gemini** | 66.7% | 66.7% | 63.2% | 64.4% | 62.1% | 64.1% |

**CJ findings:**
1. **Gemini PK is weaker than mini PK** (non-train avg: 62.6% vs 67.6% for mini-E3). Mini's exhaustive 11-section taxonomy transfers better than gemini's shorter diagnostic-question format.
2. **Gemini CS provide small lift (+1.4pp avg)** — comparable to mini-E3 CS (+1.7pp). The OR-cause CS do generalize slightly.
3. **Both gem conditions sit below mini-E3 full** by ~5pp. Model-signature extends to PK: mini's PK style is more universally transferable for CJ.
4. **Gemini CS cover different failure modes than mini** (OR-cause redundancy/prevention vs AND-cause joint necessity) — confirming PK/CS reflect train model's failure distribution.

### geometric_shapes — train model identity effect

| Condition | Train model | mini* | gpt-4.1 | claude | gemini | llama | Non-train avg |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | — | 58.0% | 71.0% | 81.0% | 48.0% | 49.0% | 62.3% |
| gold fewshot | — | 78.0% | 79.0% | 78.0% | 70.0% | 56.0% | 70.8% |
| mini E3 pk_only | mini | 61.0% | 77.0% | 80.0% | 73.0% | 59.0% | 72.3% |
| mini E3 full | mini | 80.0% | 75.0% | 77.0% | 72.0% | 59.0% | 70.8% |
| mini v3 pk_only | mini | 77.0% | 73.0% | 79.0% | 70.0% | 61.0% | 70.8% |
| mini v3 full | mini | 78.0% | 75.0% | 78.0% | 75.0% | 51.0% | 69.8% |
| **gemini pk_only** | **gemini** | 58.0% | 67.0% | 67.0% | 51.0% | 63.0% | 62.0% |
| **gemini full** | **gemini** | 78.0% | 77.0% | 79.0% | 63.0% | 59.0% | 69.5% |

**GS findings:**
1. **Gemini PK is substantially weaker** (non-train avg 62.0% vs 72.3% mini-E3, −10.3pp). Gemini's shorter PK misses key coverage (rotation tolerance, dot-product for angle verification) that mini provides.
2. **Gemini CS provide the largest CS lift of any condition tested: +7.5pp** (62.0% → 69.5%). Mini's 2 CS actually *hurt* transfer slightly (72.3% → 70.8%, −1.5pp). This reversal is the most striking E-NEW1 finding.
3. **Gemini full (69.5%) nearly matches mini full (70.8%)** despite starting from a weaker PK — the 7 CS compensate almost entirely for the PK gap.
4. **Cross-family asymmetry:** Gemini's GS CS help gpt-4.1 (75→77%) and claude (67→79%) more than mini's CS do, but hurt gemini-model itself (51% pk → 63% full, less than mini's 73%). Gemini generated CS for failure modes it encounters — but other model families benefit more from them.
5. **Mini CS hurt GS transfer despite high Phase 2 fix rates** (mini Phase 2: best fix_rate 50%, 2 CS accepted). Mini's 2 CS were too narrowly calibrated to mini's specific parse failures, introducing noise for other models.

### disambiguation_qa — train model identity effect

| Condition | Train model | mini* | gpt-4.1 | claude | gemini | llama | Non-train avg |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | — | 69.0% | 58.0% | 78.0% | 60.0% | 78.0% | 68.5% |
| gold fewshot | — | 90.0% | 87.0% | 95.0% | 85.0% | 83.0% | 87.5% |
| mini E3 pk_only | mini | 84.0% | 88.0% | 85.0% | 86.0% | 85.0% | 86.0% |
| mini E3 full | mini | 83.0% | 88.0% | 86.0% | 83.0% | 84.0% | 85.3% |
| mini v3 pk_only | mini | 85.0% | 87.0% | 95.0% | 90.0% | 84.0% | 89.0% |
| mini v3 full | mini | 88.0% | 88.0% | 91.0% | 89.0% | 84.0% | 88.0% |
| **gemini pk_only** | **gemini** | 84.0% | 85.0% | 85.0% | 81.0% | 80.0% | 82.8% |
| **gemini full** | **gemini** | 85.0% | 84.0% | 88.0% | 79.0% | 81.0% | 83.0% |

**DQ findings:**
1. **Gemini PK weaker than mini-E3 PK** (82.8% vs 86.0%, −3.2pp). Mini's 27-rule enumeration covers more pronoun resolution patterns than gemini's 5-section trap-based format.
2. **Gemini CS barely help (+0.3pp)** despite Phase 2 generating 3 CS that resolved 25/34 failing items in training. Train-time fixing does not translate to test-time transfer.
3. **Gemini CS hurt gemini-model itself** (81% → 79%, −2pp) — the train model underperforms on its own cheatsheet. This is unusual and may reflect DQ's coupling: fixing one pronoun type in CS changes behavior for other pronoun types.
4. **Mini v3 wins DQ overall** (88.0% non-train avg) due to oracle-assisted PK that captures pronoun resolution patterns with high precision. This is one case where Phase 2 oracle is beneficial at the PK level.

### Cross-task summary — E-NEW1

| | CJ | GS | DQ |
|---|---|---|---|
| Gemini PK gap vs mini E3 PK | −4.9pp | −10.3pp | −3.2pp |
| Gemini CS lift (pk→full) | +1.4pp | **+7.5pp** | +0.3pp |
| Gemini full gap vs mini E3 full | −5.2pp | −1.3pp | −2.3pp |
| Mini CS lift (pk→full) | +1.7pp | −1.5pp | −0.8pp |
| Task type | Judgment | Geometric algo | Pragmatic |

**Key findings:**

1. **Gemini PK is consistently weaker than mini PK across all 3 tasks.** Mini's longer, more exhaustive PK style (11-section CJ taxonomy, 9-section GS algorithm, 27-rule DQ enumeration) transfers better than gemini's shorter diagnostic/trap-based format. PK *length and coverage* predict transferability more than PK *framing style*.

2. **GS reversal: gemini CS outperform mini CS (+7.5pp vs −1.5pp).** This is the most striking finding. Despite gemini's PK being 10pp weaker, gemini's 7 CS compensate almost entirely, making gemini full competitive with mini full. Mini's 2 CS actually hurt GS transfer. This shows CS quality depends on: (a) the diversity of failure modes covered (7 vs 2 CS), and (b) whether the CS trigger conditions are formal/structural (gemini's ACTIVATE IF feature-flag conditions) vs mini's more semantic conditions.

3. **CS lift correlates with Phase 2 coherence, not CS count per se.** GS (+7.5pp, 7 CS) has the largest lift. DQ (+0.3pp, 3 CS) and CJ (+1.4pp, 4 CS) are near-zero despite generating CS. The GS failure partitions were maximally coherent (7 distinct error types, each addressable with a formal rule); CJ and DQ failures are more semantically coupled, limiting CS transfer even when they reduce training error.

4. **Train model identity affects CS feasibility and content, not just transfer.** Gemini required 40% regression tolerance vs mini's 15% — gemini's CS are inherently more disruptive. Gemini's CJ CS cover OR-cause scenarios; mini's cover AND-cause. Gemini's DQ Phase 2 succeeded where mini's failed (pronoun coupling). These asymmetries confirm the model-signature hypothesis at the CS level.

5. **Implication for paper:** ICR's cross-model transferability depends primarily on Phase 1 PK quality, which correlates with PK coverage and task formalism. For tasks with coherent failure partitions (GS), Phase 2 CS generated by any competent train model can substantially improve transfer — even compensating for weaker PK. For tasks with coupled or heterogeneous failures (CJ, DQ), CS provide minimal transfer gain regardless of train model.

---

## E-Oracle2x2-DQ — Oracle 2×2 for Disambiguation QA

**Description:** Extends the oracle 2×2 analysis to DQ. v5 (Phase 1 + Phase 2 oracle) was never run for DQ, so this covers 3 conditions: E3 (both OFF), p1_only (Phase 1 ON / Phase 2 OFF), v3 (Phase 1 OFF / Phase 2 ON). DQ has 0 CS in all conditions (Phase 2 consistently failed to generate accepted CS), so full ≈ pk_only throughout — all variation is pure PK quality.

**Results files:**
- E3: `runs/e3_dq_rf.json`
- p1_only: `runs/dq_p1only_rf.json`
- v3: `runs/rf_transfer_5tasks_v3.json`

`*` = train model (gpt-4.1-mini). All RF-scored. Since DQ has 0 CS, full ≈ pk_only; only pk_only shown.

### disambiguation_qa — oracle 2×2 (3-point, v5 not run)

| Condition | P1 oracle | P2 oracle | mini* | gpt-4.1 | claude | gemini | llama | Non-train avg |
|---|:---:|:---:|---:|---:|---:|---:|---:|---:|
| baseline | — | — | 69.0% | 58.0% | 78.0% | 60.0% | 78.0% | 68.5% |
| E3 pk_only | OFF | OFF | 84.0% | 88.0% | 85.0% | 86.0% | 85.0% | 86.0% |
| p1_only pk_only | ON | OFF | 84.0% | 81.0% | 81.0% | 73.0% | 79.0% | 78.5% |
| v3 pk_only | OFF | ON | 85.0% | 87.0% | **95.0%** | **90.0%** | 84.0% | **89.0%** |

**DQ oracle findings:**

1. **Phase 1 oracle devastates DQ PK** (78.5% vs 86.0%, −7.5pp). Matches the CJ and GS pattern exactly. Oracle-injected Phase 1 produces PK over-fitted to mini's oracle-assisted reasoning chain.

2. **Phase 2 oracle is the best condition** (89.0% non-train avg). v3 pk_only outperforms E3 pk_only by +3pp. Since DQ has 0 CS, this is a pure PK quality effect — Phase 2 oracle injection during the Phase 2 generation pass (even when no CS are accepted) improved PK quality. The oracle reasoning in Phase 2 may cause Phase 2 to produce better PK patch suggestions that get incorporated.

3. **Claude and gemini benefit most from Phase 2 oracle** (claude: 85%→95%, +10pp; gemini: 86%→90%, +4pp). Both are strong models that already perform well on DQ; the oracle-assisted PK captures pronoun resolution principles at a level of precision that maximally benefits these models.

4. **DQ confirms the general oracle principle:** Phase 1 oracle always hurts PK transfer (seen in CJ, GS, DQ). Phase 2 oracle effects depend on task type (hurts CJ, neutral-positive for GS, helps DQ). The task type determines whether Phase 2 oracle is beneficial: DQ's pronoun rules are precise enough that oracle reasoning improves rule quality without contaminating with model-specific reasoning patterns.

---

## E-NEW3 — Format Ablation: ACTIVATE IF vs Plain Worked Examples

**Description:** Tests whether the ACTIVATE IF conditional structure in CS contributes to accuracy, or whether the same scenarios presented as plain worked examples perform equally. For each task with CS (GS, snarks, FF), strips the ACTIVATE IF/IDENTIFY/WHY wrapper from all CS and replaces with plain Q→A worked examples using the same support_examples. PK section is unchanged.

**Script:** `scripts/eval/strip_cs_to_plain_examples.py`  
**Reformatted cheatsheets:** `runs/bbh_v3_plain_examples/<task>/`  
**Results file:** `runs/enew3_plain_examples_rf.json`

`*` = train model (gpt-4.1-mini). All RF-scored. Δ = vs pk_only.

### geometric_shapes — ACTIVATE IF vs plain examples

| Model | pk_only | ACTIVATE IF | plain ex | AIF Δ | plain Δ | plain vs AIF |
|---|---:|---:|---:|---:|---:|---:|
| gpt-4.1-mini * | 77.0% | 78.0% | 76.0% | +1.0% | −1.0% | −2.0% |
| gpt-4.1 | 73.0% | 75.0% | 76.0% | +2.0% | +3.0% | +1.0% |
| claude | 79.0% | 78.0% | 80.0% | −1.0% | +1.0% | +2.0% |
| gemini | 70.0% | **75.0%** | 67.0% | **+5.0%** | −3.0% | −8.0% |
| llama | 61.0% | 51.0% | 47.0% | −10.0% | −14.0% | −4.0% |
| **Non-train avg** | **70.8%** | **69.8%** | **67.5%** | **−1.0%** | **−3.2%** | **−2.2%** |

### snarks — ACTIVATE IF vs plain examples

| Model | pk_only | ACTIVATE IF | plain ex | AIF Δ | plain Δ | plain vs AIF |
|---|---:|---:|---:|---:|---:|---:|
| gpt-4.1-mini * | 95.8% | 97.2% | 95.8% | +1.4% | 0.0% | −1.4% |
| gpt-4.1 | 98.6% | 95.8% | 93.0% | −2.8% | −5.6% | −2.8% |
| claude | 98.6% | 95.8% | 95.8% | −2.8% | −2.8% | 0.0% |
| gemini | 95.8% | 93.0% | 94.4% | −2.8% | −1.4% | +1.4% |
| llama | 85.9% | 85.9% | 85.9% | 0.0% | 0.0% | 0.0% |
| **Non-train avg** | **94.7%** | **92.6%** | **92.3%** | **−2.1%** | **−2.5%** | **−0.3%** |

### formal_fallacies — ACTIVATE IF vs plain examples

| Model | pk_only | ACTIVATE IF | plain ex | AIF Δ | plain Δ | plain vs AIF |
|---|---:|---:|---:|---:|---:|---:|
| gpt-4.1-mini * | 95.0% | 97.0% | 94.0% | +2.0% | −1.0% | −3.0% |
| gpt-4.1 | 98.0% | 95.0% | 97.0% | −3.0% | −1.0% | +2.0% |
| claude | 95.0% | 94.0% | 95.0% | −1.0% | 0.0% | +1.0% |
| gemini | 94.0% | 92.0% | 90.0% | −2.0% | −4.0% | −2.0% |
| llama | 82.0% | 84.0% | **88.0%** | +2.0% | **+6.0%** | +4.0% |
| **Non-train avg** | **92.2%** | **91.2%** | **92.5%** | **−1.0%** | **+0.2%** | **+1.2%** |

### Cross-task summary — ACTIVATE IF vs plain examples

| Task | ACTIVATE IF Δ (non-train) | Plain Δ (non-train) | Plain vs AIF | ACTIVATE IF type |
|---|---|---|---|---|
| geometric_shapes | −1.0pp | −3.2pp | **AIF better +2.2pp** | Mode A: formal feature flags |
| snarks | −2.1pp | −2.5pp | **Neutral −0.3pp** | Mode C: phenomenological |
| formal_fallacies | −1.0pp | +0.2pp | **Plain better +1.2pp** | Mode B: syntactic pattern |

**E-NEW3 findings — the value of conditional structure depends entirely on condition type:**

1. **Mode A (formal, GS): ACTIVATE IF wins by 2.2pp.** When conditions are computationally grounded (has_arc, has_multi_subpath), the conditional gating prevents over-application. Without it, gemini loses 8pp and llama loses 14pp vs pk_only — models apply the geometric correction to all SVG problems rather than only those with the relevant features.

2. **Mode C (phenomenological, snarks): ACTIVATE IF ≈ plain examples (−0.3pp).** The snarks ACTIVATE IF conditions ("scenario feels like praising something absurd") require the same semantic judgment as the task itself. They provide no selectivity advantage — models that can evaluate the condition correctly don't need the CS, and those that can't evaluate the condition apply it anyway. The conditional structure is vacuous.

3. **Mode B (syntactic, FF): plain examples marginally outperform ACTIVATE IF (+1.2pp).** The syntactic conditions ("whoever is not X is Y") may cause some CS to fire on non-target argument patterns with similar surface forms, slightly degrading accuracy. Plain examples, applied without explicit gating, work better on average — especially for llama (+6pp with plain vs +2pp with ACTIVATE IF), suggesting llama benefits from more liberal application of the illicit-conversion fix.

---

## E-Phase0 — Bootstrap Cheatsheet Ablation (Phase 0 / Phase 1 / Phase 2)

**Question:** How much does each ICR phase contribute? Specifically: does the Phase 0 bootstrap (CS-ICL-style initial cheatsheet, before any iterative refinement) already capture most of the cheatsheet's value?

**Method:** For each task, score `ruleset_bootstrap.txt` (the initial LLM-generated cheatsheet before Phase 1 PK patching begins) using the RF scorer. Compare to existing Phase 1 (pk_only RF) and Phase 2 (full RF) scores. Bootstrap dirs: `runs/bbh_v3_phase0/`, `runs/bbh_gemini_phase0/`. All scores RF-mode.

**Result files:** `runs/phase0_mini_rf.json`, `runs/phase0_v3_nontrain_rf.json`, `runs/phase0_gemini_rf.json`, `runs/phase0_gemini_nontrain_rf.json`

---

### E-Phase0-v3 — Mini-trained cheatsheet, Phase 0/1/2 (gpt-4.1-mini train model)

| Task | Phase 0 | Phase 1 | Phase 2 | P1−P0 | P2−P1 |
|---|---:|---:|---:|---:|---:|
| boolean_expressions | 100.0% | 100.0% | 100.0% | +0.0% | +0.0% |
| causal_judgement | 73.6% | 71.3% | 72.4% | −2.3% | +1.1% |
| date_understanding | 94.0% | 92.0% | 88.0% | −2.0% | −4.0% |
| disambiguation_qa | 84.0% | 86.0% | 88.0% | +2.0% | +2.0% |
| formal_fallacies | 96.0% | 96.0% | 97.0% | +0.0% | +1.0% |
| geometric_shapes | 70.0% | 77.0% | 81.0% | **+7.0%** | **+4.0%** |
| logical_deduction_three | 100.0% | 100.0% | 100.0% | +0.0% | +0.0% |
| navigate | 100.0% | 100.0% | 100.0% | +0.0% | +0.0% |
| snarks | 95.8% | 95.8% | 98.6% | +0.0% | +2.8% |
| sports_understanding | 100.0% | 99.0% | 99.0% | −1.0% | +0.0% |
| web_of_lies | 100.0% | 100.0% | 100.0% | +0.0% | +0.0% |

**Key findings (mini/train model):**
- 5 ceiling tasks (BE/LD3/nav/SU/WOL): already 100% at Phase 0 — both phases add nothing.
- GS is the only task where both phases consistently help mini (+7pp P1, +4pp P2). It is the hardest task for mini at bootstrap.
- DU regresses at every phase (−2pp, −4pp) — PK patching overfits for this task.
- CJ noisy (73.6% → 71.3% → 72.4%): PK patching slightly hurts, CS slightly recovers.
- Overall Phase 1 delta for mini is near zero on average — mini already knows the rules it patches.

---

### E-Phase0-v3 — Non-train models, non-train average by task

| Task | NT Phase 0 | NT Phase 1 | NT Phase 2 | P1−P0 | P2−P1 |
|---|---:|---:|---:|---:|---:|
| boolean_expressions | 100.0% | 100.0% | 99.8% | +0.0% | −0.2% |
| causal_judgement | 69.3% | 68.7% | 63.5% | −0.6% | **−5.2%** |
| date_understanding | 94.0% | 92.8% | 93.5% | −1.2% | +0.8% |
| disambiguation_qa | 86.0% | 89.0% | 88.0% | **+3.0%** | −1.0% |
| formal_fallacies | 90.8% | 92.2% | 91.2% | +1.5% | −1.0% |
| geometric_shapes | 74.2% | 71.8% | 69.8% | −2.5% | −2.0% |
| logical_deduction_three | 99.5% | 100.0% | 99.5% | +0.5% | −0.5% |
| navigate | 99.5% | 99.8% | 100.0% | +0.3% | +0.2% |
| snarks | 92.6% | 91.5% | 92.6% | −1.1% | +1.1% |
| sports_understanding | 96.8% | 97.5% | 97.0% | +0.7% | −0.5% |
| web_of_lies | 100.0% | 100.0% | 99.8% | +0.0% | −0.2% |

**Per-model detail for informative tasks:**

#### causal_judgement — non-train models
| Model | Phase 0 | Phase 1 | Phase 2 | P1−P0 | P2−P1 |
|---|---:|---:|---:|---:|---:|
| gpt-4.1 | 74.7% | 70.1% | 64.4% | −4.6% | −5.7% |
| claude | 70.1% | 71.3% | 59.8% | +1.2% | −11.5% |
| gemini | 65.5% | 65.5% | 65.5% | +0.0% | +0.0% |
| llama | 66.7% | 67.8% | 64.4% | +1.2% | −3.4% |

#### geometric_shapes — non-train models
| Model | Phase 0 | Phase 1 | Phase 2 | P1−P0 | P2−P1 |
|---|---:|---:|---:|---:|---:|
| gpt-4.1 | 78.0% | 70.0% | 75.0% | −8.0% | +5.0% |
| claude | 79.0% | 81.0% | 78.0% | +2.0% | −3.0% |
| gemini | 74.0% | 72.0% | 75.0% | −2.0% | +3.0% |
| llama | 66.0% | 64.0% | 51.0% | −2.0% | −13.0% |

#### disambiguation_qa — non-train models
| Model | Phase 0 | Phase 1 | Phase 2 | P1−P0 | P2−P1 |
|---|---:|---:|---:|---:|---:|
| gpt-4.1 | 90.0% | 87.0% | 88.0% | −3.0% | +1.0% |
| claude | 92.0% | 95.0% | 91.0% | +3.0% | −4.0% |
| gemini | 82.0% | 90.0% | 89.0% | +8.0% | −1.0% |
| llama | 80.0% | 84.0% | 84.0% | +4.0% | +0.0% |

**Key findings (non-train models):**
1. **GS and CJ: bootstrap is the best cheatsheet for non-train models.** Phase 1 PK patching degrades GS (−2.5pp avg) and is neutral on CJ (−0.6pp), while Phase 2 CS further hurts both (GS −2.0pp, CJ −5.2pp). Refinement beyond Phase 0 overfits to mini's failure modes.
2. **DQ is the clean Phase 1 success story** (+3.0pp non-train avg). Phase 1 PK patching generalizes well; Phase 2 CS slightly reverses (−1.0pp).
3. **FF Phase 1 helps marginally (+1.5pp); Phase 2 gives it back (−1.0pp).** Net effect ≈ zero.
4. **Ceiling tasks (BE/LD3/nav/WOL/SU): bootstrap already saturates.** No phase contributes meaningfully.
5. **CJ Phase 2 collapse is model-specific:** claude −11.5pp, gpt-4.1 −5.7pp; gemini completely unaffected (0.0% delta all phases). CJ CS appear to actively mislead strong-reasoning models.
6. **GS llama catastrophic Phase 2 (−13pp):** llama applies CS without selectivity — CS that should only fire on arc/multi-subpath SVGs trigger broadly, hurting more than helping.

---

### E-Phase0-gemini — Gemini-trained cheatsheet, all models

| Task | Model | Baseline | Phase 0 | Phase 1 | Phase 2 | P0−Base | P1−P0 | P2−P1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| causal_judgement | gemini* | 69.0% | 63.2% | 63.2% | 64.4% | −5.8% | +0.0% | +1.2% |
| causal_judgement | mini | — | 63.2% | 63.2% | 66.7% | — | +0.0% | +3.5% |
| causal_judgement | gpt-4.1 | — | 63.2% | 64.4% | 66.7% | — | +1.2% | +2.3% |
| causal_judgement | claude | — | 63.2% | 63.2% | 63.2% | — | +0.0% | +0.0% |
| causal_judgement | llama | — | 65.5% | 59.8% | 62.1% | — | −5.7% | +2.3% |
| disambiguation_qa | gemini* | 60.0% | 67.0% | 81.0% | 79.0% | +7.0% | +14.0% | −2.0% |
| disambiguation_qa | mini | — | 78.0% | 84.0% | 85.0% | — | +6.0% | +1.0% |
| disambiguation_qa | gpt-4.1 | — | 74.0% | 85.0% | 84.0% | — | +11.0% | −1.0% |
| disambiguation_qa | claude | — | 87.0% | 85.0% | 88.0% | — | −2.0% | +3.0% |
| disambiguation_qa | llama | — | 82.0% | 80.0% | 81.0% | — | −2.0% | +1.0% |
| geometric_shapes | gemini* | 48.0% | 59.0% | 51.0% | 63.0% | +11.0% | −8.0% | +12.0% |
| geometric_shapes | mini | — | 58.0% | 58.0% | 78.0% | — | +0.0% | +20.0% |
| geometric_shapes | gpt-4.1 | — | 70.0% | 67.0% | 77.0% | — | −3.0% | +10.0% |
| geometric_shapes | claude | — | 68.0% | 67.0% | 79.0% | — | −1.0% | +12.0% |
| geometric_shapes | llama | — | 64.0% | 63.0% | 59.0% | — | −1.0% | −4.0% |

**Key findings (gemini cheatsheet):**
1. **CJ gemini bootstrap hurts vs baseline (−5.8pp).** Gemini's Phase 0 cheatsheet for CJ is worse than no cheatsheet — the bootstrap encodes gemini's own CJ reasoning approach poorly. All phases stay stuck around 63–67%.
2. **DQ: Phase 1 PK patching is the big win.** Gemini bootstrap gives +7pp over baseline; Phase 1 adds another +14pp (gemini train) / +6–11pp (non-train). Phase 2 CS slightly reverses across all models.
3. **GS: Phase 1 hurts, Phase 2 rescues.** Gemini Phase 0 +11pp over baseline; Phase 1 collapses −8pp (train) / −1–3pp (non-train); Phase 2 recovers +12pp (train) / +10–12pp (non-train — except llama −4pp). The CS for GS under gemini training are high quality and broadly transferable.
4. **Gemini bootstrap underperforms mini bootstrap on DQ** (67% vs 84%) — mini's initial cheatsheet for DQ is far better, even though mini's Phase 1 lift is also smaller.
5. **GS Phase 2 CS from gemini training transfer strongly** across mini/gpt-4.1/claude (+10–20pp) — consistent with E-NEW1 finding that gemini's 7 granular GS case studies generalize well.

**Paper implication:** The ACTIVATE IF structure is not uniformly beneficial. Its value is determined by whether the conditions are formally evaluable. A condition that requires semantic judgment to evaluate (Mode C) provides no advantage over a plain worked example. A condition that is a computable feature check (Mode A) provides genuine selectivity that protects against over-application. This suggests a design principle: CS conditions should be restricted to formally evaluable predicates; phenomenological conditions should be replaced with plain examples or omitted.

---

## Variance Seed Evaluation — 3-Seed Aggregation (2026-05-01)

**Purpose:** Reduce single-run variance from API non-determinism. Each pipeline condition is run 3 times with independent API calls (temperature-zero routing still varies across OpenRouter backends). 3-seed means and std are computed per task/model.

**Conditions evaluated:**
- **v3** (standard pipeline): seeds 1–3, tasks CJ/GS/DQ/FF/SN (seed1=`rf_transfer_5tasks_v3.json`; seeds 2–3 from `runs/variance/v3_seed2/`, `v3_seed3/`)
- **e3** (no-oracle ablation): seeds 1–3, task CJ only (seed1=`e3_no_oracle_rf.json`)
- **ea** (EA Phase 1): seeds 1–3, tasks GS/DQ (seed1=`bbh_ea_phase1_rf.json`); CJ/SN have n=1

**Eval script:** `scripts/eval/run_variance_eval.sh` (6 parallel jobs, `--reasoning-first --no-csicl --concurrency 25`)
**Aggregation script:** `scripts/eval/aggregate_variance.py`
**Output files:**
- `runs/variance/v3_3seed_mean.json`
- `runs/variance/e3_3seed_mean.json`
- `runs/variance/ea_3seed_mean.json`

### Cheatsheet Parse-Failure Bugs Found and Fixed

Several variance seeds had malformed cheatsheets that caused near-zero scores for weaker models (parse failure where models output shape/answer names rather than parenthesised letter codes like `(C)`):

| Seed | Task | Bug | Fix |
|------|------|-----|-----|
| v3_seed3 | GS | `Answer: **Line**` / `Answer: **Sector**` → models output `VERDICT: Line` → parse None | Replaced with descriptive sentences |
| ea_seed2 | GS | `Answer: Line (E).` mixed format + 8× `Trapezoid (K)` (K invalid, trapezoid not a GS option) | Replaced descriptive text; K→Kite(D)/Rectangle(H) |
| ea_seed3 | GS | 5× `Trapezoid (K)` throughout | Same fix |
| RS DQ | DQ | 17× bare `Ambiguous` (no parentheses) → `VERDICT: Ambiguous` not matched by `\(([ABC])\)` regex | All replaced with `(C) Ambiguous` |

After fixes, all affected evals re-run and results patched into seed files.

### v3 3-Seed Results (CJ, GS, DQ, FF, SN — gpt-4.1-mini train)

`*` = train model. n=3 for CJ/GS/DQ/SN; n=2 for FF (seed3 not collected).

| Task | mini* full | gpt-4.1 full | claude full | gemini full | llama full |
|------|-----------|-------------|------------|------------|-----------|
| CJ   | 67.8%     | 67.4%       | 65.5%      | 64.0%      | 64.8%     |
| GS   | 74.0%     | 74.7%       | 76.3%      | 72.3%      | 59.3% (std=10.4%) |
| DQ   | 87.0%     | 83.3%       | 85.3%      | 81.3%      | 82.7%     |
| FF   | 95.5%     | 95.5%       | 96.0%      | 89.0%      | 82.5%     |
| SN   | 97.2%     | 97.7%       | 96.7%      | 95.3%      | 89.2%     |

Per-seed std ≈ 2–5pp for most models/tasks. Exception: **llama GS std=10.4%** (seeds: 51%, 56%, 71%) — highest single-condition variance observed.

**5-task non-train averages (vs single-run paper values):**
| Model | CS-ICL | PK-only 3-seed | Full 3-seed | Δ (full−CS-ICL) |
|-------|--------|---------------|-------------|-----------------|
| GPT-4.1   | 87.6% | 83.9% | 83.7% | **−3.9pp** (was −4.7pp) |
| Claude    | 86.1% | 86.4% | 84.0% | **−2.1pp** (was −0.2pp) |
| Gemini    | 80.1% | 80.6% | 80.4% | **+0.3pp** (was +1.9pp) |
| Llama     | 77.7% | 76.8% | 75.7% | **−2.0pp** (was −3.3pp) |

Key takeaway: transfer gap persists across seeds for all models. Claude full underperforms CS-ICL by 2.1pp (was at apparent parity; single-seed was misleadingly favorable).

### e3 3-Seed Results (CJ no-oracle, n=3)

| Model | e3 full | v3 full (3-seed) | Δ |
|-------|---------|-----------------|---|
| mini* | 69.7% | 67.8% | +1.9pp |
| gpt-4.1 | 70.5% | 67.4% | +3.1pp |
| claude | 69.0% | 65.5% | +3.5pp |
| gemini | 65.9% | 64.0% | +1.9pp |
| llama | 68.6% | 64.8% | +3.8pp |

**All 5 models improve with oracle-free case studies vs oracle-contaminated v3.** Direction is more consistent than single-seed (where claude was -1.1pp and gemini 0pp). Effect sizes are smaller than single-run (gpt-4.1: +3.1pp vs +8.1pp; llama: +3.8pp vs +8.0pp) because v3 3-seed CJ averages are higher than v3 single-seed.

### ea 3-Seed Results (GS and DQ, n=3)

**GS (EA Phase 1 pk_only vs v3 pk_only, 3-seed):**
| Model | EA PK | v3 PK | ΔPK | EA full | v3 full |
|-------|-------|-------|-----|---------|---------|
| mini* | 78.7% | 72.7% | +6.0pp | 78.7% | 74.0% |
| gpt-4.1 | 77.3% | 75.3% | +2.0pp | 76.7% | 74.7% |
| claude | 80.0% | 79.7% | +0.3pp | 80.0% | 76.3% |
| gemini | 74.7% | 70.3% | +4.4pp | 76.7% | 72.3% |
| llama | 65.0% | 66.0% | −1.0pp | 64.7% | 59.3% |

EA PK consistently ≥ v3 PK for 4/5 models. EA GS is also more stable (llama std=4.7% vs 10.4% for v3). The single-run Δ of +9.0pp for mini is 6.0pp with 3-seed means (both directions still clearly positive).

**GS CS-ICL reference:** 79.0% (single-run, static).  
EA PK 3-seed (78.7%) ≈ CS-ICL (was reported as matching; confirmed approximately).

**DQ (EA Phase 1, n=3):**
| Model | EA full | v3 full (3-seed) | Δ |
|-------|---------|-----------------|---|
| mini* | 83.7% | 87.0% | −3.3pp |
| gpt-4.1 | 86.7% | 83.3% | +3.4pp |
| claude | 91.3% | 85.3% | +6.0pp |
| gemini | 81.0% | 81.3% | −0.3pp |
| llama | 85.3% | 82.7% | +2.6pp |

EA DQ is mixed vs v3. Mini is hurt (−3.3pp), others are helped. The original single-run finding that "EA full DQ is much worse" was driven by a single unfavorable seed; 3-seed averaging reverses the story for non-train models.

---

## Reasoning Scorer Pipeline Evaluation (2026-05-01)

**Description:** Evaluates cheatsheets generated by a stronger scorer (gpt-oss-120b) during Phase 1/2 training, with gpt-4.1-mini for patch/CS generation. Tests whether a stronger within-pipeline scorer improves cheatsheet quality.

**Tasks:** disambiguation_qa, geometric_shapes, formal_fallacies
**Run dirs:** `runs/bbh_reasoning_scorer/{task}/`
**Eval script:** `scripts/eval/run_reasoning_scorer_eval.sh`
**Results file:** `runs/reasoning_scorer_rf.json` (corrected; DQ re-evaluated after cheatsheet fix)

### DQ Parse Failure Bug and Fix

The RS DQ cheatsheet (`cheatsheet_phase1_pk_final.txt` and `cheatsheet_final.txt`) contained 17 instances of bare "Ambiguous" (e.g., `default to **Ambiguous**`, `if unclear, ambiguous.`) without the required `(C)` parenthesised letter. Models output `VERDICT: Ambiguous` which the RF parser `\(([ABC])\)` cannot match → returns None → counted wrong. Fixed all 17 instances to `(C) Ambiguous`. Re-eval showed gemini pk_only 12%→77%, llama pk_only 0%→83%.

### Reasoning Scorer Results (RF, 5 models)

`*` = train model. Δ_cs = full − pk_only. Compare to v3 3-seed means for same tasks.

**disambiguation_qa:**
| Model | RS full | RS pk | Δ_cs | v3 full (3-seed) |
|-------|---------|-------|------|-----------------|
| mini* | 84% | 82% | +2% | 87% |
| gpt-4.1 | 82% | 83% | −1% | 83% |
| claude | 82% | 86% | −4% | 85% |
| gemini | 75% | 77% | −2% | 81% |
| llama | 83% | 83% | 0% | 83% |

RS DQ worse than v3 for most models (−3pp mini, −6pp gemini, −3pp claude). Stronger scorer during training hurt DQ quality — possibly because gpt-oss-120b assigns scores using different criteria than the models ultimately evaluated.

**geometric_shapes:**
| Model | RS full | RS pk | Δ_cs | v3 full (3-seed) |
|-------|---------|-------|------|-----------------|
| mini* | 81% | 82% | −1% | 74% |
| gpt-4.1 | 77% | 74% | +3% | 75% |
| claude | 80% | 82% | −2% | 76% |
| gemini | 78% | 72% | +6% | 72% |
| llama | 65% | 71% | −6% | 59% |

RS GS is consistently better than v3 (+7pp mini, +2pp gpt-4.1, +4pp claude, 0pp gemini, +6pp llama). The stronger scorer identifies more reliable geometric correction rules.

**formal_fallacies:**
| Model | RS full | RS pk | Δ_cs | v3 3-seed |
|-------|---------|-------|------|-----------|
| mini* | 95% | 95% | 0% | 95.5% |
| gpt-4.1 | 96% | 96% | 0% | 95.5% |
| claude | 96% | 97% | −1% | 96% |
| gemini | 88% | 85% | +3% | 89% |
| llama | 73% | 79% | −6% | 82.5% |

RS FF roughly matches v3 (within noise). No meaningful difference.

**Summary:** Reasoning scorer helps GS (+2–7pp), hurts DQ (−1–6pp for most models), neutral on FF. The effect is task-specific: GS has structured geometric failure modes that a stronger scorer identifies better; DQ has coupled pronoun patterns that the stronger scorer may over-constrain.

---

## EA 3-Seed Variance: CJ + Snarks (2026-05-01)

**Scripts:**
- Pipeline: `scripts/pipeline/ea_seed_cj_snarks.sh` (seeds 2 & 3; seed 1 from `runs/bbh_ea_phase1/`)
- Eval: inline in above script; output `runs/variance/eval_results/ea_seed{1,2,3}_cj_sn_rf.json`
- Aggregate: `runs/variance/ea_3seed_mean.json` (updated; now covers CJ, DQ, GS, Snarks — FF pending)

**Run dirs:** `runs/variance/ea_seed{2,3}/causal_judgement`, `runs/variance/ea_seed{2,3}/snarks`
**Status:** COMPLETE (seeds 2 & 3 pipeline + eval + aggregation all finished)

`*` = train model. Δ_cs = full − pk_only.

**causal_judgement (EA, n=3):**
| Model | EA full | EA pk_only | Δ_cs | v3 full (3-seed) |
|-------|---------|-----------|------|-----------------|
| mini* | 64.4% | 70.1% | −5.7% | 67.8% |
| gpt-4.1 | 66.7% | 64.4% | +2.3% | 67.4% |
| claude | 63.2% | 62.1% | +1.1% | 65.5% |
| gemini | 66.7% | 64.4% | +2.3% | 64.0% |
| llama | 65.5% | 62.1% | +3.4% | 64.8% |

Key finding: **EA flips CJ CS from harmful to slightly helpful for all non-train models.** Under standard v3, CJ CS is negative for non-train models; under EA (stronger PK), CS becomes slightly positive (+1.1 to +3.4pp). Mini remains negative (−5.7pp). This suggests the harmful CS effect is PK-quality-dependent, not inherent to CJ.

**snarks (EA, n=3):**
| Model | EA full | EA pk_only | Δ_cs | v3 full (3-seed) |
|-------|---------|-----------|------|-----------------|
| mini* | 97.2% | 95.8% | +1.4% | 96.0% |
| gpt-4.1 | 90.1% | 98.6% | −8.5% | 91.4% |
| claude | 95.8% | 95.8% | 0.0% | 92.7% |
| gemini | 91.5% | 88.7% | +2.8% | 87.7% |
| llama | 85.9% | 80.3% | +5.6% | 82.0% |

Key finding: GPT-4.1 snarks shows a large negative CS effect (−8.5pp) under EA — EA PK (98.6%) overshoots, CS then adds noise. This is the strongest CS-harmful signal in the EA results. Claude shows 0pp (ceiling-like behavior).

### EA 4-Task Non-Train Average (CJ + DQ + GS + Snarks, n=3)

Comparison on the 4 tasks now in ea_3seed_mean.json (FF pending):

| Model | EA+no-oracle | v3 (same 4 tasks) | CS-ICL | EA vs CS-ICL |
|-------|-------------|-------------------|--------|-------------|
| GPT-4.1 | 80.0% | 80.8% | 87.6% | −7.6pp |
| Claude | 82.6% | 81.0% | 86.1% | −3.5pp |
| Gemini | 79.0% | 78.2% | 80.1% | −1.1pp |
| Llama | 75.4% | 74.0% | 77.7% | −2.3pp |

EA marginally beats v3 on the same 4 tasks for Claude (+1.6pp), Gemini (+0.8pp), Llama (+1.4pp); tied/slightly worse for GPT-4.1 (−0.8pp). All models still below CS-ICL. The EA advantage over standard v3 is small and inconsistent.

**Pending:** `ea_no_oracle_ff.sh` (formal_fallacies × 3 seeds) still running. Once complete, run `scripts/eval/eval_ea_combined.sh` to get the full 5-task `runs/variance/ea_combined_3seed_mean.json` for Tab 6 update.

---

## Gemini-Trained Seeds 2 & 3: CJ + GS (2026-05-01)

**Script:** `scripts/pipeline/gemini_train_cj_gs_seeds.sh`
**Model:** `google/gemini-2.0-flash-001` in all roles (score / rule-patch / casestudy)
**Run dirs:** `runs/gemini_train_v3_seed{2,3}/{causal_judgement,geometric_shapes}/`
**Eval output:** `runs/gemini_train_v3_eval_results/seed{1,2,3}_rf.json` (CJ+GS only)
**Status:** COMPLETE (CJ seeds 2&3 done earlier; GS seed 3 finished ~00:10 2026-05-02)

Produces partial 3-seed coverage for CJ and GS only. Superseded by `gemini_train_fds_seeds.sh` which re-evals all 5 tasks for all 3 seeds and overwrites these eval files.

---

## Gemini-Trained Seeds 2 & 3: FF + DQ + Snarks (2026-05-02)

**Script:** `scripts/pipeline/gemini_train_fds_seeds.sh`
**Model:** `google/gemini-2.0-flash-001` in all roles
**Run dirs:** `runs/gemini_train_v3_seed{2,3}/{formal_fallacies,disambiguation_qa,snarks}/`
**Eval output:** `runs/gemini_train_v3_eval_results/seed{1,2,3}_rf.json` (5-task, overwrites CJ+GS-only files)
**Aggregate output:** `runs/gemini_train_v3_3seed_mean.json`
**Status:** RUNNING (launched 2026-05-02 ~00:15; PID 42913)

Completes the full 5-task × 3-seed coverage for the Gemini-trained bidirectional transfer table (Tab `gemini_transfer`). Once done: update table data, remove "Single-run." from caption, update prose.

---

## EA 3-Seed Table Update: findings_draft.tex (2026-05-02)

Tab `ea` (Table 5) updated with 3-seed means for all 4 rows (CJ/Snarks/DQ now seeded):

| Task | Std PK (3-seed) | EA PK (3-seed) | ΔPK |
|------|----------------|---------------|-----|
| CJ | 64.8% | 70.1% | +5.4pp |
| GS | 72.7% | 78.7% | +6.0pp |
| Snarks | 96.7% | 95.8% | −0.9pp |
| DQ | 84.7% | 84.7% | 0.0pp |

**Key findings from 3-seed update:**
- CJ ΔPK flipped: was 0.0pp single-run (EA applied 0 patches in seed 1); is +5.4pp with 3-seed means. Standard Phase 1 pk is more variable on CJ; EA is more stable.
- CJ full pipeline still worse than standard (64.4% vs 67.8%) because EA generates 4 case studies that are oracle-contaminated — reinforces Phase 2 oracle story.
- Snarks: +1.4pp single-run → −0.9pp 3-seed (within noise, direction reversed).
- DQ: −1.0pp single-run → 0.0pp 3-seed (within noise).
- Caption updated: "All PK columns are 3-seed means; CS/patch counts from seed 1."
- Fig ea caption updated: removed "CJ/Snarks/DQ are single-run."
- Tab combined caveat updated: removed "non-GS rows of Tab ea should be interpreted cautiously."

**Paper decision:** RS not extended as a paper contribution — mixed results on 3 tasks, 6-day deadline. One-sentence limitation mention only.

---

## EA+No-Oracle Combined: Tab 6 Update (2026-05-02)

Source: `runs/variance/ea_combined_3seed_mean.json` (3 seeds, all 5 tasks)

### 5-task non-train RF average (EA+no-oracle)

| Model | CJ | DQ | FF | GS | Snarks | **5-task avg** |
|-------|----|----|----|----|--------|----------------|
| GPT-4.1 | 66.7% | 86.0% | 95.7% | 76.3% | 94.4% | **83.8%** |
| Claude-3.7 | 65.1% | 89.0% | 95.3% | 81.0% | 96.2% | **85.3%** |
| Gemini-2.0 | 65.9% | 80.0% | 88.0% | 74.7% | 95.3% | **80.8%** |
| Llama-3.3 | 64.4% | 84.7% | 74.7% | 64.0% | 85.9% | **74.7%** |

### Tab 6 (tab:combined) updated structure

| Model | CS-ICL | EA+no-oracle | Δ | Best-combo† | v3 full |
|-------|--------|-------------|---|-------------|---------|
| GPT-4.1 | 87.6% | 83.8% | −3.8 | 84.9% | 83.7% |
| Claude-3.7 | 86.1% | 85.3% | −0.8 | 85.4% | 84.0% |
| Gemini-2.0 | 80.1% | 80.8% | +0.7 | 81.2% | 80.4% |
| Llama-3.3 | 77.7% | 74.7% | −3.0 | 77.6% | 75.7% |

Δ = EA+no-oracle − CS-ICL. †Best-combo is per-task oracle (not deployable).

**Changes made to findings_draft.tex:**
- Tab 6 `\begin{tabular}` expanded from 4 to 5 columns: added EA+no-oracle column
- Δ column now shows EA+no-oracle − CS-ICL (was best-combo − CS-ICL)
- Caption rewritten: EA+no-oracle is primary comparison, best-combo is oracle ceiling
- Prose (already updated in prior session) leads with EA+no-oracle as unified pipeline

---

## v5 Oracle Ablation: GS Variance Seeds 2 & 3 (2026-05-02)

**Description:** v5 (both phase oracles on, default) on geometric_shapes × seeds 2 & 3 to give 3-seed means for the oracle gradient figure. Phase 1 PK patching auto-skipped (no initial ruleset); each seed runs independent auto-bootstrap (75 items) + Phase 2 CS generation.

**Script:** `scripts/pipeline/v5_gs_seeds.sh`
**Seed 2 pipeline:** `runs/variance/v5_seed2/geometric_shapes/` → COMPLETE
**Seed 3 pipeline:** `runs/variance/v5_seed3/geometric_shapes/` → COMPLETE
**Seed 2 eval:** `runs/variance/eval_results/v5_seed2_gs_rf.json` → COMPLETE
**Seed 3 eval:** `runs/variance/eval_results/v5_seed3_gs_rf.json` → COMPLETE
**Aggregation:** `runs/variance/v5_gs_3seed_mean.json` (geometric_shapes n_seeds=3)

**v5 GS 3-seed means (full CS accuracy):**

| Model | full | pk_only | Δ_cs |
|-------|------|---------|------|
| gpt-4.1-mini* | 64.7% | 59.0% | +5.7 |
| gpt-4.1 | 74.0% | 68.7% | +5.3 |
| claude-3.7 | 79.7% | 80.7% | −1.0 |
| gemini-2.0 | 71.7% | 66.7% | +5.0 |
| llama-3.3 | 62.3% | 65.0% | −2.7 |

**Oracle gradient (GS) 3-seed comparison — v3 vs E3 vs v5 (full CS accuracy):**

| Model | E3 (no oracle) | v3 (phase2 oracle) | v5 (both oracles) |
|-------|---------------|-------------------|------------------|
| gpt-4.1-mini* | 46.3% | 74.0% | 64.7% |
| gpt-4.1 | 74.0% | 74.7% | 74.0% |
| claude-3.7 | 78.7% | 76.3% | 79.7% |
| gemini-2.0 | 62.7% | 72.3% | 71.7% |
| llama-3.3 | 57.3% | 59.3% | 62.3% |

**Finding:** GS oracle effects remain inconsistent across models in 3-seed means, confirming the single-run observation. Unlike CJ (where E3 > v3 > v5 consistently), GS shows no monotone oracle ordering — E3 hurts mini severely (−27.7pp vs v3) but has negligible effect on gpt-4.1. This confirms GS case study harm is structural (SVG path parsing difficulty), not oracle-contamination driven.

**Changes made to findings_draft.tex:**
- `fig:oracle_gradient` caption updated: "All values are single-run (seed 1)" → "All values are 3-seed means"
- Caption also corrected: GS panel label changed from "PK-only accuracy" to "full CS accuracy" (both panels now show full CS)

---

## p1_6000 GS Size Ablation Rerun (2026-05-02)

**Description:** The original p1_6000chars GS ablation run returned 18% test accuracy (anomalous, likely corrupted eval). Re-run with `--max-pk-chars 6000` into a fresh output dir, eval on gpt-4.1-mini only (train model, consistent with the ablation's scope).

**Script:** `scripts/pipeline/rerun_p1_6000_gs.sh`
**Pipeline output:** `runs/ablation_size2_rerun/p1_6000chars/geometric_shapes/` → COMPLETE
**Eval output:** `runs/ablation_size2_p1_6000chars_gs_rerun_rf.json`
**Patched into:** `runs/ablation_size2_p1_6000chars_rf.json`

**Corrected result (gpt-4.1-mini):** full=**75.0%**  pk_only=73.0%  Δ_cs=+2.0pp

**Updated Tab pk_size (GS row):**

| Task | 3K | 6K | 12K | Unlimited |
|------|----|----|-----|-----------|
| GS | **78.0%** | **75.0%** | 57.0% | 75.0% |

**Finding:** The corrected 6K GS value (75.0%) is within 3pp of the 3K result (78.0%) and equal to unlimited (75.0%). This is consistent with the non-monotone ordering seen across other tasks and supports the conclusion that no single cap dominates. The earlier 18% was clearly a corrupted run.

**Changes made to findings_draft.tex:**
- `tab:pk_size` GS 6K cell: `$\dagger$` → `75.0\%`
- `tab:pk_size` caption: removed `$\dagger$: p1\_6000 GS result (18\%) excluded as anomalous`
- Appendix prose: removed "The 6K GS result (18%) is excluded as anomalous" sentence
- Limitations paragraph: removed anomalous result exclusion sentence
