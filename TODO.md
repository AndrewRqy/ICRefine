# ICRefine — Research Directions & TODO

Last updated: 2026-04-06

---

## Active Priorities

### 1. Validate new case study format on hard1
The IDENTIFY/ACTION/DOES NOT APPLY TO format was just implemented. Need to re-run
hard1 with the new prompts and compare regression rates against the previous run
(which had 23–57% regression on every bin).

- [ ] Re-run `runs/hard1_select` with new prompt format
- [ ] Compare regression rates before/after
- [ ] Check if generated case studies are actually narrower

---

### 2. Regression threshold tuning for hard1
Even with better case studies, the 10% regression threshold may be too strict for
hard1 given how small the correct pool is early in the run (7–15 items). A single
wrong item out of 10 is already 10%.

- [ ] Try `--regress-threshold 0.2` on hard1
- [ ] Consider making threshold adaptive — stricter when correct pool is large, looser when small
- [ ] Alternatively: require minimum correct pool size before regression gate fires (e.g. skip regression gate if pool < 20)

---

### 3. Full normal dataset run
We've only smoke-tested on 100 items. The pipeline needs to run on all 1000 normal
items to see whether ICR-Select can meaningfully beat the 80% baseline.

- [ ] Run ICR-Select on full normal.jsonl (after hard1 is working well)
- [ ] Eval final cheatsheet on held-out set (seed=99, k=150)
- [ ] Compare all modes: baseline / ICR-Naive / ICR-Reasoning / ICR-Select / ICR-Select+DT

---

### 4. Fix similarity gate — semantic duplicate detection
Current similarity gate misses case studies that use different wording but encode
the same rule. The new IDENTIFY-based format helps, but the prompt needs testing.

- [ ] Run a smoke test where the same failure bin is flushed twice and check if the second is correctly flagged as SKIP
- [ ] If still failing, consider embedding-based similarity as a pre-filter before the LLM call

---

### 5. DT revision — make rewrites more aggressive
The current DT revision adds clarifying sentences rather than restructuring broken
steps. On hard1, it found RULE 6 in only 1/36 failures (3%) — the step parser
missed most of the actual failure modes because they don't reference step names.

- [ ] Improve step parser to match step content, not just labels like "STEP 1"
- [ ] Make DT revision prompt require substantive changes: new sub-rules, explicit
      worked examples inline, not just added warnings
- [ ] Add a minimum change check — if the revision only adds one sentence, reject it
      and ask for a more substantial rewrite

---

## Medium-term Directions

### 6. Hard1-specific training strategy
Hard1 problems are structurally harder — the model fails on ~59% even after seeing
all 69 items. The failure modes are diverse enough that narrow case studies can't
cover them without causing regressions. Options:

- [ ] Try DT-revision-only mode (no case studies) on hard1 — the DT revision gave
      +3 points with just one pass; multiple rounds may compound
- [ ] Investigate what types of hard1 problems the model consistently gets wrong
      (read the actual failure cases and classify them manually)
- [ ] Consider whether hard1 needs a separate cheatsheet vs normal

---

### 7. Post-think quality as a signal
The step parser currently just looks for "STEP 1", "RULE 4" etc. in the reasoning.
But post-think often describes what the model did without naming the step. A better
parser would match step content semantically.

- [ ] Read 10-20 hard1 failure post-thinks manually and note what language the model
      uses to describe each step
- [ ] Update step parser regex/patterns to match common paraphrases

---

### 8. Scoring temperature experiment
All scoring is at temperature=0.0. This means the same item always gets the same
verdict — there's no way to detect borderline cases. A borderline item scored at
temperature=0.5 might flip 50% of the time, which is useful signal for which items
to prioritize in the failure bin.

- [ ] Try scoring the failure bin at temperature=0.3 and use variance across multiple
      calls as a "difficulty" signal — easy failures (consistent wrong answer) vs hard
      failures (coin-flip) may warrant different case study types

---

### 9. Eval on hard2
We haven't touched hard2 at all. Once normal and hard1 are in reasonable shape,
run baseline + best ICR variant on hard2 to get a picture of the full difficulty curve.

- [ ] Baseline eval on hard2
- [ ] Decide whether to train on hard2 separately or if normal+hard1 transfers

---

## Long-term / Research Questions

- **Does the DT revision converge?** After 3+ rounds of DT revision on a large dataset,
  does the decision tree stabilize or keep changing? Is there a ceiling?

- **Cross-dataset transfer:** Does a cheatsheet trained on normal transfer to hard1?
  Our results say no — but is this because of the training data size or something
  structural about hard1?

- **Case study count vs quality:** Is one very good case study better than five mediocre
  ones? The ablation pruning is meant to answer this but we haven't run it at scale.

- **Model dependence:** All training and eval uses gpt-oss-120b. Does a cheatsheet
  trained with one model transfer to another (e.g. GPT-4o as the eval model)?

---

### 10. Oracle A/B quantitative validation
Level 1 qualitative check shows oracle-guided CSes have better algebraic grounding.
Need level 2 to confirm this translates to better pipeline metrics.

- [ ] Re-run `eval_oracle_quality.py` with `--bin-size 8` and `--n-items 80` — 3-failure bin was too small, result was a tie
- [ ] Run `oracle_ab_control` (no oracle, 60 normal items)
- [ ] Run `oracle_ab_treatment` (with oracle, 60 normal items)
- [ ] Compare avg fix rate, avg regression rate, bins discarded across both update_logs
- [ ] If oracle clearly wins, make `--oracle-csv` the default recommendation for normal runs

---

## Completed

- [x] Build ICR_naive baseline loop
- [x] Build ICR_reasoning with post-think capture
- [x] Build ICR_select with four quality gates
- [x] Build DT revision outer loop + step parser
- [x] Align scoring parameters with SAIR eval (temperature=0.0, VERDICT-first)
- [x] Smoke test all three modes on 100 normal items
- [x] Parallelize candidate mini-evals, ablation, condensation validation
- [x] Implement merge validation gate (validate_merge flag)
- [x] Redesign case study format: IDENTIFY / ACTION / DOES NOT APPLY TO
- [x] Update similarity gate and merge prompt for new format
- [x] Write ICR_proposal_v5.tex
- [x] Oracle reasoning injection: load GPT-5.4 traces, inject as contrast signal into case study generation prompt
- [x] eval_oracle_quality.py: qualitative + quantitative comparison tool with --from-bin replay
- [x] vLLM backend support: routing, reasoning_content parsing, timeout tuning
- [x] Prescore reuse: feed SAIR eval results into ICR_select to skip redundant scoring pass
- [x] vLLM parse fix: fallback to reasoning_content when content is empty
- [x] Scoring token cap: 16K → 8K to prevent DeepSeek exhausting budget before VERDICT
- [x] Similarity gate guard: skip until ≥3 case studies exist
- [x] Integrate ICR_select into SAIR recursive refinement pipeline (icr_select updater)
