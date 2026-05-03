# ICRefine Results Classification

Generated 2026-05-01. Purpose: identify canonical, newest, and trustworthy result files
before updating findings_draft.tex. Discard superseded/partial/failed runs.

---

## ⚠️ Critical Discrepancy: Tab 1 (Phase Contribution) Sources

The existing `findings_draft.tex` Table 1 has numbers that **do not match any single eval
file**. The table appears to have been assembled from multiple sources. Summary of the conflict:

| Task | Tab 1 (P0) | Tab 1 (P1) | Tab 1 (P2) | Δ P1→P2 | E-Phase0 (P2) | v3 RF full |
|------|-----------|-----------|-----------|---------|--------------|-----------|
| CJ   | 73.6%     | 71.3%     | **73.6%** | +2.3    | 72.4%        | **66.7%** |
| GS   | 70.0%     | 77.0%     | **70.0%** | −7.0    | **81.0%**    | **78.0%** |
| DQ   | 84.0%     | 85.0%     | **84.0%** | −1.0    | 88.0%        | **88.0%** |
| FF   | 96.0%     | 96.0%     | 96.0%     | 0.0     | 97.0%        | 97.0%     |
| SN   | 95.8%     | 95.8%     | **95.8%** | 0.0     | **98.6%**    | **97.2%** |

**Root cause**: Tab 1 was written from an older CoT-mode or early-run evaluation, not the
final RF evaluations. Specifically:
- Phase 0 column ✓ matches `phase0_mini_rf.json` (Apr 30 23:30)
- Phase 1 (pk_only) ✓ matches `rf_transfer_5tasks_v3.json` for GS/DQ/SN (Apr 30 18:41)
- Phase 2 (full) ✗ does NOT match `rf_transfer_5tasks_v3.json` for GS/CJ/DQ/SN

The E-Phase0 numbers (from `runs/bbh_v3_phase0/`) represent a NEWER v3-style pipeline
run where GS CS helped (+4pp) rather than hurt (−7pp). These conflict with the paper
narrative built around GS CS hurting.

**Recommendation**: Keep Tab 1 as-is for now (the −7pp GS finding is the core narrative);
the GS result appears to be specific to the canonical `runs/bbh_v3/` run. However, the
cs_icl values and P2 values should be verified before final submission. Do NOT update Tab 1
in this pass — it requires checking which bbh_v3 run dir was the original.

---

## Canonical Result Files

### GROUP 1: v3 Baseline — 5 Non-Ceiling Tasks (Tab 2 + Tab 1 support)

**PRIMARY: `runs/rf_transfer_5tasks_v3.json`** — Apr 30 18:41 ✅ CANONICAL
- Tasks: CJ, GS, DQ, DU, SN (5 tasks, 5 models)
- gpt-4.1-mini full / pk_only / cs_icl:
  - CJ:  66.7% / 63.2% / 66.7%
  - GS:  78.0% / 77.0% / 79.0%
  - DQ:  88.0% / 85.0% / 91.0%
  - DU:  93.0% / 91.0% / 95.0%
  - SN:  97.2% / 95.8% / 95.8%
- Non-train models (full):
  - CJ:  gpt41=64.4%, claude=59.8%, gemini=65.5%, llama=64.4%
  - GS:  gpt41=75.0%, claude=78.0%, gemini=75.0%, llama=51.0%
  - DQ:  gpt41=88.0%, claude=91.0%, gemini=89.0%, llama=84.0%
  - SN:  gpt41=95.8%, claude=95.8%, gemini=93.0%, llama=85.9%

**SECONDARY (ceiling tasks): `runs/rf_transfer_6tasks_e9.json`** — Apr 30 20:06 ✅ OK
- Tasks: WOL, FF, Nav, LD3, Sports, Bool (6 remaining tasks, 5 models)
- mini full: WOL=100%, FF=97%, Nav=100%, LD3=100%, Sports=99%, Bool=100%

**SUPERSEDED** (do not use for paper):
- `rf_transfer_5tasks_v2.json` (Apr 30 17:30) — earlier version, superseded
- `rf_5tasks_full_comparison.json` (Apr 30 15:22) — partial (3 tasks only)
- `rf_full_cache.json`, `rf_full_remaining.json`, `rf_full_underperforming.json` — dev/partial runs

### GROUP 2: Phase 0 Breakdown (Tab 1 Phase 0 column)

**PRIMARY: `runs/phase0_mini_rf.json`** — Apr 30 23:30 ✅ CANONICAL
- Evaluates bootstrap-only cheatsheet (Phase 0 only) for all 11 tasks, mini
- Key values (mini, RF):
  - CJ=73.6%, GS=70.0%, DQ=84.0%, FF=96.0%, SN=95.8%
  - WOL=100%, Bool=100%, Nav=100%, LD3=100%, Sports=100%, DU=94.0%

### GROUP 3: E3 Oracle Ablation (Tab E3)

**PRIMARY: `runs/e3_no_oracle_rf.json`** — Apr 30 21:38 ✅ CANONICAL
- Condition: BOTH Phase 1 and Phase 2 oracle disabled (--no-oracle)
- Tasks: CJ, GS (2 tasks, 5 models)
- Note: This run DID generate Phase 2 CS (unlike earlier CoT-mode E3 with --no-phase2-oracle only)

gpt-4.1-mini full / pk_only / delta_cs:
- CJ:  67.8% / 62.1% / +5.7pp  ← CS generated without oracle, actually helped mini
- GS:  80.0% / 61.0% / +19.0pp ← GS CS generated without oracle strongly helped

All models CJ (full):
- mini=67.8%, gpt-4.1=71.3%, claude=66.7%, gemini=69.0%, llama=70.1%

All models GS (full):
- mini=80.0%, gpt-4.1=75.0%, claude=77.0%, gemini=72.0%, llama=59.0%

v3 full reference (for Δ comparison, from rf_transfer_5tasks_v3.json):
- CJ: mini=66.7%, gpt-4.1=64.4%, claude=59.8%, gemini=65.5%, llama=64.4%
- GS: mini=78.0%, gpt-4.1=75.0%, claude=78.0%, gemini=75.0%, llama=51.0%

**CJ E3 vs v3 Δ (E3 full − v3 full)**:
- mini: 67.8 − 66.7 = +1.1pp
- gpt-4.1: 71.3 − 64.4 = +6.9pp
- claude: 66.7 − 59.8 = +6.9pp
- gemini: 69.0 − 65.5 = +3.5pp
- llama: 70.1 − 64.4 = +5.7pp

⚠️ CONFLICT WITH EXISTING TAB E3: The existing table shows v3 full CJ for mini=64.4% (from
older rf_5tasks_full_comparison) but the canonical rf_transfer_5tasks_v3.json shows 66.7%.
Also, the existing table's "E3 PK only" = 73.6% was v3's pk_only, NOT the actual E3 run.

### GROUP 4: EA Phase 1 (Tab EA)

**PRIMARY: `runs/bbh_ea_phase1_rf.json`** — May 1 14:37 ✅ CANONICAL (most recent)
- Tasks: CJ, GS, SN, DQ (4 tasks, 5 models)
- gpt-4.1-mini full / pk_only / delta_cs / cs_icl:
  - CJ:  64.4% / 70.1% / −5.7pp / 70.1%
  - GS:  76.0% / 80.0% / −4.0pp / 80.0%
  - SN:  97.2% / 95.8% / +1.4pp / 94.4%
  - DQ:  82.0% / 86.0% / −4.0pp / 89.0%

EA vs v3 full (rf_transfer_5tasks_v3.json) for mini:
- CJ: 64.4% vs 66.7% = −2.3pp (EA full worse!)
- GS: 76.0% vs 78.0% = −2.0pp (EA full slightly worse, but EA pk_only 80% > v3 pk 77%)
- SN: 97.2% vs 97.2% = 0.0pp (tied)
- DQ: 82.0% vs 88.0% = −6.0pp (EA full significantly worse!)

EA pk_only vs v3 pk_only (both Phase 1 PK only, no CS):
- CJ: 70.1% vs 63.2% = +6.9pp (EA Phase 1 is much better PK)
- GS: 80.0% vs 77.0% = +3.0pp (EA Phase 1 slightly better)
- SN: 95.8% vs 95.8% = 0.0pp
- DQ: 86.0% vs 85.0% = +1.0pp

⚠️ KEY FINDING: EA Phase 1 produces better PK (especially CJ +6.9pp, GS +3.0pp), but Phase 2
CS in the EA run HURT significantly (CJ −5.7pp, GS −4.0pp, DQ −4.0pp). The "full" pipeline
(EA PK + Phase 2 CS) is often WORSE than EA pk_only.

The narrative that "EA addresses GS bootstrap convergence" holds at the PK level:
EA pk_only (80%) > v3 Phase 1 pk (77%) > v3 full (78% in rf_transfer_5tasks_v3 or 70% in Tab 1).
The story depends significantly on which v3 GS Phase 2 number we use (see Tab 1 conflict above).

### GROUP 5: Size Ablation PK cap (Tab 4)

All files are May 1 16:04-16:12, gpt-4.1-mini only, 4 tasks. Condition = full test accuracy:

| File | GS | FF | SN | DQ |
|------|----|----|----|-----|
| `ablation_size2_p1_3000chars_rf.json` | **78%** | **98%** | 97.2% | 70% |
| `ablation_size2_p1_6000chars_rf.json` | ⚠️18% | 97% | 97.2% | 80% |
| `ablation_size2_p1_12000chars_rf.json` | 57% | 97% | 97.2% | **85%** |
| `ablation_size2_p1_unlimited_rf.json` | 75% | 98% | 95.8% | 78% |

CS-ICL reference (varies per file): GS≈80%, FF≈94%, SN≈95%, DQ≈85-91%

⚠️ p1_6000 GS=18% is an outlier (eval script failure or severely broken cheatsheet).
The pipeline cheatsheet for p1_6000 GS may have been corrupted or contain an empty/wrong
cheatsheet. **Recommend flagging this as a failed run and excluding from paper tables, or
re-running the p1_6000 eval for GS specifically.**

Key change from train accuracy (Tab 4 currently shows train):
- Train showed 12K best for GS (89.3%). Test shows 3K best (78%).
- Train showed non-monotone pattern. Test confirms non-monotone but with different ordering.
- The GS results are all below CS-ICL on test (CS-ICL≈80% while best ablation is 78%).

### GROUP 6: Size Ablation CS count (Tab 3)

| File | GS | FF | SN | DQ |
|------|----|----|----|-----|
| `ablation_size2_p2_1cs_rf.json` | 69% | 97% | 97.2% | **88%** |
| `ablation_size2_p2_3cs_rf.json` | **74%** | 97% | 97.2% | 79% |

"Unlimited" (v3 standard): GS=78% (rf_transfer_5tasks_v3), FF=97% (rf_transfer_6tasks_e9),
SN=97.2% (rf_transfer_5tasks_v3), DQ=88% (rf_transfer_5tasks_v3)

Key change from train accuracy (Tab 3):
- GS: train showed best-of-3 (89.3%) > unlimited (83.3%) > best-of-1 (78%). Test shows
  unlimited (78%) > best-of-3 (74%) > best-of-1 (69%). Ordering reverses: unlimited wins on test.
- DQ: best-of-1 (88%) > unlimited (88%) > best-of-3 (79%) on test. Best-of-1 matches unlimited.
- SN: all conditions ~97% (ceiling effect).
- FF: all conditions ~97% (ceiling effect).

---

## Status of Ongoing Runs

### magma_large pipeline
- **Script**: `scripts/pipeline/magma_large.sh`
- **Output dir**: `runs/magma_large/`
- **Status**: Running (Phase 2 iter 1, no CS accepted yet as of 2026-05-01 20:00)
- **Eval script**: `scripts/eval/eval_magma_large.py` (ready to run after pipeline completes)
- **Result file**: `runs/magma_large_rf.json` (pending)

### reasoning_scorer (DQ/GS/FF)
- **Output dir**: `runs/bbh_reasoning_scorer/{task}/`
- **Status**: COMPLETE (2026-05-01). DQ cheatsheet had parse-failure bug (bare "Ambiguous"); fixed and re-evaluated.
- **Result file**: `runs/reasoning_scorer_rf.json` ✅ CANONICAL (post-fix)
- **Key findings**: GS +2–7pp vs v3; DQ −1–6pp vs v3; FF neutral

### variance runs
- **Dir**: `runs/variance/`
- **Status**: COMPLETE. All 6 seed evals done, parse bugs fixed, 3-seed means aggregated.
- **Eval results**: `runs/variance/eval_results/` (6 seed files, all corrected)
- **Aggregated files**: `runs/variance/v3_3seed_mean.json`, `e3_3seed_mean.json`, `ea_3seed_mean.json` ✅ CANONICAL

---

---

## GROUP 7: Variance Evaluation — 3-Seed Aggregated Results (2026-05-01)

**PRIMARY: `runs/variance/v3_3seed_mean.json`** ✅ CANONICAL (3-seed means, corrected)
- Tasks: CJ(n=3), GS(n=3), DQ(n=3), FF(n=2), SN(n=3), DU(n=1)
- Key 5-task non-train avgs: GPT-4.1 full=83.7%, Claude=84.0%, Gemini=80.4%, Llama=75.7%
- GS llama seed variance σ=10.4% (highest variance observed)

**PRIMARY: `runs/variance/e3_3seed_mean.json`** ✅ CANONICAL (3-seed means)
- Task: CJ only (n=3 seeds)
- All 5 models improve vs v3 3-seed: Δ range +1.9 to +3.8pp (direction confirmed)
- Effect sizes smaller than single-run (gpt-4.1: +3.1pp vs +8.1pp single-seed)

**PRIMARY: `runs/variance/ea_3seed_mean.json`** ✅ CANONICAL (3-seed means)
- GS(n=3), DQ(n=3); CJ/SN(n=1)
- EA GS PK 3-seed: mini=78.7%, Δ vs v3 PK = +6.0pp (was +9.0pp single-run)
- EA DQ: mixed — mini hurt (−3.3pp), most non-train models helped

**SEED FILES (do not use directly — use 3-seed means above):**
- `runs/variance/eval_results/v3_seed2_rf.json`, `v3_seed3_rf.json`
- `runs/variance/eval_results/e3_seed2_rf.json`, `e3_seed3_rf.json`
- `runs/variance/eval_results/ea_seed2_rf.json`, `ea_seed3_rf.json` (GS corrected for parse failures)

## GROUP 8: Reasoning Scorer Pipeline (2026-05-01)

**PRIMARY: `runs/reasoning_scorer_rf.json`** ✅ CANONICAL (post DQ parse-failure fix)
- Tasks: DQ, GS, FF (from `runs/bbh_reasoning_scorer/`)
- Scorer: gpt-oss-120b; Generator: gpt-4.1-mini
- Key: GS improved (+2–7pp over v3), DQ hurt (−1–6pp), FF neutral
- **Not extending for paper** — mixed results, limited tasks, deadline constraint

---

## Recommendations for Paper Update

0. **ALL main tables**: NOW USE 3-SEED MEANS from `runs/variance/v3_3seed_mean.json` (Tab 2),
   `e3_3seed_mean.json` (Tab E3), `ea_3seed_mean.json` (Tab EA GS). Done 2026-05-01.

1. **Tab 1 (Phase contribution)**: HOLD — requires reconciling which bbh_v3 run is canonical.
   The paper narrative (GS CS hurts −7pp) depends on a specific run that differs from the
   newer E-Phase0 and rf_transfer_5tasks_v3.json results. Before updating, verify which
   run dir (bbh_v3 vs bbh_v3_phase0) is the canonical source.

2. **Tab E3 (Oracle ablation)**: UPDATE — use e3_no_oracle_rf.json values. The existing
   table used incorrect/estimated numbers. Note the narrative changes: E3 DID generate CS,
   and those oracle-free CS help (not hurt) vs v3 oracle CS. Δ for mini is +1.1pp not +9.2pp.
   The "+9.2pp" claim was a within-v3-run pk_only vs full comparison, not E3 vs v3.

3. **Tab EA (Evolutionary Algorithm)**: UPDATE — replace train proxy with test accuracy from
   bbh_ea_phase1_rf.json. Add pk_only column since EA generates CS that hurt. The key finding
   is EA pk_only improvement (+3pp GS, +6.9pp CJ) not EA full improvement.

4. **Tab 4 (PK size ablation)**: UPDATE — replace train accuracy with test accuracy. Flag
   p1_6000 GS=18% as anomalous (possible eval failure). The test ordering differs from train.

5. **Tab 3 (CS count ablation)**: UPDATE — replace train accuracy with test accuracy. The
   ordering reverses for GS (unlimited now wins on test).

6. **Limitations section**: UPDATE — mention variance runs available, magma in progress,
   test results now available for EA and size ablation.
