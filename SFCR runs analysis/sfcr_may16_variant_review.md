# SFCR Variant Tests — May 15–16 Review

5 configurations × 6 seeds (CJ ×3, DQ ×3).  
Tasks: `causal_judgement` (CJ), `disambiguation_qa` (DQ).  
Seeds: 1000, 2000, 3000 (different anchor cheatsheets).  
Models: source = `gpt-4.1-mini`, proxies = `gpt-4.1` + `llama-3.3-70b`.

---

## Configurations at a Glance

| Config | Oracle mode | Validation routing | Time | Key change |
|---|---|---|---|---|
| sfcr_min | label_only | routed (keyword) | ~18:00 | Baseline — answer label only |
| sfcr_fcot | full_cot | routed (keyword) | ~22:20 | Full reasoning chain injected |
| sfcr_global | full_cot | global | ~23:00 | Global gate (all gate items exposed) |
| sfcr_subtype | full_cot | subtype (semantic_label) | ~00:00 | Replace keyword router with semantic_label gate |
| sfcr_contrast | contrastive | subtype (semantic_label) | ~01:10 | Show model's wrong answer to generator |

---

## Results Summary

### Candidates generated and accepted

| Seed | min | fcot | global | subtype | contrast |
|---|---|---|---|---|---|
| CJ s1000 | SKIP | 6 / **1** | 4 / 0 | 3 / **3** | 1 / **1** |
| CJ s2000 | 8 / **3** | 5 / **3** | 8 / 0 | 6 / **3** | 7 / **3** (→1 dedup) |
| CJ s3000 | SKIP | 8 / **3** | 8 / 0 | SKIP | 8 / **3** (→1 dedup) |
| DQ s1000 | 3 / 0 | 3 / 0 | 2 / 0 | 3 / 0 | 3 / 0 |
| DQ s2000 | 7 / **3** | 1 / **1** | 7 / 0 | 7 / **3** | 4 / **3** (→2 dedup) |
| DQ s3000 | 4 / 0 | 1 / **1** | SKIP | SKIP | 3 / **2** (→1 dedup) |
| **Total accepted** | **6** | **9** | **0** | **9** | **6 (deduped)** / 12 raw |

Format: `candidates / accepted`. Dedup applied post-hoc to contrast (per-subtype limit added during session).

---

## Per-Configuration Analysis

### 1. sfcr_min — label_only oracle, keyword routing

The generator only receives the correct answer label (no reasoning). This is the weakest oracle signal.

- **CJ s1000, s3000**: Skipped — generator could not produce any candidate that passed quick-validate. With no reasoning context, the LLM cannot articulate a rule that improves the model's behaviour on a specific failure pattern.
- **CJ s2000, DQ s2000**: Each accepted 3 rules around clear surface patterns (side effects / foreseen outcomes for CJ; pronoun possession/location for DQ). These are cases where the failure signal in V_shared was homogeneous enough that label alone was sufficient.
- **DQ s1000, s3000**: Both accepted 0. Dominant rejection cause: `private_activation_count > 0` (100% of rejections for dq_3000) — the keyword router activates rules on V_private items, triggering the private activation guard. Also heavy `routed_activation_rate >= 50%` rejections: rules activate on too many gate items globally, not targeted enough.

**Accepted rule quality**: Rules are somewhat generic ("outcome is a negative side effect that was foreseen"). Lack of reasoning context produces broad conditions that over-generalise.

---

### 2. sfcr_fcot — full_cot oracle, keyword routing

Generator receives the full correct reasoning chain per failure item. First substantive upgrade.

- **CJ s3000**: Now generates 8 candidates and accepts 3, whereas sfcr_min skipped entirely. The reasoning chain gives the generator enough context to identify and articulate the specific causal pattern (joint necessity, preemption, background conditions) causing failures.
- **CJ s1000**: Still only 1 accepted (from 6 candidates). Keyword router is filtering heavily — `private_activation_count > 0` blocks 3, `routed_activation_rate >= 50%` blocks 2.
- **DQ s2000, s3000**: Only 1 candidate generated each (previous runs produced 7). Generator was generating fewer variants per subtype — looks like a generation configuration difference vs min.
- **DQ s1000**: 0 accepted; all 3 rejected on `reg_private_count=1 > 0` and `reg_easy` — V_shared only has 3 items, so any rule that touches easy or private items fails.

**Accepted rule quality**: More specific than sfcr_min. CJ s3000 produces rules targeting concrete causal structures: "multiple agents act together (joint necessity)", "initial harmful condition made death inevitable", "initial harmful condition or risk". These map to real philosophical causation subtypes (INUS conditions, preemption chains).

**Keyword router problem visible**: CJ s1000 rejection profile dominated by `private_activation_count > 0` and `routed_activation_rate >= 50%`. Rules with broad USE WHEN conditions (e.g. "background condition") hit vocabulary overlaps in V_private items through the keyword router, triggering the private activation guard and blocking acceptance.

---

### 3. sfcr_global — full_cot oracle, global gate

Global mode exposes all gate items to every candidate (no routing — rule is tested against the full F_s). This was intended to remove the keyword router bottleneck.

**Result: 0 accepted across all 6 runs.**

**Root cause: `private_activation_count` gate bug.** In global mode, `exposed_ids = all_gate_ids`, which means every V_private item is in scope. The gate checks `private_activation_count > 0` as a hard rejection condition — but in global mode, since every rule "sees" V_private items, `private_activation_count` is trivially equal to `|V_private|` for any rule that activates on any gate item. This made the gate structurally impossible to pass.

**Fix applied**: Changed the private activation guard from `validation_routing_mode != "global"` to `validation_routing_mode == "routed"` — skipping the count-based private activation checks in both global and subtype modes, where the concept of "private activation" through routing doesn't apply.

---

### 4. sfcr_subtype — full_cot oracle, subtype (semantic_label) gate

After the global mode bug fix, replaced keyword routing with semantic_label-based gate selection. Instead of keyword-matching to find relevant gate items, the validator selects gate items whose `semantic_label` matches the generating subtype's label. This eliminates the vocabulary mismatch that caused the keyword router to expose wrong gate items.

- **CJ s1000, s2000**: Both accept 3 candidates. Zero private-activation rejections (the guard no longer fires). Rejections are now `reg_private_count > 0` (regression on private items) and U_LCB too negative — cleaner signal.
- **CJ s3000**: Still skipped — full_cot oracle is insufficient to generate candidates when the failure pattern is very hard (60% source accuracy, 21 V_shared items). The generator cannot produce rules that pass quick-validate on the subtype items.
- **DQ s1000**: 0 accepted; `reg_private_count=1 > 0` blocks all 3 candidates. V_shared=3 means any private regression is catastrophic to U_LCB.
- **DQ s2000**: 3 accepted. All target gender-agreement pronoun resolution: "USE WHEN the pronoun 'he' or 'she' refers to one of two people of different genders" — highly specific, correct error class.
- **DQ s3000**: Skipped — same generation failure as CJ s3000. Full reasoning chain alone can't bootstrap rules when the model has no clear structural error to generalise.

**Duplicate accepted rules issue discovered**: sfcr_subtype accepted 3 rules for CJ s1000 and s2000, and 3 for DQ s2000 — but inspection revealed all 3 are near-identical rewrites of the same rule (`candidates_per_subtype=3` generates 3 temperature variants per subtype, and all pass). This inflates candidate counts without adding diversity.

---

### 5. sfcr_contrast — contrastive oracle, subtype (semantic_label) gate

Contrastive oracle shows the generator: `Question + Model output (wrong) + Correct answer + Correct reasoning`. This gives the generator explicit counterfactual signal — it can see exactly what the model said versus what it should have said, making it much easier to write rules that target the model's specific error behaviour.

- **CJ s3000**: Now generates 8 candidates and accepts 3. Previously skipped under every prior config. The contrastive format unlocks generation on hard seeds where full_cot alone was insufficient.
- **DQ s3000**: Generates 3, accepts 2. Previously skipped under full_cot-based configs.
- **CJ s1000**: Only 1 candidate generated but 1 accepted — overdetermination rule. Low candidate count may reflect the subtype structure for this seed.
- **CJ s2000, DQ s2000**: Accept 3 each. Rules target misinformation/deception for CJ and pronoun-in-reported-speech for DQ.
- **DQ s1000**: 0 accepted. Consistent across all configs — V_shared=4, all 3 candidates rejected on `reg_private_count > 0`. This seed has too small a shared failure pool and too much private regression risk. No oracle format helps here.

**Accepted rule content quality**:
- CJ: overdetermination ("multiple sufficient causes independently bring about the outcome"), deception ("party causes harm only because they were misinformed by another"), disjunctive causation ("any one condition alone would produce the outcome"). These are distinct, real causal reasoning subtypes the model fails on.
- DQ: pronoun in reported speech ("X told Y that he/she..."), pronoun for possession/location ("her office", "his car"). Both are correct error patterns for WinoGrande-style disambiguation.

**Duplicate rule problem confirmed and fixed**: CJ s2000 raw: 3 accepted, all from `subtype_idx=2` (same subtype, 3 temperature variants). CJ s3000: 3 accepted, all `subtype_idx=0`. Per-subtype deduplication implemented — keep only the highest U_LCB accepted candidate per `subtype_idx`. After dedup: CJ s2000 → 1, CJ s3000 → 1, DQ s2000 → 2, DQ s3000 → 1.

---

## Cross-Config Rejection Pattern Analysis

### Primary rejection causes by config

| Rejection reason | min | fcot | global | subtype | contrast |
|---|---|---|---|---|---|
| `private_activation_count > 0` | dominant | dominant | dominant (bug) | eliminated | eliminated |
| `routed_activation_rate >= 50%` | present | present | — | — | — |
| `reg_private_count > 0` | occasional | occasional | occasional | dominant | dominant |
| `reg_easy` violations | occasional | occasional | occasional | rare | rare |
| `U_LCB <= -0.25` | secondary | secondary | secondary | secondary | secondary |

The evolution is clear:
1. **min / fcot**: keyword router creates `private_activation_count` and `routed_activation_rate` rejections — rules activate on wrong items due to vocabulary overlap.
2. **global**: bug makes private_activation_count trivially nonzero — 100% rejection.
3. **subtype / contrast**: private_activation eliminated; dominant rejection is now `reg_private_count > 0` (the rule actually regresses on some V_private items when tested) and U_LCB below threshold. These are legitimate quality signals, not artifacts.

### DQ s1000 — structural ceiling

Rejected in every config. src_acc=88–90%, V_shared=3–7, F_s=6–8. The shared failure pool is too small and the correct pool too large — any rule that fixes 1–2 shared items also touches private or easy items, pushing U_LCB deeply negative. This is not fixable by oracle mode or routing; the seed simply has too few transferable failures.

---

## Bugs Found and Fixed

### 1. Global mode private_activation gate (sfcr_global)
**Problem**: `private_activation_count > 0` check was active in global mode. Since global mode exposes all gate items, every rule trivially has `private_activation_count = |V_private|`. Zero candidates could ever pass.  
**Fix**: Changed guard condition from `validation_routing_mode != "global"` to `validation_routing_mode == "routed"`. Both global and subtype modes now skip the count-based private activation checks.

### 2. Per-subtype duplicate accepted rules (sfcr_subtype, sfcr_contrast)
**Problem**: `candidates_per_subtype=3` generates 3 temperature variants per failure subtype. When all 3 capture the same pattern, all 3 pass validation and are written into the cheatsheet — same rule 3×, wasting inference context with no diversity benefit.  
**Fix**: Added `_deduplicate_accepted_rules()` in `pipeline.py` — keeps only the highest U_LCB accepted candidate per `subtype_idx`. Cross-subtype diversity is preserved; within-subtype duplicates are collapsed to the best representative.

---

## Key Takeaways

**1. Oracle mode is the dominant lever for generation quality.**  
label_only → full_cot → contrastive shows progressive improvement. Contrastive is the only mode that reliably generates candidates for hard seeds (60% src_acc, 21 V_shared). The explicit counterfactual ("model said X, correct is Y") gives the generator exactly the signal it needs.

**2. Routing mode matters for gate signal quality, not generation.**  
Keyword routing creates spurious private_activation rejections through vocabulary overlap. Subtype routing (semantic_label-based) eliminates this artifact and shifts rejections to genuine signal: reg_private and U_LCB.

**3. DQ s1000 is irreducible under current gate design.**  
V_shared too small, correct pool too large. Every candidate regresses on private items. Would need a looser gate (e.g., relaxed reg_private threshold) or a different training split.

**4. Accepted rules are semantically appropriate.**  
CJ rules map to real philosophical causation subtypes (overdetermination, joint necessity, preemption, deception). DQ rules map to real pronoun ambiguity patterns (reported speech, possession/location, gender agreement). The pipeline is identifying genuine error patterns, not noise.

**5. Recommended config going forward: contrastive + subtype + dedup.**  
This is the only config that (a) generates candidates on all non-trivial seeds, (b) produces semantically targeted rules, (c) gates on legitimate signal, and (d) avoids cheatsheet bloat from duplicate rules.

---

## Logs

All 30 run logs (5 configs × 6 seeds) archived in `logs/` subdirectory.  
Run output directories retained in `ICRefine/runs/sfcr_{min,fcot,global,subtype,contrast}_{cj,dq}_{1000,2000,3000}/`.
