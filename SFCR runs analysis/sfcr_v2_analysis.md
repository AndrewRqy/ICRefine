# SFCR v2 Analysis — Why No Rules Are Accepted

**Experiments covered:** `sfcr_v2_llm_test` (7 jobs) · `sfcr_v2_routing_diag` (9 jobs, 3 with output)
**Date:** 2026-05-13

---

## Summary

Zero rules were accepted across all 16 jobs in the v2 experiments. Every rejection was triggered by the **count gate** (`n_count_gate = n_rejected`), not the U_LCB statistical gate. The root cause is a **complete routing failure**: rules cannot activate on the shared-failure items they are designed to fix, so the count gate sees `fixed_shared_count = 0` and immediately rejects.

---

## 1. The Acceptance Pipeline (Background)

For a candidate rule to be accepted, it must pass in sequence:

1. **Activation** — the router matches the rule's USE WHEN / DO NOT USE WHEN conditions against validation items. A rule that activates on no shared-failure items can fix nothing.
2. **Count gate** — requires `fixed_shared_count ≥ min_fix_count` (a minimum absolute number of shared failures fixed). If activation never fires on shared-failure items, this gate fails immediately.
3. **U_LCB gate** — statistical lower-confidence-bound on net utility. Only reached if the count gate passes.

---

## 2. Stage 1 — Manual Near-Oracle Rules (3 jobs)

These jobs used hand-written rules known to be semantically correct for the task, with `repair_attempts=0`. Even with oracle-quality rules, nothing was accepted.

| Job | V_shared | Candidates | activated_shared | fixed_shared | Reject reason |
|-----|----------|-----------|-----------------|--------------|---------------|
| cj_manual | 22 | 1 | 0 | 0 | count gate |
| gs_manual | 12 | 1 | 0 | 0 | count gate |
| logiqa_manual | 58 | 1 | 0 | 0 | count gate |

All three manual rules show `activated_shared_count = 0` and `activation_precision = 0.0`. The rule never matched a single shared-failure item. This is a **pure routing failure** — the USE WHEN conditions did not activate on any of the relevant items. Since generator quality is ruled out (rules were manually written), the problem is entirely in the router itself.

**Key insight:** The fact that even perfect rules cannot activate means the bug is upstream of everything else — fixing the generator or repair logic is irrelevant until routing works.

---

## 3. Stage 3 — LLM-Generated Rules (4 jobs)

### 3a. Causal Judgement — mini vs strong generator

Both generators produced 9 candidates each after repair. Activation improved over the manual baseline (the router occasionally fired) but `fixed_shared_count` remained at 0 or 1 for every candidate.

**cj_minigen (gpt-4.1-mini, 9 candidates):**

| candidate | activated_shared | fixed_shared | easy_activation | activation_precision |
|-----------|-----------------|--------------|-----------------|---------------------|
| 5906ab02 | 0 | 0 | 0 | 0.00 |
| 10327305 | 2 | 0 | 0 | 0.67 |
| e49dbbb5 | 0 | 0 | 0 | 0.00 |
| 3c219b4f | 1 | 0 | 0 | 1.00 |
| ac77a048 | 0 | 0 | 0 | 0.00 |
| 27fe7943 | 2 | 0 | 0 | 0.67 |
| 63e4456e | 3 | 0 | 1 | 0.75 |
| dcb078b5 | 8 | **1** | 2 | 0.62 |
| d8e5b51a | 6 | 0 | 4 | 0.46 |

Best candidate (`dcb078b5`): activated 8 shared-failure items, precision 0.62, but only fixed 1. The count gate requires more than 1 fix; with V_shared=22 failures to address, fixing a single item is well below threshold.

**cj_stronggen (gpt-4.1, 9 candidates):**

| candidate | activated_shared | fixed_shared | easy_activation | activation_precision |
|-----------|-----------------|--------------|-----------------|---------------------|
| 707b58f8 | 8 | **1** | 1 | 0.67 |
| 2593a56d | 0 | 0 | 0 | 0.00 |
| 2b1c0d5e | 0 | 0 | 0 | 0.00 |
| cb8c55ba | 1 | 0 | 0 | 1.00 |
| 52b73b97 | 3 | **1** | 1 | 0.50 |
| 17bc603b | 0 | 0 | 0 | 0.00 |
| fa1f2f83 | 0 | 0 | 0 | 0.00 |
| e04532c3 | 0 | 0 | 0 | 0.00 |
| e237f2ef | 1 | 0 | 3 | 0.25 |

Strongest candidate (`707b58f8`): activated 8/24 shared failures, precision 0.67, but fixed only 1. Same pattern as mini.

**gpt-4.1 vs gpt-4.1-mini on CJ:** No meaningful difference. Both produce the same best-case outcome (1 fix, count gate rejection). Generator strength is not the bottleneck here.

### 3b. LogiQA — mini vs strong generator

LogiQA has a much larger failure pool (V_shared = 58–65) which should make it easier to accumulate fixes, but the pattern holds.

**logiqa_minigen (6 candidates):**

| candidate | activated_shared | fixed_shared | easy_activation | activation_precision |
|-----------|-----------------|--------------|-----------------|---------------------|
| 442a2eb3 | 0 | 0 | 1 | 0.00 |
| ff779642 | 1 | 0 | 0 | 1.00 |
| 52050e04 | 1 | 0 | 1 | 0.50 |
| c2f709b4 | 2 | 0 | 0 | 1.00 |
| 30348f2d | 6 | 0 | 10 | 0.33 |
| 3a13412d | 1 | 0 | 0 | 0.33 |

Best candidate (`30348f2d`): activated 6 shared failures, but fixed 0. The rule activates broadly but does not change the model's answer on any item it fires on. High `easy_activation_count` (10) suggests the rule is latching onto easy-correct items rather than targeting the actual failure pattern.

**logiqa_stronggen (9 candidates):**

| candidate | activated_shared | fixed_shared | easy_activation | activation_precision |
|-----------|-----------------|--------------|-----------------|---------------------|
| 3875dfdf | 0 | 0 | 1 | 0.00 |
| 3c3886b7 | 2 | 0 | 2 | 0.50 |
| 9da6f10a | 0 | 0 | 0 | 0.00 |
| 903df604 | 3 | 0 | 2 | 0.43 |
| 1008e0f3 | 3 | 0 | 5 | 0.25 |
| b3acc507 | 5 | **1** | 5 | 0.42 |
| 735c219b | 11 | 0 | 8 | 0.38 |
| e6cc89c4 | 6 | 0 | 10 | 0.29 |
| 23a403bc | 7 | 0 | 7 | 0.35 |

Best activation candidate (`735c219b`): activated 11 shared failures — the highest of any candidate across all jobs — but still fixed 0. Across 9 candidates, only `b3acc507` fixed 1 item. The rule activates on failures but cannot actually correct them.

---

## 4. Routing Diagnostic Runs (sfcr_v2_routing_diag)

The routing_diag experiment tested three parameter combinations on GS (CJ and logiqa_en dirs are empty — those jobs did not produce output files):

| Fix | min_veto_matches | router_min_matches | Meaning |
|-----|-----------------|-------------------|---------|
| A | 2 | 2 | Harder to veto, same activation threshold |
| B | 1 | 1 | Easier to activate, same veto threshold |
| AB | 2 | 1 | Harder to veto AND easier to activate |

**Results for GS:**

| Fix | V_shared | activated_shared | fixed_shared | accepted |
|-----|----------|-----------------|--------------|---------|
| A | 10 | 0 | 0 | 0 |
| B | 16 | 0 | 0 | 0 |
| AB | 11 | 0 | 0 | 0 |

All three GS routing variants still show `activated_shared_count = 0`. Neither relaxing the veto threshold (Fix A) nor lowering the activation threshold (Fix B) nor both together (Fix AB) caused the rule to activate on any shared-failure item. The routing fixes had no effect on GS.

Note: V_shared varies across fixes (10 / 16 / 11) because scoring is stochastic — the shared-failure set is re-computed each run. This variance is noise, not a signal difference between fixes.

---

## 5. Root Cause Summary

Three distinct failure modes are present, layered on top of each other:

### Failure Mode 1: Zero activation (manual rules, GS routing_diag)
The USE WHEN conditions never match any validation item at all — `activated_shared_count = 0`. This is the most severe form: the router completely fails to route. The manual rules in Stage 1 and all GS routing_diag runs exhibit this. The routing parameter fixes (min_veto_matches, router_min_matches) do not help because the issue is not about veto storm or empty activation threshold — the semantic matching itself is broken for these rule types.

### Failure Mode 2: Activation without fixing (logiqa Stage 3)
Rules activate on shared-failure items (`activated_shared_count > 0`, up to 11 for logiqa_stronggen) but `fixed_shared_count = 0`. The router fires correctly but the rule content does not change the model's answer. Two sub-causes:
- Rules activate on items where the model's error is not the type the rule addresses.
- High `easy_activation_count` (up to 10) shows the rule fires broadly on easy-correct items, indicating the USE WHEN conditions are too coarse.

### Failure Mode 3: Insufficient fix count (CJ Stage 3)
The best CJ candidates fix exactly 1 shared-failure item. The count gate threshold is higher than 1. With V_shared = 22–24 and only 1 fix achievable, the count gate will never pass unless either (a) the threshold is lowered to 1, or (b) rule content is substantially improved so more failures are corrected.

---

## 6. What Needs to Change

| Failure mode | Affected jobs | Proposed fix |
|---|---|---|
| Zero activation | cj_manual, gs_manual, logiqa_manual, GS routing_diag | Debug the USE WHEN semantic matching — the string-match or embedding logic is not finding relevant items. Log which items the router evaluates and why none match. |
| Activation without fixing | logiqa Stage 3 | Rule content needs to target the specific reasoning error, not just surface patterns. Stronger oracle context or explicit failure-mode subtypes may help. |
| Insufficient fix count | CJ Stage 3 | Either lower `min_fix_count` to 1 for tasks with small V_shared, or improve rule recall (wider USE WHEN conditions, multiple rules per subtype). |

The routing_diag experiment (Fix A / B / AB) addressed neither Failure Mode 1 nor 2 — the parameter changes are orthogonal to the actual bugs. The next diagnostic step should be adding router decision logging to trace why `activated_shared_count = 0` even when manually crafted rules are evaluated against items they are semantically designed to match.
