# SF-CR Pilot Runs — Detailed Analysis

Runs completed: 7 (all seed 1000)  
Total candidates evaluated: 56 (8 per run)  
Total accepted: 0  

summary: Every candidate across all seven SF-CR runs was rejected. The primary failure mode is negative U_LCB caused by high `reg_easy` (regression on already-correct examples), which overwhelms marginal `delta_shared` gains. A secondary mode is non-zero `reg_private` pushing U_LCB further negative. The single exception — GS llama_ho candidate `fdd11388` — achieved positive U_LCB (+0.027) but was rejected by the `private_activation_rate > 10%` guard (100%). 

---
Current implementation:

1. Partitions the training split into three regions under the source-model baseline cheatsheet:
   - **V_shared** — examples where the source model *and* every proxy model all fail
   - **V_private** — examples where the source model fails but all proxy models pass
   - **V_easy** — examples where the source model *and* all proxy models pass
2. Generates 8 candidate rules targeting V_shared, instructed to avoid V_private activation
3. Validates each candidate on each proxy model using three signals:
   - `delta_shared` — accuracy gain on V_shared under the candidate rule
   - `reg_private` — regression rate on V_private
   - `reg_easy` — regression rate on V_easy
4. Computes per-proxy **U_LCB** (utility lower-confidence bound incorporating all three signals)
5. Accepts if `U_LCB > 0` for **all** proxies **and** `private_activation_rate ≤ 10%`


---

## Run Configurations at a Glance

| Run ID | Task | Held-out | CS-ICL init | Source acc | V_shared | V_private | V_easy | Accepted |
|---|---|---|---|---|---|---|---|---|
| `sfcr_cj_1000` | causal_judgement | llama | holistic-v7 best | 75.0% | 13 | 2 | 40 | 0/8 |
| `sfcr_cj_1000_llama_v2` | causal_judgement | llama | v2 cheatsheet | 85.0% | 9 | 0 | 34 | 0/8 |
| `sfcr_cj_csicl_1000` | causal_judgement | gemini | CS-ICL baseline | 78.3% | 11 | 2 | 39 | 0/8 |
| `sfcr_geometric_shapes_1000` | geometric_shapes | gemini | holistic-v7 best | 94.7% | 4 | 0 | 60 | 0/8 |
| `sfcr_geometric_shapes_1000_llama_ho` | geometric_shapes | llama | holistic-v7 best | 92.0% | 5 | 1 | 61 | 0/8 |
| `sfcr_logiqa_en_1000` | agieval_logiqa_en | gemini | holistic-v7 best | 63.0% | 33 | 4 | 42 | 0/8 |
| `sfcr_lsat_ar_1000` | agieval_lsat_ar | gemini | holistic-v7 best | 66.7% | 20 | 0 | 10 | 0/8 |

### Jaccard failure-overlap matrices

| Run | mini↔gpt-4.1 | mini↔gemini | mini↔llama |
|---|---|---|---|
| `sfcr_cj_1000` | 0.563 | 0.500 | 0.611 |
| `sfcr_cj_1000_llama_v2` | 0.727 | 0.471 | 0.444 |
| `sfcr_cj_csicl_1000` | 0.600 | 0.471 | 0.500 |
| `sfcr_geometric_shapes_1000` | 0.667 | 0.429 | 0.182 |
| `sfcr_geometric_shapes_1000_llama_ho` | 0.714 | 0.375 | 0.273 |
| `sfcr_logiqa_en_1000` | 0.523 | 0.646 | 0.511 |
| `sfcr_lsat_ar_1000` | 0.423 | 0.529 | 0.311 |

Notably, `sfcr_geometric_shapes_1000` has mini↔llama Jaccard of only 0.182 — the source and llama barely share failures, which explains llama's near-zero `delta_shared` across all 8 candidates.

---

## Run 1: `sfcr_cj_1000` — CJ, held-out=llama, holistic init

**Configuration:** source_acc=75.0%, V_shared=13, V_private=2, V_easy=40  
**Init cheatsheet:** holistic v7 best (refined from CS-ICL gpt-4.1)

The holistic-v7 run had already achieved reasonably high Jaccard with all three proxies (0.50–0.61), meaning shared failures are genuine consensus failures. Despite this, every candidate was rejected due to U_LCB ≤ 0.

### Critical structural issue: reg_private = 0.6576 across all candidates

The V_private region has only **2 examples**. Across all 8 candidates, every proxy shows `reg_private = 0.6576`. This exact value equals 1.3152/2 scaled — it means the candidates are **regressing roughly 1.3 out of 2 private examples on average**. This may reflect that the generated rules are broad enough to activate on private examples despite the `DO NOT USE WHEN` constraint.

This is a major amplification effect: with only 2 private examples, a single activation causing a wrong prediction produces `reg_private ≈ 0.66`, which is a catastrophic signal in the U_LCB formula regardless of `delta_shared`.

### Per-candidate detail

| ID | Rule (truncated) | U_LCB | priv_act | Reject reason |
|---|---|---|---|---|
| `b500c42d` | "When multiple sufficient conditions exist, typical causation judgments depend on whether the questioned cause is perceiv..." | -1.7024 | 0.50 | U_LCB ≤ 0 |
| `c1a54c73` | "When multiple sufficient conditions exist, typical causation judgments focus on whether the questioned cause is the most..." | -1.7191 | 0.00 | U_LCB ≤ 0 |
| `13400852` | "When multiple sufficient conditions exist, typical causation judgments depend on whether the questioned cause is the..." | -1.8068 | 0.00 | U_LCB ≤ 0 |
| `d58f5ae0` | "When multiple sufficient conditions exist, typical causation judgments depend on whether the questioned cause is the..." | -1.8072 | 0.00 | U_LCB ≤ 0 |
| `bcc3b2ed` | "When multiple sufficient conditions exist, typical causation judgments focus on whether the questioned cause is the..." | -1.8205 | 1.00 | U_LCB ≤ 0 |
| `613eb179` | "When multiple sufficient conditions exist, typical causation judgments focus on whether the questioned cause is the..." | -1.8240 | 0.00 | U_LCB ≤ 0 |
| `2c179057` | "When multiple sufficient conditions exist, if the outcome depends on at least one condition being present..." | -1.9114 | 0.50 | U_LCB ≤ 0 |
| `9962d4a8` | "When multiple conditions or actions together produce an outcome, attribute cause only to those actions..." | -2.2144 | 1.00 | U_LCB ≤ 0 |

**Per-proxy stats across all 8 candidates (representative range):**

| Proxy | delta_shared range | reg_private (all) | reg_easy range |
|---|---|---|---|
| gpt-4.1 | 0.000–0.016 | **0.658** (constant) | 0.176–0.258 |
| gemini-2.0-flash | 0.000–0.090 | **0.658–0.906** | 0.176–0.392 |
| llama-3.3-70b | 0.013–0.117 | **0.658** (constant) | 0.176–0.258 |

**Interpretation:** All candidate rules are semantic variations on the same "multiple sufficient causes" pattern — the LLM converged to nearly identical rule content across 8 generations. The entire batch is likely activating the same examples (hence constant reg_private). The delta_shared gain (1–3 examples out of 13) cannot compensate for losing 1.3 private examples per proxy every time. The U_LCB floor at around −1.7 is essentially determined by the reg_private penalty.

---

## Run 2: `sfcr_cj_1000_llama_v2` — CJ, held-out=llama, v2 init

**Configuration:** source_acc=85.0%, V_shared=9, V_private=0, V_easy=34  
**Init cheatsheet:** v2 cheatsheet (higher accuracy baseline than holistic run)

With **V_private=0** (no source-private failures), the reg_private term should be zero — and indeed all proxies show `reg_private = 0.000` for 5 out of 8 candidates. Yet all 8 are rejected. The culprit here is **reg_easy**, not reg_private.

### Critical issue: reg_easy dominates despite V_private=0

All candidates show `reg_easy` of 0.215–0.315 across proxies. With V_easy=34 examples, that means roughly 7–11 previously-correct examples are regressed per candidate per proxy. The small `delta_shared` (0.000–0.102 on 9 shared examples, so 0–1 example fixed) cannot offset this.

Notably, U_LCB scores are much less negative here (−0.19 to −0.83) compared to run 1, reflecting the absence of a reg_private catastrophe. This suggests the algorithm is working as designed — it's just that the rules genuinely cannot help V_shared without hurting V_easy.

### Per-candidate detail

| ID | Rule (truncated) | U_LCB | priv_act | gpt-4.1 Δshared | gemini Δshared | llama Δshared | Mean reg_easy |
|---|---|---|---|---|---|---|---|
| `ee2fe4ad` | "In cases where an outcome results from multiple independently sufficient causes present simultaneously..." | -0.1882 | 0.00 | +0.043 | +0.102 | +0.000 | 0.248 |
| `7a0cc50a` | "In cases where an outcome results from multiple independent sufficient causes present simultaneously..." | -0.2371 | 0.00 | +0.000 | +0.035 | +0.000 | 0.248 |
| `6017b7eb` | "In cases where an outcome results from multiple independent sufficient causes present simultaneously..." | -0.7191 | 0.00 | +0.014 | +0.035 | +0.016 | 0.281 |
| `d4b7131d` | "In cases where an outcome results from multiple independent sufficient causes present simultaneously..." | -0.7211 | 0.50 | +0.014 | +0.035 | +0.000 | 0.248 |
| `49d3bd7a` | "In cases where an outcome results from multiple independent sufficient causes (disjunctive conditions)..." | -0.7237 | 0.00 | +0.014 | +0.035 | +0.016 | 0.248 |
| `a6838dd5` | "In cases where an outcome results from multiple independently sufficient causes present simultaneously..." | -0.7310 | 0.00 | +0.014 | +0.011 | +0.000 | 0.215 |
| `bde1f07c` | "In cases where an outcome results from multiple independent sufficient causes present simultaneously..." | -0.8168 | 0.50 | +0.014 | +0.102 | +0.000 | 0.281 |
| `d705067f` | "In cases where an outcome results from multiple independent sufficient causes or disjunctive conditions..." | -0.8302 | 0.00 | +0.000 | +0.066 | +0.000 | 0.248 |

**Interpretation:** Candidate `ee2fe4ad` is the closest to acceptance (U_LCB=−0.188): gemini gains +10.2% and gpt-4.1 gains +4.3% on shared examples, but llama gains nothing on its own shared failures. Since llama is the held-out target, the proxy panel still includes llama, and llama's zero delta_shared combined with ~31% reg_easy is sufficient to make U_LCB negative.

The consistent reg_easy around 0.21–0.31 across all candidates suggests the rule class (multiple-sufficient-causes framing) is too general — it activates on easy examples that don't need it and causes incorrect answers there.

---

## Run 3: `sfcr_cj_csicl_1000` — CJ, held-out=gemini, CS-ICL init

**Configuration:** source_acc=78.3%, V_shared=11, V_private=2, V_easy=39  
**Init cheatsheet:** CS-ICL baseline (gpt-4.1-mini generated, iter0)

Similar to runs 1–2 but with gemini as the held-out target (excluded from proxy panel). U_LCB values (−0.27 to −0.36) are considerably less negative than run 1, reflecting zero reg_private for most candidates (one exception: candidate `89466f9c` has gemini reg_private=1.000).

### Per-candidate detail

| ID | Rule (truncated) | U_LCB | priv_act | gpt-4.1 Δshared | gemini Δshared | llama Δshared | Notes |
|---|---|---|---|---|---|---|---|
| `9bd930c1` | "When multiple agents or actions contribute to a harmful joint outcome but only some acted contrary..." | -0.2656 | 1.00 | +0.051 | +0.152 | +0.127 | Best delta_shared; high priv_act kills it |
| `252ac138` | "When multiple agents' independent actions combine to produce a harmful outcome only if both occur..." | -0.3004 | 0.00 | +0.016 | +0.051 | +0.043 | |
| `567dc8dc` | "When multiple agents independently act but only one intentionally or knowingly violates instructions..." | -0.3007 | 1.00 | +0.016 | +0.051 | +0.082 | |
| `c28c4457` | "When multiple agents' independent actions combine to produce a harmful outcome only if both act, and..." | -0.3009 | 1.00 | +0.016 | +0.016 | +0.082 | |
| `b8910209` | "When multiple agents or actions contribute to a joint outcome through independent, non-overlapping..." | -0.3032 | 1.00 | +0.016 | +0.051 | +0.014 | |
| `89466f9c` | "When multiple agents independently act but only one follows the correct instruction or intended plan..." | -0.3310 | 1.00 | +0.051 | +0.097 | +0.082 | gemini reg_private=1.000 |
| `d062afa4` | "When multiple agents or actions contribute to a harmful or undesired joint outcome that only occurs..." | -0.3617 | 1.00 | +0.016 | +0.051 | +0.043 | |
| `1b8de5de` | "When multiple agents independently perform actions that combine to cause harm, but only one agent..." | -0.3620 | 0.00 | +0.051 | +0.016 | +0.014 | |

**Interpretation:** The candidates are semantically diverse (multiple agents / joint causation framing) and some achieve reasonable delta_shared gains (0.05–0.15 range). However, reg_easy is uniformly ~0.16–0.36 across proxies. The rule class is again too broad. Candidate `9bd930c1` achieves the best gains across all three proxies but has priv_act=1.00, meaning it activates on both private examples — a fatal signal that the rule doesn't respect the specificity constraint. The absence of V_private regression (reg_private=0.000 for most) is better than run 1, but reg_easy cannot be reduced below ~0.16 with this rule class.

---

## Run 4: `sfcr_geometric_shapes_1000` — GS, held-out=gemini, holistic init

**Configuration:** source_acc=94.7%, V_shared=4, V_private=0, V_easy=60  
**Init cheatsheet:** holistic v7 best

Near-ceiling accuracy (94.7%) creates an extremely small V_shared region (only **4 examples**). Any candidate rule that fixes 1–2 of those 4 shared examples gains delta_shared of 0.25–0.50 per proxy — but the large V_easy=60 pool means even small absolute reg_easy (~1–3 examples) registers as 0.09–0.23 fractional rate.

### Critical observation: llama Jaccard = 0.182

The mini↔llama Jaccard is only 0.182. This means llama barely overlaps with the source on failures — the 4 shared failures are shared with gpt-4.1 and gemini (Jaccard 0.667 and 0.429) but not llama. As a result, llama shows `delta_shared = 0.000` on every single candidate (the 4 "shared" failures aren't actually llama failures). This zero contribution pulls U_LCB negative even when gpt-4.1 and gemini show strong delta_shared gains.

### Per-candidate detail

| ID | Rule (truncated) | U_LCB | priv_act | gpt-4.1 Δshared | gemini Δshared | llama Δshared | gemini reg_private |
|---|---|---|---|---|---|---|---|
| `0d6a49b2` | "When a closed quadrilateral has all four sides equal or nearly equal but does not have right angles..." | -0.1447 | 1.00 | +0.231 | +0.301 | **0.000** | 0.500 |
| `4bbe135c` | "When a closed quadrilateral has all four sides equal or nearly equal but does not have all right ang..." | -0.1448 | 1.00 | +0.118 | +0.301 | **0.000** | 0.500 |
| `ff579288` | "When a closed four-sided shape has two pairs of opposite sides equal in length and parallel but does..." | -0.1740 | 1.00 | +0.000 | +0.150 | **0.000** | 0.500 |
| `6ad1b314` | "When a closed quadrilateral has all four sides equal or nearly equal but does not have all right ang..." | -0.1822 | 1.00 | +0.118 | +0.301 | **0.000** | 0.000 |
| `e3faf45f` | "When a closed path uses two elliptical arc commands with equal radii and zero rotation that form a f..." | -0.1863 | 1.00 | +0.231 | +0.301 | **0.000** | 0.500 |
| `343d43da` | "When a closed polygon has four sides and all angles are right but the shape is not perfectly axis-al..." | -0.2233 | 1.00 | +0.036 | +0.046 | **0.000** | 0.500 |
| `6cd93774` | "When a closed polygon has four sides and all angles are right but the answer choices include both re..." | -0.6005 | 1.00 | +0.376 | +0.510 | +0.046 | 0.500 |
| `8e533a83` | "When a closed polygon has four sides and all angles are right but the shape is not axis-aligned, or..." | -0.6845 | 1.00 | +0.231 | +0.150 | **0.000** | 0.500 |

**Interpretation:** Candidates `0d6a49b2`, `4bbe135c`, `6ad1b314`, `e3faf45f` all achieve excellent gpt-4.1 and gemini gains (+0.10–+0.30), but llama's zero delta_shared suppresses U_LCB below zero. The shared failure region apparently doesn't include the llama failure mode — the 4 examples where all models fail are actually driven by a different SVG pattern from what llama struggles with. This is a Jaccard-interpretation problem: V_shared is defined as source ∩ (union of proxy failures), not source ∩ (intersection). When Jaccard is as low as 0.182 (mini↔llama), llama failures barely overlap the target region.

gemini's `reg_private=0.500` on several candidates (despite V_private=0 for the source) may reflect that the proxy panel's "private" computation under the candidate is different — this warrants investigation.

---

## Run 5: `sfcr_geometric_shapes_1000_llama_ho` — GS, held-out=llama, holistic init

**Configuration:** source_acc=92.0%, V_shared=5, V_private=1, V_easy=61  
**Init cheatsheet:** holistic v7 best

This run produced the **only candidate across all 7 runs to pass the U_LCB gate**: `fdd11388` with U_LCB=+0.0266. It was ultimately rejected by `private_activation_rate = 100% > 10%`.

### The near-miss candidate: `fdd11388`

| Metric | gpt-4.1 | gemini | llama |
|---|---|---|---|
| delta_shared | +0.1876 | +0.2307 | **+0.000** |
| reg_private | 0.000 | 0.000 | 0.000 |
| reg_easy | 0.118 | 0.151 | 0.240 |

**Rule:** "If a closed quadrilateral does not have all right angles and does not meet kite conditions, but has equal-length diagonals or is symmetric about one axis..."

With llama as held-out target (excluded from proxy panel), the proxy panel is only gpt-4.1 and gemini. Both show positive delta_shared (+0.19, +0.23) with zero reg_private. The reg_easy values (~0.12–0.15 for gpt-4.1/gemini; 0.24 for llama) are still high but small enough for U_LCB to come out positive (+0.027).

**However:** `private_activation_rate = 1.000`. The 1 private example (where the source model fails but all proxies pass) triggers the `USE WHEN` condition of this rule. This means the rule would specifically target an example that proxies already handle correctly — exactly the scenario SFCR is designed to avoid. The guard correctly rejects it.

### Speculation on why fdd11388 activated the private example

The private example likely involves a quadrilateral that is source-private (mini fails, gpt-4.1/gemini/llama all pass). The rule's condition "does not have all right angles and does not meet kite conditions but has equal-length diagonals or axis symmetry" happens to describe the private example's SVG path, even though the intended target was the 5 shared failures. This is a precision problem in the rule's USE WHEN clause — it's not specific enough to exclude the one private example.

### Remaining 7 candidates

| ID | U_LCB | priv_act | Notes |
|---|---|---|---|
| `fdd11388` | **+0.0266** | 1.000 | Passed U_LCB, failed priv_act gate |
| `6e3dd754` | -0.0988 | 1.000 | Just below threshold; 2nd best |
| `ae2b7f86` | -0.2494 | 1.000 | gemini strong Δshared=+0.376 but llama 0 |
| `77688d9c` | -0.2526 | 1.000 | gemini Δshared=+0.118 |
| `76c896b4` | -0.3300 | 1.000 | Low delta across board |
| `700a4fce` | -0.4346 | 1.000 | gpt-4.1 reg_private=0.333 |
| `11ede173` | -0.4968 | 1.000 | Both gpt-4.1/700a and 11ede have reg_private |
| `c1784dd7` | -0.5807 | 1.000 | gpt-4.1 reg_private=0.333 |

All 8 candidates have `priv_act = 1.000`. This is a striking pattern — the single private example (a specific SVG quadrilateral) matches the USE WHEN conditions of every generated rule. The private example may be unusually prominent in the shared-failure context examples provided to the rule generator, causing all rules to inadvertently describe it.

---

## Run 6: `sfcr_logiqa_en_1000` — LogiQA-en, held-out=gemini, holistic init

**Configuration:** source_acc=63.0%, V_shared=33, V_private=4, V_easy=42  
**Init cheatsheet:** holistic v7 best

This run has the largest V_shared region (33 examples) and the best-balanced partition. With high Jaccard across all pairs (0.511–0.646), the shared failures are genuinely representative. U_LCB values (−0.61 to −0.95) are more negative than the CJ runs despite better Jaccard — the culprit is reg_private.

### Critical issue: reg_private is high despite V_private=4 being small

All proxies show reg_private between 0.28 and 0.83, meaning the candidates regularly regress 1–3 of the 4 private examples. The generated rules use generic logical verification language ("Explicitly verify that all relevant quantitative comparisons..."), which is broad enough to activate on private examples that test the same logical faculties.

### Per-candidate detail

| ID | Rule (truncated) | U_LCB | priv_act | gpt-4.1 reg_priv | gemini reg_priv | llama reg_priv |
|---|---|---|---|---|---|---|
| `a8b841ad` | "Explicitly verify that all relevant quantitative or qualitative comparisons are logically consistent..." | -0.6057 | 0.70 | 0.510 | 0.510 | 0.278 |
| `be88f010` | "Explicitly verify that comparisons or conclusions about quantities, qualities, or causal effects..." | -0.7016 | 0.20 | 0.603 | 0.404 | 0.278 |
| `f9dcdd2d` | "Explicitly verify that all relevant conditions and constraints are fully integrated into the reasoning..." | -0.8146 | 0.50 | 0.687 | 0.603 | 0.404 |
| `a1f27c16` | "Explicitly verify that all conditionals and logical equivalences are correctly interpreted, including..." | -0.8179 | 0.60 | 0.687 | 0.510 | 0.404 |
| `e232eefa` | "Explicitly verify that all relevant quantitative or qualitative comparisons are logically consistent..." | -0.8440 | 0.20 | 0.687 | 0.404 | 0.404 |
| `99c0e158` | "Explicitly verify that all relevant quantitative or qualitative comparisons are logically consistent..." | -0.8752 | 0.20 | 0.763 | 0.603 | 0.278 |
| `a072cdd0` | "Explicitly verify that all conditionals and logical equivalences in the argument are correctly interp..." | -0.9359 | 0.50 | 0.832 | 0.404 | 0.603 |
| `8de21914` | "Explicitly verify that all conditionals and logical constraints are fully integrated when drawing..." | -0.9487 | 0.40 | 0.832 | 0.510 | 0.278 |

**Interpretation:** All 8 candidates are variants of the same meta-verification instruction pattern ("Explicitly verify that..."). This rule class is too abstract — it tells the model to check its reasoning, not what specifically to check. As a result, the rule activates broadly: on shared failures, private examples, and easy examples alike. The best candidate (`a8b841ad`) has the lowest U_LCB at −0.61, with moderate reg_private (~0.28–0.51). The improvement delta_shared per proxy is consistent but small (0.014–0.154), while reg_private overwhelms it.

A more specific rule narrowed to the exact logical error type in V_shared (e.g., comparative quantity misinterpretation) would be needed to pass validation.

---

## Run 7: `sfcr_lsat_ar_1000` — LSAT-AR, held-out=gemini, holistic init

**Configuration:** source_acc=66.7%, V_shared=20, V_private=0, V_easy=10  
**Init cheatsheet:** holistic v7 best

This run has the most extreme partition: **V_easy=10** (only 10 examples all models already get right) and **V_shared=20** (the majority of examples are shared failures). With such a small V_easy, the absolute number of regressions is tiny (0–7 examples) but the fractional reg_easy is enormous (0.324–0.694).

### Critical issue: extreme reg_easy due to tiny V_easy

With only 10 easy examples, a rule that causes 3–7 additional errors produces reg_easy of 0.32–0.69. Every candidate shows this pattern. Combined with zero private examples (V_private=0), U_LCB is determined almost entirely by the balance of delta_shared vs. reg_easy.

gemini also shows `reg_private = 1.000` on candidates `1580ac39`, `1474e5d1`, `d855d5d0`, `0c16f925` — but with V_private=0 from the source's perspective, this likely means the proxy's own "private" region under the rule is being triggered, or there is a computation artifact.

### Per-candidate detail

| ID | Rule (truncated) | U_LCB | priv_act | gpt-4.1 Δshared | gemini Δshared | llama Δshared | Mean reg_easy |
|---|---|---|---|---|---|---|---|
| `1580ac39` | "Explicitly incorporate the interplay of multiple conditional chains and compound constraints..." | -0.4810 | 1.00 | +0.118 | 0.000 | 0.000 | 0.496 |
| `b38c18e9` | "Explicitly incorporate the indirect and chained consequences of conditional statements by fully prop..." | -0.4810 | 1.00 | +0.118 | 0.000 | 0.000 | 0.511 |
| `860b736b` | "Explicitly consider the indirect and chained consequences of conditional statements, including..." | -0.5761 | 1.00 | +0.118 | 0.000 | +0.026 | 0.536 |
| `1474e5d1` | "Explicitly incorporate the logical consequences of conditional chains and mutual exclusivity..." | -0.5807 | 1.00 | +0.118 | 0.000 | +0.026 | 0.554 |
| `36f28322` | "Explicitly incorporate the indirect and chained consequences of conditional statements, including co..." | -0.6030 | 1.00 | 0.000 | +0.026 | 0.000 | 0.505 |
| `d855d5d0` | "Explicitly incorporate the contrapositive and biconditional reasoning for all conditional statements..." | -0.6034 | 1.00 | +0.118 | +0.026 | 0.000 | 0.401 |
| `12f3ecb6` | "Explicitly incorporate the contrapositive and biconditional reasoning for all conditional statements..." | -0.6036 | 1.00 | +0.118 | 0.000 | 0.000 | 0.491 |
| `0c16f925` | "Explicitly incorporate the interplay of multiple conditional chains and compound constraints by syst..." | -0.7053 | 1.00 | +0.036 | 0.000 | 0.000 | 0.621 |

**Structural observation:** gpt-4.1 gains delta_shared on 6/8 candidates (+0.118 = ~2.4 examples out of 20), but gemini and llama gain nothing on 6/8 candidates. This is a severe alignment failure — the rule class (conditional chain / contrapositive reasoning) specifically helps gpt-4.1's failure patterns but doesn't overlap with gemini's or llama's shared failures. With gemini as held-out target and also showing zero delta_shared as a proxy panel member, U_LCB is guaranteed negative.

The low mini↔llama Jaccard (0.311) is consistent: llama's failure region doesn't overlap well with the source, so a rule targeting source shared-failures will miss llama's errors entirely.

---

## Cross-Run Patterns and Rejection Analysis

### Pattern 1: reg_easy is the dominant blocker (5 of 7 runs)

In all runs except `sfcr_cj_1000` (where reg_private dominates), reg_easy is the primary driver of negative U_LCB. The generated rules are too broad — they target a reasoning pattern (multiple causation, logical verification, conditional chains) that is also present in correctly-answered easy examples. When the rule fires on easy examples and changes the model's reasoning, it frequently produces wrong answers.

**Root cause:** The LLM rule generator tends to produce general heuristics rather than narrow discriminative rules. A rule like "Explicitly verify conditional chains" activates on every logical reasoning question, not just the shared-failure subset.

### Pattern 2: Small absolute regions amplify fractional rates

The V_easy and V_private regions are often tiny (2–10 examples), while the rule validation uses fractional rates. A single incorrect prediction out of 2 private examples = 50% reg_private. Three wrong out of 10 easy examples = 30% reg_easy. The U_LCB formula cannot distinguish "catastrophically broad rule" from "unlucky small sample" — both look the same.

**Implication:** The SFCR validation protocol needs minimum-size guards before comparing fractional rates. Runs with V_private < 5 or V_easy < 20 are effectively operating with very high variance estimates.

### Pattern 3: Semantic convergence across candidates in the same run

In `sfcr_cj_1000`, all 8 candidates are near-identical variants of "multiple sufficient conditions → typical causation judgments." In `sfcr_logiqa_en_1000`, all 8 are "Explicitly verify [some logical consistency]." The LLM rule generator, given a consistent shared-failure region, converges to a single semantic frame and generates variations rather than fundamentally different hypotheses.

**Consequence:** If the dominant semantic frame fails validation (reg_easy too high), all 8 candidates fail. The diversity budget of 8 candidates is wasted.

### Pattern 4: Proxy panel misalignment (GS and LSAT-AR)

For geometric_shapes and lsat_ar, some proxies consistently show zero delta_shared across all candidates. This happens when a proxy's shared-failure overlap with the source is low (Jaccard ≤ 0.27), meaning the "shared failures" identified in V_shared don't actually overlap with that proxy's failures. Since U_LCB requires positive expected gain across all proxies, one zero-gain proxy is enough to push U_LCB negative even if other proxies benefit substantially.

**This is a correct behavior** of the safety mechanism — a rule that helps gpt-4.1 but does nothing for llama is not transferable to the llama regime. However, it also exposes that V_shared computed as source ∩ (union of proxy failures) may be too inclusive; if any proxy fails an example, it enters V_shared even if other proxies don't.

### Pattern 5: priv_act universally high in GS runs

In both geometric_shapes runs, every candidate has `private_activation_rate = 1.000`. The single private example (V_private=1 in the llama_ho run) appears to match the USE WHEN condition of every generated rule. This suggests the rule generator is effectively memorizing the private example's description when constructing USE WHEN clauses, likely because V_private examples are included as anti-activation context and the LLM interprets this as a description to encode rather than a constraint to exclude.

---

## Summary Statistics

### U_LCB distribution across all 56 candidates

| Run | Min U_LCB | Max U_LCB | Median U_LCB |
|---|---|---|---|
| sfcr_cj_1000 | -2.214 | -1.702 | -1.824 |
| sfcr_cj_1000_llama_v2 | -0.830 | -0.188 | -0.727 |
| sfcr_cj_csicl_1000 | -0.362 | -0.266 | -0.308 |
| sfcr_geometric_shapes_1000 | -0.685 | -0.145 | -0.184 |
| sfcr_geometric_shapes_1000_llama_ho | -0.581 | **+0.027** | -0.376 |
| sfcr_logiqa_en_1000 | -0.949 | -0.606 | -0.828 |
| sfcr_lsat_ar_1000 | -0.705 | -0.481 | -0.581 |

### Primary rejection cause by run

| Run | Primary cause | Secondary cause |
|---|---|---|
| sfcr_cj_1000 | reg_private=0.658 constant (2 private examples, broadly activating rule) | reg_easy=0.18–0.39 |
| sfcr_cj_1000_llama_v2 | reg_easy=0.21–0.31 (llama delta_shared=0 on all candidates) | — |
| sfcr_cj_csicl_1000 | reg_easy=0.16–0.36 | priv_act=1.0 on 6/8 candidates |
| sfcr_geometric_shapes_1000 | llama delta_shared=0.000 (low Jaccard 0.182) | gemini reg_private=0.50 |
| sfcr_geometric_shapes_1000_llama_ho | priv_act=1.000 on sole passing candidate | — |
| sfcr_logiqa_en_1000 | reg_private=0.28–0.83 (abstract rules activate broadly) | reg_easy=0.10–0.15 |
| sfcr_lsat_ar_1000 | reg_easy=0.32–0.69 (only 10 easy examples; gemini delta_shared=0) | gemini reg_private=1.0 on 4 candidates |

---

## Recommendations for Future SFCR Runs

1. **Add minimum region-size guards**: Skip SFCR if V_private < 5 or V_easy < 20. Operating on fractional rates with 1–4 examples is statistically meaningless and produces rejection artifacts.

2. **Temperature sweep for diversity**: Generate candidates at temperature 0.2 (conservative rule), 0.6 (moderate variation), and 1.0 (exploratory) rather than all at temperature ~0.5. This breaks semantic convergence.

3. **Force structural diversity**: Prompt the rule generator to explicitly produce rules in different format categories: (a) a discriminative case-based rule, (b) a step-level procedural rule, (c) a calibration/heuristic rule. This prevents 8 near-identical variations.

4. **Proxy-weighted U_LCB**: When a proxy's failure overlap with V_shared is below Jaccard 0.25, down-weight its reg_easy contribution — it's not a meaningful signal that the rule hurts easy examples for a model whose failures barely overlap the target region.

5. **Refine private_activation_rate definition**: Currently computed as fraction of V_private examples matching USE WHEN. With V_private=1, this is always 0% or 100%. Consider using a soft activation-score threshold or requiring V_private ≥ 3 before applying the guard.

6. **Investigate rule generator memorization**: The pattern where all candidates have priv_act=1.000 (GS llama_ho run) suggests the generator is encoding the private example's features into the rule rather than generating a genuinely exclusive condition. One fix: omit V_private examples from the generation context entirely and instead describe them abstractly ("your rule should not apply to examples involving simple quadrilaterals with SVG right-angle markers").

---
