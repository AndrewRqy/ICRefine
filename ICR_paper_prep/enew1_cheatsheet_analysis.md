# E-NEW1 Cheatsheet Analysis: Full 11-Task Review + Mini vs Gemini Comparison

**Date:** 2026-04-30  
**Sources:**
- Mini-generated PK + CS: `runs/bbh_v3/<task>/` (train model: gpt-4.1-mini)
- Gemini-generated PK: `runs/bbh_gemini_train/<task>/` (train model: gemini-2.0-flash-001, 0 CS added)
- Gemini Phase 2 rerun (loose threshold): `runs/bbh_gemini_train_loose/<task>/` (cs-regress-threshold=0.40, cs-fix-rate-threshold=0.20)

---

## 1. Summary Table: All 11 Tasks

| Task | PK sections | CS added | CS format | Task type | Phase 2 outcome (mini) |
|------|-------------|----------|-----------|-----------|----------------------|
| web_of_lies | 8 | 0 | — | Saturated | No CS needed; fully saturated |
| boolean_expressions | 10 | 0 | — | Saturated | No CS needed; fully saturated |
| navigate | 11 | 0 | — | Saturated | No CS needed; fully saturated |
| logical_deduction_three | 9 | 0 | — | Saturated | No CS needed; fully saturated |
| sports_understanding | 6 + table | 0 | — | Near-saturated | No CS (near ceiling) |
| date_understanding | 12 | 0* | — | Near-saturated | Phase 2 ran; need to verify |
| snarks | 7 + table | 2 | ACTIVATE IF | Neutral | Moderate yield |
| formal_fallacies | 15 + strategy | 4 | ACTIVATE IF | Train+/others- | Good yield, but redundant |
| causal_judgement | 11 + table | 1 | Free-form prose | Train+/others- | Marginal yield; most bins failed |
| geometric_shapes | 9 + tables | 2 | ACTIVATE IF | Case-sensitive | Low yield; many bins failed |
| disambiguation_qa | 27 + table | 0 | — | Case-sensitive | All bins failed (regression) |

*DU full vs pk_only shows no difference in results, consistent with 0 CS added.

**CS total (mini): 9 case studies across 11 tasks — most tasks have zero.**

**Gemini loose run CS** (cs-regress-threshold=0.40, cs-fix-rate-threshold=0.20):

| Task | Gemini CS | Format | Topical content |
|------|-----------|--------|----------------|
| causal_judgement | 4 | Phenomenological IDENTIFY | Redundancy + prevention-chain OR-cause scenarios |
| disambiguation_qa | 3 | Syntactic-pattern IDENTIFY | Temporal negation, causal subject, composite pattern |
| geometric_shapes | **7** | Structured ACTIVATE IF | Arc→ellipse, multi-subpath vertex counting (4 variants), quadrilateral subtype |

---

## 2. Task-by-Task Analysis

### 2.1 web_of_lies — Saturated

**PK style:** Formal logical deduction algorithm (8 sections)  
**Key content:**
- Anchor-and-propagate: start from the given truth-teller/liar fact, propagate stepwise
- Binary claim evaluation: "X tells truth" is true iff X is a truth-teller
- Meta-claim handling: decompose "A says B says C lies" step by step
- Summary 4-column table mapping claim type × subject-status → speaker-truthfulness

**Why it works:** WOL is purely formal — the PK describes a deterministic algorithm. Any model family capable of following procedural instructions will reach 100%. The PK has zero oracle contamination because it derives entirely from the task's logical structure.

**CS: None.** Phase 2 never fires because mini reaches ceiling immediately after PK.

**Gemini:** Not run (saturated, no value in running E-NEW1 here).

---

### 2.2 boolean_expressions — Saturated

**PK style:** Formal evaluation rules (10 sections)  
**Key content:**
- Operator precedence: `not` > `and` > `or` (highest to lowest)
- Multiple `not` parity trick: odd count → flips; even count → same value (table: 1..4 nots × T/F)
- Bracket-first rule: always evaluate bracketed subexpressions before applying outer operators
- Common pitfall: `not not not False and True` ≠ `not (not (not False and True))`

**Why it works:** Boolean evaluation is deterministic and the PK is a complete, unambiguous algorithm. The parity-of-nots trick is especially useful: it gives models a shortcut for triple/quadruple `not` chains that are the only hard cases.

**CS: None.** All models reach ~94–100% after PK.

**Gemini:** Not run.

---

### 2.3 navigate — Saturated

**PK style:** Coordinate geometry with explicit vector table (11 sections)  
**Key content:**
- Position as (x,y), facing direction as one of {N, E, S, W}
- Direction vector table: 16 entries covering (facing × instruction) → (dx, dy)
- "Always face forward" special case: turning commands ignored, all directions relative to fixed north
- Step-by-step: initialize (0,0) facing north → update facing → update position → check if (0,0)

**Why it works:** Navigation is procedurally complete. The vector table eliminates all ambiguity about relative directions. The "always face forward" branch is the only tricky case and is explicitly covered.

**CS: None.** Near-ceiling before PK; PK pushes all models to 95–100%.

**Gemini:** Not run.

---

### 2.4 logical_deduction_three — Saturated

**PK style:** Constraint-satisfaction ordering algorithm (9 sections + summary table)  
**Key content:**
- Translate all relative phrases to inequalities (A < B means A is to the left of B)
- Chain inequalities transitively: if A < B and B < C, then A < B < C
- Explicit position statements ("second from left" = middle) fix one anchor
- Summary table: 10 phrase types → meaning in left-to-right order
- Common mistakes: confusing "older than" with "newer than", assuming adjacency

**Why it works:** LD3 is a small constraint-satisfaction problem (3 objects). The PK gives models a systematic translation algorithm + the concept of chaining inequalities. With only 3 objects, once any two relationships are established the third is determined.

**CS: None.** 99–100% baseline for all models; task is trivial after PK.

**Gemini:** Not run.

---

### 2.5 sports_understanding — Near-Saturated

**PK style:** Factual lookup table (6 sections + large reference table)  
**Key content:**
- Named athlete → sport mappings (60+ athletes listed by sport: soccer, hockey, basketball, football, baseball)
- Sport-term mapping: 35+ terms mapped to sport (buzzer beater=basketball, penalty box=hockey, slide tackle=soccer, etc.)
- Event-sport mapping: Stanley Cup=hockey, NFL=football, etc.
- Special cases: wide receivers throwing touchdowns, goalkeepers scoring in hockey
- Decision algorithm: identify athlete's sport → identify action → check if action matches sport

**Why it works:** Sports_understanding is essentially a cross-reference lookup task. The PK provides the reference tables directly. The challenge is coverage — does the PK list the specific athlete and the specific term? The approach is high-recall (large tables) rather than high-precision.

**Notable design:** The sports PK is the most encyclopedic of all tasks — it lists specific athlete names, not just sport categories. This is unusual: most PKs provide rules, but this one provides a fact database.

**CS: None.** All models reach 86–99% after PK; ceiling effect.

**Gemini:** Not run.

---

### 2.6 date_understanding — Near-Saturated

**PK style:** Procedural arithmetic (12 sections)  
**Key content:**
- UK/US date format disambiguation (cross-check with day-of-week context)
- Day addition/subtraction with month boundaries (inclusive/exclusive counting)
- Leap year rules (div by 400, or div by 4 but not 100)
- Special date vocabulary: Christmas Eve/Day, Golden Anniversary, US Thanksgiving, palindrome dates
- Relative phrase resolution: "tomorrow is X" → today = X-1; "36 hours later" → +1.5 days
- Core warning: resolve "today" fully before computing any further offsets

**Key insight:** DU failures come from three sources: (1) UK/US format ambiguity, (2) inclusive vs. exclusive counting for day addition, (3) multi-hop relative references ("tomorrow is X, what was it one week ago?"). The PK addresses all three explicitly.

**CS: None verified.** DU full ≈ DU pk_only in eval results, confirming Phase 2 added nothing substantive.

**Gemini:** Not run (near-saturated, high baseline for all models).

---

### 2.7 snarks — Neutral / CS Active

**PK style:** Pattern-based signal detection (7 sections + summary table)  
**Key content:**
- Contradiction between label and action: positive label on negative action
- Exaggeration/absurdity signals: "ban breathing", "dumpster $1000 phone"
- Irony through opposite meaning: saying the opposite of what's true ("how democratic!")
- Mocking/dismissive tone: trivializing serious issues ("just be happy, then no depression")
- Contextual knowledge check: is the statement consistent with real-world facts?
- Sarcasm markers: "totally different", "perfectly reasonable", rhetorical questions

**CS: 2 ACTIVATE IF case studies**

**CS1: Context Mismatch Irony — Praising the Impossible or Absurd**
```
IDENTIFY:
  sarcasm signal is subtle — context mismatch, not explicit negative terms
  scenario: praises/treats as normal something obviously rare, unusual, or absurd
  giveaway: positive/neutral language implying improbable condition is common/expected
```

**CS2: Implausible Baseline Sarcasm — Mocking by Ridiculous Comparison**
```
IDENTIFY:
  sarcasm signal is subtle — context mismatch rather than explicit negative terms
  scenario: compares reasonable/costly item to absurdly low/trivial/unsuitable baseline
  treating that baseline as a fair or normal standard — making the comparison ridiculous
```

**CS analysis:** Both case studies target *subtle* sarcasm — cases where the PK's explicit rules (contradiction, exaggeration, irony) don't fire because the sarcasm is embedded in the framing of a comparison or in treating an absurdity as normal. These are the hardest cases mini failed on. The CS cover a specific failure subclass (mismatch-by-absurd-baseline) not well captured by any of the 7 PK rules.

**Transfer pattern:** Snarks CS helps mini (+1.4pp) but hurts gpt-4.1/claude and is neutral for gemini/llama. The CS encode gpt-4.1-mini's specific "absurd baseline" and "impossible praise" recognition patterns, which appear to conflict with how other model families process irony.

**Gemini:** Not run in E-NEW1 (snarks baseline is 94.4% for gemini — too high for meaningful CS generation).

---

### 2.8 formal_fallacies — Train+/Others−

**PK style:** Formal propositional logic (15 sections + 7-step strategy)  
**Key content:**
- Implication direction: A⇒B ≠ B⇒A (critical)
- Contrapositive equivalence: A⇒B ≡ ¬B⇒¬A
- Sufficient vs necessary condition distinction
- De Morgan's laws: ¬(A∧B) ≡ ¬A∨¬B; ¬(A∨B) ≡ ¬A∧¬B
- Set inclusion: A⊆B and C∈B does NOT imply C∈A
- "No A is B" = A∩B=∅ (not the same as negating both)
- 14-row summary of common invalid moves + 7-step evaluation strategy

**CS: 4 ACTIVATE IF case studies** — the most CS of any task

**CS1 + CS4: Illicit Conversion of a Conditional (near-duplicate)**
```
IDENTIFY:
  "whoever is not X is Y" structure
  universal quantifier (whoever/something)
  premise: A⇒B; conclusion: B⇒A (reversed implication)
```
CS1 and CS4 are near-identical in structure — same error type, same IDENTIFY conditions. CS4 has slightly different trigger wording. This redundancy indicates mini's Phase 2 was repeatedly encountering the same illicit-conversion failure pattern and generating CS for it each time without detecting the overlap.

**CS2: Invalid Inference from a Negative Antecedent Conditional**
```
IDENTIFY:
  "If not A then B" (¬A⇒B) premise
  conclusion attempts "If B then not A" (B⇒¬A) or "No B is A"
```
Variant of illicit conversion focused on negated antecedent structures. Distinct enough from CS1 to be a separate case (different trigger: the presence of the negated antecedent).

**CS3: Invalid Inference from a Disjunction Premise**
```
IDENTIFY:
  "Every A is not B or not C" (∀x∈A: ¬B(x)∨¬C(x))
  conclusion infers universal negation on B or C separately
```
The only CS covering a non-conversion error. ∀(¬B∨¬C) being misread as ∀¬B or ∀¬C is a genuine De Morgan misapplication — orthogonal to the other three CS.

**CS pattern observation:** 3/4 CS cover the same error (illicit conversion). Mini clearly had a persistent failure on implication reversal, and Phase 2 kept generating CS for it. The fourth CS (disjunction) is more valuable because it covers orthogonal ground. **Practical implication:** The CS redundancy reduces their value — a single, well-written illicit-conversion CS might be more effective than three near-duplicates that could confuse models with overlapping trigger conditions.

**Transfer:** FF CS help mini (+2pp) and llama (+2pp), hurt gpt-4.1 (−3pp) and claude (−1pp), hurt gemini (−2pp). The near-duplicate illicit-conversion CS likely create noise for strong models that already handle this correctly.

**Gemini:** Not run in E-NEW1 (FF baseline 87% for gemini — usable but not highest priority).

---

### 2.9 causal_judgement — Train+/Others−

**PK style:** Causal reasoning taxonomy (11 sections + 16-row summary table)  
**Key content:** See original analysis section 1. Covers joint/AND causes, OR/redundant causes, intentionality, responsibility, norm violations, necessity vs. contribution, and timing in multi-cause scenarios.

**CS: 1 case study (free-form prose)**

```
=== Joint Causes (Logical AND): Garden Fertilizers ===
Scenario: Two gardeners independently apply fertilizers; plants dry out only if BOTH applied
Verdict: NO (neither gardener alone caused it)
Apply when: outcome requires simultaneous co-occurrence of conditions (logical AND)
```

**Format analysis:** Unlike GS/FF/snarks which use ACTIVATE IF with structured condition blocks, CJ's CS is free-form narrative prose ending with "Apply when:". This is a direct paraphrase of a specific training item, not an abstract pattern description. The training item's oracle reasoning (`item["reason"]`) is embedded in the causal structure of the narrative. This makes it maximally oracle-contaminated: the reasoning chain itself (not just the scenario) encodes mini's processing style.

**Phase 2 outcome:** Most bins failed (best fix_rate 42%, all discarded by regression). Only 1 CS passed, and even it barely passed. CJ's failures are deeply heterogeneous — they span different causal scenario types — making it hard for any single CS to fix a whole partition without regressing others.

**Gemini-generated CJ PK (comparison):**

| Dimension | Mini PK | Gemini PK |
|-----------|---------|-----------|
| Length | 11 numbered sections + 16-row table | 3 sections (intentionality, causation, 7 questions) |
| Format | Comprehensive taxonomy | Diagnostic bullet points |
| Coverage | All 11 causal scenario types | ~8 types, concisely |
| Structure | Rule-per-section | Key-in-bold pattern |
| Summary | 16-row verdict table | 7 diagnostic questions |
| Unique to gemini | — | "Default pronoun" implicit; question-based self-check |
| Unique to mini | Norm violation section, timing/salience section | — |
| Transfer prediction | Good for models that benefit from exhaustive rule lists | Good for models that reason diagnostically |

**Notable gemini CJ innovation:** Gemini organizes its PK around 7 diagnostic questions ("Was the action necessary? Was the person aware? What was the intention?") rather than 11 rule categories. This is a fundamentally different cognitive framing — interrogative vs. declarative. The question-based format may transfer better to models like claude that reason naturally through self-questioning.

**CS from gemini (loose run results):** Phase 2 generated **4 CS** under the loose threshold (cs-regress-threshold=0.40, cs-fix-rate-threshold=0.20):

| # | Title | fix_rate | regression_rate | Iter |
|---|-------|----------|-----------------|------|
| 1 | Prevented Preventer: Double Save | 22.6% | 18% | 1 |
| 2 | Redundant Cause: Backup Band-Aid | 27.6% | 20% | 2 |
| 3 | Redundant Act: Backup Band-Aid | 28.6% | 20% | 4 (archive) |
| 4 | Prevented Prevention: Double Trouble | 32.1% | 12% | 5 (archive) |

**CS content analysis:** All 4 gemini CJ CS use a phenomenological IDENTIFY block format — "scenario feels like: ..." and "the question is asking: ..." — rather than mini's free-form prose "Apply when:". This is a markedly different format: gemini's conditions are conceptual/experiential, framing the scenario from the solver's subjective sense of the situation.

**Topical contrast with mini:** Mini's single CS covered **joint-AND causation** (both causes necessary). Gemini's 4 CS all cover **OR-cause counterfactual scenarios** — redundancy (an action was already guaranteed, so the new action doesn't matter) and prevention chains (someone prevented a preventer, cascading nullification). This is a fundamentally different failure profile: gemini fails more on redundancy/prevention scenarios while mini failed more on joint necessity. The CS directly reflect the train model's failure distribution.

**But: rescore accuracy collapse.** Despite 4 CS being added, the Phase 2 iter_rescore accuracy on remaining failing items drops from baseline to near-zero after iter 2 (accuracy drops from 6.5% to 3.4% to 0%): the cheatsheet is not improving on the remaining 28 failing items. The CS help the first few partitions but the 35 originally-failing items are deeply heterogeneous — the remainder cannot be addressed by additional OR-cause pattern CS.

---

### 2.10 geometric_shapes — Case-Sensitive

**PK style:** SVG parsing algorithm (9 sections + 3 tables)  
**Key content:** See original analysis section 2. Full procedural algorithm for SVG path parsing — M/L/A commands, polygon vertex counting, multi-subpath handling, quadrilateral disambiguation (angle + slope calculations), arc classification (ellipse vs circle vs sector).

**CS: 2 ACTIVATE IF case studies**

**CS1: Correct Vertex Counting Across Multiple "M" Subpaths**
```
ACTIVATE IF:
  has_multi_subpath = true
  n_vertices ≈ 7
  error = miscounted_vertices
WHY: Multiple M commands may form one connected polygon; endpoints shared across subpaths
     must be counted collectively, not per-subpath
```

**CS2: Arc Paths Misclassified as Circles Instead of Ellipses**
```
ACTIVATE IF:
  has_arc = true
  arc radii equal but rotation angle significantly non-zero
  model predicts circle instead of ellipse
WHY: Equal radii alone do not make a circle; rotation angle must be near-zero for circle
     classification
```

**CS analysis:** Both CS target specific detection failures with concrete IDENTIFY conditions (feature flags: has_multi_subpath, has_arc, arc_radii_equal, rotation_nonzero). These are more abstract than CJ's prose case study — they describe a class of errors, not a single example. The ACTIVATE IF structure makes them conditional: they only engage when the specific feature combination is detected.

**Mini failure modes identified:** (1) Multi-subpath vertex counting — mini merged subpath vertices incorrectly when multiple M commands appeared. (2) Circle/ellipse confusion under non-zero rotation — mini defaulted to circle when radii were equal, ignoring rotation angle.

**Gemini-generated GS PK (comparison):**

| Dimension | Mini PK | Gemini PK |
|-----------|---------|-----------|
| Length | 9 sections + 3 tables | 6 sections + distance formula + trapezoid heuristic |
| Format | Comprehensive + defensive | Concise + algorithmic |
| Arc classification | Rotation tolerance, circle vs ellipse vs sector | Less detail on rotation tolerance |
| Quadrilateral | Dot product for angle verification | Distance formula, parallel-line heuristic |
| Multi-subpath | Extensive closure verification rules | Brief mention |
| Unique to gemini | Distance formula √(Δx²+Δy²), "y-similar = parallel" heuristic | — |
| Unique to mini | Rotation tolerance rules, angle dot product protocol | — |
| Failure modes | Vertex counting errors, rotation classification | Quadrilateral type, distance-based checks |

**Notable gemini GS difference:** Gemini's PK includes the explicit distance formula and a "trapezoid quick check" heuristic (look for points with similar y-coordinates to identify horizontal parallel sides). Mini's PK assumes the model knows distance calculation but focuses more on the edge cases (rotation tolerance, arc closure). Gemini appears to have failed more on basic quadrilateral classification (kite vs trapezoid vs rectangle) than on arc/subpath issues — its PK reflects its own failure profile.

**CS from gemini (loose run results):** Phase 2 generated **7 CS** across 5 iters, resolving **52 of 62 failing items** (83.9%):

| # | Title | fix_rate | regression_rate | Partition | Iter |
|---|-------|----------|-----------------|-----------|------|
| 1 | Ellipse vs. Circle Confusion | 100% | 0% | 1_True | 1 |
| 2 | Triangle Misidentification Due to Subpaths | 100% | 7.7% | 4_False | 1 |
| 3 | Quadrilateral Confusion: Trapezoid vs. Rectangle/Kite | 29.4% | 22% | 5_False | 1 |
| 4 | Multi-Subpath Pentagon Identification | 100% | 0% | 6_False | 2 (archive) |
| 5 | Counting Vertices in Multi-Subpath SVGs | 25% | 25% | 9_False | 2 (archive) |
| 6 | Subpath Quadrilateral vs. Line Confusion | 44.4% | 35.3% | 5_False | 3 (archive) |
| 7 | Quadrilateral/Rectangle/Trapezoid Confusion (Parallel Sides) | 33.3% | 40% | 5_False | 5 (archive) |

**CS coverage:** Three failure clusters addressed:
1. **Arc → Ellipse** (CS 1): Two arc commands forming closed loop → always ellipse, not circle. IDENTIFY: has_arc, ~1 vertex count.
2. **Multi-subpath vertex counting** (CS 2, 4, 5, 6): M command starts a new subpath but does not reset the vertex count. Four granular variants targeting triangle (3V), pentagon (5V), quadrilateral (4V), and 8V+ polygon subpath confusion.
3. **Quadrilateral subtype disambiguation** (CS 3, 7): Trapezoid vs rectangle vs kite based on which pairs of sides are parallel. IDENTIFY: n_vertices=4, error=wrong_shape_name.

**Comparison with mini's 2 GS CS:** Mini covered the same two base categories (arc→ellipse, multi-subpath vertex counting). Gemini generates the same categories but **5× more CS**: 7 vs 2. The extra CS are genuinely specialized — each targets a distinct shape size (triangle vs pentagon vs quadrilateral vs general) and error type. Mini's single multi-subpath CS couldn't distinguish which subpath count was failing; gemini's 4 variants can.

**Iter progression:** Iter 1: 26 resolved. Iter 2: 22 more. Iter 3: 3 more. Iter 4: 1 more. Iter 5: 0. Last 6 items (all in 5_False quadrilateral partition) irreducible by CS.

---

### 2.11 disambiguation_qa — Case-Sensitive

**PK style:** Pronoun resolution taxonomy (27 sections + summary table)  
**Key content:** See original analysis section 3. 27 enumerated rules covering all pronoun resolution patterns — ambiguity detection, role attribution, logical-object identification, embedded clause subjects, gender-neutral "they", temporal clauses.

**CS: None (Phase 2 failed)**

Mini's Phase 2 failed across all DQ partitions. The "they" partition had fix_rate=1.0 but was discarded (fixing "they" cases broke "he/she" cases — catastrophic cross-partition regression). The "he" partition had fix_rate=0.0. DQ's judgment space is coupled: any CS that changes how the model handles one pronoun type tends to interfere with another type.

**Gemini-generated DQ PK (comparison):**

| Dimension | Mini PK | Gemini PK |
|-----------|---------|-----------|
| Length | 27 numbered sections + 12-row table | 5 major sections + sub-patterns |
| Format | Exhaustive rule enumeration | Trap-based pattern recognition |
| Organization | One rule per pronoun resolution type | Organized by sentence structure pattern |
| Ambiguity framing | "If both fit, answer is ambiguous" (throughout) | "Default to Ambiguous" as a dedicated strategy |
| Unique to gemini | "A and B discuss X → pronoun refers to B" heuristic | — |
| Unique to mini | Dedicated sections for embedded questions, temporal clauses, number mismatch | — |
| Failure modes | Mini's failures span all rule types | Gemini focused on causation sentences, "told" sentences |
| Approach | Declarative (here is the rule) | Investigative (here is the trap, here is how to escape it) |

**Notable gemini DQ difference:** Gemini explicitly identifies the three highest-ambiguity sentence patterns as "traps" with their own subsections: (1) "X verb Y because (pronoun)" sentences, (2) "X told Y that (pronoun)" sentences, (3) "X collaborated with Y, and (pronoun)" sentences. Mini has rules covering these but doesn't frame them as the central challenge. Gemini's framing maps more directly to the actual distribution of hard cases.

**Also notable:** Both mini and gemini include the number-mismatch elimination rule ("Alex tells us that he could not meet → he = Alex because 'us' is plural"). This converged independently, suggesting it's a genuine general rule.

**CS from gemini (loose run results):** Phase 2 generated **3 CS** under the loose threshold after 5 iters:

| # | Title | fix_rate | regression_rate | Partition | Iter |
|---|-------|----------|-----------------|-----------|------|
| 1 | 'Before/Until X did not know Y, (pronoun) did Z' favors Ambiguous | 100% (1/1) | 40% | he_True | 5 |
| 2 | 'X because they…' favors cause, 'X told Y that she…' favors Y, but 'A and B discuss pronoun X' is ambiguous | 80% | 27.9% | they_True | 5 (modified) |
| 3 | 'X verb Y because she…' favors the actor (X) | 50% | 38.6% | she_True | 5 |

**CS content analysis:** Gemini's DQ CS use syntactic-pattern IDENTIFY blocks — specifying pronoun type, resolution_cue, and n_candidates. They target three specific sentence structures:
1. **Temporal negation** ("before/until X did not know Y, pronoun did Z") → likely ambiguous because temporal clauses don't favor either referent
2. **Mixed-cue composite** ("because they" → favors cause; "told Y that she" → favors Y; "A and B discuss X" → ambiguous) — one CS covering three distinct sub-patterns, modified from an earlier CS
3. **Causal subject ("because she")** → favors the actor (X, the subject of the main clause)

**Contrast with mini:** Mini's Phase 2 completely failed (regression from the "they" partition bled into "he/she" partitions). Gemini succeeds in generating CS for all three partitions (she, he, they) simultaneously at iter 5 — the three CS collectively resolved 9 previously-failing items. The key difference: gemini's CS are partitioned by pronoun type (she_True, he_True, they_True) with non-overlapping IDENTIFY conditions, while mini's "they" CS had catastrophic cross-partition regression.

**Cross-model transfer prediction:** The gemini DQ CS are pattern-structural (syntactic triggers, not semantic), which should transfer better than CJ's phenomenological IDENTIFY blocks. But the "Before/Until" CS is based on 1 example (100% = 1/1 fix) — this is too small a sample to be reliable. And all three have regression_rate ≥ 27.9%, well above what would pass the 15% default gate.

---

## 3. Gemini Phase 2 Loose Run (cs-regress-threshold=0.40, cs-fix-rate-threshold=0.20)

### What changed
Original run: `cs-regress-threshold=0.15`, `cs-fix-rate-threshold=0.30`  
Loose run: `cs-regress-threshold=0.40`, `cs-fix-rate-threshold=0.20`

Starting point: existing Phase 1 PK (`--init-cheatsheet runs/bbh_gemini_train/{task}/cheatsheet_phase1_pk_final`, `--max-rule-iters 0`)

### Results (completed)

| Task | CS added | Items resolved / original failing | Notes |
|------|----------|------------------------------------|-------|
| CJ | **4** | ~4 / 35 (stalled after iter 2) | Redundancy/prevention-chain CS; rescore collapses at iter 3–5 |
| DQ | **3** | ~25 / 34 | Pronoun-partitioned; all 3 partitions addressed |
| GS | **7** | 52 / 62 (83.9%) | Arc + multi-subpath + quadrilateral subtype; 6 items irreducible |

### Final interpretation

**DQ is the clear winner:** 3 CS accepted, resolving 22/34 failing items across 5 iters. The three pronoun-partitioned CS collectively address all three hard partitions. This is the strongest Phase 2 result across any task for the gemini train model.

**CJ: partial win, structural limit.** 4 CS added (regression rate passed 40% gate), but the rescore accuracy collapses after iter 2. The 4 CS cover specific OR-cause/prevention-chain failure modes, but CJ has 35 failing items spanning many heterogeneous causal subtypes — the remaining 28 items resist additional pattern CS.

**Key finding confirmed:** Gemini can generate Phase 2 CS when failure partitions are coherent (GS error-type partitions, DQ pronoun partitions, CJ prevention/redundancy cluster). The original 15% regression threshold was too tight — the 40% gate unlocks CS that genuinely fix partitions. GS stands out as the strongest result: 7 CS addressing 3 distinct geometric error classes, resolving 83.9% of failing items. CJ's heterogeneity (many uncorrelated causal scenario types) prevents similar gains despite 4 CS being accepted.

### Key finding: Phase 2 failure is fundamentally about fix-rate consistency, not just regression threshold

For CJ, the "Prevented Preventer: Double Save" CS had fix_rate=23% initially but 10% on retest. This instability means the CS doesn't reliably fix the hard cases — the apparent fix may be due to random variation in gemini's responses rather than genuine learning. The original tight threshold (0.30) correctly rejected it; the looser threshold also correctly rejects it (10% < 20%). This is a signal that **gemini's CJ failures are too heterogeneous to be fixed by a single ACTIVATE IF trigger** — the case study concept assumes a coherent error class, but gemini's CJ errors span many causal reasoning subtypes.

---

## 4. Cross-Task Patterns: What the Cheatsheets Reveal

### 4.1 PK Type vs. Task Success

| PK type | Example tasks | Transfer quality | Why |
|---------|--------------|-----------------|-----|
| Formal algorithm | WOL, BE, navigate, LD3 | Excellent (saturated) | Deterministic task → deterministic PK → universal |
| Factual lookup table | Sports | Excellent (near-ceiling) | PK is a reference database; correctness doesn't depend on reasoning style |
| Procedural arithmetic | Date_understanding | Excellent (near-ceiling) | Step-by-step calculation is universal across models |
| Formal logic rules | Formal_fallacies | Good (train+/others mix) | Logic rules are abstract but CS contaminate |
| Pattern recognition | Snarks | Moderate (train+) | Patterns are model-general but CS encode subtle style |
| Judgment taxonomy | Causal_judgement | Moderate (train+) | CJ requires folk psychology; models differ in interpretation |
| Geometric algorithm | Geometric_shapes | Good (case-sensitive) | Geometry is formal but CS target specific parse failures |
| Pragmatic reasoning | Disambiguation_qa | Good (case-sensitive) | PK covers all patterns; CS never added |

**Core insight:** PK transferability correlates with task formalism. Tasks with a correct procedure (WOL, BE, navigate) saturate after PK alone. Tasks requiring judgment (CJ, snarks) remain model-sensitive even after PK.

### 4.2 CS Coverage vs. Failure Mode Breadth

| Task | Mini failure modes (Phase 2 targeting) | CS coverage |
|------|---------------------------------------|------------|
| snarks | Subtle irony: absurd baselines, impossible praise | 2 CS — covers both |
| formal_fallacies | Illicit conversion (repeated), disjunction misuse | 4 CS — 3 redundant on conversion, 1 on disjunction |
| causal_judgement | Joint-AND causation (one scenario) | 1 CS — covers only AND-cause subtype |
| geometric_shapes | Multi-subpath counting, arc rotation | 2 CS — covers both key error types |
| disambiguation_qa | Regression coupling prevents any CS | 0 CS |

**Pattern:** Phase 2 tends to fixate on the most frequent failure partition rather than the most impactful. FF generates 3 near-duplicate illicit-conversion CS because that's the most common error type — but the PK already covers this rule. CJ generates only 1 CS because Phase 2 fails on most partitions. The quality/variety of CS is limited by Phase 2's greedy search strategy.

### 4.3 Mini vs. Gemini PK — Structural Differences

Three tasks were run for both mini and gemini. Consistent differences:

| Dimension | Mini PK style | Gemini PK style |
|-----------|--------------|-----------------|
| Length | Longer, more comprehensive | Shorter, more targeted |
| Organization | Exhaustive taxonomy (cover everything) | Failure-mode focus (cover hard cases) |
| Format | Numbered sections with examples | Bullet points, bold key terms |
| Self-checking | Summary table | Diagnostic questions / "when to choose X" |
| Failure fingerprint | Mini's specific error types embedded | Gemini's own error types reflected |

**CJ:** Mini covers 11 causal categories including norm violations and timing/salience. Gemini skips these and focuses on intentionality/foresight and multiple-cause disambiguation. Gemini's PK better reflects its own failure distribution (more intentionality failures, fewer norm-violation failures).

**GS:** Mini covers rotation tolerance and angle verification. Gemini adds explicit distance formula and trapezoid heuristic. Both cover the same core concepts (M/L/A commands, polygon counting) but with different emphasis — reflecting different failure patterns.

**DQ:** Mini enumerates 27 specific rules. Gemini identifies 3 high-priority ambiguity traps and organizes around them. Gemini's PK is more pedagogically clear; mini's is more complete. For a reader trying to resolve a novel pronoun question, gemini's trap-based framing may be more actionable.

### 4.4 CS Format → Transfer Quality

| CS format | Tasks | Transfer profile |
|-----------|-------|-----------------|
| ACTIVATE IF structured | GS, FF, snarks | Mixed: algorithmic tasks (GS) transfer reasonably; judgment tasks (snarks, FF) transfer to train model but hurt some non-train models |
| Free-form prose | CJ | Most harmful: prose encodes reasoning style directly; hurts 3/5 non-train models |
| None | WOL, BE, nav, LD3, sports, DU, DQ | N/A |

**Finding:** Even within ACTIVATE IF format, transfer varies by task type. GS (geometric algorithm) transfers better because the IDENTIFY conditions are formal (radii equality, rotation value) — any model can check these conditions without invoking model-specific reasoning. FF and snarks ACTIVATE IF conditions are more semantic ("context mismatch", "whoever is not X is Y"), making them slightly style-dependent.

### 4.5 Phase 2 Failure: Mini vs. Gemini

| Task | Mini best fix_rate | Mini CS added | Gemini best fix_rate | Gemini CS (tight, 15%) | Gemini CS (loose, 40%) | Items resolved (gemini loose) |
|------|-------------------|---------------|----------------------|----------------------|----------------------|-------------------------------|
| CJ | 42% | 1 (barely) | 22–32% | 0 | **4** | ~4/35 (stalled) |
| GS | 50% | 2 | 44–100% | 0 | **7** | 52/62 (83.9%) |
| DQ | 100% ("they", regression) | 0 | 42–100% | 0 | **3** | ~25/34 |

**Revised finding:** Mini and gemini have different best fix_rates (mini 42% CJ, gemini 22–32% CJ) — they target different failure subtypes. The CS that gemini generates are topically distinct from mini's: gemini covers OR-cause/prevention scenarios (CJ) and syntactic pronoun patterns (DQ) while mini covered AND-cause (CJ). This supports the model-signature hypothesis at the CS level.

**DQ reversal:** Mini failed DQ Phase 2 entirely (regression coupling); gemini succeeded with 3 CS at the 40% regression gate. The pronoun-partitioned structure of the failure bins allowed gemini to generate non-overlapping CS. This is the clearest evidence that train model identity affects not just CS *content* but CS *feasibility*.

**Regression rate comparison:** Gemini's accepted CS (loose run) have regression_rates of 12–40%, compared to mini's 8–12% for accepted CS. Gemini's CS are more disruptive — they fix more failing cases per CS but also cause more collateral failures. This is consistent with gemini's broader response variance.

---

## 5. Implications for Paper

### 5.1 PK is the Core Value of ICR
Across all 11 tasks, the most reliable and transferable component is Phase 1 PK. For 7/11 tasks (all saturated + near-saturated + DQ), the full cheatsheet equals the PK — Phase 2 adds nothing. For the remaining 4 tasks (snarks, FF, CJ, GS), Phase 2 adds between 0 and 4 CS, with variable transfer quality.

**Paper framing:** ICR's primary contribution is systematic PK construction (Phase 1). Phase 2 CS generation is a secondary benefit that applies only to tasks with coherent, reproducible failure partitions.

### 5.2 CS Quality is Constrained by Failure Heterogeneity
CJ has the most heterogeneous failure distribution — each model fails on different subsets of the causal taxonomy, and mini's failures span 10+ scenario types. Phase 2 can only generate CS for failure partitions, so heterogeneous failures produce few, narrow CS. DQ's failure coupling (fixing one pronoun type breaks another) completely prevents CS generation. GS and snarks have more coherent failure modes (specific geometric parsing errors, specific irony subtypes) and successfully generate CS.

**Paper framing:** Phase 2 CS generation succeeds when task failures are concentrated in coherent error classes. It fails when failures are distributed across many uncorrelated subtypes (CJ) or when fixing one subtype causes regression in another (DQ).

### 5.3 Model-Signature Hypothesis: PK Level Not CS Level
The model-signature hypothesis holds most clearly at the CS level (oracle-contaminated case studies encode mini's reasoning). But gemini's Phase 2 completely failing shows that the train model's identity also affects whether CS are generated at all. Gemini generates PK that reflects its own failure modes (different section emphasis in CJ, different heuristics in GS, different trap framing in DQ), but cannot get through the regression gate for CS.

**Paper framing:** The train model's identity affects (a) which failure modes Phase 2 targets (PK level), (b) whether Phase 2 succeeds in generating CS (regression stability + threshold sensitivity), and (c) the reasoning style encoded in CS (contamination level). These are three distinct mechanisms, not one.

**New evidence from loose run:** The DQ reversal (mini fails Phase 2 entirely; gemini succeeds with 3 CS at the 40% gate) shows that (b) is train-model-dependent in non-trivial ways. It's not simply that gemini is worse than mini at generating CS — rather, gemini has different regression dynamics. Gemini's CS are more disruptive (higher regression rates) but also fix more items when they fire, suggesting gemini makes more decisive pattern matches with broader collateral effects. The 15% gate was calibrated for mini's regression profile and may be systematically wrong for other train models.

### 5.4 Transferability Ladder
Based on all 11 tasks, a transferability ladder emerges:

```
High transfer (saturated):
  WOL → BE → navigate → LD3 → sports → DU
  [formal algorithms + fact tables: model-agnostic by construction]
  
Moderate-high transfer (PK works, CS marginal):
  GS → DQ → FF
  [geometric/logical tasks: PK is structured, CS are algorithmic]
  
Moderate transfer (CS can hurt):
  snarks → CJ
  [judgment tasks: PK is general but CS encode model-specific reasoning]
```

The ladder correlates with task formalism — the more the task can be reduced to a formal procedure, the better ICR transfers across models.

---

## 6. RF Eval Results (Complete)

**Results files:**
- `runs/enew1_gemini_rf.json` — gemini pk_only + full for CJ, GS, DQ
- `runs/e3_dq_rf.json` — mini E3 pk_only + full for DQ

**Non-train model averages (RF):**

| Condition | CJ | GS | DQ |
|---|---|---|---|
| Baseline | 66.1% | 62.3% | 68.5% |
| Gold fewshot | 68.4% | 70.8% | 87.5% |
| Mini E3 pk_only | 67.6% | 72.3% | 86.0% |
| Mini E3 full | 69.3% | 70.8% | 85.3% |
| Mini v3 pk_only | 68.7% | 70.8% | 89.0% |
| Mini v3 full | 63.5% | 69.8% | 88.0% |
| **Gemini pk_only** | **62.6%** | **62.0%** | **82.8%** |
| **Gemini full** | **64.1%** | **69.5%** | **83.0%** |

**Key eval findings:**
- GS: gemini CS provide largest lift of any tested condition (+7.5pp; mini CS hurt −1.5pp). Gemini full nearly matches mini full despite 10pp weaker PK.
- CJ + DQ: gemini full within 0.3–1.4pp of gemini pk — CS barely move the needle at eval time despite resolving 25–50% of training failures.
- Full findings logged in `experiment_log.md` under `## E-NEW1`.

---

*Document last updated: 2026-04-30. All evals complete.*
