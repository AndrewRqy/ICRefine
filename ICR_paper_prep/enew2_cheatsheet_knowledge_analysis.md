# E-NEW2: What Do ICR Cheatsheets Actually Contain?
## A Knowledge-Type Analysis of v3 Cheatsheets Across All 11 Tasks

**Date:** 2026-04-30  
**Source:** `runs/bbh_v3/<task>/cheatsheet_final.json` and `cheatsheet_phase1_pk_final.txt`  
**Eval data:** `runs/rf_transfer_5tasks_v3.json`, `runs/rf_transfer_6tasks_e9.json`

---

## 1. The Central Question

ICR produces two artifacts per task: a **Prior Knowledge section** (Phase 1) and **Case Studies** (Phase 2). The standard description calls both "knowledge" — but are they the same kind of knowledge? Does it matter?

This analysis argues they are fundamentally different in nature, origin, and transferability:

- **PK** encodes *task knowledge* — principles about what the task requires to answer correctly.
- **CS** encode *failure signatures* — records of where a specific model fails, and a conditional patch.

The distinction has direct empirical consequences: PK transfers across model families, CS mostly don't. But the story is more nuanced than "task knowledge = good, model knowledge = bad." Some PK is model-specific; some CS are model-agnostic. The determining factor is not *which phase generated it* but *what epistemological type of knowledge it is*.

---

## 2. A Taxonomy of Knowledge Types in v3 Cheatsheets

Reading across all 11 tasks, four distinct knowledge types appear — spanning both PK and CS sections.

### Type I: Complete Algorithms

**Tasks:** WOL, BE, navigate, LD3  
**Characteristic:** Provides a full decision procedure that, if followed correctly, guarantees the right answer.

| Task | Core algorithm |
|------|---------------|
| WOL | Anchor from known truth-teller/liar → propagate claim-by-claim; each step binary |
| BE | Innermost brackets → `not` → `and` → `or`; odd `not` count flips, even preserves |
| Navigate | Track (x,y) position + facing direction; apply 16-entry direction-change table |
| LD3 | Translate relative phrases to inequalities; chain transitively to derive ordering |

These PKs are **closed systems** — they require no external knowledge and leave no judgment calls. A model following the algorithm is executing deterministic computation, not making inferences.

**Transfer property: universal.** Every model family that can follow procedural instructions achieves ceiling. Under RF, all four tasks hit 100% for all models with any cheatsheet. The PK contribution is real but becomes invisible once models are already near-ceiling without it.

**Why ICR produces this structure:** For rule-sufficient tasks, mini's training failures arise from execution errors (applying operator precedence wrong, losing track of position). The PK iteratively identifies these errors and adds clarifying rules until mini executes correctly. The result is a complete algorithm because the task *has* a complete algorithm — ICR converges to it.

### Type II: Reference Databases

**Tasks:** sports_understanding, date_understanding (partially)  
**Characteristic:** Provides domain facts that the task requires and that may not be reliably encoded in model weights.

Sports PK example structure:
```
Soccer: Marcelo, Gerard Pique, Neymar, Santi Cazorla, Giorgio Chiellini...
Ice Hockey: Elias Lindholm, John Carlson, John Tavares, Frederik Andersen...
Sport terms: buzzer beater=basketball, penalty box=hockey, slide tackle=soccer...
```

Date PK provides: days per month, leap year rules, UK/US format disambiguation logic, special-date vocabulary (Golden Anniversary = 50 years).

**Transfer property: high, bounded by coverage.** Any model that can perform lookup benefits from having the lookup table. The limiting factor is whether the specific athlete/term in the test item appears in the PK. Models with stronger memorized sports knowledge (likely larger models) benefit less — their internal database already covers the test items.

**Key structural choice:** Sports PK is unusually encyclopedic — it lists specific athlete names, not just sport categories. This is necessary because the task turns on exact athlete-sport mappings that models may have memorized inconsistently. The PK compensates for coverage gaps in model pretraining, not reasoning gaps.

### Type III: Disambiguation Taxonomies

**Tasks:** DQ, snarks, CJ  
**Characteristic:** Provides rules for resolving cases where the correct answer depends on subtle cues that models systematically misread.

These tasks share a structure: a set of surface patterns, each associated with a resolution principle, with explicit handling for edge cases. But the resolution principles differ in nature:

**DQ (27 rules):** Mostly formal-ish — each rule describes a specific syntactic configuration and states which pronoun referent it implies. Example: "If X told Y that (pronoun) did something → speaker X more likely referent, because X is informing Y of something Y doesn't already know." This is pragmatics, but stated as a rule.

**Snarks (7 signals):** Phenomenological — describes the *feel* of sarcasm rather than its formal structure. Example: "Positive label applied to negative action" is stated as a rule, but identifying what counts as "negative action" requires semantic judgment. The signals cascade in confidence: explicit contradiction (high certainty) → exaggeration/absurdity → irony through opposite meaning → mocking tone → contextual knowledge.

**CJ (11 types):** A full taxonomy of causal reasoning scenarios — joint-AND causes, multiple-sufficient causes, omission, prevention, intentionality, norm violation, etc. Each type has a verdict rule (e.g., "if both causes are individually insufficient, neither alone caused the outcome").

**Transfer property: moderate, model-dependent.** DQ transfers reasonably because pronoun resolution rules are syntactically grounded and mostly model-agnostic. Snarks transfer moderately — the explicit signals (contradiction, absurdity) work for all models, but models diverge on the subtler signals (contextual knowledge, mocking tone). CJ transfers the least reliably: the causal taxonomy describes folk-psychological principles that models weigh differently. For example, the "norm violation" principle (the norm-violating actor bears more causal responsibility) is weighted differently by claude vs llama vs mini.

**The deeper issue with CJ:** The causal taxonomy is not purely task knowledge — it's the *folk psychology of causation*, which is itself contested and model-family-dependent. Mini's PK faithfully describes human folk-causal intuitions. But different models have absorbed different distributions of text about causation, and their "intuitions" diverge. This is a case where the task itself doesn't have a clean ground truth — different annotators (human or model) genuinely disagree.

### Type IV: Error-Class Patches

**Tasks:** GS (CS), FF (CS), BE (CS), DU (CS), LD3 (CS), sports (CS), snarks (CS)  
**Characteristic:** CS that describe a specific class of mistake and provide a corrective case.

This is the CS section's primary content. But a critical distinction emerges when examining the ACTIVATE IF conditions across tasks:

---

## 3. The ACTIVATE IF Conditions Reveal Three Activation Modes

The ACTIVATE IF mechanism is designed to be conditional — a model applies the CS only when the described conditions are met. But the conditions themselves span a spectrum that determines transferability:

### Mode A: Formal Feature Flags (structural, model-agnostic)

```
GS CS1:  "path has approximately 7 vertices"
         "has_multi_subpath = true"
         "n_vertices = total unique connected points"

GS CS2:  "path contains arc (A) commands"
         "has_arc = true"
         "arc radii equal but rotation angle significantly non-zero"

BE CS:   "expression contains NOT operator"
         "has_not"
         "nested_not"

Sports:  "sport: hockey"
         "action_type: scoring or timing"
```

These conditions are computable from the input. Determining whether `has_arc = true` requires no judgment — any model scanning the SVG path string will agree. The ACTIVATE IF acts as a guard gate with zero ambiguity. If the condition is satisfied, the patch is applied; if not, it's ignored.

**Transfer consequence:** Mode A CS transfer well because the *condition itself* is model-agnostic. Whether to apply the GS arc-rotation CS doesn't depend on the model's internal reasoning — it depends on whether the input contains A commands and non-zero rotation. Different models can apply the same CS with the same selectivity.

### Mode B: Syntactic Pattern Matching (structural template, mostly model-agnostic)

```
FF CS1:  "argument uses 'whoever is not X is Y' structure"
         "argument structure looks like: 'If A then B'; therefore 'If B then A'"

LD3 CS:  "model reverses direction of a comparative phrase"
         "'A is to the right of B' interpreted as A < B instead of B < A"

DU CS1:  "question involves relative date phrases like 'tomorrow is X date'"
         "error: model uses given relative date as 'today'"
```

These conditions are syntactic — the model checks for a specific surface form or argument pattern. Slightly more abstract than feature flags, but still largely model-agnostic: "does this argument have the structure A⇒B therefore B⇒A?" is close to a formal verification task.

**Transfer consequence:** Mode B CS transfer well for strong models (which can detect syntactic patterns reliably) but may be inconsistently applied by weaker models that struggle with the pattern-matching itself. The FF CS for illicit conversion should fire for any model that correctly identifies implication reversal in the argument structure — but a model that can reliably do this probably doesn't need the CS.

### Mode C: Phenomenological Conditions (semantic, model-specific)

```
Snarks CS1: "sarcasm signal is subtle — context mismatch"
            "scenario feels like: one statement praises something obviously rare or absurd"
            "the giveaway is: sarcastic option uses positive language for an improbable condition"

Snarks CS2: "scenario feels like: compares reasonable item to absurdly low baseline"
            "the giveaway is: literal comparison ignores common sense about fair benchmarks"
```

These conditions require semantic judgment to evaluate. "Sarcasm signal is subtle" is itself a judgment, not a feature. "Scenario feels like praising something absurd" depends on the model's sense of what is absurd — which differs across model families and training data distributions.

**The ACTIVATE IF condition is itself a task instance.** To decide whether CS1 should activate, the model must evaluate whether the scenario "praises something obviously rare" — which is essentially asking the model to solve a simpler version of the sarcasm classification task. If the model already has the judgment capacity to evaluate the condition, it arguably doesn't need the CS. If it lacks that capacity, it can't reliably evaluate the condition either.

**Transfer consequence:** Mode C CS are most likely to transfer narrowly — to models whose semantic intuitions about subtlety, absurdity, and sarcasm align with mini's. Different model families have different distributions over "what feels obviously absurd," making the condition firing rate variable across models. This explains why snarks CS help mini and llama but hurt gpt-4.1 and claude.

---

## 4. The PK-CS Epistemological Split

The three-mode taxonomy above applies to CS. But looking at the PK sections, a parallel structure appears:

| PK knowledge type | Tasks | Whether it could appear as CS |
|---|---|---|
| Complete algorithm | WOL, BE, navigate, LD3 | No — no residual failure class to patch after PK is complete |
| Reference database | Sports, DU | No — the CS would just be more facts |
| Formal disambiguation rule | FF, DQ | Yes — CS for specific syntactic edge cases |
| Folk-psychological taxonomy | CJ, snarks | Yes — CS for phenomenological subtypes |
| Geometric/algorithmic | GS | Yes — CS for specific parsing edge cases |

**The insight:** PK contains *task definitions* — what the correct answer requires. CS contain *residual failure instances* — what mini still gets wrong after the PK is in place. The type of CS generated depends entirely on what type of PK failure remains.

For formal/algorithmic tasks (GS, FF), PK handles the general rules and CS patch specific edge-case failures with formal conditions. The CS are targeted, specific, and model-agnostic precisely because the PK already handles the general case.

For folk-psychological tasks (CJ, snarks), PK attempts to describe the judgment principles, but the judgment principles themselves are heterogeneous across models. Mini's residual failures are *different failure modes than other models' failures* — and the CS generated for mini's failures don't patch other models' failures. The CS are phenomenological because the task *is phenomenological*.

---

## 5. The Self-Referential Construction Loop

ICR's Phase 1 is a self-referential process: mini writes rules to fix mini's failures, using mini's understanding of its own mistakes. This creates a feedback loop whose output depends on what type of errors mini makes and how it reasons about them.

For Type I tasks (algorithms): mini's training failures are execution errors. The self-referential loop correctly identifies "I failed because I applied precedence wrong" and adds a rule clarifying precedence. The rule is correct because the failure was a clear procedural error with an unambiguous fix.

For Type III tasks (disambiguation taxonomies): mini's training failures are judgment errors. The self-referential loop identifies "I failed because I didn't recognize this was an AND-cause scenario" and adds a rule for AND-cause scenarios. But the rule describes the AND-cause scenario through mini's lens — using mini's conceptual vocabulary, mini's choice of examples, mini's implicit weighting of which cues matter. The rule may be correct but is framed in a way that specifically addresses mini's gap.

**Key finding:** The self-referential loop is most productive when the failure has an objective cause (algorithm error → objective fix) and least productive when the failure reflects a judgment difference (folk-psychological disagreement → model-specific framing). The former produces universally-applicable knowledge; the latter produces model-stamped approximations.

---

## 6. Why Knowledge Structures Look the Way They Do

### 6.1 Length correlates with task difficulty, not with knowledge type

DQ PK has 27 rules; WOL PK has 8 sections. DQ is harder (more failure modes), not more complex (both are well-defined). The PK grows to cover each failure mode mini encounters, making length a proxy for mini's training error distribution, not for task complexity.

### 6.2 Formal structure appears when the task has formal structure

BE PK has parity tables and precedence rules because boolean logic has formal structure. Sports PK has athlete name lists because the task is a lookup. Snarks PK has phenomenological descriptions because the task *is* phenomenological. The cheatsheet format mirrors the task's own epistemic structure.

### 6.3 CS format reflects residual failure type, not a design choice

Mini uses free-form prose for CJ's CS ("Apply when: the outcome requires simultaneous co-occurrence...") because CJ failures are narrative — they involve interpreting a story. Mini uses formal feature flags for GS's CS because GS failures are computable — they involve parsing a string. The ACTIVATE IF conditions don't reflect a deliberate choice by a system designer; they reflect how mini naturally describes the pattern it's trying to catch.

**This has a profound implication:** The CS format is diagnostic. Phenomenological ACTIVATE IF conditions signal that the CS encodes a judgment, not a rule — and judgments don't transfer. Formal ACTIVATE IF conditions signal that the CS encodes a detection procedure, not a judgment — and detection procedures transfer. A future system could predict CS transferability from the lexical properties of the ACTIVATE IF conditions before running eval.

### 6.4 Redundant CS reveal Phase 2's greedy search pathology

FF has 4 CS, 3 of which share near-identical ACTIVATE IF conditions (all target "whoever is not X is Y" structure). Phase 2 generates CS greedily per partition and doesn't globally deduplicate. The result: the most common failure mode (illicit conversion) is over-addressed while orthogonal failures (disjunction misapplication) get only one CS. This creates a CS set that's both redundant and incomplete — too many patches for the same bug, too few for different bugs.

The redundancy has a practical cost: when a model reads 3 near-duplicate CS, it receives a conflicting signal about which one to apply. The condition matching is noisy (Mode B, not Mode A), so all three may fire simultaneously, producing incoherent combined activation.

---

## 7. The Hardest Tasks Resist All Knowledge Types

CJ and snarks fail to saturate under any condition. The oracle 2×2 analysis shows that adding more oracle information doesn't help — it hurts. Why?

**CJ's failure distribution is heterogeneous across models.** Each model family fails on different causal scenario types. Mini fails more on AND-cause joint necessity. Gemini fails more on OR-cause redundancy (as confirmed by E-NEW1: gemini's CS cover redundancy/prevention chains while mini's covers joint necessity). Claude fails on norm-violation and responsibility attribution. There is no single cheatsheet content that addresses all model families' failure modes simultaneously.

This is not a failure of ICR — it reflects genuine task ambiguity. CJ is testing folk-psychological intuition about causation. The BBH labels reflect a specific human annotation convention. Different model families, trained on different text distributions, have absorbed different causal intuitions. A cheatsheet written from mini's training failures cannot patch a different model's different intuitions.

**Implication:** For tasks where the ground truth reflects human folk psychology (CJ, aspects of snarks and DQ), cross-model transfer is fundamentally limited not by cheatsheet quality but by genuine disagreement about what the correct answer is. PK can provide the taxonomy and the resolution rules, but models may weigh these rules differently — and no amount of explicit PK text changes a model's implicit weighting.

---

## 8. Summary: A Transferability Ladder from First Principles

| Knowledge type | Representative task | Transfer mechanism | Transferability |
|---|---|---|---|
| Complete algorithm | WOL, BE, nav, LD3 | Procedural instruction, model-agnostic | Universal |
| Reference database | Sports, DU | Lookup table, model-agnostic | High (bounded by coverage) |
| Formal CS (Mode A) | GS | Feature-flag activation, computable | High |
| Syntactic CS (Mode B) | FF, LD3, DU | Pattern matching, near-computable | Moderate-high |
| Formal disambiguation | DQ | Pragmatic rules, mostly syntactic | Moderate |
| Folk-psychology taxonomy | CJ, snarks PK | Judgment principles, model-weighted | Moderate-low |
| Phenomenological CS (Mode C) | Snarks CS | Semantic judgment activation, model-specific | Low |
| Prose failure patch | CJ CS | No formal condition, implicit activation | Very low |

**The ladder predicts RF transfer results almost exactly.** Tasks in the top rows show strong PK lift (WOL +0pp because already at ceiling; sports +5–12pp from PK alone; GS CS +7.5pp for gemini's formal CS). Tasks in the bottom rows show weak CS transfer and model-dependent PK effects (CJ: mixed; snarks: train-model-only CS benefit).

---

## 9. Research-Worthy Findings

**1. ACTIVATE IF condition type is a transferability signal.**
The lexical properties of ACTIVATE IF conditions — specifically whether they are computable from input features (formal) vs. require semantic judgment (phenomenological) — predict CS transfer without running eval. This opens a path to CS quality assessment before expensive evaluation.

**2. The PK self-reference loop produces universal knowledge iff task failures are objective.**
ICR Phase 1 is most powerful for tasks with objective error modes (algorithm execution). For judgment tasks, it produces model-specific knowledge disguised as general rules. The system cannot distinguish between "I failed because of a clear procedural error" and "I failed because of a genuine judgment disagreement" — both trigger PK patching with the same mechanism.

**3. Phase 2 CS are better understood as failure mode documentation than as knowledge.**
CS don't teach models what to know — they document where a specific model failed and provide a single corrective example. This is fundamentally different from PK's explanatory knowledge. The practical implication: CS should be evaluated not for correctness but for *failure mode coverage* — do they address the target model's actual failure distribution, or a different one?

**4. Cross-task CS format variation is mechanistically informative, not stylistic.**
Mini uses prose for CJ CS, feature flags for GS CS, and phenomenological conditions for snarks CS not because of a prompt design choice, but because each reflects the residual failure type after PK. The CS format is an emergent property of the task structure. This makes format analysis a diagnostic tool for understanding what remains unresolved in each task after PK.

**5. The folk-psychological ceiling exists independently of ICR.**
For tasks grounded in human folk psychology (CJ, aspects of snarks), model families have absorbed different intuitive weightings from pretraining. A cheatsheet written by one model's self-analysis cannot resolve the inter-model disagreement because the disagreement is not a knowledge gap — it's a genuine difference in internalized folk-psychological priors. This is a fundamental limit of in-context knowledge transfer, not just ICR.

---

*Document last updated: 2026-04-30. E-NEW3 plain-examples eval and Oracle 2×2 DQ running in parallel.*
