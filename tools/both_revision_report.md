# Prompt Revision Report

**Dataset**: datasets/mixed_n100_h1_h2.jsonl  
**Cheatsheet**: runs/partition_mixed_n100h1h2/cheatsheet_current.txt  
**Scoring model**: openai/gpt-4o-mini  
**Revision model**: openai/gpt-4o  
**Test model**: openai/gpt-4o  
**Failures used**: 5  
**Variants requested**: 3  


================================================================================
# CASE STUDY PROMPT REVISION
================================================================================


## Failures used (5 items)

```
[1] E1 = x = (y * (x * (z * x))) * y  |  E2 = x * y = x * ((z * w) * z)
    Expected: TRUE  Predicted: ?
    Weak model reasoning (excerpt):
      **Analyzing equational implications**

      I need to tackle solving equational implications in magmas, specifically determining if identity E1 implies E2 universally. E1 states that for all x, y, and z, x = (y * (x * (z * x))) * y. I’ll denote the binary operation as "*", noting there’s no associativity present. I should explore the consequences this might enforce, like possibly a trivial structure. It’s clear I need to derive or find a counterexample where E1 holds, but E2 does not. Let’s see where this leads!**Deriving simplifications**

      I'm attempting to derive simplifications for the equation ...

[2] E1 = x = (y * (x * (z * x))) * y  |  E2 = x * y = x * ((z * w) * z)
    Expected: TRUE  Predicted: ?
    Weak model reasoning (excerpt):
      VERDICT: FALSE
      REASONING: There is an infinite magma satisfying Equation 1 but not Equation 2. A convenient counterexample is built from normal forms of a terminating rewrite system whose single rule is exactly the pattern from Equation 1.

      PROOF: 

      COUNTEREXAMPLE: Let \(T\) be the set of all binary terms over distinct atoms \(p,q,r,s,\dots\) with binary symbol \(*\). Consider the rewrite rule
      \[
      ((u*(v*(w*v)))*u)\to v.
      \]
      This strictly shortens terms, so repeated leftmost-outermost rewriting always terminates. Let \(N\subseteq T\) be the set of irreducible terms, and define a binary operation...

[3] E1 = x = (y * x) * (y * (z * y))  |  E2 = x = ((y * z) * w) * (y * w)
    Expected: TRUE  Predicted: ?
    Weak model reasoning (excerpt):
      VERDICT: FALSE
      REASONING: Equation 1 does not force Equation 2. A counterexample can be built on a free term algebra with tagged constructors so that every value of the form y*(z*y) acts as a right inverse for the map x ↦ y*x, but the term in Equation 2 does not collapse to x.

      PROOF:

      COUNTEREXAMPLE: Let T be the set of all finite terms generated from one constant e using binary constructors S, T, L, M. Define a binary operation * on T by the following cases, in this order:

      1. S(y,x) * T(y,d) = x
      2. T(y,L(z,a)) * T(y,d) = S(z,a)
      3. T(y,M(z,a)) * T(y,d) = T(z,a)
      4. p * S(z,a) = T(p,L(z,a)) if...

[4] E1 = x = ((y * x) * (y * z)) * y  |  E2 = x = ((y * z) * (y * w)) * u
    Expected: TRUE  Predicted: ?
    Weak model reasoning (excerpt):
      VERDICT: FALSE
      REASONING: Equation 2 forces every magma to be trivial: for fixed y,z,w,u, the term ((y*z)*(y*w))*u is independent of x, so x=((y*z)*(y*w))*u for all x implies all elements are equal. Thus it is enough to show that Equation 1 has a nontrivial model. It does: one can obtain such a model by a standard compactness construction. Hence Equation 1 does not imply Equation 2.
      PROOF: 
      COUNTEREXAMPLE: Let the language have one binary operation * and two distinguished constants a,b. Add also a constant c_t for every formal binary term t built from a,b. Consider the theory T consisting of:
      ...

[5] E1 = x = ((y * x) * (y * z)) * y  |  E2 = x = ((y * z) * (y * w)) * u
    Expected: TRUE  Predicted: ?
    Weak model reasoning (excerpt):
      VERDICT: FALSE
      REASONING: Equation 1 does not force all magmas to satisfy Equation 2. One can build a countable magma that is engineered to satisfy every instance of \(x=((y*x)*(y*z))*y\), while many values of \(((y*z)*(y*w))*u\) remain unconstrained and can be kept different from \(x\).

      PROOF: 

      COUNTEREXAMPLE: Take a countable set built recursively, starting with five distinct elements \(0,1,2,3,4\). We define a binary operation \(*\) by a fair stage construction over all triples \((y,x,z)\) of elements already created.

      At a stage for a triple \((y,x,z)\), do the following:
      1. If \(y*x\) i...
```


## Original Prompt

```
You are an expert in universal algebra working on equational theories of magmas.
A magma is a set with a single binary operation * and no other axioms.

You are writing a TEACHING NOTE for a weaker reasoning model that keeps making the same mistake. Your job is to diagnose WHY it fails and teach the exact fix — either a key piece of missing algebraic knowledge, or a wrong/missing reasoning pattern compared to how a stronger model handles the same case.

The cheatsheet the model uses has two parts:

=== REASONING ROADMAP ===
{roadmap}
=== END REASONING ROADMAP ===

=== EXISTING CASE STUDIES ===
{case_studies}
=== END CASE STUDIES ===

=== PATTERNS ALREADY COVERED — YOUR CASE STUDY MUST NOT RESTATE THESE ===
{already_covered}
Your new case study MUST address a gap NOT covered above. If your ACTIVATE IF
conditions would fire on the same equation pairs as any pattern listed above,
you are restating existing knowledge — discard that idea and find a genuinely
new pattern in the failure examples.
=== END ALREADY COVERED ===

The following examples were ALL predicted INCORRECTLY by the weaker model.
The ground-truth verdict and the weaker model's wrong reasoning are shown.
Where available, a CORRECT oracle reasoning trace from a stronger model is shown for contrast — this is your primary signal for what the weaker model is missing.

=== FAILURES WITH INCORRECT MODEL REASONING ===

{failure_lines}

=== YOUR TASK ===

{polarity_instruction}

Step 0 — DIAGNOSE the failure type. Choose exactly one:
  TYPE A — MISSING KNOWLEDGE: The weaker model's reasoning strategy was reasonable
    but it lacks a key algebraic fact (a lemma, identity, or structural property).
    Signal: the oracle trace invokes a fact or consequence that the weaker model
    never considers, even though the weaker model followed a plausible path.
    Example: model doesn't know "absorbing E1 forces all elements equal → TRUE always".
  TYPE B — WRONG/MISSING REASONING PATTERN: The weaker model has the relevant tools
    but applies the wrong one, stops too early, skips a necessary check, or follows
    a bad heuristic that a stronger model avoids.
    Signal: the oracle trace and the weaker model both start similarly, but the
    oracle takes a different branch or performs an extra check the weaker model skips.

Step 1 — For TYPE A: State the missing lemma or algebraic fact precisely in one
  sentence. Explain what structural condition triggers it and what it implies.
  For TYPE B: Quote or paraphrase the exact wrong move from the weaker model's trace.
  Name the correct move the stronger model takes instead, at the same decision point.

Step 2 — Find the CORRECT MOVE: the specific mechanical check that produces the
  right answer. It must be something the model can execute by direct inspection
  of the equation syntax — no proof, no judgment, just structural reading.
  For TYPE A this is often: "check whether [lemma condition] holds → if yes, verdict
  follows immediately." For TYPE B this is the alternative branch/check to take.

Step 3 — Find the TRIGGER: the precise structural conditions that distinguish these
  equations from superficially similar ones where the same mistake would not occur.
  Be narrow. A trigger that fires on too many cases causes regressions and is worse
  than no case study at all. Prefer a trigger that fires on 2–3 cases correctly
  over one that fires on 10 cases and gets half wrong.

Step 4 — Find the ANTI-TRIGGER: 1–2 structurally similar cases where this teaching
  note should NOT fire (the shortcut or the weaker model's approach is actually fine).

Now produce TWO outputs:

OUTPUT 1 — CASE STUDY (max 900 chars)
Write the teaching note in EXACTLY this format, with these exact field names:

=== CASE STUDY: [short title — name the missing lemma (TYPE A) or the mistaken shortcut/structural trap (TYPE B)] ===
FAILURE_TYPE: A or B
ACTIVATE IF:
  - [condition 1 — one structural fact about E1 or E2 that must be true]
  - [condition 2 — ...]
  (All conditions must hold. If any is false, do not use this note.)
DO NOT ACTIVATE IF: [1 sentence — the closest structural case where the shortcut is actually correct]
COMMON WRONG MOVE: [1 sentence — TYPE A: "Does not apply [missing lemma/fact]..."; TYPE B: start with a verb: "Applies...", "Stops at...", "Treats...", "Ignores..."]
NEXT CHECK: [the one mechanical thing to verify instead — must be answerable by direct inspection; end with what it means: "If yes → TRUE. If no → FALSE." or "If yes → proceed to STEP N."]
WHY THIS WORKS: [TYPE A: state the missing lemma explicitly and why it resolves the case. TYPE B: explain why the stronger model's reasoning branch is correct and the weaker model's is not. 1–2 sentences.]
SUPPORT:
  • E1 = ...  |  E2 = ...  |  Answer: TRUE/FALSE  — [one phrase: what structural fact the trigger catches]
  • E1 = ...  |  E2 = ...  |  Answer: TRUE/FALSE  — [one phrase]
TARGET_STEP: [roadmap aspect this corrects, e.g. "ASPECT 2" or "ASPECT 4"]

OUTPUT 2 — ROADMAP PATCH (max 800 chars)
One or more targeted corrections to the reasoning roadmap that would have prevented
these failures. Write only aspects that need to be ADDED or MODIFIED — do not
rewrite the whole roadmap. Name the aspect being refined.

=== ROADMAP PATCH ===
[ASPECT X EXCEPTION / INSERT AFTER ASPECT Y / NEW ASPECT Z]
[corrected or new rule text]
...
=== END PATCH ===

Output ONLY these two sections — no preamble, no sign-off.
```


## Meta-revision raw response

```
### VARIANT 1
CHANGE: Emphasizes the use of precomputed feature vectors in the ACTIVATE IF and NEXT CHECK sections.

You are an expert in universal algebra working on equational theories of magmas. A magma is a set with a single binary operation * and no other axioms.

You are writing a TEACHING NOTE for a weaker reasoning model that keeps making the same mistake. Your job is to diagnose WHY it fails and teach the exact fix — either a key piece of missing algebraic knowledge, or a wrong/missing reasoning pattern compared to how a stronger model handles the same case.

The cheatsheet the model uses has two parts:

=== REASONING ROADMAP ===
{roadmap}
=== END REASONING ROADMAP ===

=== EXISTING CASE STUDIES ===
{case_studies}
=== END CASE STUDIES ===

=== PATTERNS ALREADY COVERED — YOUR CASE STUDY MUST NOT RESTATE THESE ===
{already_covered}
Your new case study MUST address a gap NOT covered above. If your ACTIVATE IF conditions would fire on the same equation pairs as any pattern listed above, you are restating existing knowledge — discard that idea and find a genuinely new pattern in the failure examples.
=== END ALREADY COVERED ===

The following examples were ALL predicted INCORRECTLY by the weaker model. The ground-truth verdict and the weaker model's wrong reasoning are shown. Where available, a CORRECT oracle reasoning trace from a stronger model is shown for contrast — this is your primary signal for what the weaker model is missing.

=== FAILURES WITH INCORRECT MODEL REASONING ===

{failure_lines}

=== YOUR TASK ===

{polarity_instruction}

Step 0 — DIAGNOSE the failure type. Choose exactly one:
  TYPE A — MISSING KNOWLEDGE: The weaker model's reasoning strategy was reasonable but it lacks a key algebraic fact (a lemma, identity, or structural property).
  Signal: the oracle trace invokes a fact or consequence that the weaker model never considers, even though the weaker model followed a plausible path.
  Example: model doesn't know "absorbing E1 forces all elements equal → TRUE always".
  TYPE B — WRONG/MISSING REASONING PATTERN: The weaker model has the relevant tools but applies the wrong one, stops too early, skips a necessary check, or follows a bad heuristic that a stronger model avoids.
  Signal: the oracle trace and the weaker model both start similarly, but the oracle takes a different branch or performs an extra check the weaker model skips.

Step 1 — For TYPE A: State the missing lemma or algebraic fact precisely in one sentence. Explain what structural condition triggers it and what it implies. For TYPE B: Quote or paraphrase the exact wrong move from the weaker model's trace. Name the correct move the stronger model takes instead, at the same decision point.

Step 2 — Find the CORRECT MOVE: Use PRECOMPUTED FEATURES to determine the specific mechanical check that produces the right answer. It must be something the model can execute by direct inspection of the equation syntax — no proof, no judgment, just structural reading. For ...
```


## Variant 1: Emphasizes the use of precomputed feature vectors in the ACTIVATE IF and NEXT CHECK sections.

### Proposed Prompt
```
You are an expert in universal algebra working on equational theories of magmas. A magma is a set with a single binary operation * and no other axioms.

You are writing a TEACHING NOTE for a weaker reasoning model that keeps making the same mistake. Your job is to diagnose WHY it fails and teach the exact fix — either a key piece of missing algebraic knowledge, or a wrong/missing reasoning pattern compared to how a stronger model handles the same case.

The cheatsheet the model uses has two parts:

=== REASONING ROADMAP ===
{roadmap}
=== END REASONING ROADMAP ===

=== EXISTING CASE STUDIES ===
{case_studies}
=== END CASE STUDIES ===

=== PATTERNS ALREADY COVERED — YOUR CASE STUDY MUST NOT RESTATE THESE ===
{already_covered}
Your new case study MUST address a gap NOT covered above. If your ACTIVATE IF conditions would fire on the same equation pairs as any pattern listed above, you are restating existing knowledge — discard that idea and find a genuinely new pattern in the failure examples.
=== END ALREADY COVERED ===

The following examples were ALL predicted INCORRECTLY by the weaker model. The ground-truth verdict and the weaker model's wrong reasoning are shown. Where available, a CORRECT oracle reasoning trace from a stronger model is shown for contrast — this is your primary signal for what the weaker model is missing.

=== FAILURES WITH INCORRECT MODEL REASONING ===

{failure_lines}

=== YOUR TASK ===

{polarity_instruction}

Step 0 — DIAGNOSE the failure type. Choose exactly one:
  TYPE A — MISSING KNOWLEDGE: The weaker model's reasoning strategy was reasonable but it lacks a key algebraic fact (a lemma, identity, or structural property).
  Signal: the oracle trace invokes a fact or consequence that the weaker model never considers, even though the weaker model followed a plausible path.
  Example: model doesn't know "absorbing E1 forces all elements equal → TRUE always".
  TYPE B — WRONG/MISSING REASONING PATTERN: The weaker model has the relevant tools but ...
```

### Sample Output
```
OUTPUT 1 — CASE STUDY

=== CASE STUDY: Ignoring Triviality Implied by E1 ===
FAILURE_TYPE: A
ACTIVATE IF:
  - E1 has depth 2+ with * operators
  - x appears multiple times or non-terminally in E1's RHS
DO NOT ACTIVATE IF: E1 does not force a single-element magma in all cases.
COMMON WRONG MOVE: Does not apply the triviality lemma that E1 forces a single-element magma, which would make E2 trivially true.
NEXT CHECK: Check if E1 forces a single-element magma (triviality) → if yes, verdict is TRUE. If no, further analysis needed.
WHY THIS WORKS: The triviality lemma states that if E1 collapses every magma to a single element, then E2 holds trivially since it must hold in a single-element magma.
SUPPORT:
  • E1 = x = (y * (x * (z * x))) * y  |  E2 = x * y = x * ((z * w) * z)  |  Answer: TRUE  — E1 forces triviality
  • E1 = x = ((y * x) * (y * z)) * y  |  E2 = x = ((y * z) * (y * w)) * u  |  Answer: TRUE  — E1 forces triviality
TARGET_STEP: ASPECT 2

OUTPUT 2 — ROADMAP PATCH

=== ROADMAP PATCH ===
INSERT AFTER ASPECT 2
If E1 has depth 2+ with * operators and x appears multiple times or non-terminally in E1's RHS, check if E1 forces a single-element magma. If it does, E2 holds trivially. This ensures that cases where E1 inherently forces triviality are correctly identified.
=== END PATCH ===
```


## Variant 2: Enhances clarity in TYPE A diagnosis by emphasizing the contrast between oracle and weaker model.

### Proposed Prompt
```
You are an expert in universal algebra working on equational theories of magmas. A magma is a set with a single binary operation * and no other axioms.

You are writing a TEACHING NOTE for a weaker reasoning model that keeps making the same mistake. Your job is to diagnose WHY it fails and teach the exact fix — either a key piece of missing algebraic knowledge, or a wrong/missing reasoning pattern compared to how a stronger model handles the same case.

The cheatsheet the model uses has two parts:

=== REASONING ROADMAP ===
{roadmap}
=== END REASONING ROADMAP ===

=== EXISTING CASE STUDIES ===
{case_studies}
=== END CASE STUDIES ===

=== PATTERNS ALREADY COVERED — YOUR CASE STUDY MUST NOT RESTATE THESE ===
{already_covered}
Your new case study MUST address a gap NOT covered above. If your ACTIVATE IF conditions would fire on the same equation pairs as any pattern listed above, you are restating existing knowledge — discard that idea and find a genuinely new pattern in the failure examples.
=== END ALREADY COVERED ===

The following examples were ALL predicted INCORRECTLY by the weaker model. The ground-truth verdict and the weaker model's wrong reasoning are shown. Where available, a CORRECT oracle reasoning trace from a stronger model is shown for contrast — this is your primary signal for what the weaker model is missing.

=== FAILURES WITH INCORRECT MODEL REASONING ===

{failure_lines}

=== YOUR TASK ===

{polarity_instruction}

Step 0 — DIAGNOSE the failure type. Choose exactly one:
  TYPE A — MISSING KNOWLEDGE: The weaker model's reasoning strategy was reasonable but it lacks a key algebraic fact (a lemma, identity, or structural property).
  Signal: the oracle trace invokes a fact or consequence that the weaker model never considers, even though the weaker model followed a plausible path.
  Example: model doesn't know "absorbing E1 forces all elements equal → TRUE always".
  TYPE B — WRONG/MISSING REASONING PATTERN: The weaker model has the relevant tools but ...
```

### Sample Output
```
OUTPUT 1 — CASE STUDY

=== CASE STUDY: Wrong Reasoning Pattern in Non-Trivial Magmas ===
FAILURE_TYPE: B
ACTIVATE IF:
  - E1 is in a form that does not force triviality (e.g., non-terminal variables in RHS)
  - E2 is not trivially true in a non-trivial magma
  (All conditions must hold. If any is false, do not use this note.)
DO NOT ACTIVATE IF: E1 forces a single-element magma trivially.
COMMON WRONG MOVE: Treats E1 as forcing triviality without verifying implications in non-trivial magmas.
NEXT CHECK: Verify if E1 can be satisfied in a non-trivial magma. If yes → FALSE. If no → TRUE.
WHY THIS WORKS: The stronger model correctly checks for non-trivial models of E1 before concluding triviality, whereas the weaker model prematurely assumes triviality without this verification.
SUPPORT:
  • E1 = x = (y * (x * (z * x))) * y  |  E2 = x * y = x * ((z * w) * z)  |  Answer: TRUE  — E1 can hold non-trivially
  • E1 = x = (y * x) * (y * (z * y))  |  E2 = x = ((y * z) * w) * (y * w)  |  Answer: TRUE  — E1 can hold non-trivially
TARGET_STEP: ASPECT 3

OUTPUT 2 — ROADMAP PATCH

=== ROADMAP PATCH ===
INSERT AFTER ASPECT 3
- When E1 does not trivially force a single-element magma, verify if E1 can be satisfied in a non-trivial magma. If it can, E2 may not hold universally.
=== END PATCH ===
```


## Variant 3: Focuses on improving clarity in the SUPPORT section by detailing the structural condition the ACTIVATE IF caught.

### Proposed Prompt
```
You are an expert in universal algebra working on equational theories of magmas. A magma is a set with a single binary operation * and no other axioms.

You are writing a TEACHING NOTE for a weaker reasoning model that keeps making the same mistake. Your job is to diagnose WHY it fails and teach the exact fix — either a key piece of missing algebraic knowledge, or a wrong/missing reasoning pattern compared to how a stronger model handles the same case.

The cheatsheet the model uses has two parts:

=== REASONING ROADMAP ===
{roadmap}
=== END REASONING ROADMAP ===

=== EXISTING CASE STUDIES ===
{case_studies}
=== END CASE STUDIES ===

=== PATTERNS ALREADY COVERED — YOUR CASE STUDY MUST NOT RESTATE THESE ===
{already_covered}
Your new case study MUST address a gap NOT covered above. If your ACTIVATE IF conditions would fire on the same equation pairs as any pattern listed above, you are restating existing knowledge — discard that idea and find a genuinely new pattern in the failure examples.
=== END ALREADY COVERED ===

The following examples were ALL predicted INCORRECTLY by the weaker model. The ground-truth verdict and the weaker model's wrong reasoning are shown. Where available, a CORRECT oracle reasoning trace from a stronger model is shown for contrast — this is your primary signal for what the weaker model is missing.

=== FAILURES WITH INCORRECT MODEL REASONING ===

{failure_lines}

=== YOUR TASK ===

{polarity_instruction}

Step 0 — DIAGNOSE the failure type. Choose exactly one:
  TYPE A — MISSING KNOWLEDGE: The weaker model's reasoning strategy was reasonable but it lacks a key algebraic fact (a lemma, identity, or structural property).
  Signal: the oracle trace invokes a fact or consequence that the weaker model never considers, even though the weaker model followed a plausible path.
  Example: model doesn't know "absorbing E1 forces all elements equal → TRUE always".
  TYPE B — WRONG/MISSING REASONING PATTERN: The weaker model has the relevant tools but ...
```

### Sample Output
```
OUTPUT 1 — CASE STUDY

=== CASE STUDY: Wrong Reasoning Pattern in Non-Triviality Detection ===
FAILURE_TYPE: B
ACTIVATE IF:
  - E1 has a non-terminal occurrence of x on the right side.
  - E2 has a different structural pattern from E1 and does not trivially follow from E1.
  (All conditions must hold. If any is false, do not use this note.)
DO NOT ACTIVATE IF: E1 forces a trivial magma by collapsing all elements to a single value.
COMMON WRONG MOVE: Stops at assuming E1 implies a trivial magma without verifying non-trivial models.
NEXT CHECK: Verify if E1 can be satisfied in a non-trivial magma by constructing or imagining a non-trivial model → If yes, E1 does not imply E2 trivially. If no, E1 implies E2 trivially.
WHY THIS WORKS: The stronger model explores the possibility of non-trivial models for E1, ensuring that E1 does not trivially imply E2 by default. This prevents incorrect assumptions about triviality.
SUPPORT:
  • E1 = x = (y * (x * (z * x))) * y  |  E2 = x * y = x * ((z * w) * z)  |  Answer: TRUE  — E1 does not force triviality.
  • E1 = x = (y * x) * (y * (z * y))  |  E2 = x = ((y * z) * w) * (y * w)  |  Answer: TRUE  — E1 does not force triviality.
TARGET_STEP: ASPECT 4

OUTPUT 2 — ROADMAP PATCH

=== ROADMAP PATCH ===
INSERT AFTER ASPECT 3
Check for non-trivial models: If E1 does not trivially collapse the magma, explore non-trivial models to verify if E2 follows. If a non-trivial model exists where E1 holds but E2 does not, E1 does not imply E2 trivially.
=== END PATCH ===
```



================================================================================
# PK SECTION PROMPT REVISION
================================================================================


## Original Prompt

```
You are improving a mathematical reasoning guide for equational theories of magmas.
A magma is a set with a binary operation * with no other axioms.

TARGET FAILURE CLASS
  Structural partition : {partition_label}
  E1 form              : {form_e1}
  E2 form              : {form_e2}
  Expected answer      : {polarity}
  Depth bucket         : {depth_desc}

The following {n_examples} items ALL belong to this class and are CONSISTENTLY WRONG:
{failure_examples}

EXISTING PRIOR KNOWLEDGE SECTIONS (titles only — do NOT duplicate):
{existing_titles}

YOUR TASK
Write ONE new prior knowledge section — a GENERAL RULE or HEURISTIC — for this
structural class.  This is NOT a case study.  Do not use IDENTIFY/ACTION/WHY/EXAMPLES
fields.  Instead, write in the same style as the existing sections above: a numbered
or bulleted set of decision rules with brief mathematical justification.

CORRECT FORMAT (imitate this style exactly):
  === STEP N: YOUR RULE TITLE ===

  RULE X: <statement of the rule> → TRUE or FALSE.
    WHY: <one-sentence mathematical justification>.
    Example: E1 = "..." | E2 = "..." → <answer and reason>.

  RULE X+1: <another rule if needed>
    WHY: ...

Requirements:
  • Open with === YOUR SECTION TITLE ===
  • State rules that apply to EVERY item in this structural class, not just the examples
  • Each rule must have a WHY line explaining the mathematical reason
  • Include at least one worked example inline (not as a separate block)
  • Length: 100–300 words
  • Do NOT use IDENTIFY / ACTION / DOES NOT APPLY TO headings

Write ONLY the new section text, starting with ===.
```


## Meta-revision raw response

```
### VARIANT 1
CHANGE: Instructs the model to name rules using the feature vector and explicitly mentions separator and collapse conditions.

---

You are improving a mathematical reasoning guide for equational theories of magmas. A magma is a set with a binary operation * with no other axioms.

TARGET FAILURE CLASS
  Structural partition : {partition_label}
  E1 form              : {form_e1}
  E2 form              : {form_e2}
  Expected answer      : {polarity}
  Depth bucket         : {depth_desc}

The following {n_examples} items ALL belong to this class and are CONSISTENTLY WRONG:
{failure_examples}

EXISTING PRIOR KNOWLEDGE SECTIONS (titles only — do NOT duplicate):
{existing_titles}

YOUR TASK
Write ONE new prior knowledge section — a GENERAL RULE or HEURISTIC — for this structural class. This is NOT a case study. Do not use IDENTIFY/ACTION/WHY/EXAMPLES fields. Instead, write in the same style as the existing sections above: a numbered or bulleted set of decision rules with brief mathematical justification.

CORRECT FORMAT (imitate this style exactly):
  === STEP N: YOUR RULE TITLE ===

  RULE X: <statement of the rule using feature vector> → TRUE or FALSE.
    WHY: <one-sentence mathematical justification referencing feature vector>.
    Example: RULE: bare(E1)=TRUE, vars(E1)>=4, vars(E2)=2 → TRUE.
    WHY: A bare equation with 4 or more variables enforces a collapse that implies any 2-variable equation.
    Example: E1 = "x = x * y", E2 = "x = y" | Features: bare(E1)=TRUE, vars(E1)=2, vars(E2)=1 → TRUE.

  RULE X+1: <another rule using separator or collapse conditions if needed>
    WHY: ...

Requirements:
  • Open with === YOUR SECTION TITLE ===
  • State rules using feature vector and mention separator/collapse explicitly
  • Each rule must have a WHY line explaining the mathematical reason
  • Include at least one worked example inline, showing feature values alongside equations
  • Length: 100–300 words
  • Do NOT use IDENTIFY / ACTION / DOES NOT APPLY TO headings

---

### VARIANT 2
CHANGE: Encourages step-ordering and instructs to STOP after a rule fires.

---

You are improving a mathematical reasoning guide for equational theories of magmas. A magma is a set with a binary operation * with no other axioms.

TARGET FAILURE CLASS
  Structural partition : {partition_label}
  E1 form              : {form_e1}
  E2 form              : {form_e2}
  Expected answer      : {polarity}
  Depth bucket         : {depth_desc}

The following {n_examples} items ALL belong to this class and are CONSISTENTLY WRONG:
{failure_examples}

EXISTING PRIOR KNOWLEDGE SECTIONS (titles only — do NOT duplicate):
{existing_titles}

YOUR TASK
Write ONE new prior knowledge section — a GENERAL RULE or HEURISTIC — for this structural class. This is NOT a case study. Do not use IDENTIFY/ACTION/WHY/EXAMPLES fields. Instead, write in the same style as the existing sections above: a numbered or bulleted set of decision rules with brief mathematical justification.

CORREC...
```
