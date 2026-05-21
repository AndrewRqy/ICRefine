# SFCR ABC Experiment Review
Date: 2026-05-14

## Overview

13 runs across experiment groups A1, B1, C1, C3. Every run returned **0 accepted rules**.
Two distinct failure modes were observed.

---

## Failure Mode 1 — Count Gate Rejects All Candidates (A1 / B1 / C1)

All 11 runs in groups A1, B1, C1 generated candidates but rejected 100% of them at the
count gate. The dominant rejection reasons across runs:

| Rejection reason                        | Runs affected        |
|-----------------------------------------|----------------------|
| `fixed_shared_count=0 < threshold`      | All 11               |
| `private_activation_count > 0`          | CJ runs (B1, C1)     |
| `reg_private_count > 0`                 | CJ runs (B1, C1)     |

### Root cause A: shared failure set is too small to hit

The shared failure set (cases where *multiple* proxy models fail, not just the source model)
is tiny: 13–15 items for CJ, similar for GS and LogiQA. At the same time, the rules being
evaluated are general-purpose — they describe broad causal reasoning patterns rather than
surgical fixes for precisely those 13–15 cases.

Result: rules activate on many items in the full training set but happen to not fix any
item in the shared failure set specifically. `fixed_shared_count` stays 0 even when the
rule is logically correct for the failure pattern it targets.

The count gate requires `fixed_shared_count >= 1` (diagnostic profile) or
`fixed_shared_count >= 2` (large profile) as a hard floor. With only 13–15 shared
failures and rules written at coarse semantic granularity, this floor is very difficult
to clear by chance.

### Root cause B: private activation veto is too strict

The gate also rejects any candidate that activates on even one model-private failure
(a case only the source model gets wrong, not the proxies). This makes intuitive sense
as a generalization signal, but in practice:

- CJ has 4–6 private failures
- Any rule that describes a common causal reasoning pattern will plausibly activate on
  some private failures, since those failures often arise from the same underlying patterns
  as shared ones — the distinction is statistical, not semantic

The combination of `fixed_shared_count >= 1` AND `private_activation_count == 0` is
doubly hard to satisfy. A rule that is broad enough to fix shared cases is almost
certain to also fire on a few private cases.

### Root cause C: gate profile escalation (C1 large)

The C1 large gate raises the shared-fix threshold from 1 to 2. With a shared set of
only ~14 items, requiring 2 confirmed shared fixes from a single manually written rule
is essentially impossible without extremely targeted rules. C1 large rejected even the
candidate that achieved `fixed_shared_count=1` (accepted in other profiles).

### Summary of the count gate problem

The gate was designed assuming rules are highly targeted, surgical corrections derived
from LLM generation guided by shared failure cases specifically. The current manual
rules are semantic generalizations that do not reliably land on the exact 13–15 shared
failures. The gate is correct in principle but miscalibrated for the granularity of
rules being evaluated.

**Fixes to consider:**
- Lower `fixed_shared_count` threshold to 0 with a soft signal (rate-based rather
  than count-based) for small shared sets
- Replace hard `private_activation_count == 0` veto with a precision condition:
  `shared_activations / total_activations >= min_precision`
- Pool shared failures across multiple seeds before gating, so the effective
  shared set is larger and easier to hit

---

## Failure Mode 2 — Generation Produces No Valid Candidates (C3)

Both C3 runs (formal_fallacies, snarks) failed before the gate:
`skip_reason: "generation produced no valid candidates"`

No candidates were parsed from the LLM output at all. Likely causes:

1. **Format mismatch**: The generation prompt for the C3 configuration (keyword router,
   rule_check_example memory format) may produce output that does not match the expected
   parse pattern, resulting in empty candidate sets.
2. **Task mismatch**: formal_fallacies and snarks may have very few shared failures
   (or zero), giving the generator no grounded examples to work from, causing it to
   produce off-format or empty output.
3. **Keyword router cold-start**: If no keywords fire on the shared failure cases,
   the router may route zero items to any region, leaving the generator with an empty
   input block.

These failures are upstream of the gate and require separate debugging: inspect the
raw generation output in the C3 logs (`logs/sfcr_C3_*.log`) to identify whether the
issue is format, empty input, or a task-specific edge case.

---

## Variant Comparison (within A1/B1/C1)

Across the router type and memory format variants, rejection patterns were consistent:
all variants failed for the same underlying reasons. No variant was notably closer to
acceptance than others.

| Group | Variable tested       | Outcome |
|-------|-----------------------|---------|
| A1    | router: keyword vs feature | Both 0 accepted; keyword slightly more candidates |
| B1    | memory format: rule vs rule_check_example | Both 0 accepted; same rejection reasons |
| C1    | gate profile: small/medium/large | All 0 accepted; large profile most restrictive |

The experiment design is sound — the variants are testing meaningful axes — but the
underlying gate calibration issue means no signal reaches the acceptance stage for any
variant.

---

## Recommended Next Steps

1. **Relax the count gate** for small shared failure sets: introduce a minimum shared
   set size below which precision-based gating replaces count-based gating.
2. **Replace `private_activation_count == 0` with a precision floor** (e.g., ≥ 50%
   of activations must be on shared or easy items).
3. **Debug C3 generation** by reading the raw LLM output from the C3 logs to identify
   why parsing fails.
4. **Rerun A1/B1/C1 with relaxed gate** once calibration is fixed before concluding
   that router type or memory format variants make no difference — currently no signal
   reaches the gate exit.
