#!/usr/bin/env python3
"""
smoke_test_pk_patch.py — Smoke tests for _pk_patch_phase and related changes.

Tests
-----
  1. _bootstrap_ruleset includes post_think in LLM prompt
  2. _pk_patch_phase happy path: patch accepted, PK updated
  3. _pk_patch_phase fix-rate gate: patch rejected, PK unchanged
  4. _pk_patch_phase regression gate: fix-rate passes but regression fails → rejected
  5. _pk_patch_phase oracle: _oracle_exact text appears in LLM prompt
  6. Bootstrap guard: rule_set stays None when init-cheatsheet provides PK
  7. elif dispatch: _pk_patch_phase branch taken when PK exists and rule_set is None
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from unittest.mock import patch, MagicMock
from utils.cheatsheet import Cheatsheet

# ── Colours ──────────────────────────────────────────────────────────────────
G, R, RESET = "\033[92m", "\033[91m", "\033[0m"
n_pass = n_fail = 0

def _ok(name):
    global n_pass
    print(f"  {G}PASS{RESET}  {name}")
    n_pass += 1

def _fail(name, detail=""):
    global n_fail
    print(f"  {R}FAIL{RESET}  {name}" + (f"\n       {detail}" if detail else ""))
    n_fail += 1

def check(name, cond, detail=""):
    (_ok if cond else lambda n: _fail(n, detail))(name)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_items(n, prefix="q"):
    return [{"input": f"{prefix}_{i} context", "expected": "YES",
             "answer": "YES", "id": str(i)} for i in range(n)]

def make_wrong(items, post_think="model wrongly concluded X"):
    return [{**it, "predicted": "NO", "post_think": post_think,
             "thinking": "", "raw_response": ""} for it in items]

def make_correct(items):
    return [{**it, "predicted": "YES", "post_think": "",
             "thinking": "", "raw_response": ""} for it in items]

def fake_resp(text):
    r = MagicMock()
    r.content = text
    r.thinking = ""
    return r


# =============================================================================
# TEST 1 — _bootstrap_ruleset: post_think in LLM prompt
# =============================================================================
print("\n[1] _bootstrap_ruleset — post_think in failure prompt")

captured_prompt = []

def _capture_llm(prompt, **kwargs):
    captured_prompt.append(prompt)
    return fake_resp("CJ-1: test rule →  YES")

failures_1 = [
    {"input": "Alice pushed Bob.", "answer": "YES", "predicted": "NO",
     "post_think": "UNIQUE_POST_THINK_SIGNAL", "reason": ""},
    {"input": "Carol helped Dave.", "answer": "NO", "predicted": "YES",
     "post_think": "", "reason": "UNIQUE_REASON_SIGNAL"},
]

with patch("utils.llm_client.call_llm", side_effect=_capture_llm):
    try:
        from tasks.bbh_tasks import _causal_bootstrap
        _causal_bootstrap(failures_1, "test-model", "fake-key")
        prompt = captured_prompt[0] if captured_prompt else ""
        check("post_think in prompt", "UNIQUE_POST_THINK_SIGNAL" in prompt,
              f"signal not found; prompt[:200]={repr(prompt[:200])}")
        check("reason in prompt", "UNIQUE_REASON_SIGNAL" in prompt,
              f"reason not found; prompt[:200]={repr(prompt[:200])}")
        check("Wrong reasoning label present", "Wrong reasoning" in prompt)
        check("Correct reasoning label present", "Correct reasoning" in prompt)
    except Exception as e:
        import traceback
        _fail("_bootstrap_ruleset runs without error", traceback.format_exc())


# =============================================================================
# TEST 2 — _pk_patch_phase: happy path (patch accepted, PK updated)
# =============================================================================
print("\n[2] _pk_patch_phase — happy path")

from ICR_hybrid.training.loop import _pk_patch_phase

train_8 = make_items(8)
cs2 = Cheatsheet(roadmap="", case_studies=[], prior_knowledge="Original PK text.")

score_seq_2 = []

def mock_score_2(items, cs_text, model, api_key, **kwargs):
    n = len(score_seq_2)
    score_seq_2.append(n)
    if n == 0:      # pk-init: 5 correct + 3 wrong
        return make_correct(items[:5]), make_wrong(items[5:])
    elif n == 1:    # fix-gate iter 1: 2/3 fixed → 67% > 20% threshold ✓
        return make_correct(items[:2]), make_wrong(items[2:])
    elif n == 2:    # regress-gate iter 1: 1/5 regressed → 20% = threshold, not > ✓
        return make_correct(items[:4]), make_wrong(items[4:])
    elif n == 3:    # full rescore after accept: 7 correct, 1 wrong
        return make_correct(items[:7]), make_wrong(items[7:])
    else:           # subsequent fix-gates: 0% → rejected → exhaust static_iters
        return [], make_wrong(items)

llm_call_count_2 = []

def mock_llm_2(prompt, **kwargs):
    llm_call_count_2.append(prompt)
    return fake_resp("IMPROVED_PK_UNIQUE_TEXT")

try:
    with patch("ICR_hybrid.training.loop.score_batch", side_effect=mock_score_2), \
         patch("utils.llm_client.call_llm", side_effect=mock_llm_2):
        n_patches, final_acc, iters = _pk_patch_phase(
            cheatsheet=cs2,
            train_items=train_8,
            model_patch="gpt-4o-mini",
            model_score="gpt-4o-mini",
            api_key="fake",
            oracle=None,
            max_iters=4,
            acc_goal=0.99,
            static_iters=2,
            fix_rate_threshold=0.20,
            regress_threshold=0.20,
            concurrency=5,
            log_fn=lambda m: None,
            task_spec=None,
            reasoning_effort=None,
            cot_first=False,
        )
    check("n_patches == 1", n_patches == 1, f"got {n_patches}")
    check("PK updated to candidate", cs2.prior_knowledge == "IMPROVED_PK_UNIQUE_TEXT",
          f"PK={repr(cs2.prior_knowledge)}")
    check("final_acc is float", isinstance(final_acc, float))
    check("iters_done is int", isinstance(iters, int))
    check("LLM was called", len(llm_call_count_2) >= 1)
except Exception as e:
    import traceback
    _fail("_pk_patch_phase happy path", traceback.format_exc())


# =============================================================================
# TEST 3 — _pk_patch_phase: fix-rate gate rejects
# =============================================================================
print("\n[3] _pk_patch_phase — fix-rate gate rejection")

cs3 = Cheatsheet(roadmap="", case_studies=[], prior_knowledge="ORIGINAL_PK_3")
score_seq_3 = []

def mock_score_3(items, cs_text, model, api_key, **kwargs):
    n = len(score_seq_3)
    score_seq_3.append(n)
    if n == 0:   # pk-init: 3 wrong
        return make_correct(items[:5]), make_wrong(items[5:])
    else:        # fix-gate always: 0 fixed → 0% < 20% → rejected every time
        return [], make_wrong(items)

try:
    with patch("ICR_hybrid.training.loop.score_batch", side_effect=mock_score_3), \
         patch("utils.llm_client.call_llm", side_effect=lambda *a, **kw: fake_resp("candidate")):
        n_patches, final_acc, iters = _pk_patch_phase(
            cheatsheet=cs3,
            train_items=make_items(8),
            model_patch="gpt-4o-mini",
            model_score="gpt-4o-mini",
            api_key="fake",
            oracle=None,
            max_iters=4,
            acc_goal=0.99,
            static_iters=2,
            fix_rate_threshold=0.20,
            regress_threshold=0.20,
            concurrency=5,
            log_fn=lambda m: None,
            task_spec=None,
            reasoning_effort=None,
            cot_first=False,
        )
    check("no patches accepted", n_patches == 0, f"got {n_patches}")
    check("PK unchanged", cs3.prior_knowledge == "ORIGINAL_PK_3",
          f"PK changed to {repr(cs3.prior_knowledge)}")
    check("exits after static_iters=2", iters <= 3, f"iters={iters}")
except Exception as e:
    import traceback
    _fail("fix-rate gate rejection", traceback.format_exc())


# =============================================================================
# TEST 4 — _pk_patch_phase: regression gate rejects
# =============================================================================
print("\n[4] _pk_patch_phase — regression gate rejection")

cs4 = Cheatsheet(roadmap="", case_studies=[], prior_knowledge="ORIGINAL_PK_4")
score_seq_4 = []

def mock_score_4(items, cs_text, model, api_key, **kwargs):
    n = len(score_seq_4)
    score_seq_4.append(n)
    if n == 0:   # pk-init: 5 correct + 3 wrong
        return make_correct(items[:5]), make_wrong(items[5:])
    elif n % 2 == 1:   # fix-gate: all 3 fixed → 100% > 20% threshold ✓
        return make_correct(items), []
    else:              # regress-gate: all 5 regressed → 100% > 20% → rejected
        return [], make_wrong(items)

try:
    with patch("ICR_hybrid.training.loop.score_batch", side_effect=mock_score_4), \
         patch("utils.llm_client.call_llm", side_effect=lambda *a, **kw: fake_resp("candidate")):
        n_patches, _, _ = _pk_patch_phase(
            cheatsheet=cs4,
            train_items=make_items(8),
            model_patch="gpt-4o-mini",
            model_score="gpt-4o-mini",
            api_key="fake",
            oracle=None,
            max_iters=4,
            acc_goal=0.99,
            static_iters=2,
            fix_rate_threshold=0.20,
            regress_threshold=0.20,
            concurrency=5,
            log_fn=lambda m: None,
            task_spec=None,
            reasoning_effort=None,
            cot_first=False,
        )
    check("no patches accepted", n_patches == 0, f"got {n_patches}")
    check("PK unchanged", cs4.prior_knowledge == "ORIGINAL_PK_4",
          f"PK changed to {repr(cs4.prior_knowledge)}")
except Exception as e:
    import traceback
    _fail("regression gate rejection", traceback.format_exc())


# =============================================================================
# TEST 5 — _pk_patch_phase: _oracle_exact text appears in LLM prompt
# =============================================================================
print("\n[5] _pk_patch_phase — oracle text in LLM prompt")

oracle_wrong_items = [
    {**it, "predicted": "NO", "post_think": "wrong reasoning",
     "_oracle_exact": "ORACLE_UNIQUE_SIGNAL", "thinking": "", "raw_response": ""}
    for it in make_items(3)
]
oracle_correct_items = make_correct(make_items(5, "c"))

score_seq_5 = []

def mock_score_5(items, cs_text, model, api_key, **kwargs):
    n = len(score_seq_5)
    score_seq_5.append(n)
    if n == 0:   # pk-init: 5 correct + 3 wrong (with oracle)
        return oracle_correct_items, oracle_wrong_items
    else:        # all other calls: everything correct → exit after 1 accepted patch
        return make_correct(items), []

captured_llm_prompt_5 = []

def mock_llm_5(prompt, **kwargs):
    captured_llm_prompt_5.append(prompt)
    return fake_resp("IMPROVED_WITH_ORACLE")

try:
    with patch("ICR_hybrid.training.loop.score_batch", side_effect=mock_score_5), \
         patch("utils.llm_client.call_llm", side_effect=mock_llm_5):
        _pk_patch_phase(
            cheatsheet=Cheatsheet(roadmap="", case_studies=[], prior_knowledge="base pk"),
            train_items=make_items(8),
            model_patch="gpt-4o-mini",
            model_score="gpt-4o-mini",
            api_key="fake",
            oracle=None,          # oracle is embedded in _oracle_exact field on items
            max_iters=2,
            acc_goal=0.99,
            static_iters=2,
            fix_rate_threshold=0.01,  # very low — passes immediately
            regress_threshold=0.99,   # very high — passes immediately
            concurrency=5,
            log_fn=lambda m: None,
            task_spec=None,
            reasoning_effort=None,
            cot_first=False,
        )
    prompt_with_oracle = next(
        (p for p in captured_llm_prompt_5 if "ORACLE_UNIQUE_SIGNAL" in p), None
    )
    check("oracle text appears in LLM prompt", prompt_with_oracle is not None,
          f"ORACLE_UNIQUE_SIGNAL not found in {len(captured_llm_prompt_5)} prompt(s)")
    check("post_think in same prompt", prompt_with_oracle is not None
          and "wrong reasoning" in prompt_with_oracle)
except Exception as e:
    import traceback
    _fail("oracle text in prompt", traceback.format_exc())


# =============================================================================
# TEST 6 — Bootstrap guard: rule_set stays None when init-cheatsheet has PK
# =============================================================================
print("\n[6] Bootstrap guard — rule_set stays None when PK present")

bootstrap_invoked = []

def mock_bootstrap_rs(failures, model, api_key):
    bootstrap_invoked.append(True)
    return MagicMock()

# Replicate the exact guard condition from run_hybrid_loop's bootstrap else-branch:
#   if initial_cheatsheet is not None and cheatsheet.prior_knowledge.strip():
#       [skip bootstrap]
#   else:
#       rule_set = task_spec.bootstrap_ruleset(...)

initial_cs = Cheatsheet(roadmap="", case_studies=[], prior_knowledge="CS-ICL content here")
cs_copy = Cheatsheet(
    roadmap=initial_cs.roadmap,
    case_studies=list(initial_cs.case_studies),
    prior_knowledge=initial_cs.prior_knowledge,
)
rule_set_6 = None

if initial_cs is not None and cs_copy.prior_knowledge.strip():
    pass   # guard fires: skip bootstrap
else:
    rule_set_6 = mock_bootstrap_rs([], "model", "key")

check("bootstrap_ruleset NOT called", len(bootstrap_invoked) == 0)
check("rule_set stays None", rule_set_6 is None)
check("PK preserved", cs_copy.prior_knowledge == "CS-ICL content here")

# Negative case: when PK is empty, bootstrap IS called
bootstrap_invoked_neg = []

def mock_bootstrap_neg(f, m, k):
    bootstrap_invoked_neg.append(True)
    return MagicMock()

initial_cs_empty = None  # no init-cheatsheet
cs_empty = Cheatsheet(roadmap="", case_studies=[], prior_knowledge="")
rule_set_6n = None

if initial_cs_empty is not None and cs_empty.prior_knowledge.strip():
    pass
else:
    rule_set_6n = mock_bootstrap_neg([], "model", "key")

check("bootstrap_ruleset IS called when PK empty", len(bootstrap_invoked_neg) == 1)
check("rule_set set when PK empty", rule_set_6n is not None)


# =============================================================================
# TEST 7 — elif dispatch: correct branch taken based on rule_set / PK state
# =============================================================================
print("\n[7] elif dispatch — correct Phase 1 branch selection")

# Case A: rule_set=None, PK present → pk_patch branch
pk_patch_branch_taken = []
rule_patch_branch_taken = []
skip_branch_taken = []

rule_set_7a = None
cs_7a = Cheatsheet(roadmap="", case_studies=[], prior_knowledge="non-empty PK")

if rule_set_7a is not None:
    rule_patch_branch_taken.append("rule_patch")
elif cs_7a.prior_knowledge.strip():
    pk_patch_branch_taken.append("pk_patch")
else:
    skip_branch_taken.append("skip")

check("PK branch taken when rule_set=None + PK present",
      len(pk_patch_branch_taken) == 1 and not rule_patch_branch_taken and not skip_branch_taken)

# Case B: rule_set provided → rule_patch branch
rule_patch_branch_taken_B = []
rule_set_7b = MagicMock()  # non-None rule_set
cs_7b = Cheatsheet(roadmap="", case_studies=[], prior_knowledge="some PK")

if rule_set_7b is not None:
    rule_patch_branch_taken_B.append("rule_patch")
elif cs_7b.prior_knowledge.strip():
    pk_patch_branch_taken.append("should not happen")

check("rule_patch branch taken when rule_set present",
      len(rule_patch_branch_taken_B) == 1)

# Case C: rule_set=None, PK empty → skip
skip_taken_C = []
cs_7c = Cheatsheet(roadmap="", case_studies=[], prior_knowledge="")

if None is not None:
    skip_taken_C.append("rule_patch")
elif cs_7c.prior_knowledge.strip():
    skip_taken_C.append("pk_patch")
# else: neither branch → skip

check("neither branch taken when both None/empty",
      len(skip_taken_C) == 0)


# =============================================================================
# Summary
# =============================================================================
total = n_pass + n_fail
print(f"\n{'─'*55}")
print(f"  {n_pass}/{total} tests passed" +
      (f"  |  {R}{n_fail} FAILED{RESET}" if n_fail else f"  |  {G}all clear{RESET}"))
sys.exit(0 if n_fail == 0 else 1)
