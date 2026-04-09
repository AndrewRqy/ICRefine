"""
ICR_select/training/utility_gate.py — Utility-based candidate acceptance.

Replaces the separate fix-rate and regression threshold gates with a single
continuous score:

    U(c) = ΔAccS(Vmatch; c) + λ·ΔAccS(Vgap; c) − μ·RegressS(Veasy; c) − ν·chars(c)/1000

Slice semantics
---------------
Vmatch  Val items whose structural feature tokens overlap the candidate's
        feature_signature.  Built per-candidate at flush time by pure string
        matching against val_items — no API call.

Vgap    Teacher-correct / student-wrong held-out items.  Populated from the
        disagree_bin reservoir (vgap_reserve) maintained by the training loop.
        Items in the current failures set are excluded to prevent the candidate
        from being evaluated on the items that generated it.

Veasy   Previously-correct training items drawn from correct_pool.  Measures
        stability: a good candidate must not break what already works.

Scoring strategy
----------------
Instead of 2N score_batch calls (N candidates × 2 for without/with), we make
N+1 calls:
  1. One shared "without" call  — baseline for all slices, all candidates.
  2. N parallel "with" calls    — one per candidate (different cheatsheet render).

This halves the API cost vs naïve per-candidate scoring.

Fallback
--------
When Vmatch or Vgap has fewer than config.min_slice items, score_utility()
returns a UtilityResult with fell_back=True.  The caller (_process_flush)
then falls back to the existing fix_rate + regression gate pair unchanged.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from utils.case_study import CaseStudy
from utils.cheatsheet import Cheatsheet, _sig_tokens, extract_query_features
from utils.scorer import score_batch


# ---------------------------------------------------------------------------
# Config & result
# ---------------------------------------------------------------------------

@dataclass
class UtilityConfig:
    lam:       float = 0.5    # λ — weight for Vgap improvement
    mu:        float = 1.0    # μ — weight for regression on Veasy
    nu:        float = 0.1    # ν — length penalty (per 1,000 chars rendered)
    threshold: float = 0.0    # accept candidate when U > threshold
    min_slice: int   = 5      # min items per required slice (Vmatch, Vgap);
                              # below this → fell_back=True


@dataclass
class UtilityResult:
    utility:        float
    delta_vmatch:   float   # ΔAcc on Vmatch (positive = improvement)
    delta_vgap:     float   # ΔAcc on Vgap   (positive = improvement)
    regress_veasy:  float   # accuracy drop on Veasy (non-negative; 0 = stable)
    length_penalty: float   # ν · chars / 1000
    vmatch_size:    int
    vgap_size:      int
    veasy_size:     int
    fell_back:      bool = False   # True → slice too small; caller should use old gates


# ---------------------------------------------------------------------------
# Max size for the rolling Vgap reserve (maintained by training loop)
# ---------------------------------------------------------------------------

VGAP_RESERVE_MAX = 40


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def build_vmatch(candidate: CaseStudy, val_items: list[dict]) -> list[dict]:
    """
    Return val_items whose structural feature tokens have any overlap with
    the candidate's feature_signature token set.

    Pure string matching — no API call.  Returns [] when the candidate
    has no feature_signature (caller detects this via len < min_slice).
    """
    if not candidate.feature_signature:
        return []
    cs_tokens = _sig_tokens(candidate.feature_signature)
    if not cs_tokens:
        return []

    matched = []
    for item in val_items:
        try:
            q_tokens = extract_query_features(item).tokens()
        except Exception:
            continue
        if q_tokens & cs_tokens:
            matched.append(item)
    return matched


def _tag(items: list[dict], slice_name: str) -> list[dict]:
    """Shallow-copy items, adding a _slice tag so results can be split."""
    return [{**it, "_slice": slice_name} for it in items]


def _acc_for_slice(
    correct: list[dict],
    all_scored: list[dict],
    slice_name: str,
) -> tuple[float, int]:
    """Return (accuracy, n_total) for items tagged with slice_name."""
    n_correct = sum(1 for it in correct    if it.get("_slice") == slice_name)
    n_total   = sum(1 for it in all_scored if it.get("_slice") == slice_name)
    acc = n_correct / n_total if n_total > 0 else 0.0
    return acc, n_total


# ---------------------------------------------------------------------------
# Baseline scoring (shared across all candidates in a flush)
# ---------------------------------------------------------------------------

def score_baseline(
    vmatch: list[dict],
    vgap:   list[dict],
    veasy:  list[dict],
    cheatsheet_text: str,
    model:  str,
    api_key: str,
    concurrency:      int = 10,
    reasoning_effort: str | None = "low",
    cot_first:        bool = True,
) -> dict[str, tuple[float, int]]:
    """
    Score all slices against the current cheatsheet (without any candidate).

    Returns {slice_name: (accuracy, n_total)} for vmatch, vgap, veasy.
    Called once per flush, shared across all N candidates.
    """
    tagged = _tag(vmatch, "vmatch") + _tag(vgap, "vgap") + _tag(veasy, "veasy")
    if not tagged:
        return {"vmatch": (0.0, 0), "vgap": (0.0, 0), "veasy": (0.0, 0)}

    correct, wrong = score_batch(
        tagged, cheatsheet_text, model, api_key,
        concurrency=concurrency,
        reasoning_effort=reasoning_effort,
        cot_first=cot_first,
        progress_label="utility-baseline",
    )
    all_scored = correct + wrong
    return {
        "vmatch": _acc_for_slice(correct, all_scored, "vmatch"),
        "vgap":   _acc_for_slice(correct, all_scored, "vgap"),
        "veasy":  _acc_for_slice(correct, all_scored, "veasy"),
    }


# ---------------------------------------------------------------------------
# Per-candidate utility computation
# ---------------------------------------------------------------------------

def score_utility_one(
    candidate:        CaseStudy,
    cheatsheet:       Cheatsheet,
    vmatch:           list[dict],
    vgap:             list[dict],
    veasy:            list[dict],
    baseline:         dict[str, tuple[float, int]],   # from score_baseline()
    config:           UtilityConfig,
    model:            str,
    api_key:          str,
    concurrency:      int = 10,
    reasoning_effort: str | None = "low",
    cot_first:        bool = True,
) -> UtilityResult:
    """
    Compute U(c) for one candidate given the pre-computed baseline.

    Makes one score_batch call (cheatsheet + candidate) and computes
    the delta against the shared baseline.
    """
    # Check slice sizes before paying any API cost
    vm_acc_wo, vm_n = baseline["vmatch"]
    vg_acc_wo, vg_n = baseline["vgap"]
    ve_acc_wo, ve_n = baseline["veasy"]

    if vm_n < config.min_slice or vg_n < config.min_slice:
        return UtilityResult(
            utility=0.0, delta_vmatch=0.0, delta_vgap=0.0,
            regress_veasy=0.0, length_penalty=0.0,
            vmatch_size=vm_n, vgap_size=vg_n, veasy_size=ve_n,
            fell_back=True,
        )

    # Build cheatsheet text with candidate appended
    cs_with = Cheatsheet(
        roadmap=cheatsheet.roadmap,
        case_studies=[*cheatsheet.case_studies, candidate],
        prior_knowledge=cheatsheet.prior_knowledge,
    )
    text_with = cs_with.render()

    tagged = _tag(vmatch, "vmatch") + _tag(vgap, "vgap") + _tag(veasy, "veasy")
    correct_w, wrong_w = score_batch(
        tagged, text_with, model, api_key,
        concurrency=concurrency,
        reasoning_effort=reasoning_effort,
        cot_first=cot_first,
    )
    all_w = correct_w + wrong_w

    vm_acc_w, _ = _acc_for_slice(correct_w, all_w, "vmatch")
    vg_acc_w, _ = _acc_for_slice(correct_w, all_w, "vgap")
    ve_acc_w, _ = _acc_for_slice(correct_w, all_w, "veasy")

    delta_vmatch  = vm_acc_w - vm_acc_wo
    delta_vgap    = vg_acc_w - vg_acc_wo
    regress_veasy = max(0.0, ve_acc_wo - ve_acc_w)
    length_penalty = config.nu * len(candidate.render()) / 1000.0

    utility = (
        delta_vmatch
        + config.lam * delta_vgap
        - config.mu  * regress_veasy
        - length_penalty
    )

    return UtilityResult(
        utility=utility,
        delta_vmatch=delta_vmatch,
        delta_vgap=delta_vgap,
        regress_veasy=regress_veasy,
        length_penalty=length_penalty,
        vmatch_size=vm_n,
        vgap_size=vg_n,
        veasy_size=ve_n,
    )


# ---------------------------------------------------------------------------
# Batch utility scoring — N candidates, N+1 API calls total
# ---------------------------------------------------------------------------

def score_utility_batch(
    candidates:       list[CaseStudy],
    cheatsheet:       Cheatsheet,
    vmatch:           list[dict],
    vgap:             list[dict],
    veasy:            list[dict],
    config:           UtilityConfig,
    model:            str,
    api_key:          str,
    concurrency:      int = 10,
    reasoning_effort: str | None = "low",
    cot_first:        bool = True,
    log_fn=None,
) -> list[UtilityResult]:
    """
    Compute U(c) for each candidate.  N+1 score_batch calls total:
      1 shared baseline call (without any candidate)
      N parallel "with" calls (one per candidate)

    Returns results in the same order as candidates.
    Falls back gracefully (fell_back=True) when slices are too small.
    """
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)
        else:
            print(msg, file=sys.stderr, flush=True)

    # Step 1: shared baseline (one call)
    baseline = score_baseline(
        vmatch, vgap, veasy,
        cheatsheet_text=cheatsheet.render(),
        model=model, api_key=api_key,
        concurrency=concurrency,
        reasoning_effort=reasoning_effort,
        cot_first=cot_first,
    )
    vm_n = baseline["vmatch"][1]
    vg_n = baseline["vgap"][1]
    ve_n = baseline["veasy"][1]
    _log(
        f"  [gate:utility] slices: vmatch={vm_n}  vgap={vg_n}  "
        f"veasy={ve_n}  min_slice={config.min_slice}"
    )

    # Early-exit if slices too small — all candidates fall back
    if vm_n < config.min_slice or vg_n < config.min_slice:
        _log(
            f"  [gate:utility] slices too small → falling back to "
            f"fix_rate+regression gates for all candidates"
        )
        return [
            UtilityResult(
                utility=0.0, delta_vmatch=0.0, delta_vgap=0.0,
                regress_veasy=0.0, length_penalty=0.0,
                vmatch_size=vm_n, vgap_size=vg_n, veasy_size=ve_n,
                fell_back=True,
            )
            for _ in candidates
        ]

    # Step 2: N parallel "with" calls
    results: list[UtilityResult | None] = [None] * len(candidates)

    def _eval(args: tuple[int, CaseStudy]) -> tuple[int, UtilityResult]:
        i, cand = args
        r = score_utility_one(
            cand, cheatsheet, vmatch, vgap, veasy, baseline, config,
            model, api_key, concurrency, reasoning_effort, cot_first,
        )
        return i, r

    with ThreadPoolExecutor(max_workers=len(candidates)) as ex:
        futures = {ex.submit(_eval, (i, c)): i for i, c in enumerate(candidates)}
        for fut in as_completed(futures):
            i, r = fut.result()
            results[i] = r
            _log(
                f"  [gate:utility] candidate {i+1}: "
                f"U={r.utility:+.4f}  "
                f"ΔVmatch={r.delta_vmatch:+.2%}  "
                f"ΔVgap={r.delta_vgap:+.2%}  "
                f"Regress={r.regress_veasy:.2%}  "
                f"len={r.length_penalty:.3f}"
            )

    return [r for r in results if r is not None]
