"""ICR_sfcr/rule_validator.py — Validate candidate rules via U_LCB scoring.

For each candidate rule, the validator:
  1. Constructs anchor_cheatsheet + candidate rule as a single text.
  2. Scores the gate split under this combined cheatsheet for each model.
  3. Computes per-model metrics relative to the gate baseline.
  4. Applies either the count-aware pilot gate or the U_LCB gate depending on
     region sizes, then records a failure_profile for the repair loop.

Gate choice (count-aware vs U_LCB):
  If |V_private| < MIN_PRIVATE or |V_easy| < MIN_EASY, Wilson UCBs become
  uninformative.  In that regime we fall back to a stricter count-based gate:
    fixed_shared_count >= 2  (on benefit panel for the candidate's subtype)
    reg_private_count == 0
    reg_easy_count <= 1  OR  reg_easy_rate <= 5%
    private_activation_count == 0

U_LCB formula (subtype-aware, section 2 of sfcr_v2_plan.md):
  U_LCB(c) = max_g  min_{j ∈ B_g} LCB(Δ_shared_j on subtype g)
             - lambda * max_{j ∈ S} UCB(Reg_private_j)
             - mu     * max_{j ∈ S} UCB(Reg_easy_j)
             - nu     * (len(rule) / 1000)

  B_g = benefit panel = proxies with ≥ 1 baseline failure in the candidate's
        target subtype.  S = safety panel = all proxy models.

Acceptance (U_LCB gate):
  U_LCB > 0
  private_activation_rate <= 0.10
  Reg_easy_j <= 0.05  for every proxy model j
  len(rule) <= 800 characters

Acceptance (count-aware gate, triggered when denominators are too small):
  fixed_shared_count >= 2   (across benefit panel)
  reg_private_count == 0
  reg_easy_count <= 1  OR  reg_easy_rate <= 0.05
  private_activation_count == 0
  len(rule) <= 800 characters
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from utils.scorer import score_batch
from utils.task_spec import TaskSpec

from .failure_regions import FailureRegions

_RULE_SECTION = "\n\n--- ADDITIONAL RULE ---\n"

# Thresholds that switch the gate from U_LCB to count-aware
MIN_PRIVATE = 5
MIN_EASY    = 20


# ---------------------------------------------------------------------------
# Wilson confidence interval helpers
# ---------------------------------------------------------------------------

def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return (LCB, UCB) of the Wilson interval for proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _lcb(k: int, n: int) -> float:
    return _wilson(k, n)[0]


def _ucb(k: int, n: int) -> float:
    return _wilson(k, n)[1]


# ---------------------------------------------------------------------------
# Cheatsheet construction
# ---------------------------------------------------------------------------

def build_cheatsheet_with_rule(anchor: str, rule: dict) -> str:
    """Append a structured rule block to the anchor cheatsheet."""
    lines = [
        _RULE_SECTION,
        f"RULE: {rule['rule']}",
    ]
    if rule.get("use_when"):
        lines.append(f"USE WHEN: {rule['use_when']}")
    if rule.get("do_not_use_when"):
        lines.append(f"DO NOT USE WHEN: {rule['do_not_use_when']}")
    if rule.get("check"):
        lines.append(f"CHECK: {rule['check']}")
    return anchor.rstrip() + "\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProxyStats:
    # Rate-based metrics (Wilson-adjusted where denominators allow)
    delta_shared: float          # accuracy gain on V_shared_j (or subtype) for this proxy
    reg_private:  float          # regression rate on V_private for this proxy
    reg_easy:     float          # regression rate on V_easy for this proxy
    activation_rate: float       # fraction of gate items matching USE WHEN
    private_activation_rate: float  # fraction of V_private matching USE WHEN

    # Region sizes
    n_shared: int
    n_private: int
    n_easy: int

    # Absolute counts (v2)
    fixed_shared_count:      int = 0  # items in V_shared_j: baseline wrong → candidate correct
    reg_easy_count:          int = 0  # items in V_easy: correct → wrong under candidate
    reg_private_count:       int = 0  # items in V_private: correct → wrong under candidate
    private_activation_count: int = 0 # items in V_private matching USE WHEN
    activation_count:        int = 0  # total gate items matching USE WHEN


@dataclass
class ValidationResult:
    rule:                    dict
    accepted:                bool
    u_lcb:                   float          # mean U_LCB across seeds (or single-seed value)
    private_activation_rate: float
    reg_easy_worst:          float
    reject_reason:           str | None
    per_proxy_stats:         dict[str, ProxyStats] = field(default_factory=dict)
    u_lcb_per_seed:          list[float]    = field(default_factory=list)
    count_gate_used:         bool           = False  # True when count-aware gate was applied
    # Repair loop data
    failure_profile:         dict | None    = None


# ---------------------------------------------------------------------------
# Gate baseline
# ---------------------------------------------------------------------------

@dataclass
class GateBaseline:
    """
    Pre-computed per-model correctness on gate items under the anchor.
    """
    correct_by_model:   dict[str, dict[str, bool]]   # model -> id -> correct
    annotated_by_model: dict[str, dict[str, dict]]   # model -> id -> annotated item


def compute_gate_baseline(
    gate_items: list[dict],
    anchor_cheatsheet: str,
    source_model: str,
    proxy_models: list[str],
    api_key: str,
    task_spec: TaskSpec,
    concurrency: int = 30,
) -> GateBaseline:
    """Score all models on gate items under anchor (called once before validation)."""
    all_models = [source_model] + proxy_models

    def _score_one(model: str):
        short = model.split("/")[-1]
        c, w = score_batch(
            gate_items, anchor_cheatsheet, model, api_key,
            concurrency=concurrency, temperature=0.0,
            progress_label=f"[gate-baseline] {short}",
            reasoning_effort=None, cot_first=True, task_spec=task_spec,
        )
        return model, c, w

    correct_by_model:   dict[str, dict[str, bool]] = {}
    annotated_by_model: dict[str, dict[str, dict]] = {}

    with ThreadPoolExecutor(max_workers=len(all_models)) as pool:
        for model, c, w in pool.map(_score_one, all_models):
            cb: dict[str, bool] = {}
            ann: dict[str, dict] = {}
            for it in c:
                iid = it["_sfcr_id"]
                cb[iid]  = True
                ann[iid] = it
            for it in w:
                iid = it["_sfcr_id"]
                cb[iid]  = False
                ann[iid] = it
            correct_by_model[model]   = cb
            annotated_by_model[model] = ann

    return GateBaseline(
        correct_by_model=correct_by_model,
        annotated_by_model=annotated_by_model,
    )


# ---------------------------------------------------------------------------
# Gate-split region membership
# ---------------------------------------------------------------------------

def _gate_regions(
    gate_items: list[dict],
    baseline: GateBaseline,
    source_model: str,
    proxy_models: list[str],
) -> dict:
    """
    Derive failure-region id-sets from the gate baseline.
    """
    src_cb = baseline.correct_by_model[source_model]
    F_s = {it["_sfcr_id"] for it in gate_items if not src_cb.get(it["_sfcr_id"], False)}
    K_s = {it["_sfcr_id"] for it in gate_items if src_cb.get(it["_sfcr_id"], False)}

    proxy_correct_inter: set | None = None
    proxy_fail: dict[str, set] = {}
    for pm in proxy_models:
        pm_cb = baseline.correct_by_model[pm]
        pm_correct = {it["_sfcr_id"] for it in gate_items if pm_cb.get(it["_sfcr_id"], False)}
        pm_wrong   = {it["_sfcr_id"] for it in gate_items if not pm_cb.get(it["_sfcr_id"], False)}
        proxy_fail[pm] = pm_wrong
        if proxy_correct_inter is None:
            proxy_correct_inter = pm_correct.copy()
        else:
            proxy_correct_inter &= pm_correct
    if proxy_correct_inter is None:
        proxy_correct_inter = set()

    V_private = F_s & proxy_correct_inter
    V_easy    = K_s & proxy_correct_inter

    # Per-proxy V_shared_j: source fails AND this proxy fails
    V_shared_per_proxy = {pm: F_s & proxy_fail[pm] for pm in proxy_models}

    return {
        "F_s": F_s, "K_s": K_s,
        "V_private": V_private, "V_easy": V_easy,
        "V_shared_per_proxy": V_shared_per_proxy,
    }


# ---------------------------------------------------------------------------
# Activation rate (simple keyword match — V1)
# ---------------------------------------------------------------------------

def _activation_rate_for_ids(
    item_ids: set,
    gate_items: list[dict],
    use_when: str,
) -> tuple[float, int]:
    """Fraction + absolute count of items (by id) whose input matches USE WHEN."""
    from .activation import matches_use_when
    if not item_ids:
        return 0.0, 0
    targets = [it for it in gate_items if it["_sfcr_id"] in item_ids]
    if not targets:
        return 0.0, 0
    hits = sum(1 for it in targets if matches_use_when(use_when, it.get("input", "")))
    return hits / len(targets), hits


# ---------------------------------------------------------------------------
# Benefit panel helper
# ---------------------------------------------------------------------------

def _benefit_panel(
    proxy_models: list[str],
    V_shared_per_proxy: dict[str, set],
    subtype_ids: set | None,
) -> list[str]:
    """
    Return proxies in the benefit panel for the candidate's target subtype.

    The benefit panel B_g contains proxies that have at least one baseline
    failure in the target subtype.  If no subtype info is available, return
    all proxy models (legacy behaviour).
    """
    if not subtype_ids:
        return proxy_models
    panel = [pm for pm in proxy_models
             if V_shared_per_proxy.get(pm, set()) & subtype_ids]
    return panel if panel else proxy_models  # guard: at least one proxy


# ---------------------------------------------------------------------------
# Validate a single candidate
# ---------------------------------------------------------------------------

def _validate_one(
    rule: dict,
    gate_items: list[dict],
    gate_baseline: GateBaseline,
    gate_region_ids: dict,
    anchor_cheatsheet: str,
    source_model: str,
    proxy_models: list[str],
    acceptance_proxies: list[str],
    api_key: str,
    task_spec: TaskSpec,
    concurrency: int,
    lambda_w: float,
    mu_w: float,
    nu_w: float,
    private_activation_ceiling: float,
    reg_easy_ceiling: float,
    max_rule_chars: int,
) -> ValidationResult:
    # Hard gate: rule length
    if len(rule["rule"]) > max_rule_chars:
        return ValidationResult(
            rule=rule, accepted=False, u_lcb=-999.0,
            private_activation_rate=0.0, reg_easy_worst=0.0,
            reject_reason=f"rule too long ({len(rule['rule'])} > {max_rule_chars} chars)",
        )

    cs_with_rule = build_cheatsheet_with_rule(anchor_cheatsheet, rule)

    # Score gate items under anchor+rule for all proxy models in parallel
    def _score_proxy(model: str):
        short = model.split("/")[-1]
        c, w = score_batch(
            gate_items, cs_with_rule, model, api_key,
            concurrency=concurrency, temperature=0.0,
            progress_label=f"[validate] {short}",
            reasoning_effort=None, cot_first=True, task_spec=task_spec,
        )
        return model, {it["_sfcr_id"]: True for it in c} | {it["_sfcr_id"]: False for it in w}

    all_eval = [source_model] + proxy_models
    with ThreadPoolExecutor(max_workers=len(all_eval)) as pool:
        cand_results: dict[str, dict[str, bool]] = {}
        for model, cb in pool.map(_score_proxy, all_eval):
            cand_results[model] = cb

    V_private      = gate_region_ids["V_private"]
    V_easy         = gate_region_ids["V_easy"]
    V_shared_pp    = gate_region_ids["V_shared_per_proxy"]
    baseline_cb    = gate_baseline.correct_by_model

    # Subtype target ids from candidate (if clustered generation was used)
    subtype_ids: set | None = (
        set(rule["subtype_items"]) if rule.get("subtype_items") else None
    )

    per_proxy_stats: dict[str, ProxyStats] = {}

    for pm in proxy_models:
        pm_baseline = baseline_cb[pm]
        pm_cand     = cand_results[pm]

        # V_shared_j for delta: restricted to candidate's subtype if available
        v_shared_j_full = V_shared_pp.get(pm, set())
        v_shared_j = (v_shared_j_full & subtype_ids) if subtype_ids else v_shared_j_full

        n_shared = len(v_shared_j)
        if n_shared > 0:
            # k0: proxy was wrong at baseline (should be all of v_shared_j by definition)
            k0 = sum(1 for iid in v_shared_j if not pm_baseline.get(iid, True))
            k1 = sum(1 for iid in v_shared_j if pm_cand.get(iid, False))
            fixed = k1  # items that moved wrong→correct
            delta_shared = _lcb(k1, n_shared) - (k0 / n_shared)
        else:
            fixed = 0
            delta_shared = 0.0

        # Reg_private_j
        n_private = len(V_private)
        k_reg_prv = sum(
            1 for iid in V_private
            if pm_baseline.get(iid, False) and not pm_cand.get(iid, False)
        )
        if n_private >= MIN_PRIVATE:
            reg_private = _ucb(k_reg_prv, n_private)
        elif n_private > 0:
            reg_private = k_reg_prv / n_private
        else:
            reg_private = 0.0

        # Reg_easy_j
        n_easy = len(V_easy)
        k_reg_easy = sum(
            1 for iid in V_easy
            if pm_baseline.get(iid, False) and not pm_cand.get(iid, False)
        )
        reg_easy = _ucb(k_reg_easy, n_easy) if n_easy > 0 else 0.0

        # Activation stats
        all_gate_ids = {it["_sfcr_id"] for it in gate_items}
        act_rate, act_count  = _activation_rate_for_ids(all_gate_ids, gate_items, rule["use_when"])
        prv_act_rate, prv_act_count = _activation_rate_for_ids(V_private, gate_items, rule["use_when"])

        per_proxy_stats[pm] = ProxyStats(
            delta_shared=delta_shared,
            reg_private=reg_private,
            reg_easy=reg_easy,
            activation_rate=act_rate,
            private_activation_rate=prv_act_rate,
            n_shared=n_shared,
            n_private=n_private,
            n_easy=n_easy,
            fixed_shared_count=fixed,
            reg_easy_count=k_reg_easy,
            reg_private_count=k_reg_prv,
            private_activation_count=prv_act_count,
            activation_count=act_count,
        )

    # ── Build benefit panel for U_LCB ─────────────────────────────────────
    benefit_panel = _benefit_panel(acceptance_proxies, V_shared_pp, subtype_ids)
    # Safety panel = all acceptance proxies
    safety_panel  = acceptance_proxies

    ap_stats    = [per_proxy_stats[pm] for pm in acceptance_proxies if pm in per_proxy_stats]
    bp_stats    = [per_proxy_stats[pm] for pm in benefit_panel if pm in per_proxy_stats]
    if not ap_stats:
        return ValidationResult(
            rule=rule, accepted=False, u_lcb=-999.0,
            private_activation_rate=0.0, reg_easy_worst=0.0,
            reject_reason="no acceptance proxy stats available",
        )

    # ── Choose gate regime ─────────────────────────────────────────────────
    n_prv = next(iter(ap_stats)).n_private
    n_esy = next(iter(ap_stats)).n_easy
    use_count_gate = (n_prv < MIN_PRIVATE or n_esy < MIN_EASY)

    # Aggregated safety metrics (worst-case over safety panel)
    max_reg_private  = max((s.reg_private  for s in ap_stats), default=0.0)
    max_reg_easy     = max((s.reg_easy     for s in ap_stats), default=0.0)
    private_act_rate = max((s.private_activation_rate for s in ap_stats), default=0.0)
    reg_easy_worst   = max_reg_easy

    # U_LCB (subtype-aware: max benefit over proxy benefit panel)
    min_delta_benefit = min((s.delta_shared for s in bp_stats), default=0.0)
    length_cost       = len(rule["rule"]) / 1000.0
    u_lcb = (
        min_delta_benefit
        - lambda_w * max_reg_private
        - mu_w     * max_reg_easy
        - nu_w     * length_cost
    )

    # Count-gate aggregates
    max_fixed_shared       = max((s.fixed_shared_count      for s in bp_stats), default=0)
    total_reg_private_cnt  = max((s.reg_private_count       for s in ap_stats), default=0)
    total_reg_easy_cnt     = max((s.reg_easy_count          for s in ap_stats), default=0)
    total_prv_act_cnt      = max((s.private_activation_count for s in ap_stats), default=0)

    # ── Acceptance decision ────────────────────────────────────────────────
    reject_reason = None

    if use_count_gate:
        # Count-aware pilot gate
        if max_fixed_shared < 2:
            reject_reason = (
                f"count-gate: fixed_shared_count={max_fixed_shared} < 2 "
                f"(|V_private|={n_prv}, |V_easy|={n_esy})"
            )
        elif total_reg_private_cnt > 0:
            reject_reason = (
                f"count-gate: reg_private_count={total_reg_private_cnt} > 0"
            )
        elif total_reg_easy_cnt > 1 and max_reg_easy > reg_easy_ceiling:
            reject_reason = (
                f"count-gate: reg_easy_count={total_reg_easy_cnt} > 1 "
                f"and reg_easy_rate={max_reg_easy:.2%} > {reg_easy_ceiling:.0%}"
            )
        elif total_prv_act_cnt > 0:
            reject_reason = (
                f"count-gate: private_activation_count={total_prv_act_cnt} > 0"
            )
    else:
        # U_LCB gate
        if u_lcb <= 0:
            reject_reason = f"U_LCB={u_lcb:.4f} ≤ 0"
        elif private_act_rate > private_activation_ceiling:
            reject_reason = (
                f"private_activation_rate={private_act_rate:.2%} "
                f"> {private_activation_ceiling:.0%}"
            )
        elif reg_easy_worst > reg_easy_ceiling:
            reject_reason = (
                f"reg_easy_worst={reg_easy_worst:.2%} > {reg_easy_ceiling:.0%}"
            )

    accepted = reject_reason is None

    # ── Build failure_profile for repair loop ─────────────────────────────
    failure_profile: dict | None = None
    if not accepted:
        ann_by_model = gate_baseline.annotated_by_model

        # Items in V_easy or V_private that the candidate mis-triggered
        mis_easy_items: list[dict] = []
        mis_prv_items:  list[dict] = []
        no_gain_proxies: list[str] = []

        for pm in acceptance_proxies:
            s = per_proxy_stats.get(pm)
            if s is None:
                continue
            if s.reg_easy_count > 0:
                ann = ann_by_model.get(pm, {})
                mis_easy_items = [ann[iid] for iid in V_easy
                                  if baseline_cb[pm].get(iid, False)
                                  and not cand_results[pm].get(iid, False)
                                  and iid in ann][:4]
            if s.private_activation_count > 0:
                ann = ann_by_model.get(pm, {})
                mis_prv_items = [ann[iid] for iid in V_private
                                 if iid in ann][:4]
            if s.fixed_shared_count == 0:
                no_gain_proxies.append(pm.split("/")[-1])

        failure_profile = {
            "reject_reason":         reject_reason,
            "mis_triggered_easy":    mis_easy_items,
            "mis_triggered_private": mis_prv_items,
            "no_gain_proxies":       no_gain_proxies,
        }

    return ValidationResult(
        rule=rule,
        accepted=accepted,
        u_lcb=u_lcb,
        private_activation_rate=private_act_rate,
        reg_easy_worst=reg_easy_worst,
        reject_reason=reject_reason,
        per_proxy_stats=per_proxy_stats,
        count_gate_used=use_count_gate,
        failure_profile=failure_profile,
    )


# ---------------------------------------------------------------------------
# Validate all candidates
# ---------------------------------------------------------------------------

def validate_candidates(
    candidates: list[dict],
    gate_items: list[dict],
    gate_baseline: GateBaseline,
    anchor_cheatsheet: str,
    source_model: str,
    proxy_models: list[str],
    held_out_target: str | None,
    api_key: str,
    task_spec: TaskSpec,
    concurrency: int = 30,
    lambda_w: float = 1.0,
    mu_w: float = 1.0,
    nu_w: float = 0.05,
    max_accepted: int = 3,
    private_activation_ceiling: float = 0.10,
    reg_easy_ceiling: float = 0.05,
    max_rule_chars: int = 800,
    repair_fn: Callable | None = None,
    repair_attempts: int = 1,
) -> list[ValidationResult]:
    """
    Validate all candidates and return results sorted by U_LCB descending.

    `held_out_target` is a substring matched against model names to exclude
    one family from U_LCB computation (leave-one-out eval protocol).

    `repair_fn` is called for rejected candidates before final discard.
    Signature: repair_fn(rule, failure_profile) -> dict | None.
    If it returns a new rule dict the repaired candidate is validated once more.
    Set repair_attempts to control how many repair rounds are allowed.
    """
    if held_out_target:
        acceptance_proxies = [
            pm for pm in proxy_models
            if held_out_target.lower() not in pm.lower()
        ]
        excluded = [pm for pm in proxy_models if pm not in acceptance_proxies]
        if excluded:
            print(f"[validate] held-out target '{held_out_target}': "
                  f"excluded from acceptance: {[m.split('/')[-1] for m in excluded]}")
    else:
        acceptance_proxies = proxy_models

    gate_region_ids = _gate_regions(gate_items, gate_baseline, source_model, proxy_models)
    print(
        f"[validate] gate regions: "
        f"|V_shared_per_proxy|=[{', '.join(str(len(v)) for v in gate_region_ids['V_shared_per_proxy'].values())}]  "
        f"|V_private|={len(gate_region_ids['V_private'])}  "
        f"|V_easy|={len(gate_region_ids['V_easy'])}"
    )

    common_kwargs = dict(
        gate_items=gate_items,
        gate_baseline=gate_baseline,
        gate_region_ids=gate_region_ids,
        anchor_cheatsheet=anchor_cheatsheet,
        source_model=source_model,
        proxy_models=proxy_models,
        acceptance_proxies=acceptance_proxies,
        api_key=api_key,
        task_spec=task_spec,
        concurrency=concurrency,
        lambda_w=lambda_w,
        mu_w=mu_w,
        nu_w=nu_w,
        private_activation_ceiling=private_activation_ceiling,
        reg_easy_ceiling=reg_easy_ceiling,
        max_rule_chars=max_rule_chars,
    )

    results: list[ValidationResult] = []
    n_accepted = 0

    for i, rule in enumerate(candidates):
        print(f"\n[validate] candidate {i+1}/{len(candidates)}: "
              f"{rule['rule'][:60]}...")

        if n_accepted >= max_accepted:
            print(f"[validate] max_accepted={max_accepted} reached — skipping remaining candidates")
            results.append(ValidationResult(
                rule=rule, accepted=False, u_lcb=-999.0,
                private_activation_rate=0.0, reg_easy_worst=0.0,
                reject_reason="max_accepted reached",
            ))
            continue

        result = _validate_one(rule=rule, **common_kwargs)

        # ── Repair loop ────────────────────────────────────────────────────
        if not result.accepted and repair_fn is not None and result.failure_profile is not None:
            for rep in range(repair_attempts):
                repaired_rule = repair_fn(rule, result.failure_profile)
                if repaired_rule is None:
                    break
                print(f"[validate] repair attempt {rep+1}: re-validating repaired candidate...")
                result = _validate_one(rule=repaired_rule, **common_kwargs)
                if result.accepted:
                    print(f"[validate] repaired candidate ACCEPTED after {rep+1} repair(s)")
                    break
                rule = repaired_rule  # try repairing the already-repaired version

        status = "ACCEPTED" if result.accepted else f"REJECTED ({result.reject_reason})"
        gate_tag = " [count-gate]" if result.count_gate_used else ""
        print(
            f"[validate] {status}{gate_tag}  U_LCB={result.u_lcb:.4f}  "
            f"prv_act={result.private_activation_rate:.2%}  "
            f"reg_easy={result.reg_easy_worst:.2%}"
        )

        results.append(result)
        if result.accepted:
            n_accepted += 1

    results.sort(key=lambda r: r.u_lcb, reverse=True)

    print(
        f"\n[validate] done: {n_accepted}/{len(candidates)} candidates accepted "
        f"(max_accepted={max_accepted})"
    )
    return results


# ---------------------------------------------------------------------------
# Multi-seed validation
# ---------------------------------------------------------------------------

def validate_candidates_multiseed(
    candidates: list[dict],
    gate_splits: list[tuple[list[dict], GateBaseline]],
    anchor_cheatsheet: str,
    source_model: str,
    proxy_models: list[str],
    held_out_target: str | None,
    api_key: str,
    task_spec: TaskSpec,
    concurrency: int = 30,
    lambda_w: float = 1.0,
    mu_w: float = 1.0,
    nu_w: float = 0.05,
    max_accepted: int = 3,
    private_activation_ceiling: float = 0.10,
    reg_easy_ceiling: float = 0.05,
    max_rule_chars: int = 800,
    repair_fn: Callable | None = None,
    repair_attempts: int = 1,
) -> list[ValidationResult]:
    """
    Validate candidates across multiple gate splits and average U_LCB.

    Hard gates (private_activation_rate, reg_easy, count-gate) use worst case
    across seeds so a rule that fails on any seed is still rejected.

    `repair_fn` and `repair_attempts` are forwarded to the single-seed logic
    after aggregation: if mean_u_lcb <= 0 due to one bad seed, a repair is
    attempted before final discard.
    """
    if held_out_target:
        acceptance_proxies = [
            pm for pm in proxy_models
            if held_out_target.lower() not in pm.lower()
        ]
        excluded = [pm for pm in proxy_models if pm not in acceptance_proxies]
        if excluded:
            print(
                f"[validate-ms] held-out '{held_out_target}': "
                f"excluded from acceptance: {[m.split('/')[-1] for m in excluded]}"
            )
    else:
        acceptance_proxies = proxy_models

    n_seeds = len(gate_splits)
    print(f"[validate-ms] {len(candidates)} candidates × {n_seeds} gate seeds")

    gate_region_ids_per_seed = [
        _gate_regions(gate_items, baseline, source_model, proxy_models)
        for gate_items, baseline in gate_splits
    ]

    results: list[ValidationResult] = []
    n_accepted = 0

    for i, rule in enumerate(candidates):
        print(f"\n[validate-ms] candidate {i+1}/{len(candidates)}: "
              f"{rule['rule'][:60]}...")

        if n_accepted >= max_accepted:
            results.append(ValidationResult(
                rule=rule, accepted=False, u_lcb=-999.0,
                private_activation_rate=0.0, reg_easy_worst=0.0,
                reject_reason="max_accepted reached",
            ))
            continue

        if len(rule["rule"]) > max_rule_chars:
            results.append(ValidationResult(
                rule=rule, accepted=False, u_lcb=-999.0,
                private_activation_rate=0.0, reg_easy_worst=0.0,
                reject_reason=f"rule too long ({len(rule['rule'])} > {max_rule_chars} chars)",
            ))
            continue

        def _validate_on_seed(s_idx: int) -> ValidationResult:
            gate_items, gate_baseline = gate_splits[s_idx]
            return _validate_one(
                rule=rule,
                gate_items=gate_items,
                gate_baseline=gate_baseline,
                gate_region_ids=gate_region_ids_per_seed[s_idx],
                anchor_cheatsheet=anchor_cheatsheet,
                source_model=source_model,
                proxy_models=proxy_models,
                acceptance_proxies=acceptance_proxies,
                api_key=api_key,
                task_spec=task_spec,
                concurrency=concurrency,
                lambda_w=lambda_w,
                mu_w=mu_w,
                nu_w=nu_w,
                private_activation_ceiling=1.0,  # applied at aggregate level below
                reg_easy_ceiling=1.0,
                max_rule_chars=max_rule_chars + 1,
            )

        seed_results = [_validate_on_seed(s) for s in range(n_seeds)]
        for s_idx, sr in enumerate(seed_results):
            print(
                f"  [seed {s_idx+1}/{n_seeds}] U_LCB={sr.u_lcb:.4f}  "
                f"prv_act={sr.private_activation_rate:.2%}  "
                f"reg_easy={sr.reg_easy_worst:.2%}"
                + (" [count-gate]" if sr.count_gate_used else "")
            )

        u_lcbs       = [sr.u_lcb for sr in seed_results]
        mean_u_lcb   = sum(u_lcbs) / len(u_lcbs)
        max_prv_act  = max(sr.private_activation_rate for sr in seed_results)
        max_reg_easy = max(sr.reg_easy_worst for sr in seed_results)
        any_count_gate = any(sr.count_gate_used for sr in seed_results)

        # Count-gate aggregates (worst case)
        max_fixed   = max((max(s.fixed_shared_count for s in sr.per_proxy_stats.values())
                          for sr in seed_results if sr.per_proxy_stats), default=0)
        max_reg_prv = max((max(s.reg_private_count for s in sr.per_proxy_stats.values())
                          for sr in seed_results if sr.per_proxy_stats), default=0)
        max_reg_esy = max((max(s.reg_easy_count for s in sr.per_proxy_stats.values())
                          for sr in seed_results if sr.per_proxy_stats), default=0)
        max_prv_act_cnt = max((max(s.private_activation_count for s in sr.per_proxy_stats.values())
                               for sr in seed_results if sr.per_proxy_stats), default=0)

        # Merge per_proxy_stats: average numeric fields across seeds
        merged_stats: dict[str, ProxyStats] = {}
        for pm in proxy_models:
            per_seed_pm = [sr.per_proxy_stats[pm] for sr in seed_results if pm in sr.per_proxy_stats]
            if per_seed_pm:
                merged_stats[pm] = ProxyStats(
                    delta_shared            = sum(s.delta_shared for s in per_seed_pm) / len(per_seed_pm),
                    reg_private             = sum(s.reg_private  for s in per_seed_pm) / len(per_seed_pm),
                    reg_easy                = sum(s.reg_easy     for s in per_seed_pm) / len(per_seed_pm),
                    activation_rate         = sum(s.activation_rate         for s in per_seed_pm) / len(per_seed_pm),
                    private_activation_rate = sum(s.private_activation_rate for s in per_seed_pm) / len(per_seed_pm),
                    n_shared  = sum(s.n_shared  for s in per_seed_pm),
                    n_private = sum(s.n_private for s in per_seed_pm),
                    n_easy    = sum(s.n_easy    for s in per_seed_pm),
                    fixed_shared_count       = max(s.fixed_shared_count       for s in per_seed_pm),
                    reg_easy_count           = max(s.reg_easy_count           for s in per_seed_pm),
                    reg_private_count        = max(s.reg_private_count        for s in per_seed_pm),
                    private_activation_count = max(s.private_activation_count for s in per_seed_pm),
                    activation_count         = max(s.activation_count         for s in per_seed_pm),
                )

        # Apply aggregated acceptance decision
        reject_reason = None
        if any_count_gate:
            if max_fixed < 2:
                reject_reason = f"count-gate: max fixed_shared={max_fixed} < 2 across seeds"
            elif max_reg_prv > 0:
                reject_reason = f"count-gate: reg_private_count={max_reg_prv} > 0"
            elif max_reg_esy > 1 and max_reg_easy > reg_easy_ceiling:
                reject_reason = (
                    f"count-gate: reg_easy_count={max_reg_esy} > 1 "
                    f"and reg_easy_rate={max_reg_easy:.2%} > {reg_easy_ceiling:.0%}"
                )
            elif max_prv_act_cnt > 0:
                reject_reason = f"count-gate: private_activation_count={max_prv_act_cnt} > 0"
        else:
            if mean_u_lcb <= 0:
                reject_reason = (
                    f"mean U_LCB={mean_u_lcb:.4f} ≤ 0  "
                    f"(per-seed: [{', '.join(f'{v:.4f}' for v in u_lcbs)}])"
                )
            elif max_prv_act > private_activation_ceiling:
                reject_reason = (
                    f"max private_activation_rate={max_prv_act:.2%} "
                    f"> {private_activation_ceiling:.0%}"
                )
            elif max_reg_easy > reg_easy_ceiling:
                reject_reason = (
                    f"max reg_easy={max_reg_easy:.2%} > {reg_easy_ceiling:.0%}"
                )

        accepted = reject_reason is None

        # Build failure_profile for repair
        failure_profile: dict | None = None
        if not accepted:
            fp_from_seed = next(
                (sr.failure_profile for sr in seed_results if sr.failure_profile), None
            )
            if fp_from_seed:
                failure_profile = {**fp_from_seed, "reject_reason": reject_reason}

        # Repair loop (post-aggregation)
        if not accepted and repair_fn is not None and failure_profile is not None:
            for rep in range(repair_attempts):
                repaired_rule = repair_fn(rule, failure_profile)
                if repaired_rule is None:
                    break
                print(f"[validate-ms] repair attempt {rep+1}: re-validating repaired candidate across {n_seeds} seeds...")
                rep_seed_results = [_validate_one(
                    rule=repaired_rule,
                    gate_items=gate_splits[s][0],
                    gate_baseline=gate_splits[s][1],
                    gate_region_ids=gate_region_ids_per_seed[s],
                    anchor_cheatsheet=anchor_cheatsheet,
                    source_model=source_model,
                    proxy_models=proxy_models,
                    acceptance_proxies=acceptance_proxies,
                    api_key=api_key,
                    task_spec=task_spec,
                    concurrency=concurrency,
                    lambda_w=lambda_w, mu_w=mu_w, nu_w=nu_w,
                    private_activation_ceiling=1.0,
                    reg_easy_ceiling=1.0,
                    max_rule_chars=max_rule_chars + 1,
                ) for s in range(n_seeds)]
                rep_mean_u = sum(sr.u_lcb for sr in rep_seed_results) / n_seeds
                rep_max_prv = max(sr.private_activation_rate for sr in rep_seed_results)
                rep_max_easy = max(sr.reg_easy_worst for sr in rep_seed_results)
                if (rep_mean_u > 0
                        and rep_max_prv <= private_activation_ceiling
                        and rep_max_easy <= reg_easy_ceiling):
                    reject_reason = None
                    accepted = True
                    mean_u_lcb = rep_mean_u
                    max_prv_act = rep_max_prv
                    max_reg_easy = rep_max_easy
                    rule = repaired_rule
                    print(f"[validate-ms] repaired candidate ACCEPTED after {rep+1} repair(s)")
                    break
                rule = repaired_rule

        status = "ACCEPTED" if accepted else f"REJECTED ({reject_reason})"
        gate_tag = " [count-gate]" if any_count_gate else ""
        print(
            f"[validate-ms] {status}{gate_tag}  mean_U_LCB={mean_u_lcb:.4f}  "
            f"max_prv_act={max_prv_act:.2%}  max_reg_easy={max_reg_easy:.2%}"
        )

        results.append(ValidationResult(
            rule=rule,
            accepted=accepted,
            u_lcb=mean_u_lcb,
            private_activation_rate=max_prv_act,
            reg_easy_worst=max_reg_easy,
            reject_reason=reject_reason,
            per_proxy_stats=merged_stats,
            u_lcb_per_seed=u_lcbs,
            count_gate_used=any_count_gate,
            failure_profile=failure_profile,
        ))

        if accepted:
            n_accepted += 1

    results.sort(key=lambda r: r.u_lcb, reverse=True)
    print(
        f"\n[validate-ms] done: {n_accepted}/{len(candidates)} accepted "
        f"across {n_seeds} seeds (max_accepted={max_accepted})"
    )
    return results
