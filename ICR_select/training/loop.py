"""
ICR_select/training/loop.py — Selective training loop with quality gates.

Every candidate case study must pass all active gates before entering the
cheatsheet. Useless entries are pruned periodically, and the cheatsheet is
condensed when it grows too large.

Gates (in order):
  1. Candidate competition  — generate N candidates, score each on the failure
                              batch, keep the best.
  2. Fix-rate gate          — best candidate must fix >= fix_rate_threshold of
                              the failure batch.
  3. Regression gate        — candidate must not break > regress_threshold of
                              the correct pool.
  4. Similarity gate        — LLM dedup: skip if duplicate, merge if overlap,
                              add if genuinely new.

Periodic maintenance (see maintenance.py):
  5. Ablation pruning       — every ablation_every flushes, remove case studies
                              with zero contribution.
  6. Condensation           — when len(case_studies) >= condense_at, rewrite
                              to a denser set and validate before swapping.

Gate helpers live in gates.py; maintenance functions in maintenance.py.
"""

from __future__ import annotations

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from utils.cheatsheet import Cheatsheet, extract_query_features
from utils.data import DisagreementBin, FailureBin
from utils.oracle_index import OracleIndex
from utils.scorer import score_batch, test_cheatsheet
from ICR_reasoning.core.oracle import OracleDict
from ..generators.case_study import generate_candidates
from ..prompts.templates import CORRECT_POOL_MAX
from .gates import (
    _MIN_CS_FOR_SIMILARITY,
    _apply_prescore,
    _mini_eval,
    _mini_eval_full,
    _regression_check,
    _replace_eval,
    _similarity_gate,
    _merge_case_studies,
)
from .maintenance import _ablation_prune, _condense
from .utility_gate import (
    UtilityConfig,
    VGAP_RESERVE_MAX,
    build_vmatch,
    score_utility_batch,
)


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    cheatsheet: Cheatsheet
    n_case_studies_added: int
    n_bins_discarded: int
    n_bins_skipped: int
    n_merges: int
    n_ablation_pruned: int
    n_condensations: int
    train_accuracy: float
    val_accuracy: float | None
    update_log: list[dict]
    # Disagreement mining stats
    n_disagree: int = 0    # wrong items matched to an oracle nearest neighbour
    n_both_wrong: int = 0  # wrong items with no oracle signal
    # Utility gate stats
    n_utility_accepted: int = 0   # bins accepted via utility gate
    n_utility_fallbacks: int = 0  # bins where slices were too small → fell back to old gates


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _save_checkpoint(
    cheatsheet: Cheatsheet,
    update_log: list[dict],
    output_dir: Path,
    tag: int | str = "",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cheatsheet.save(output_dir / "cheatsheet_current")
    if tag != "":
        name = f"cheatsheet_update_{tag:02d}" if isinstance(tag, int) \
               else f"cheatsheet_{tag}"
        cheatsheet.save(output_dir / name)
    (output_dir / "update_log.json").write_text(
        json.dumps(update_log, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_training_loop(
    cheatsheet: Cheatsheet,
    train_items: list[dict],
    val_items: list[dict] | None,
    model_score: str,
    model_casestudy: str,
    api_key: str,
    # Bin
    bin_threshold: int = 5,
    batch_size: int = 10,
    concurrency: int = 10,
    # Candidate generation
    n_candidates: int = 3,
    candidate_rounds: int = 3,
    flush_strategy: str = "default",   # "default" | "retry"
    oracle: OracleDict | None = None,
    oracle_min_similarity: float = 0.25,   # Jaccard threshold for nearest-oracle match
    prescore_map: dict | None = None,
    # Gates
    fix_rate_threshold: float = 0.30,
    regress_threshold: float = 0.15,
    min_pool_for_regression: int = 10,
    similarity_gate: bool = True,
    validate_merge: bool = False,
    # Maintenance
    ablation_every: int = 5,
    condense_at: int = 6,
    # Utility gate (replaces fix-rate + regression gates when slices are large enough)
    utility_gate: bool = False,
    utility_config: UtilityConfig | None = None,
    # Other
    flush_remainder: bool = True,
    cot_first: bool = True,
    reasoning_effort: str | None = "low",
    output_dir: Path | None = None,
    log: bool = True,
    skip_final_val: bool = False,  # skip test_cheatsheet when outer pipeline will re-evaluate
) -> TrainingResult:

    def _log(msg: str) -> None:
        if log:
            print(msg, file=sys.stderr, flush=True)

    # Build oracle nearest-neighbour index for disagreement mining
    oracle_index: OracleIndex | None = None
    if oracle:
        oracle_index = OracleIndex(oracle, min_similarity=oracle_min_similarity)

    # Cluster-aware bins keyed by E1 structural form (TRIVIAL / SINGLETON /
    # ABSORBING / STANDARD / GENERAL).  Each cluster flushes independently so
    # the case study generator always receives a structurally homogeneous batch.
    # Disagreement clusters (teacher✓ / student✗) have priority over both-wrong.
    disagree_bins:   dict[str, DisagreementBin] = {}
    both_wrong_bins: dict[str, FailureBin]      = {}

    def _cluster_key(item: dict) -> str:
        """Coarse structural key for bin routing: E1 equation form."""
        try:
            return extract_query_features(item).form_e1
        except Exception:
            return "GENERAL"  # safe fallback for malformed items
    correct_pool   : list[dict] = []
    train_seen     : list[dict] = []
    update_log     : list[dict] = []

    n_added        = 0
    n_discarded    = 0
    n_skipped      = 0
    n_merges       = 0
    n_pruned_total = 0
    n_condensed    = 0
    flush_count    = 0

    # Vgap rolling reservoir for the utility gate (populated from disagree_bin items)
    vgap_reserve:    list[dict] = []
    vgap_seen_count: int = 0        # total items ever offered (for reservoir sampling)

    n_disagree_total   = 0   # items routed to disagree_bin across all batches
    n_both_wrong_total = 0   # items routed to both_wrong_bin
    n_utility_accepted_total  = 0
    n_utility_fallbacks_total = 0

    total_correct = 0
    total_scored  = 0
    total_batches = (len(train_items) + batch_size - 1) // batch_size

    _cfg = utility_config or UtilityConfig()
    _log(
        f"\n{'='*60}\n"
        f"ICR_select Training loop\n"
        f"  items={len(train_items)}  batch={batch_size}  bin={bin_threshold}\n"
        f"  n_candidates={n_candidates}  fix_rate≥{fix_rate_threshold}  "
        f"regress≤{regress_threshold}  min_pool={min_pool_for_regression}\n"
        f"  validate_merge={validate_merge}  "
        f"ablation_every={ablation_every}  condense_at={condense_at}\n"
        f"  utility_gate={'on (λ='+str(_cfg.lam)+' μ='+str(_cfg.mu)+' ν='+str(_cfg.nu)+' thresh='+str(_cfg.threshold)+')' if utility_gate else 'off'}\n"
        f"  oracle_index={'yes ('+str(len(oracle_index))+' entries)' if oracle_index else 'none'}\n"
        f"  model_score={model_score}\n"
        f"  model_casestudy={model_casestudy}\n"
        f"{'='*60}"
    )

    # ---- shared gate kwargs ------------------------------------------------
    _gkw = dict(
        model_score=model_score, api_key=api_key,
        concurrency=concurrency, reasoning_effort=reasoning_effort, cot_first=cot_first,
    )

    def _process_flush(failures: list[dict], batch_num: int) -> None:
        nonlocal n_added, n_discarded, n_skipped, n_merges, flush_count
        nonlocal n_utility_accepted_total, n_utility_fallbacks_total

        _log(f"\n  [flush] {len(failures)} failures → generating {n_candidates} candidates ...")

        # Step 1: candidate competition
        try:
            candidates = generate_candidates(
                failures, cheatsheet, model_casestudy, api_key,
                n=n_candidates, oracle=oracle,
            )
        except RuntimeError as exc:
            _log(f"  [flush] candidate generation failed: {exc} — discarding bin.")
            n_discarded += 1
            return

        # Steps 2–4: score candidates and apply quality gates
        # Two paths: utility gate (continuous score) or classic mini-eval + threshold gates.
        best_cand  = None
        best_fix_rate = 0.0
        reg_rate   = 0.0

        use_utility = utility_gate and utility_config is not None
        used_utility_path = False   # True when utility gate actually ran (not fell_back)

        if use_utility:
            # Build shared Vmatch slice: union of val items matching any candidate
            failure_keys: set[tuple] = {
                (it.get("equation1", ""), it.get("equation2", "")) for it in failures
            }
            vmatch_seen: set[tuple] = set()
            vmatch: list[dict] = []
            for cand in candidates:
                for it in build_vmatch(cand, val_items or []):
                    k = (it.get("equation1", ""), it.get("equation2", ""))
                    if k not in vmatch_seen:
                        vmatch_seen.add(k)
                        vmatch.append(it)

            vgap  = [it for it in vgap_reserve
                     if (it.get("equation1", ""), it.get("equation2", "")) not in failure_keys]
            veasy = list(correct_pool)

            u_results = score_utility_batch(
                candidates, cheatsheet, vmatch, vgap, veasy, utility_config,
                model_score, api_key, concurrency, reasoning_effort, cot_first,
                log_fn=_log,
            )

            if not u_results[0].fell_back:
                used_utility_path = True
                best_idx  = max(range(len(u_results)), key=lambda i: u_results[i].utility)
                best_cand = candidates[best_idx]
                best_u    = u_results[best_idx]

                if best_u.utility <= utility_config.threshold:
                    _log(
                        f"  [gate:utility] best U={best_u.utility:+.4f} ≤ "
                        f"threshold={utility_config.threshold} — discarding."
                    )
                    n_discarded += 1
                    update_log.append({
                        "event": "bin_discarded", "batch": batch_num,
                        "reason": "utility_below_threshold",
                        "best_utility": best_u.utility,
                    })
                    return

                _log(
                    f"  [gate:utility] accepted  U={best_u.utility:+.4f}  "
                    f"ΔVmatch={best_u.delta_vmatch:+.2%}  ΔVgap={best_u.delta_vgap:+.2%}  "
                    f"Regress={best_u.regress_veasy:.2%}"
                )
                n_utility_accepted_total += 1
                best_fix_rate = best_u.delta_vmatch   # proxy for downstream log compat
                reg_rate      = best_u.regress_veasy
            else:
                _log("  [gate:utility] slices too small — falling back to fix_rate+regression gates.")
                n_utility_fallbacks_total += 1

        if not used_utility_path:
            # Classic path: mini-eval all candidates in parallel, pick best
            scored: list[tuple[float, object] | None] = [None] * len(candidates)

            def _eval_candidate(args: tuple[int, object]) -> tuple[int, float, object]:
                i, cand = args
                fix_rate = _mini_eval(
                    cand, failures, cheatsheet,
                    label=f"mini-eval cand {i+1}/{len(candidates)}", **_gkw,
                )
                return i, fix_rate, cand

            with ThreadPoolExecutor(max_workers=len(candidates)) as ex:
                futures = {ex.submit(_eval_candidate, (i, c)): i for i, c in enumerate(candidates)}
                for fut in as_completed(futures):
                    i, fix_rate, cand = fut.result()
                    scored[i] = (fix_rate, cand)
                    _log(f"  [mini-eval] candidate {i+1}: fix_rate={fix_rate:.0%}")

            scored_valid = [s for s in scored if s is not None]
            scored_valid.sort(key=lambda x: x[0], reverse=True)
            best_fix_rate, best_cand = scored_valid[0]

            # Fix-rate gate
            if best_fix_rate < fix_rate_threshold:
                _log(
                    f"  [gate:fix_rate] best={best_fix_rate:.0%} < "
                    f"threshold={fix_rate_threshold:.0%} — discarding bin."
                )
                n_discarded += 1
                update_log.append({
                    "event": "bin_discarded", "batch": batch_num,
                    "reason": "fix_rate_below_threshold",
                    "best_fix_rate": best_fix_rate,
                })
                return

            # Regression gate
            reg_rate = 0.0
            if correct_pool and len(correct_pool) >= min_pool_for_regression:
                reg_rate = _regression_check(best_cand, correct_pool, cheatsheet, **_gkw)
                _log(f"  [gate:regression] regression_rate={reg_rate:.0%}")
                if reg_rate > regress_threshold:
                    _log(
                        f"  [gate:regression] {reg_rate:.0%} > "
                        f"threshold={regress_threshold:.0%} — discarding."
                    )
                    n_discarded += 1
                    update_log.append({
                        "event": "bin_discarded", "batch": batch_num,
                        "reason": "regression_above_threshold",
                        "regression_rate": reg_rate,
                    })
                    return
            elif correct_pool:
                _log(
                    f"  [gate:regression] skipped — pool too small "
                    f"({len(correct_pool)} < min_pool={min_pool_for_regression})"
                )

        # Step 5: similarity gate (optional)
        if similarity_gate and len(cheatsheet.case_studies) >= _MIN_CS_FOR_SIMILARITY:
            action, merge_idx = _similarity_gate(best_cand, cheatsheet, model_casestudy, api_key)
            _log(f"  [gate:similarity] action={action}" +
                 (f" (merge into CS {merge_idx+1})" if merge_idx is not None else ""))

            if action == "SKIP":
                n_skipped += 1
                update_log.append({"event": "bin_skipped", "batch": batch_num, "reason": "duplicate"})
                return

            if action == "MERGE" and merge_idx is not None:
                existing = cheatsheet.case_studies[merge_idx]
                merged   = _merge_case_studies(existing, best_cand, model_casestudy, api_key)

                if validate_merge:
                    correct_base, _ = score_batch(
                        failures, cheatsheet.render(), model_score, api_key,
                        concurrency=concurrency, reasoning_effort=reasoning_effort,
                        cot_first=cot_first, progress_label="merge-eval-baseline",
                    )
                    existing_fix_rate = len(correct_base) / len(failures) if failures else 0.0
                    merged_fix_rate   = _replace_eval(merged, merge_idx, failures, cheatsheet, **_gkw)
                    _log(
                        f"  [gate:merge_validate] existing={existing_fix_rate:.0%}  "
                        f"merged={merged_fix_rate:.0%}"
                    )
                    if merged_fix_rate < existing_fix_rate:
                        _log("  [gate:merge_validate] merged worse — adding as new entry.")
                        update_log.append({
                            "event": "merge_rejected", "batch": batch_num,
                            "merge_idx": merge_idx + 1,
                            "existing_fix_rate": existing_fix_rate,
                            "merged_fix_rate": merged_fix_rate,
                        })
                        # fall through to ADD
                    else:
                        cheatsheet.case_studies[merge_idx] = merged
                        n_merges += 1
                        flush_count += 1
                        _log(f"  [merge] validated and updated CS {merge_idx+1}.")
                        update_log.append({
                            "event": "bin_merged", "batch": batch_num,
                            "merged_into": merge_idx + 1,
                            "fix_rate": best_fix_rate, "regression_rate": reg_rate,
                        })
                        return
                else:
                    cheatsheet.case_studies[merge_idx] = merged
                    n_merges += 1
                    flush_count += 1
                    _log(f"  [merge] updated CS {merge_idx+1} in-place.")
                    update_log.append({
                        "event": "bin_merged", "batch": batch_num,
                        "merged_into": merge_idx + 1,
                        "fix_rate": best_fix_rate, "regression_rate": reg_rate,
                    })
                    return

        # ADD
        best_cand.creation_fix_rate  = best_fix_rate
        best_cand.historical_fix_rate = best_fix_rate
        cheatsheet.add_case_study(best_cand)
        n_added     += 1
        flush_count += 1
        _log(
            f"  [added] CS {len(cheatsheet.case_studies)} '{best_cand.title}' — "
            f"fix_rate={best_fix_rate:.0%}  regress={reg_rate:.0%}"
        )
        update_log.append({
            "event": "bin_added", "batch": batch_num,
            "title": best_cand.title,
            "fix_rate": best_fix_rate, "regression_rate": reg_rate,
            "n_case_studies_total": len(cheatsheet.case_studies),
        })

    def _process_flush_retry(failures: list[dict], batch_num: int) -> None:
        """Retry flush — like _process_flush but retries up to candidate_rounds times."""
        nonlocal n_added, n_discarded, n_skipped, n_merges, flush_count
        nonlocal n_utility_accepted_total, n_utility_fallbacks_total

        prev_attempt: dict | None = None

        for attempt in range(1, candidate_rounds + 1):
            if attempt == 1:
                _log(f"\n  [flush/retry] {len(failures)} failures → generating {n_candidates} candidates ...")
            else:
                _log(f"\n  [flush/retry] attempt {attempt}/{candidate_rounds} (prev failed: {prev_attempt['reason']}) ...")

            try:
                candidates = generate_candidates(
                    failures, cheatsheet, model_casestudy, api_key,
                    n=n_candidates, oracle=oracle, prev_attempt=prev_attempt,
                )
            except RuntimeError as exc:
                _log(f"  [flush/retry] generation failed: {exc} — {'retrying' if attempt < candidate_rounds else 'discarding'}.")
                continue

            # Utility gate path (no retry on threshold failure — utility is a soft signal)
            if utility_gate and utility_config is not None:
                failure_keys: set[tuple] = {
                    (it.get("equation1", ""), it.get("equation2", "")) for it in failures
                }
                vmatch_seen: set[tuple] = set()
                vmatch: list[dict] = []
                for cand in candidates:
                    for it in build_vmatch(cand, val_items or []):
                        k = (it.get("equation1", ""), it.get("equation2", ""))
                        if k not in vmatch_seen:
                            vmatch_seen.add(k)
                            vmatch.append(it)
                vgap  = [it for it in vgap_reserve
                         if (it.get("equation1", ""), it.get("equation2", "")) not in failure_keys]
                veasy = list(correct_pool)

                u_results = score_utility_batch(
                    candidates, cheatsheet, vmatch, vgap, veasy, utility_config,
                    model_score, api_key, concurrency, reasoning_effort, cot_first,
                    log_fn=_log,
                )

                if not u_results[0].fell_back:
                    best_idx  = max(range(len(u_results)), key=lambda i: u_results[i].utility)
                    best_cand = candidates[best_idx]
                    best_u    = u_results[best_idx]
                    best_still_wrong: list[dict] = []  # not available on utility path

                    if best_u.utility <= utility_config.threshold:
                        _log(
                            f"  [gate:utility] best U={best_u.utility:+.4f} ≤ "
                            f"threshold={utility_config.threshold} — "
                            f"{'retrying' if attempt < candidate_rounds else 'discarding'}."
                        )
                        prev_attempt = {"candidate": best_cand, "still_wrong": [], "reason": "utility"}
                        n_utility_fallbacks_total += 1
                        continue

                    _log(
                        f"  [gate:utility] accepted  U={best_u.utility:+.4f}  "
                        f"ΔVmatch={best_u.delta_vmatch:+.2%}  ΔVgap={best_u.delta_vgap:+.2%}  "
                        f"Regress={best_u.regress_veasy:.2%}"
                    )
                    n_utility_accepted_total += 1
                    best_fix_rate = best_u.delta_vmatch
                    reg_rate      = best_u.regress_veasy
                    # Skip to similarity gate below
                    if similarity_gate and len(cheatsheet.case_studies) >= _MIN_CS_FOR_SIMILARITY:
                        action, merge_idx = _similarity_gate(best_cand, cheatsheet, model_casestudy, api_key)
                        _log(f"  [gate:similarity] action={action}" +
                             (f" (merge into CS {merge_idx+1})" if merge_idx is not None else ""))

                        if action == "SKIP":
                            n_skipped += 1
                            update_log.append({"event": "bin_skipped", "batch": batch_num, "reason": "duplicate"})
                            return

                        if action == "MERGE" and merge_idx is not None:
                            existing = cheatsheet.case_studies[merge_idx]
                            merged   = _merge_case_studies(existing, best_cand, model_casestudy, api_key)
                            cheatsheet.case_studies[merge_idx] = merged
                            n_merges += 1
                            flush_count += 1
                            _log(f"  [merge] updated CS {merge_idx+1} in-place.")
                            update_log.append({
                                "event": "bin_merged", "batch": batch_num,
                                "merged_into": merge_idx + 1,
                                "fix_rate": best_fix_rate, "regression_rate": reg_rate,
                                "attempt": attempt,
                            })
                            return

                    best_cand.creation_fix_rate   = best_fix_rate
                    best_cand.historical_fix_rate  = best_fix_rate
                    cheatsheet.add_case_study(best_cand)
                    n_added     += 1
                    flush_count += 1
                    _log(
                        f"  [added] CS {len(cheatsheet.case_studies)} '{best_cand.title}' — "
                        f"U={best_u.utility:+.4f}  attempt={attempt}"
                    )
                    update_log.append({
                        "event": "bin_added", "batch": batch_num,
                        "title": best_cand.title,
                        "fix_rate": best_fix_rate, "regression_rate": reg_rate,
                        "n_case_studies_total": len(cheatsheet.case_studies),
                        "attempt": attempt,
                    })
                    return
                else:
                    _log("  [gate:utility] slices too small — falling back to fix_rate+regression gates.")
                    n_utility_fallbacks_total += 1
                    # Fall through to classic gates below

            # Classic path: mini-eval full (returns still-wrong items for next retry context)
            scored: list[tuple[float, list[dict], object] | None] = [None] * len(candidates)

            def _eval_full(args: tuple[int, object]) -> tuple[int, float, list[dict], object]:
                i, cand = args
                fix_rate, still_wrong = _mini_eval_full(
                    cand, failures, cheatsheet,
                    label=f"mini-eval cand {i+1}/{len(candidates)}", **_gkw,
                )
                return i, fix_rate, still_wrong, cand

            with ThreadPoolExecutor(max_workers=len(candidates)) as ex:
                futures = {ex.submit(_eval_full, (i, c)): i for i, c in enumerate(candidates)}
                for fut in as_completed(futures):
                    i, fix_rate, still_wrong, cand = fut.result()
                    scored[i] = (fix_rate, still_wrong, cand)
                    _log(f"  [mini-eval] candidate {i+1}: fix_rate={fix_rate:.0%}")

            scored_valid = [s for s in scored if s is not None]
            scored_valid.sort(key=lambda x: x[0], reverse=True)
            best_fix_rate, best_still_wrong, best_cand = scored_valid[0]

            if best_fix_rate < fix_rate_threshold:
                _log(
                    f"  [gate:fix_rate] best={best_fix_rate:.0%} < threshold={fix_rate_threshold:.0%} — "
                    f"{'retrying' if attempt < candidate_rounds else 'discarding'}."
                )
                prev_attempt = {"candidate": best_cand, "still_wrong": best_still_wrong, "reason": "fix_rate"}
                continue

            reg_rate = 0.0
            if correct_pool and len(correct_pool) >= min_pool_for_regression:
                reg_rate = _regression_check(best_cand, correct_pool, cheatsheet, **_gkw)
                _log(f"  [gate:regression] regression_rate={reg_rate:.0%}")
                if reg_rate > regress_threshold:
                    _log(
                        f"  [gate:regression] {reg_rate:.0%} > threshold={regress_threshold:.0%} — "
                        f"{'retrying' if attempt < candidate_rounds else 'discarding'}."
                    )
                    prev_attempt = {"candidate": best_cand, "still_wrong": best_still_wrong, "reason": "regression"}
                    continue
            elif correct_pool:
                _log(
                    f"  [gate:regression] skipped — pool too small "
                    f"({len(correct_pool)} < min_pool={min_pool_for_regression})"
                )

            if similarity_gate and len(cheatsheet.case_studies) >= _MIN_CS_FOR_SIMILARITY:
                action, merge_idx = _similarity_gate(best_cand, cheatsheet, model_casestudy, api_key)
                _log(f"  [gate:similarity] action={action}" +
                     (f" (merge into CS {merge_idx+1})" if merge_idx is not None else ""))

                if action == "SKIP":
                    n_skipped += 1
                    update_log.append({"event": "bin_skipped", "batch": batch_num, "reason": "duplicate"})
                    return

                if action == "MERGE" and merge_idx is not None:
                    existing = cheatsheet.case_studies[merge_idx]
                    merged   = _merge_case_studies(existing, best_cand, model_casestudy, api_key)
                    cheatsheet.case_studies[merge_idx] = merged
                    n_merges += 1
                    flush_count += 1
                    _log(f"  [merge] updated CS {merge_idx+1} in-place.")
                    update_log.append({
                        "event": "bin_merged", "batch": batch_num,
                        "merged_into": merge_idx + 1,
                        "fix_rate": best_fix_rate, "regression_rate": reg_rate,
                        "attempt": attempt,
                    })
                    return

            # ADD
            best_cand.creation_fix_rate   = best_fix_rate
            best_cand.historical_fix_rate  = best_fix_rate
            cheatsheet.add_case_study(best_cand)
            n_added     += 1
            flush_count += 1
            _log(
                f"  [added] CS {len(cheatsheet.case_studies)} '{best_cand.title}' — "
                f"fix_rate={best_fix_rate:.0%}  regress={reg_rate:.0%}  attempt={attempt}"
            )
            update_log.append({
                "event": "bin_added", "batch": batch_num,
                "title": best_cand.title,
                "fix_rate": best_fix_rate, "regression_rate": reg_rate,
                "n_case_studies_total": len(cheatsheet.case_studies),
                "attempt": attempt,
            })
            return

        # All rounds exhausted
        n_discarded += 1
        _log(f"  [flush/retry] all {candidate_rounds} rounds failed — discarding bin.")
        update_log.append({
            "event": "bin_discarded", "batch": batch_num,
            "reason": f"all_{candidate_rounds}_rounds_failed",
            "last_reason": prev_attempt["reason"] if prev_attempt else None,
            "best_fix_rate": scored_valid[0][0] if scored_valid else None,
        })

    def _maybe_maintain(batch_num: int) -> None:
        nonlocal n_pruned_total, n_condensed

        if flush_count > 0 and flush_count % ablation_every == 0 and train_seen:
            _log(f"\n  [ablation] running after flush #{flush_count} ...")
            pruned_cs, n_pruned = _ablation_prune(
                cheatsheet, train_seen, model_score, api_key,
                concurrency, reasoning_effort, cot_first, _log,
            )
            cheatsheet.roadmap = pruned_cs.roadmap
            cheatsheet.case_studies  = pruned_cs.case_studies
            n_pruned_total += n_pruned
            update_log.append({
                "event": "ablation", "batch": batch_num,
                "n_pruned": n_pruned,
                "n_remaining": len(cheatsheet.case_studies),
            })

        if len(cheatsheet.case_studies) >= condense_at:
            _log(f"\n  [condense] {len(cheatsheet.case_studies)} case studies ≥ condense_at={condense_at} ...")
            condensed = _condense(
                cheatsheet, train_seen, model_casestudy, model_score, api_key,
                concurrency, reasoning_effort, cot_first, _log,
            )
            if condensed.case_studies != cheatsheet.case_studies:
                cheatsheet.roadmap = condensed.roadmap
                cheatsheet.case_studies  = condensed.case_studies
                n_condensed += 1
                update_log.append({
                    "event": "condensation", "batch": batch_num,
                    "n_after": len(cheatsheet.case_studies),
                })

    # ── Main training loop ────────────────────────────────────────────────

    for batch_start in range(0, len(train_items), batch_size):
        batch     = train_items[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1

        _total_dis = sum(len(b) for b in disagree_bins.values())
        _total_bw  = sum(len(b) for b in both_wrong_bins.values())
        _log(
            f"\n[batch {batch_num}/{total_batches}]  "
            f"items {batch_start+1}–{min(batch_start+len(batch), len(train_items))}  "
            f"disagree={_total_dis}  both_wrong={_total_bw}  "
            f"(threshold={bin_threshold}/cluster  clusters={len(disagree_bins)}d+{len(both_wrong_bins)}bw)"
        )

        if prescore_map is not None:
            correct, wrong = _apply_prescore(batch, prescore_map)
            _log(f"  [prescore] {len(correct)} correct  {len(wrong)} wrong  (no API call)")
        else:
            correct, wrong = score_batch(
                batch, cheatsheet.render(), model_score, api_key, concurrency,
                reasoning_effort=reasoning_effort, cot_first=cot_first,
            )
        total_correct += len(correct)
        total_scored  += len(batch)
        train_seen    += batch
        running_acc    = total_correct / total_scored

        for item in correct:
            if len(correct_pool) < CORRECT_POOL_MAX:
                correct_pool.append(item)
            else:
                correct_pool[random.randrange(CORRECT_POOL_MAX)] = item

        _log(
            f"  correct={len(correct)}  wrong={len(wrong)}  "
            f"running_accuracy={running_acc:.1%}  "
            f"correct_pool={len(correct_pool)}"
        )

        # Route wrong items into cluster-keyed bins.
        # Disagreement items (oracle nearest found) go to disagree_bins[key];
        # both-wrong items go to both_wrong_bins[key].  key = E1 structural form.
        n_dis = n_bw = 0
        for item in wrong:
            key = _cluster_key(item)
            if oracle_index:
                match = oracle_index.find_nearest(item)
                if match:
                    nearest_entry, sim = match
                    item = {**item,
                            "oracle_nearest": nearest_entry.to_dict(),
                            "oracle_sim":     round(sim, 3)}
                    disagree_bins.setdefault(
                        key, DisagreementBin(threshold=bin_threshold)
                    ).add(item)
                    n_dis += 1
                    # Reservoir sampling: maintain a rolling Vgap reserve for utility gate
                    vgap_seen_count += 1
                    if len(vgap_reserve) < VGAP_RESERVE_MAX:
                        vgap_reserve.append(item)
                    else:
                        j = random.randrange(vgap_seen_count)
                        if j < VGAP_RESERVE_MAX:
                            vgap_reserve[j] = item
                else:
                    both_wrong_bins.setdefault(
                        key, FailureBin(threshold=bin_threshold)
                    ).add(item)
                    n_bw += 1
            else:
                # No oracle index — all items go to the both-wrong cluster
                both_wrong_bins.setdefault(
                    key, FailureBin(threshold=bin_threshold)
                ).add(item)
                n_bw += 1

        n_disagree_total   += n_dis
        n_both_wrong_total += n_bw
        if oracle_index and (n_dis or n_bw):
            _log(
                f"  [disagree] routed {n_dis} disagreement / {n_bw} both-wrong  "
                f"(clusters: {sorted(disagree_bins)} d / {sorted(both_wrong_bins)} bw)"
            )

        # Retry strategy is only meaningful on the classic gate path — when the utility
        # gate is active, still_wrong is always [] so retry generates blind (no extra
        # signal). Force default strategy to save N*(candidate_rounds-1) scoring calls.
        # TODO: fix retry on utility path by running a cheap mini-eval to get still_wrong
        #       and passing it as context even when utility gate is the primary gate.
        _effective_strategy = "default" if (utility_gate and utility_config is not None) else flush_strategy
        _flush_fn = _process_flush_retry if _effective_strategy == "retry" else _process_flush

        # Flush full disagree clusters first (any key, deterministic order)
        for _key in sorted(disagree_bins):
            while disagree_bins[_key].is_full():
                failures = disagree_bins[_key].flush()
                _log(f"  [flush:disagree:{_key}] {len(failures)} teacher✓/student✗ pairs")
                _flush_fn(failures, batch_num)
                _maybe_maintain(batch_num)
                if output_dir:
                    _save_checkpoint(cheatsheet, update_log, output_dir, n_added)

        # Flush full both-wrong clusters only when no disagree cluster is full
        if not any(b.is_full() for b in disagree_bins.values()):
            for _key in sorted(both_wrong_bins):
                while both_wrong_bins[_key].is_full():
                    failures = both_wrong_bins[_key].flush()
                    _log(f"  [flush:both-wrong:{_key}] {len(failures)} items (no oracle signal)")
                    _flush_fn(failures, batch_num)
                    _maybe_maintain(batch_num)
                    if output_dir:
                        _save_checkpoint(cheatsheet, update_log, output_dir, n_added)

    # ── Remainder flush ───────────────────────────────────────────────────
    # Each non-empty cluster is flushed separately — the batch stays homogeneous.
    # Disagree clusters first, then both-wrong clusters, both in sorted key order.

    if flush_remainder:
        for _key in sorted(disagree_bins):
            if len(disagree_bins[_key]) > 0:
                failures = disagree_bins[_key].flush()
                _log(
                    f"\n[remainder:disagree:{_key}] flushing {len(failures)} failures ..."
                )
                _flush_fn(failures, batch_num=-1)
                _maybe_maintain(batch_num=-1)
                if output_dir:
                    _save_checkpoint(cheatsheet, update_log, output_dir, "remainder")

        for _key in sorted(both_wrong_bins):
            if len(both_wrong_bins[_key]) > 0:
                failures = both_wrong_bins[_key].flush()
                _log(
                    f"\n[remainder:both-wrong:{_key}] flushing {len(failures)} failures ..."
                )
                _flush_fn(failures, batch_num=-1)
                _maybe_maintain(batch_num=-1)
                if output_dir:
                    _save_checkpoint(cheatsheet, update_log, output_dir, "remainder")

    train_accuracy = total_correct / total_scored if total_scored > 0 else 0.0
    _log(
        f"\n{'='*60}\n"
        f"Training complete.\n"
        f"  train_accuracy={train_accuracy:.1%}\n"
        f"  case_studies_added={n_added}  merges={n_merges}\n"
        f"  bins_discarded={n_discarded}  bins_skipped={n_skipped}\n"
        f"  ablation_pruned={n_pruned_total}  condensations={n_condensed}\n"
        f"  {cheatsheet.summary()}\n"
        f"{'='*60}"
    )

    val_accuracy: float | None = None
    if val_items and not skip_final_val:
        _log(f"\nValidating on {len(val_items)} held-out items ...")
        result = test_cheatsheet(
            cheatsheet_text=cheatsheet.render(),
            val_items=val_items,
            model=model_score,
            api_key=api_key,
            concurrency=concurrency,
            reasoning_effort=reasoning_effort,
            cot_first=cot_first,
        )
        val_accuracy = result.accuracy
        _log(f"  {result.summary()}")

    if output_dir:
        _save_checkpoint(cheatsheet, update_log, output_dir, "final")

    return TrainingResult(
        cheatsheet=cheatsheet,
        n_case_studies_added=n_added,
        n_bins_discarded=n_discarded,
        n_bins_skipped=n_skipped,
        n_merges=n_merges,
        n_ablation_pruned=n_pruned_total,
        n_condensations=n_condensed,
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
        update_log=update_log,
        n_disagree=n_disagree_total,
        n_both_wrong=n_both_wrong_total,
        n_utility_accepted=n_utility_accepted_total,
        n_utility_fallbacks=n_utility_fallbacks_total,
    )
