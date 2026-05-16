"""
ICR_holistic/training/loop.py — Main iterative refinement loop.

Each iteration:
  1. Score all training items with the current cheatsheet (model_score).
  2. Partition failures into structural bins (ICR_partition style).
  3. For each active bin (in parallel, bin_concurrency workers):
       a. Enrich failures with nearest-neighbour oracle traces.
       b. Generate a RULE or EXAMPLE via model_gen (bin_generator).
       c. Score bin failures WITH (current_cs + new content) → fix_rate.
       d. Score bin correct_pool WITH (current_cs + new content) → record regressions.
       e. Accept bin if fix_rate >= fix_rate_threshold (regression never gates).
  4. Holistic rewrite: one model_gen call sees current_cs + accepted outputs +
     all regressed cases → produces the new cheatsheet.
  5. New cheatsheet replaces current_cs. Repeat.

All intermediate artefacts (bin outputs, analysis, iteration cheatsheets) are
written to output_dir for inspection.
"""
from __future__ import annotations

import dataclasses
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from utils.scorer import score_batch
from utils.oracle_index import OracleIndex
from ICR_partition.training.partition import (
    build_partitions, partition_label,
)

from ..bin_generator import BinGeneratorOutput, generate_bin_content
from ..cheatsheet_rewriter import rewrite_cheatsheet, _last_failed_raw

# Separator used when temporarily appending a bin's content for fix/regression scoring.
_CONTENT_SEP = "\n\n=== ADDITIONAL GUIDANCE (UNDER EVALUATION) ===\n"


def _test_cheatsheet(base_cs: str, new_content: str) -> str:
    """Append new_content to base_cs for per-bin fix/regression scoring."""
    return base_cs.rstrip() + _CONTENT_SEP + new_content.strip()


@dataclass
class HolisticLoopConfig:
    model_score:        str           # scoring model (measures real accuracy)
    model_gen:          str           # generation model (writes rules + rewrites)
    api_key:            str
    output_dir:         Path
    task_spec:          object        # TaskSpec for scoring + partition key
    oracle:             dict          # loaded oracle CSV {id → reasoning}
    max_iters:          int   = 5
    bin_threshold:      int   = 3     # min failures to activate a bin
    fix_rate_threshold: float = 0.10  # fraction of bin failures that must be fixed
    regression_pool_size: int = 100   # hard cap on regression pool per bin
    regression_pool_fraction: float = 0.10  # pool = min(cap, fraction * n_train)
    score_concurrency:  int   = 50
    bin_concurrency:    int   = 4     # parallel bin workers
    rollback_to_best:   bool  = False # if True, roll back to best CS when acc drops
    bin_retry:          int   = 2     # max generation attempts per bin before discarding
    min_pool_for_net_gate: int = 4   # skip net-score gate when correct_pool < this
    beam_size:          int   = 2    # holistic rewrite candidates: 1=A only, 2=A+A2(t=0.3), 3=+A3(t=0.5)…
    fix_rate_escalation: float = 0.0  # added to fix_rate_threshold each iteration
    bin_threshold_escalation: int = 0 # added to bin_threshold each iteration
    early_stop_patience: int  = 0    # stop after N iters with no new best (0 = disabled)
    val_split:          float = 0.0  # fraction of train held out for acceptance gating (0 = disabled)
    val_seed:           int   = 42   # RNG seed for val/opt split
    val_gate_threshold: float = 0.0  # absolute val accuracy floor (0 = use relative comparison)
    bin_max_tokens:     int   = 900  # max tokens for bin content generation
    rewriter_max_tokens: int  = 3000 # max tokens for holistic cheatsheet rewrite
    rewriter_cs_max_chars: int = 4000 # max chars of current cheatsheet fed to rewriter
    no_oracle_injection: bool = False # if True, skip injecting correct reasoning into bin generator
    rewrite_secondary_model: str | None = None   # if set, gate holistic rewrite on this model's accuracy
    rewrite_secondary_tolerance: float = 0.02   # max accuracy drop allowed on secondary model (e.g. 0.02 = 2%)
    slowandsteady:      bool = False       # merge ≤3 candidates per iter; carry rest as pending
    rewrite_min_fix:    int  = 3          # min wrong items a rewrite must fix to pass gate (0 = disabled)
    rewrite_gate_retries: int = 3         # max rewrite attempts before accepting best seen
    rewrite_max_broken: int  = -1         # max correct items a rewrite may break (-1 = disabled)
    rewrite_min_net_gain: int = -999      # min (n_fixed − n_broken) required to pass gate (-999 = disabled)


@dataclass
class HolisticLoopResult:
    final_cheatsheet: str
    final_accuracy:   float
    n_iters:          int
    best_cheatsheet:  str   = ""
    best_accuracy:    float = 0.0
    best_iter:        int   = 0   # 0 = initial CS-ICL, N = after iteration N
    history:          list[dict] = field(default_factory=list)


def run_holistic_loop(
    initial_cheatsheet: str,
    train_items: list[dict],
    cfg: HolisticLoopConfig,
) -> HolisticLoopResult:

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_cs   = initial_cheatsheet
    oracle_index = OracleIndex(cfg.oracle)
    history: list[dict] = []
    _ts = cfg.task_spec
    # Use the RF scoring prompt (REASONING-first, VERDICT-last) when available.
    if _ts is not None and getattr(_ts, "build_scoring_prompt_rf", None) is not None:
        _ts = dataclasses.replace(_ts, build_scoring_prompt=_ts.build_scoring_prompt_rf)
        print(f"[holistic] Using RF scoring prompt for task '{_ts.task_name}'")
    _part_key_fn = _ts.partition_key if _ts is not None else None

    (output_dir / "cheatsheet_iter0.txt").write_text(current_cs, encoding="utf-8")

    # ── Val / opt split ───────────────────────────────────────────────────────
    if cfg.val_split > 0.0:
        rng = random.Random(cfg.val_seed)
        shuffled = list(train_items)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * cfg.val_split))
        val_items_held: list[dict] = shuffled[:n_val]
        loop_items: list[dict]     = shuffled[n_val:]
        print(f"[holistic] Val split: {len(loop_items)} opt / {len(val_items_held)} val "
              f"({cfg.val_split:.0%} held out, seed={cfg.val_seed})")
    else:
        val_items_held = []
        loop_items     = train_items

    # Regression pool size: proportional to training set, capped at config hard cap
    _reg_pool_size = min(
        cfg.regression_pool_size,
        max(1, int(len(loop_items) * cfg.regression_pool_fraction)),
    )
    print(f"[holistic] Regression pool: {_reg_pool_size} per bin "
          f"({cfg.regression_pool_fraction:.0%} of {len(loop_items)}, "
          f"cap={cfg.regression_pool_size})")

    best_cs  = current_cs
    best_acc = -1.0
    best_iter_idx = 0
    best_update_count = 0
    _sec_best = [-1.0]    # [0] secondary model best accuracy (cell for closure mutation)
    consecutive_no_bins = 0
    iters_without_improvement = 0
    pending_pool: list = []  # carries deferred bin candidates across iterations (slowandsteady)

    def _update_best(cs: str, acc: float, iter_idx: int) -> None:
        nonlocal best_cs, best_acc, best_iter_idx, best_update_count
        if acc >= best_acc:
            improved       = acc > best_acc
            best_cs        = cs
            best_acc       = acc
            best_iter_idx  = iter_idx
            best_update_count += 1
            (output_dir / "cheatsheet_best.txt").write_text(cs, encoding="utf-8")
            label = "New best" if improved else "Same acc — adopting"
            print(f"[holistic] {label} cheatsheet: iter={iter_idx}  acc={acc:.1%}  "
                  f"(update #{best_update_count})")

    for outer_iter in range(cfg.max_iters):
        # ── Escalating thresholds ─────────────────────────────────────────────
        iter_fix_rate   = cfg.fix_rate_threshold   + cfg.fix_rate_escalation   * outer_iter
        iter_bin_thresh = cfg.bin_threshold + cfg.bin_threshold_escalation * outer_iter
        if outer_iter > 0 and (cfg.fix_rate_escalation > 0 or cfg.bin_threshold_escalation > 0):
            print(f"[holistic] Escalated thresholds: "
                  f"fix_rate={iter_fix_rate:.1%}  bin_threshold={iter_bin_thresh}")

        print(f"\n[holistic] === Iteration {outer_iter + 1}/{cfg.max_iters} ===")

        # ── Step 1: Score opt items (+ val items in parallel if val_split) ────
        print(f"[holistic] Scoring {len(loop_items)} opt items...")

        _iter_tag = f"iter{outer_iter + 1}"

        def _score_opt():
            return score_batch(
                items=loop_items, cheatsheet_text=current_cs,
                model=cfg.model_score, api_key=cfg.api_key,
                concurrency=cfg.score_concurrency, temperature=0.0,
                task_spec=_ts, cot_first=True,
                progress_label=f"[{_iter_tag}:opt]",
            )

        def _score_val():
            return score_batch(
                items=val_items_held, cheatsheet_text=current_cs,
                model=cfg.model_score, api_key=cfg.api_key,
                concurrency=cfg.score_concurrency, temperature=0.0,
                task_spec=_ts, cot_first=True,
                progress_label=f"[{_iter_tag}:val]",
            )

        def _score_secondary():
            return score_batch(
                items=loop_items, cheatsheet_text=current_cs,
                model=cfg.rewrite_secondary_model, api_key=cfg.api_key,
                concurrency=cfg.score_concurrency, temperature=0.0,
                task_spec=_ts, cot_first=True,
                progress_label=f"[{_iter_tag}:sec]",
            )

        _n_score_workers = 1 + bool(val_items_held) + bool(cfg.rewrite_secondary_model)
        with ThreadPoolExecutor(max_workers=_n_score_workers) as _s1_pool:
            _fut_opt = _s1_pool.submit(_score_opt)
            _fut_val = _s1_pool.submit(_score_val) if val_items_held else None
            _fut_sec = _s1_pool.submit(_score_secondary) if cfg.rewrite_secondary_model else None
            correct_items, wrong_items = _fut_opt.result()
            if _fut_val:
                _val_correct_curr, _ = _fut_val.result()
                val_acc_curr = len(_val_correct_curr) / len(val_items_held)
                print(f"[holistic] Val accuracy (current): "
                      f"{len(_val_correct_curr)}/{len(val_items_held)} ({val_acc_curr:.1%})")
            else:
                val_acc_curr = 0.0
            if _fut_sec:
                _sec_correct_curr, _ = _fut_sec.result()
                _sec_n_curr = len(_sec_correct_curr)
                _sec_correct_ids_curr = {it["id"] for it in _sec_correct_curr}
                _sec_acc_curr = _sec_n_curr / len(loop_items)
                print(f"[holistic] Secondary accuracy (current): "
                      f"{_sec_n_curr}/{len(loop_items)} ({_sec_acc_curr:.1%})")
            else:
                _sec_n_curr = 0
                _sec_correct_ids_curr = set()
                _sec_acc_curr = 0.0

        acc = len(correct_items) / len(loop_items)
        print(f"[holistic] Accuracy: {len(correct_items)}/{len(loop_items)} ({acc:.1%})  "
              f"| {len(wrong_items)} failures")

        prev_best = best_acc
        _update_best(current_cs, acc, outer_iter)  # iter 0 = initial CS-ICL
        if best_acc > prev_best:
            iters_without_improvement = 0
        else:
            iters_without_improvement += 1

        rolled_back = False
        if cfg.rollback_to_best and acc < best_acc:
            print(f"[holistic] Rollback: {acc:.1%} < best {best_acc:.1%}  "
                  f"(iter {best_iter_idx}) — re-scoring best cheatsheet as base...")
            current_cs = best_cs
            correct_items, wrong_items = score_batch(
                items=loop_items,
                cheatsheet_text=current_cs,
                model=cfg.model_score,
                api_key=cfg.api_key,
                concurrency=cfg.score_concurrency,
                temperature=0.0,
                task_spec=_ts,
                cot_first=True,
                progress_label=f"[{_iter_tag}:rollback]",
            )
            acc = len(correct_items) / len(loop_items)
            print(f"[holistic] Best cheatsheet re-scored: "
                  f"{len(correct_items)}/{len(loop_items)} ({acc:.1%})")
            rolled_back = True
            if val_items_held:
                _vrb, _ = score_batch(
                    items=val_items_held, cheatsheet_text=current_cs,
                    model=cfg.model_score, api_key=cfg.api_key,
                    concurrency=cfg.score_concurrency, temperature=0.0,
                    progress_label=f"[{_iter_tag}:rollback-val]",
                )
                val_acc_curr = len(_vrb) / len(val_items_held)
                print(f"[holistic] Val accuracy (after rollback): {val_acc_curr:.1%}")

        iter_log: dict = {
            "iter": outer_iter + 1, "accuracy": acc,
            "n_wrong": len(wrong_items), "n_correct": len(correct_items),
            "bins_active": 0, "bins_accepted": 0, "n_regressed": 0,
            "rolled_back": rolled_back,
            "best_update_count": best_update_count,
            "iters_without_improvement": iters_without_improvement,
        }
        if val_items_held:
            iter_log["val_acc_curr"] = val_acc_curr

        # ── Early stopping ────────────────────────────────────────────────────
        if (cfg.early_stop_patience > 0
                and iters_without_improvement >= cfg.early_stop_patience
                and outer_iter > 0):
            print(f"[holistic] Early stop: no improvement for "
                  f"{iters_without_improvement} consecutive iterations "
                  f"(patience={cfg.early_stop_patience}).")
            history.append(iter_log)
            _save_history(history, output_dir)
            break

        if not wrong_items:
            print("[holistic] No failures — converged.")
            history.append(iter_log)
            _save_history(history, output_dir)
            break

        # ── Step 2: Partition failures into structural bins ───────────────────
        bins = build_partitions(
            wrong_items, correct_items, iter_bin_thresh,
            partition_key_fn=_part_key_fn,
        )
        active_bins = {k: v for k, v in bins.items()
                       if len(v.failures) >= iter_bin_thresh}
        print(f"[holistic] {len(active_bins)} active bins "
              f"(≥{iter_bin_thresh} failures each) out of {len(bins)} total")
        iter_log["bins_active"] = len(active_bins)

        if not active_bins:
            print("[holistic] No active bins — stopping.")
            history.append(iter_log)
            _save_history(history, output_dir)
            break

        # ── Step 3: Solve bins in parallel ────────────────────────────────────
        accepted_outputs: list[tuple[str, BinGeneratorOutput]] = []
        all_regressed:    list[dict] = []

        def _solve_bin(bin_key, pb) -> Optional[tuple]:
            label = partition_label(bin_key)
            n_fail = len(pb.failures)
            n_pool = len(pb.correct_pool)
            print(f"  [bin] {label}  ({n_fail} failures, {n_pool} correct pool)")

            # Enrich failures with oracle reasoning (idempotent; skipped when no_oracle_injection).
            if not cfg.no_oracle_injection:
                for item in pb.failures:
                    if "_oracle_exact" not in item:
                        own_reason = item.get("reason") or item.get("oracle_reasoning")
                        if own_reason:
                            item["_oracle_exact"] = own_reason
                    if "oracle_nearest" not in item:
                        nn = oracle_index.find_nearest(item)
                        if nn:
                            entry, sim = nn
                            item["oracle_nearest"] = {
                                "eq1": entry.eq1, "eq2": entry.eq2,
                                "reasoning": entry.reasoning, "similarity": sim,
                            }

            correct_sample = list(pb.correct_pool)[:_reg_pool_size]
            bin_items = pb.failures + correct_sample
            fail_ids  = {f["id"] for f in pb.failures}
            pool_ids  = {c["id"] for c in correct_sample}

            prev_attempt = None
            for attempt in range(1, cfg.bin_retry + 1):
                # Generate rule or example (with feedback from previous failed attempt)
                out = generate_bin_content(
                    bin_label=label,
                    failures=pb.failures,
                    model=cfg.model_gen,
                    api_key=cfg.api_key,
                    task_spec=_ts,
                    previous_attempt=prev_attempt,
                    bin_max_tokens=cfg.bin_max_tokens,
                )
                if out is None:
                    print(f"  [bin] {label} — generation failed (attempt {attempt}/{cfg.bin_retry})")
                    prev_attempt = None
                    continue

                # Score failures + regression pool with test cheatsheet
                test_cs = _test_cheatsheet(current_cs, out.content)
                bc, bw = score_batch(
                    items=bin_items,
                    cheatsheet_text=test_cs,
                    model=cfg.model_score,
                    api_key=cfg.api_key,
                    concurrency=min(cfg.score_concurrency, len(bin_items) + 1),
                    temperature=0.0,
                    task_spec=_ts,
                    cot_first=True,
                    progress_label=f"[bin:{label[:12]}:primary]",
                )

                fixed     = [i for i in bc if i["id"] in fail_ids]
                regressed = [i for i in bw if i["id"] in pool_ids]
                fix_rate  = len(fixed) / n_fail
                net_score = len(fixed) - len(regressed)

                print(f"  [bin] {label} — attempt {attempt}/{cfg.bin_retry}  "
                      f"{out.content_type}  fix={fix_rate:.1%}  net={net_score:+d}")

                pool_too_small = len(correct_sample) < cfg.min_pool_for_net_gate
                rejection_reason = None

                if fix_rate < iter_fix_rate:
                    rejection_reason = (
                        f"fix_rate {fix_rate:.1%} below threshold {iter_fix_rate:.1%} "
                        f"({len(fixed)}/{n_fail} failures fixed)"
                    )
                    print(f"  [bin] {label} — below fix threshold "
                          f"({iter_fix_rate:.1%}), retrying...")
                elif net_score <= 0 and not pool_too_small:
                    rejection_reason = (
                        f"net score {net_score:+d} ≤ 0 "
                        f"({len(fixed)} fixed − {len(regressed)} regressions)"
                    )
                    print(f"  [bin] {label} — net score non-positive, retrying...")

                if rejection_reason is not None:
                    prev_attempt = {
                        "content":          out.content,
                        "content_type":     out.content_type,
                        "fix_rate":         fix_rate,
                        "net_score":        net_score,
                        "n_fixed":          len(fixed),
                        "n_fail":           n_fail,
                        "n_regressed":      len(regressed),
                        "rejection_reason": rejection_reason,
                    }
                    continue

                if net_score <= 0 and pool_too_small:
                    print(f"  [bin] {label} — net gate skipped (pool={len(correct_sample)} "
                          f"< {cfg.min_pool_for_net_gate}), accepting on fix rate alone")

                for r in regressed:
                    r["_regression_source_bin"] = label
                print(f"  [bin] {label} — ACCEPTED  "
                      f"fix={fix_rate:.1%}  net={net_score:+d}"
                      f"  regressions={len(regressed)}")
                return label, out, regressed

            print(f"  [bin] {label} — all {cfg.bin_retry} attempts failed, discarding")
            return None

        with ThreadPoolExecutor(max_workers=cfg.bin_concurrency) as pool:
            futures = {pool.submit(_solve_bin, k, pb): k
                       for k, pb in active_bins.items()}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    label, out, regressed = result
                    accepted_outputs.append((label, out))
                    all_regressed.extend(regressed)

        iter_log["bins_accepted"] = len(accepted_outputs)
        iter_log["n_regressed"]   = len(all_regressed)
        print(f"\n[holistic] {len(accepted_outputs)} bins accepted, "
              f"{len(all_regressed)} regressions recorded")

        # Save bin outputs before rewrite (for inspection)
        iter_tag = f"iter{outer_iter + 1}"
        _save_bin_outputs(accepted_outputs, all_regressed, output_dir, iter_tag)

        if not accepted_outputs:
            consecutive_no_bins += 1
            print(f"[holistic] No bins accepted "
                  f"({consecutive_no_bins}/2 consecutive — "
                  f"{'stopping' if consecutive_no_bins >= 2 else 'continuing'}).")
            history.append(iter_log)
            _save_history(history, output_dir)
            if consecutive_no_bins >= 2:
                break
            continue
        consecutive_no_bins = 0

        # ── Step 4: Holistic rewrite (beam) with fix gate ────────────────────
        beam_size = cfg.beam_size
        _rewrite_min_fix    = cfg.rewrite_min_fix
        _any_gate_active    = (_rewrite_min_fix > 0
                               or cfg.rewrite_max_broken >= 0
                               or cfg.rewrite_min_net_gain > -999)
        _rewrite_max_tries  = cfg.rewrite_gate_retries if _any_gate_active else 1
        _warm_temps = [0.3, 0.5, 0.7]

        def _run_rewrite(label: str, prompt_template: Optional[str], temperature: float,
                         caution_cases: list | None = None,
                         failed_cheatsheet: str | None = None):
            r = rewrite_cheatsheet(
                current_cheatsheet=current_cs,
                accepted_bin_outputs=accepted_outputs,
                regressed_cases=all_regressed,
                model=cfg.model_gen,
                api_key=cfg.api_key,
                prompt_template=prompt_template,
                temperature=temperature,
                cheatsheet_max_chars=cfg.rewriter_cs_max_chars,
                rewriter_max_tokens=cfg.rewriter_max_tokens,
                pending_pool=pending_pool if cfg.slowandsteady else None,
                caution_cases=caution_cases or [],
                failed_cheatsheet=failed_cheatsheet,
            )
            if r is None and _last_failed_raw[0]:
                (output_dir / f"rewrite_raw_{iter_tag}_{label}_FAILED.txt").write_text(
                    _last_failed_raw[0], encoding="utf-8")
                _last_failed_raw[0] = ""
            return r

        def _score_candidate(cs_text, tag):
            correct, _ = score_batch(
                items=wrong_items,
                cheatsheet_text=cs_text,
                model=cfg.model_score,
                api_key=cfg.api_key,
                concurrency=min(cfg.score_concurrency, len(wrong_items) + 1),
                temperature=0.0,
                task_spec=_ts,
                cot_first=True,
                progress_label=f"[{_iter_tag}:beam-{tag}]",
            )
            return len(correct)

        # Sample of correct items used to detect regressions introduced by the rewrite.
        _rw_correct_sample = (
            random.sample(correct_items, min(50, len(correct_items)))
            if correct_items else []
        )

        # (chosen_tag, rewrite_obj, n_fixed, n_broken, beam_scores_dict)
        _best_attempt: tuple | None = None
        _rw_caution_cases: list[dict] = []   # grows across retries
        _rw_failed_cs: str | None = None     # most recent rejected cheatsheet

        for _rw_try in range(_rewrite_max_tries):
            _try_lbl = (f"attempt {_rw_try + 1}/{_rewrite_max_tries}"
                        if _rewrite_max_tries > 1 else "")
            beam_specs: list[tuple[str, Optional[str], float]] = [
                ("A", None, 0.0)
            ]
            for _bi in range(1, beam_size):
                _base = _warm_temps[_bi - 1] if _bi - 1 < len(_warm_temps) else 0.7
                beam_specs.append((f"A{_bi + 1}", None, _base))

            _lbl_suffix = f" ({_try_lbl})" if _try_lbl else ""
            print(f"[holistic] Running beam holistic rewrite "
                  f"(beam_size={beam_size}){_lbl_suffix}...")

            with ThreadPoolExecutor(max_workers=beam_size) as rw_pool:
                _rw_futures = {
                    rw_pool.submit(
                        _run_rewrite, *spec, _rw_caution_cases, _rw_failed_cs
                    ): spec[0]
                    for spec in beam_specs
                }
                _beam_results: dict[str, object] = {}
                for _fut in as_completed(_rw_futures):
                    _beam_results[_rw_futures[_fut]] = _fut.result()

            candidates = [
                (spec[0], _beam_results[spec[0]])
                for spec in beam_specs
                if _beam_results.get(spec[0]) is not None
            ]

            if not candidates:
                print(f"[holistic] All rewrites failed{_lbl_suffix}.")
                continue

            # Score all candidates on wrong items
            print(f"[holistic] Scoring {len(candidates)} candidate(s) on wrong items...")
            with ThreadPoolExecutor(max_workers=len(candidates)) as sc_pool:
                _sc_futures = {
                    sc_pool.submit(_score_candidate, rw.cheatsheet, tag): tag
                    for tag, rw in candidates
                }
                beam_scores: dict[str, int] = {}
                for _sfut in as_completed(_sc_futures):
                    beam_scores[_sc_futures[_sfut]] = _sfut.result()

            for tag, _ in candidates:
                print(f"[holistic] Rewrite {tag}: "
                      f"{beam_scores[tag]}/{len(wrong_items)} wrong items fixed")

            chosen_tag, chosen_rw = max(candidates, key=lambda t: beam_scores[t[0]])
            n_fixed = beam_scores[chosen_tag]

            # Check how many correct items this rewrite breaks
            n_broken = 0
            newly_broken: list[dict] = []
            if _rw_correct_sample and (cfg.rewrite_max_broken >= 0
                                        or cfg.rewrite_min_net_gain > -999):
                _brk_correct, _brk_wrong = score_batch(
                    items=_rw_correct_sample,
                    cheatsheet_text=chosen_rw.cheatsheet,
                    model=cfg.model_score,
                    api_key=cfg.api_key,
                    concurrency=min(cfg.score_concurrency, len(_rw_correct_sample) + 1),
                    temperature=0.0,
                    task_spec=_ts,
                    cot_first=True,
                    progress_label=f"[{_iter_tag}:rw-broken{_lbl_suffix}]",
                )
                newly_broken = _brk_wrong   # were correct, now wrong under new cheatsheet
                n_broken = len(newly_broken)
                print(f"[holistic] Rewrite {chosen_tag}: "
                      f"{n_broken}/{len(_rw_correct_sample)} correct items broken")

            # Track best across attempts (prioritise fixing more, then breaking less)
            if (_best_attempt is None
                    or n_fixed > _best_attempt[2]
                    or (n_fixed == _best_attempt[2] and n_broken < _best_attempt[3])):
                _best_attempt = (chosen_tag, chosen_rw, n_fixed, n_broken,
                                 {t: beam_scores[t] for t, _ in candidates})

            # Gate check: fix floor, broken ceiling, net-gain floor
            net_gain    = n_fixed - n_broken
            fix_fail    = _rewrite_min_fix > 0 and n_fixed < _rewrite_min_fix
            broken_fail = cfg.rewrite_max_broken >= 0 and n_broken > cfg.rewrite_max_broken
            gain_fail   = cfg.rewrite_min_net_gain > -999 and net_gain < cfg.rewrite_min_net_gain
            gate_failed = fix_fail or broken_fail or gain_fail

            if gate_failed and _rw_try < _rewrite_max_tries - 1:
                reasons = []
                if fix_fail:
                    reasons.append(f"fixed {n_fixed} < min {_rewrite_min_fix}")
                if broken_fail:
                    reasons.append(f"broke {n_broken} > max {cfg.rewrite_max_broken}")
                if gain_fail:
                    reasons.append(f"net_gain {net_gain:+d} < min {cfg.rewrite_min_net_gain:+d}")
                # Accumulate newly broken cases and save failed cheatsheet for next attempt
                for item in newly_broken:
                    if item not in _rw_caution_cases:
                        _rw_caution_cases.append(item)
                _rw_failed_cs = chosen_rw.cheatsheet
                print(f"[holistic] Rewrite gate FAILED ({'; '.join(reasons)}) — retrying "
                      f"with {len(_rw_caution_cases)} caution case(s)...")
                continue

            if gate_failed:
                print(f"[holistic] Rewrite gate: all {_rewrite_max_tries} attempts failed — "
                      f"using best (fixed={_best_attempt[2]}, broken={_best_attempt[3]}, "
                      f"net={_best_attempt[2]-_best_attempt[3]:+d}).")
            else:
                _reasons = []
                if _rewrite_min_fix > 0:
                    _reasons.append(f"fixed={n_fixed}>={_rewrite_min_fix}")
                if cfg.rewrite_max_broken >= 0:
                    _reasons.append(f"broken={n_broken}<={cfg.rewrite_max_broken}")
                if cfg.rewrite_min_net_gain > -999:
                    _reasons.append(f"net={net_gain:+d}>={cfg.rewrite_min_net_gain:+d}")
                if _reasons:
                    print(f"[holistic] Rewrite gate PASSED ({', '.join(_reasons)})")

            # ── Secondary model rewrite gate ───────────────────────────────
            # Score ALL training items with secondary model under new cheatsheet.
            # If secondary accuracy drops more than tolerance, retry the rewrite
            # with regressed secondary items injected as caution.
            if not gate_failed and cfg.rewrite_secondary_model and _sec_n_curr > 0:
                _sec_new_correct, _sec_new_wrong = score_batch(
                    items=loop_items,
                    cheatsheet_text=chosen_rw.cheatsheet,
                    model=cfg.rewrite_secondary_model,
                    api_key=cfg.api_key,
                    concurrency=cfg.score_concurrency,
                    temperature=0.0,
                    task_spec=_ts,
                    cot_first=True,
                    progress_label=f"[{_iter_tag}:sec-rw-gate]",
                )
                _sec_new_n = len(_sec_new_correct)
                _sec_new_acc = _sec_new_n / len(loop_items)
                _sec_drop_pct = _sec_acc_curr - _sec_new_acc
                _sec_floor = _sec_acc_curr - cfg.rewrite_secondary_tolerance
                print(f"[holistic] Secondary rewrite gate: "
                      f"{_sec_acc_curr:.1%}→{_sec_new_acc:.1%} ({-_sec_drop_pct:+.1%})  "
                      f"floor={_sec_floor:.1%}  tol={cfg.rewrite_secondary_tolerance:.1%}")

                # Track secondary best
                if _sec_new_acc >= _sec_best[0]:
                    _sec_best[0] = _sec_new_acc
                    (output_dir / "cheatsheet_sec_best.txt").write_text(
                        chosen_rw.cheatsheet, encoding="utf-8")
                    print(f"[holistic] New secondary best: {_sec_new_acc:.1%}")

                if (_sec_new_acc < _sec_floor and _rw_try < _rewrite_max_tries - 1):
                    # Find items secondary model regressed on
                    _sec_new_wrong_ids = {it["id"] for it in _sec_new_wrong}
                    _sec_regressed = [
                        it for it in loop_items
                        if it["id"] in _sec_correct_ids_curr
                        and it["id"] in _sec_new_wrong_ids
                    ]
                    for _it in _sec_regressed[:10]:
                        if _it not in _rw_caution_cases:
                            _rw_caution_cases.append(_it)
                    _rw_failed_cs = chosen_rw.cheatsheet
                    print(f"[holistic] Secondary gate FAILED "
                          f"({_sec_new_acc:.1%} < floor {_sec_floor:.1%}) — "
                          f"retrying with {len(_sec_regressed)} secondary regressed cases...")
                    continue
                elif _sec_new_acc < _sec_floor:
                    print(f"[holistic] Secondary gate FAILED but no retries left — accepting anyway")

            break

        if _best_attempt is None:
            print("[holistic] All rewrites failed — keeping current cheatsheet, stopping.")
            history.append(iter_log)
            _save_history(history, output_dir)
            break

        chosen_tag, rewrite, _n_fixed, _n_broken, _final_beam_scores = _best_attempt
        iter_log["rewrite_chosen"]       = chosen_tag
        iter_log["beam_scores"]          = _final_beam_scores
        iter_log["rewrite_n_fixed"]      = _n_fixed
        iter_log["rewrite_n_broken"]     = _n_broken
        iter_log["rewrite_gate_retries"] = _rw_try + 1

        # ── Val gate (applied when val_split > 0) ────────────────────────────
        if val_items_held:
            print(f"[holistic] Scoring chosen rewrite on {len(val_items_held)} val items...")
            _val_correct_cand, _ = score_batch(
                items=val_items_held,
                cheatsheet_text=rewrite.cheatsheet,
                model=cfg.model_score,
                api_key=cfg.api_key,
                concurrency=min(cfg.score_concurrency, len(val_items_held) + 1),
                temperature=0.0,
                task_spec=_ts,
                cot_first=True,
                progress_label=f"[{_iter_tag}:val-gate]",
            )
            val_acc_cand = len(_val_correct_cand) / len(val_items_held)
            print(f"[holistic] Val gate: current={val_acc_curr:.1%}  "
                  f"candidate={val_acc_cand:.1%}")
            iter_log["val_acc_cand"] = val_acc_cand
            if cfg.val_gate_threshold > 0.0:
                gate_pass = val_acc_cand >= cfg.val_gate_threshold
                gate_desc = (f"threshold={cfg.val_gate_threshold:.1%}  "
                             f"candidate={val_acc_cand:.1%}")
            else:
                gate_pass = val_acc_cand > val_acc_curr
                gate_desc = f"current={val_acc_curr:.1%}  candidate={val_acc_cand:.1%}"
            print(f"[holistic] Val gate: {gate_desc}")
            if not gate_pass:
                print(f"[holistic] Val gate REJECTED — keeping current CS")
                (output_dir / f"cheatsheet_{iter_tag}_rejected.txt").write_text(
                    rewrite.cheatsheet, encoding="utf-8")
                history.append(iter_log)
                _save_history(history, output_dir)
                continue
            print(f"[holistic] Val gate PASSED")
            _old_val_acc = val_acc_curr
            val_acc_curr = val_acc_cand

        # Save iteration outputs
        (output_dir / f"cheatsheet_{iter_tag}.txt").write_text(
            rewrite.cheatsheet, encoding="utf-8")
        (output_dir / f"analysis_{iter_tag}.txt").write_text(
            rewrite.analysis, encoding="utf-8")
        (output_dir / f"rewrite_raw_{iter_tag}.txt").write_text(
            rewrite.raw_response, encoding="utf-8")

        history.append(iter_log)
        _save_history(history, output_dir)
        current_cs = rewrite.cheatsheet

        # ── Pending pool update (slowandsteady mode) ──────────────────────────
        if cfg.slowandsteady:
            accepted_map = {lbl: out for lbl, out in accepted_outputs}
            pending_map  = {lbl: out for lbl, out in pending_pool}
            combined_map = {**pending_map, **accepted_map}
            new_pending = [
                (lbl, combined_map[lbl])
                for lbl in rewrite.deferred_labels
                if lbl in combined_map
            ]
            if new_pending:
                print(f"[holistic] Pending pool: {len(new_pending)} candidates deferred "
                      f"to next iter: " + ", ".join(lbl for lbl, _ in new_pending))
            else:
                print("[holistic] Pending pool: empty (all candidates merged or unknown labels)")
            pending_pool = new_pending
            iter_log["n_pending"] = len(pending_pool)

    # ── Final full eval ───────────────────────────────────────────────────────
    print("\n[holistic] Final evaluation on opt items...")
    fc, fw = score_batch(
        items=loop_items,
        cheatsheet_text=current_cs,
        model=cfg.model_score,
        api_key=cfg.api_key,
        concurrency=cfg.score_concurrency,
        temperature=0.0,
        task_spec=_ts,
        cot_first=True,
        progress_label="[final:opt]",
    )
    final_acc = len(fc) / len(loop_items)
    print(f"[holistic] Final accuracy: {len(fc)}/{len(loop_items)} ({final_acc:.1%})")
    (output_dir / "cheatsheet_final.txt").write_text(current_cs, encoding="utf-8")
    _update_best(current_cs, final_acc, len(history))

    print(f"[holistic] Best cheatsheet: iter={best_iter_idx}  "
          f"train_acc={best_acc:.1%}  → cheatsheet_best.txt  "
          f"(updated {best_update_count}x)")

    return HolisticLoopResult(
        final_cheatsheet=current_cs,
        final_accuracy=final_acc,
        n_iters=len(history),
        best_cheatsheet=best_cs,
        best_accuracy=best_acc,
        best_iter=best_iter_idx,
        history=history,
    )


# ─── Persistence helpers ──────────────────────────────────────────────────────

def _save_bin_outputs(
    accepted: list[tuple[str, BinGeneratorOutput]],
    regressed: list[dict],
    output_dir: Path,
    tag: str,
) -> None:
    data = {
        "accepted_bins": [
            {
                "label":        lbl,
                "type":         o.content_type,
                "content":      o.content,
                "reasoning":    o.reasoning,
                "raw_response": o.raw_response,
            }
            for lbl, o in accepted
        ],
        "regressed_cases": [
            {
                "id":           r.get("id"),
                "equation1":    r.get("equation1"),
                "equation2":    r.get("equation2"),
                "answer":       r.get("answer"),
                "source_bin":   r.get("_regression_source_bin"),
            }
            for r in regressed
        ],
    }
    (output_dir / f"bin_outputs_{tag}.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8")


def _save_history(history: list[dict], output_dir: Path) -> None:
    (output_dir / "training_log.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
