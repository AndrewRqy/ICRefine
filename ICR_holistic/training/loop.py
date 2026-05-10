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
from ..prompts import HOLISTIC_REWRITE_PROMPT_CONSERVATIVE

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
    beam_size:          int   = 2    # holistic rewrite candidates: 1=A only, 2=A+B, 3+=A+B+warm-A…
    fix_rate_escalation: float = 0.0  # added to fix_rate_threshold each iteration
    bin_threshold_escalation: int = 0 # added to bin_threshold each iteration
    early_stop_patience: int  = 0    # stop after N iters with no new best (0 = disabled)
    val_split:          float = 0.0  # fraction of train held out for acceptance gating (0 = disabled)
    val_seed:           int   = 42   # RNG seed for val/opt split
    val_gate_threshold: float = 0.0  # absolute val accuracy floor (0 = use relative comparison)


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
    consecutive_no_bins = 0
    iters_without_improvement = 0

    def _update_best(cs: str, acc: float, iter_idx: int) -> None:
        nonlocal best_cs, best_acc, best_iter_idx, best_update_count
        if acc > best_acc:
            best_cs        = cs
            best_acc       = acc
            best_iter_idx  = iter_idx
            best_update_count += 1
            (output_dir / "cheatsheet_best.txt").write_text(cs, encoding="utf-8")
            print(f"[holistic] New best cheatsheet: iter={iter_idx}  acc={acc:.1%}  "
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

        def _score_opt():
            return score_batch(
                items=loop_items, cheatsheet_text=current_cs,
                model=cfg.model_score, api_key=cfg.api_key,
                concurrency=cfg.score_concurrency, temperature=0.0,
                task_spec=_ts, cot_first=True,
            )

        def _score_val():
            return score_batch(
                items=val_items_held, cheatsheet_text=current_cs,
                model=cfg.model_score, api_key=cfg.api_key,
                concurrency=cfg.score_concurrency, temperature=0.0,
                task_spec=_ts, cot_first=True,
            )

        if val_items_held:
            with ThreadPoolExecutor(max_workers=2) as _s1_pool:
                _fut_opt = _s1_pool.submit(_score_opt)
                _fut_val = _s1_pool.submit(_score_val)
                correct_items, wrong_items = _fut_opt.result()
                _val_correct_curr, _ = _fut_val.result()
            val_acc_curr = len(_val_correct_curr) / len(val_items_held)
            print(f"[holistic] Val accuracy (current): "
                  f"{len(_val_correct_curr)}/{len(val_items_held)} ({val_acc_curr:.1%})")
        else:
            val_acc_curr = 0.0
            correct_items, wrong_items = _score_opt()

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
                    task_spec=_ts, cot_first=True,
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

            # Enrich failures with oracle reasoning (idempotent).
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
                      f"fix={fix_rate:.1%}  net={net_score:+d}  regressions={len(regressed)}")
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

        # ── Step 4: Holistic rewrite (beam) ──────────────────────────────────
        beam_size = cfg.beam_size
        print(f"[holistic] Running beam holistic rewrite (beam_size={beam_size})...")

        # beam_specs: (tag, prompt_template, temperature)
        beam_specs: list[tuple[str, Optional[str], float]] = [("A", None, 0.0)]
        if beam_size >= 2:
            beam_specs.append(("B", HOLISTIC_REWRITE_PROMPT_CONSERVATIVE, 0.0))
        for _bi in range(2, beam_size):
            beam_specs.append((f"A{_bi}", None, 0.3))

        def _run_rewrite(label: str, prompt_template: Optional[str], temperature: float):
            r = rewrite_cheatsheet(
                current_cheatsheet=current_cs,
                accepted_bin_outputs=accepted_outputs,
                regressed_cases=all_regressed,
                model=cfg.model_gen,
                api_key=cfg.api_key,
                prompt_template=prompt_template,
                temperature=temperature,
            )
            if r is None and _last_failed_raw[0]:
                (output_dir / f"rewrite_raw_{iter_tag}_{label}_FAILED.txt").write_text(
                    _last_failed_raw[0], encoding="utf-8")
                _last_failed_raw[0] = ""
            return r

        with ThreadPoolExecutor(max_workers=beam_size) as rw_pool:
            _rw_futures = {rw_pool.submit(_run_rewrite, *spec): spec[0] for spec in beam_specs}
            _beam_results: dict[str, object] = {}
            for _fut in as_completed(_rw_futures):
                _beam_results[_rw_futures[_fut]] = _fut.result()

        candidates = [
            (spec[0], _beam_results[spec[0]])
            for spec in beam_specs
            if _beam_results.get(spec[0]) is not None
        ]

        if not candidates:
            print("[holistic] All rewrites failed — keeping current cheatsheet, stopping.")
            history.append(iter_log)
            _save_history(history, output_dir)
            break

        if len(candidates) == 1:
            chosen_tag, rewrite = candidates[0]
            iter_log["rewrite_chosen"] = chosen_tag
            if beam_size > 1:
                print(f"[holistic] Only rewrite {chosen_tag} succeeded — using it.")
        else:
            print(f"[holistic] Scoring {len(candidates)} beam candidates on wrong items...")

            def _score_candidate(cs_text):
                correct, _ = score_batch(
                    items=wrong_items,
                    cheatsheet_text=cs_text,
                    model=cfg.model_score,
                    api_key=cfg.api_key,
                    concurrency=min(cfg.score_concurrency, len(wrong_items) + 1),
                    temperature=0.0,
                    task_spec=_ts,
                    cot_first=True,
                )
                return len(correct)

            with ThreadPoolExecutor(max_workers=len(candidates)) as sc_pool:
                _sc_futures = {
                    sc_pool.submit(_score_candidate, rw.cheatsheet): tag
                    for tag, rw in candidates
                }
                beam_scores: dict[str, int] = {}
                for _sfut in as_completed(_sc_futures):
                    beam_scores[_sc_futures[_sfut]] = _sfut.result()

            for tag, _ in candidates:
                print(f"[holistic] Rewrite {tag}: "
                      f"{beam_scores[tag]}/{len(wrong_items)} wrong items fixed")
            chosen_tag, rewrite = max(candidates, key=lambda t: beam_scores[t[0]])
            print(f"[holistic] Chose rewrite {chosen_tag} "
                  f"(score={beam_scores[chosen_tag]})")
            iter_log["rewrite_chosen"] = chosen_tag
            iter_log["beam_scores"] = {t: beam_scores[t] for t, _ in candidates}

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
