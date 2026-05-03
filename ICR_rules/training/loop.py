"""
ICR_rules training loop — Rule-patch variant of ICR_partition's loop.

Algorithm
---------
Pre-pass:
  Ablation: disable each rule, score test set, rank by accuracy delta.
  Rules with positive delta (removing them helps) are priority patch targets.

Outer iterations:
  1. Score test set with current RuleSet (SAIR-style, no feature injection).
  2. Build partition bins from failures (same PartitionKey as ICR_partition).
  3. For each active bin (concurrently):
     a. Identify the dominant triggered rule from model reasoning traces.
     b. Enrich failures with oracle nearest-neighbour traces.
     c. Generate a RulePatch via GPT-5.4.
     d. Apply patch → score only bin items with patched RuleSet.
     e. Fix-rate gate: fraction of bin failures fixed >= threshold.
     f. Regression gate: correct_pool accuracy with patch >= (1 - threshold).
     g. Stage accepted patches (one per target rule to avoid conflicts).
  4. Apply staged patches (highest fix-rate wins per target rule).
  5. Checkpoint. Re-score. Refresh bins.
  6. Repeat until converged or max_iters reached.
"""
from __future__ import annotations

import builtins
import functools
import json
import signal
import sys
import threading
from collections import Counter

builtins.print = functools.partial(builtins.print, flush=True)
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ICR_partition.training.partition import (
    PartitionBin, build_partitions, refresh_partitions, partition_label,
)
from ICR_reasoning.core.oracle import OracleDict, load_oracle_csv
from utils.oracle_index import OracleIndex

from ..rules.rule import RuleSet, RulePatch
from ..rules.parser import identify_triggered_rule
from ..generators.rule_patch import generate_rule_patch
from .scorer import score_batch_sair
from .ablation import run_ablation, print_ablation_report


# ---------------------------------------------------------------------------
# Config / Result
# ---------------------------------------------------------------------------

@dataclass
class RuleLoopConfig:
    model_score: str         # model used to score items (e.g. gemma)
    model_patch: str         # model used to generate patches (e.g. gpt-5.4)
    api_key: str
    output_dir: Path
    max_outer_iters: int = 6
    bin_threshold: int = 3           # min failures to activate a bin
    fix_rate_threshold: float = 0.20 # fraction of bin failures that must be fixed
    regress_threshold: float = 0.20  # max fraction of correct_pool allowed to regress
    partition_concurrency: int = 4   # parallel bin solvers
    score_concurrency: int = 50
    run_ablation_prepass: bool = True
    correct_pool_max: int = 30
    max_static_iters: int = 2        # stop after this many consecutive iterations with no patch accepted


@dataclass
class RuleLoopResult:
    rule_set: RuleSet
    patch_log: List[dict]
    ablation_results: dict
    final_accuracy: float
    n_patches_applied: int


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_rule_loop(
    initial_rule_set: RuleSet,
    train_items: list[dict],
    oracle: OracleDict,
    cfg: RuleLoopConfig,
) -> RuleLoopResult:

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_rs = initial_rule_set
    patch_log: List[dict] = []
    ablation_results: dict = {}
    _shutdown = threading.Event()

    def _handle_signal(sig, frame):
        print("\n[ICR_rules] Interrupt — saving checkpoint and exiting...")
        _shutdown.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    oracle_index = OracleIndex(oracle)

    # ── Pre-pass: ablation ──────────────────────────────────────────────────
    if cfg.run_ablation_prepass:
        print("\n[ICR_rules] Running ablation pre-pass...")
        baseline_correct, baseline_wrong = score_batch_sair(
            train_items, current_rs, cfg.model_score, cfg.api_key,
            concurrency=cfg.score_concurrency,
        )
        ablation_results = run_ablation(
            current_rs, train_items, cfg.model_score, cfg.api_key,
            concurrency=cfg.score_concurrency,
            baseline_correct=baseline_correct,
            baseline_wrong=baseline_wrong,
        )
        print_ablation_report(ablation_results)
        _save_ablation(ablation_results, output_dir)
    else:
        baseline_correct, baseline_wrong = score_batch_sair(
            train_items, current_rs, cfg.model_score, cfg.api_key,
            concurrency=cfg.score_concurrency,
        )

    correct_items = baseline_correct
    wrong_items = baseline_wrong
    n_total = len(train_items)
    print(f"\n[ICR_rules] Initial accuracy: {len(correct_items)}/{n_total} ({len(correct_items)/n_total:.1%})")

    # ── Outer loop ──────────────────────────────────────────────────────────
    bins: Dict[tuple, PartitionBin] = build_partitions(wrong_items, correct_items, cfg.bin_threshold)
    print(f"[ICR_rules] {len(bins)} active bins after initial scoring")
    static_iters = 0  # consecutive iterations with no patch accepted

    for outer_iter in range(cfg.max_outer_iters):
        if _shutdown.is_set():
            break

        print(f"\n[ICR_rules] === Iteration {outer_iter + 1}/{cfg.max_outer_iters} ===")
        active_bins = {k: v for k, v in bins.items() if len(v.failures) >= cfg.bin_threshold}
        if not active_bins:
            print("[ICR_rules] No active bins — converged.")
            break

        # Enrich failures with oracle nearest-neighbour
        for pb in active_bins.values():
            for item in pb.failures:
                if "oracle_nearest" not in item:
                    nn = oracle_index.find_nearest(item)
                    if nn:
                        entry, sim = nn
                        item["oracle_nearest"] = {"eq1": entry.eq1, "eq2": entry.eq2, "reasoning": entry.reasoning, "similarity": sim}

        # ── Solve bins concurrently ─────────────────────────────────────────
        staged_patches: list[tuple[tuple, RulePatch, RuleSet, float]] = []  # (bin_key, patch, patched_rs, fix_rate)
        rs_lock = threading.Lock()

        def _solve_bin(bin_key: tuple, pb: PartitionBin) -> Optional[tuple]:
            label = partition_label(bin_key)
            print(f"  [bin] {label}  ({len(pb.failures)} failures)")

            # Find dominant triggered rule
            triggered = _find_dominant_triggered_rule(pb.failures)
            if triggered is None:
                print(f"  [bin] {label} — could not identify triggered rule, skipping")
                return None

            target_rule = current_rs.get_rule(triggered)
            if target_rule is None:
                print(f"  [bin] {label} — rule {triggered} not found in RuleSet, skipping")
                return None

            print(f"  [bin] {label} — targeting rule {triggered}")

            # Generate patch
            patch = generate_rule_patch(
                target_rule=target_rule,
                rule_set=current_rs,
                failures=pb.failures,
                correct_pool=list(pb.correct_pool)[:cfg.correct_pool_max],
                oracle=oracle,
                model=cfg.model_patch,
                api_key=cfg.api_key,
            )
            if patch is None:
                print(f"  [bin] {label} — patch generation failed")
                return None

            patch.bin_key = str(label)

            # Apply patch and score bin items
            try:
                patched_rs = current_rs.apply_patch(patch)
            except Exception as e:
                print(f"  [bin] {label} — patch application error: {e}")
                return None

            bin_items = pb.failures + list(pb.correct_pool)[:len(pb.failures)]
            bc, bw = score_batch_sair(
                bin_items, patched_rs, cfg.model_score, cfg.api_key,
                concurrency=min(cfg.score_concurrency, len(bin_items) + 1),
            )
            newly_fixed = [i for i in bc if i["id"] in {f["id"] for f in pb.failures}]
            fix_rate = len(newly_fixed) / len(pb.failures) if pb.failures else 0
            patch.bin_fix_rate = fix_rate

            if fix_rate < cfg.fix_rate_threshold:
                print(f"  [bin] {label} — fix_rate {fix_rate:.1%} < threshold, discarding")
                return None

            # Regression check on correct pool
            pool_items = list(pb.correct_pool)[:cfg.correct_pool_max]
            if pool_items:
                pc, pw = score_batch_sair(
                    pool_items, patched_rs, cfg.model_score, cfg.api_key,
                    concurrency=min(cfg.score_concurrency, len(pool_items) + 1),
                )
                reg_rate = len(pw) / len(pool_items)
                if reg_rate > cfg.regress_threshold:
                    print(f"  [bin] {label} — regression {reg_rate:.1%} > threshold, discarding")
                    return None

            print(f"  [bin] {label} — ACCEPTED patch={patch.patch_type} rule={triggered} fix_rate={fix_rate:.1%}")
            return (bin_key, patch, patched_rs, fix_rate)

        with ThreadPoolExecutor(max_workers=cfg.partition_concurrency) as pool:
            futures = {pool.submit(_solve_bin, k, pb): k for k, pb in active_bins.items()}
            for future in as_completed(futures):
                if _shutdown.is_set():
                    break
                result = future.result()
                if result is not None:
                    staged_patches.append(result)

        if not staged_patches:
            static_iters += 1
            print(f"[ICR_rules] No patches accepted this iteration ({static_iters}/{cfg.max_static_iters} static).")
            if static_iters >= cfg.max_static_iters:
                print("[ICR_rules] max_static_iters reached — stopping.")
                break
            continue

        # ── Apply patches (one per target rule, highest fix_rate wins) ──────
        staged_patches.sort(key=lambda x: x[3], reverse=True)
        applied_targets: set[str] = set()
        for bin_key, patch, patched_rs, fix_rate in staged_patches:
            if patch.target_rule_id not in applied_targets:
                current_rs = patched_rs
                applied_targets.add(patch.target_rule_id)
                patch_log.append({
                    "iteration": outer_iter + 1,
                    "bin_key": str(bin_key),
                    "target_rule": patch.target_rule_id,
                    "patch_type": patch.patch_type,
                    "new_rules": patch.new_rules,
                    "fix_rate": fix_rate,
                    "reasoning": patch.reasoning,
                })
                print(f"[ICR_rules] Applied patch: {patch.target_rule_id} → {patch.patch_type}  (fix_rate={fix_rate:.1%})")

        static_iters = 0  # reset on any accepted patch

        # ── Checkpoint ──────────────────────────────────────────────────────
        _save_checkpoint(current_rs, patch_log, output_dir, tag=f"iter{outer_iter+1}")

        # ── Re-score and refresh bins ────────────────────────────────────────
        all_active_failures = [item for pb in active_bins.values() for item in pb.failures]
        rc, rw = score_batch_sair(
            all_active_failures, current_rs, cfg.model_score, cfg.api_key,
            concurrency=cfg.score_concurrency,
        )
        print(f"[ICR_rules] After iter {outer_iter+1}: {len(rc)}/{len(all_active_failures)} active-bin items correct")
        bins = refresh_partitions(bins, rw, rc, retirement_threshold=cfg.bin_threshold)

    # ── Final full eval ──────────────────────────────────────────────────────
    print("\n[ICR_rules] Final full evaluation...")
    fc, fw = score_batch_sair(train_items, current_rs, cfg.model_score, cfg.api_key, concurrency=cfg.score_concurrency)
    final_acc = len(fc) / len(train_items)
    print(f"[ICR_rules] Final accuracy: {len(fc)}/{len(train_items)} ({final_acc:.1%})")

    _save_checkpoint(current_rs, patch_log, output_dir, tag="final")

    return RuleLoopResult(
        rule_set=current_rs,
        patch_log=patch_log,
        ablation_results=ablation_results,
        final_accuracy=final_acc,
        n_patches_applied=len(patch_log),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_dominant_triggered_rule(failures: list[dict]) -> Optional[str]:
    """Return the most commonly triggered rule across all failure reasoning traces."""
    counts: Counter = Counter()
    for item in failures:
        reasoning = item.get("reasoning") or item.get("post_think") or ""
        rule_id = identify_triggered_rule(reasoning)
        if rule_id:
            counts[rule_id] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _save_checkpoint(rule_set: RuleSet, patch_log: list, output_dir: Path, tag: str) -> None:
    cheatsheet_path = output_dir / f"cheatsheet_{tag}.jinja2"
    cheatsheet_path.write_text(rule_set.render(), encoding="utf-8")

    log_path = output_dir / f"patch_log_{tag}.json"
    log_path.write_text(json.dumps(patch_log, indent=2), encoding="utf-8")

    print(f"[ICR_rules] Checkpoint saved: {cheatsheet_path} ({rule_set.byte_size()/1024:.1f} KB)")


def _save_ablation(ablation_results: dict, output_dir: Path) -> None:
    path = output_dir / "ablation.json"
    serializable = {
        k: {"rule_id": v.rule_id, "accuracy_baseline": v.accuracy_baseline,
            "accuracy_without": v.accuracy_without, "delta": v.delta, "n_items": v.n_items}
        for k, v in ablation_results.items()
    }
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"[ICR_rules] Ablation results saved: {path}")
