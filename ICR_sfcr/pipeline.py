"""
ICR_sfcr/pipeline.py — Shared-Failure Conservative Refinement (SFCR) pipeline.

Algorithm:
  1. Split training data into rule_gen / gate (disjoint).
  2. Score source + proxy models on rule_gen under anchor cheatsheet C0.
     Optionally use n_eval_seeds > 1 for soft probability estimation.
  3. Compute failure regions: V_shared, V_private, V_easy.
  4. Skip guard — exit with anchor unchanged if conditions are unmet.
  5. Cluster V_shared into subtypes; generate 2-3 candidates per subtype.
  6. Pre-compute gate baseline (score all models on gate under anchor).
  7. Validate each candidate via U_LCB / count-aware gate on the gate split.
     Rejected candidates enter the repair loop (--repair-attempts controls depth).
  8. Accept up to max_accepted rules (default 3).
  9. Write outputs: accepted_rules.json, final_cheatsheet_{mode}.txt, sfcr_log.jsonl.

Usage:
    python -m ICR_sfcr.pipeline \\
        --task              causal_judgement \\
        --dataset           datasets/bbh/causal_judgement_train_labeled.jsonl \\
        --anchor-cheatsheet CS_ICL_Initial_Prompt/bbh_causal_judgement/gen_gpt-4.1-mini_0_1000.txt \\
        --output-dir        runs/sfcr_cj_1000 \\
        --model-source      openai/gpt-4.1-mini \\
        --models-proxy      openai/gpt-4.1,google/gemini-2.0-flash,meta-llama/llama-3.3-70b \\
        --held-out-target   claude \\
        --oracle-mode       label_only \\
        --routing-mode      routed \\
        --seed              1000 \\
        --concurrency       30

    # Soft probability regions (3 repeated evals per model):
    python -m ICR_sfcr.pipeline ... --n-eval-seeds 3

    # Enable repair loop (1 repair attempt per rejected candidate):
    python -m ICR_sfcr.pipeline ... --repair-attempts 1

    # Disable subtype clustering (flat-pool generation, original behaviour):
    python -m ICR_sfcr.pipeline ... --no-subtypes

    # With full oracle CoT (for ablation):
    python -m ICR_sfcr.pipeline ... --oracle-mode full_cot

    # Global prepend mode (no routing):
    python -m ICR_sfcr.pipeline ... --routing-mode global
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data import load_jsonl
from utils.llm_client import get_api_key
from utils.scorer import score_batch
from tasks.registry import get_task, TASK_REGISTRY

from .activation import activation_summary, build_cheatsheet
from .failure_regions import compute_failure_regions
from .logger import SFCRLogger, log_condition, rule_id
from .rule_generator import generate_candidates, load_manual_rules, repair_candidate
from .rule_validator import (
    build_cheatsheet_with_rule,
    compare_global_routed_results,
    compute_gate_baseline,
    validate_candidates,
)
from .splits import make_splits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Shared-Failure Conservative Refinement (SFCR)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    ap.add_argument("--task",               required=True,
                    help=f"Task name. Known: {', '.join(sorted(TASK_REGISTRY))}")
    ap.add_argument("--dataset",            required=True,
                    help="Path to training .jsonl file")
    ap.add_argument("--anchor-cheatsheet",  required=True,
                    help="Path to anchor CS-ICL cheatsheet .txt file")
    ap.add_argument("--output-dir",         required=True,
                    help="Directory for output files")
    ap.add_argument("--model-source",       required=True,
                    help="Source model used for failure elicitation (e.g. openai/gpt-4.1-mini)")
    ap.add_argument("--models-proxy",       required=True,
                    help="Comma-separated proxy model IDs for U_LCB validation")

    # Optional: protocol
    ap.add_argument("--held-out-target",    default="",
                    help="Family substring to exclude from U_LCB acceptance (leave-one-out)")
    ap.add_argument("--oracle-mode",        default="label_only",
                    choices=["none", "label_only", "compressed", "full_cot"],
                    help="Information provided to the rule generator")
    ap.add_argument("--routing-mode",       default="routed",
                    choices=["global", "routed"],
                    help="How accepted rules are applied at inference")
    ap.add_argument("--validation-routing-mode", default="routed",
                    choices=["global", "routed", "both"],
                    help="How candidate rules are validated; 'both' writes side-by-side metrics.")
    ap.add_argument("--gate-mode",          default="hybrid",
                    choices=["ulcb", "count_aware", "hybrid"],
                    help="Candidate acceptance gate.")
    ap.add_argument("--router-min-matches", type=int, default=2,
                    help="Minimum USE WHEN content-term matches required by the router.")
    ap.add_argument("--min-veto-matches",   type=int, default=1,
                    help="Minimum DO NOT USE WHEN content-term matches required to veto an item. "
                         "Default 1 = original behaviour (any match vetoes). "
                         "Set to 2 to prevent broad domain vocabulary from vetoing every item.")
    ap.add_argument("--allow-empty-use-when-global", action="store_true",
                    help="Allow empty USE WHEN to activate globally in routed mode.")
    ap.add_argument("--router-type",        default="keyword",
                    choices=["keyword", "feature"],
                    help="USE WHEN routing strategy. 'keyword'=lexical (default); "
                         "'feature'=task-aware tag router (Line A).")
    ap.add_argument("--memory-format",      default="rule",
                    choices=["rule", "rule_check", "rule_check_example"],
                    help="Memory atom format for rule generation and cheatsheet rendering (Line B). "
                         "rule=B1, rule_check=B2, rule_check_example=B3.")
    ap.add_argument("--gate-profile",       default="auto",
                    choices=["auto", "small", "medium", "large", "diagnostic"],
                    help="Acceptance gate profile (Line C). 'auto' selects based on |V_shared|.")

    # Optional: failure region estimation
    ap.add_argument("--n-eval-seeds",       type=int, default=1,
                    help="Number of repeated evals per model for soft probability regions. "
                         "1 = hard binary (original). 3 = majority-vote soft regions.")
    ap.add_argument("--eval-temperature",   type=float, default=0.7,
                    help="Temperature used for repeated evals when --n-eval-seeds > 1")
    ap.add_argument("--tau-s",              type=float, default=0.5,
                    help="Source failure threshold for soft regions (p_s >= tau_s → fails)")
    ap.add_argument("--tau-p",              type=float, default=0.5,
                    help="Proxy failure threshold for V_shared (max_j p_j >= tau_p)")
    ap.add_argument("--tau-low",            type=float, default=0.33,
                    help="'Consistently correct' threshold (p <= tau_low)")

    # Optional: split sizes
    ap.add_argument("--rule-gen-n",         type=int, default=60)
    ap.add_argument("--gate-n",             type=int, default=40)
    ap.add_argument("--seed",               type=int, default=1000)

    # Optional: generation
    ap.add_argument("--n-candidates",       type=int, default=8)
    ap.add_argument("--candidates-per-subtype", type=int, default=3,
                    help="Candidates to generate per failure subtype")
    ap.add_argument("--temperatures",       default="0.2,0.5,0.8",
                    help="Comma-separated temperatures for candidate generation")
    ap.add_argument("--model-gen",          default="",
                    help="Model for rule generation (defaults to --model-source)")
    ap.add_argument("--generator-model",    default="",
                    help="Alias for --model-gen used by SF-CR v2 experiment scripts")
    ap.add_argument("--candidate-source",   default="generator",
                    choices=["generator", "manual"],
                    help="Generate candidates with an LLM or read --manual-rules-file")
    ap.add_argument("--manual-rules-file",  default="",
                    help="YAML/JSON file of manual candidate rules")
    ap.add_argument("--max-rule-chars",     type=int, default=800)
    ap.add_argument("--no-subtypes",        action="store_true",
                    help="Disable subtype clustering; use flat-pool generation (original behaviour)")
    ap.add_argument("--no-quick-validate",  action="store_true",
                    help="Disable per-candidate quick-validation on subtype items before gate scoring")

    # Optional: repair loop
    ap.add_argument("--repair-attempts",    type=int, default=0,
                    help="Repair attempts per rejected candidate (0 = disabled)")

    # Optional: validation weights
    ap.add_argument("--lambda-w",           type=float, default=1.0,
                    help="Penalty weight on private regression in U_LCB")
    ap.add_argument("--mu-w",              type=float, default=1.0,
                    help="Penalty weight on easy regression in U_LCB")
    ap.add_argument("--nu-w",              type=float, default=0.05,
                    help="Penalty weight on rule length in U_LCB")
    ap.add_argument("--max-accepted",       type=int, default=3)
    ap.add_argument("--private-act-ceil",   type=float, default=0.10)
    ap.add_argument("--reg-easy-ceil",      type=float, default=0.05)

    # Optional: infra
    ap.add_argument("--concurrency",        type=int, default=30)

    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key      = get_api_key()
    task_spec    = get_task(args.task)
    anchor_text  = Path(args.anchor_cheatsheet).read_text(encoding="utf-8").strip()
    all_items    = load_jsonl(Path(args.dataset))
    proxy_models = [m.strip() for m in args.models_proxy.split(",") if m.strip()]
    gen_model    = args.generator_model or args.model_gen or args.model_source
    temperatures = [float(t) for t in args.temperatures.split(",")]
    use_subtypes = not args.no_subtypes

    print(f"\n{'='*60}")
    print(f" SFCR  task={args.task}  seed={args.seed}")
    print(f" source={args.model_source}")
    print(f" proxies={[m.split('/')[-1] for m in proxy_models]}")
    print(f" oracle_mode={args.oracle_mode}  routing_mode={args.routing_mode}  "
          f"validation={args.validation_routing_mode}  gate={args.gate_mode}")
    print(f" router_type={args.router_type}  memory_format={args.memory_format}  "
          f"gate_profile={args.gate_profile}")
    print(f" n_eval_seeds={args.n_eval_seeds}  use_subtypes={use_subtypes}")
    print(f" repair_attempts={args.repair_attempts}")
    print(f"{'='*60}\n")

    # ── 1. Split ──────────────────────────────────────────────────────────
    splits = make_splits(
        all_items,
        rule_gen_n=args.rule_gen_n,
        gate_n=args.gate_n,
        seed=args.seed,
    )

    from .failure_regions import _tag_ids
    _tag_ids(splits.rule_gen)
    _tag_ids(splits.gate)

    # ── 2-3. Failure regions on rule_gen ─────────────────────────────────
    print("[pipeline] Computing failure regions on rule_gen split...")
    rg_regions = compute_failure_regions(
        items=splits.rule_gen,
        anchor_cheatsheet=anchor_text,
        source_model=args.model_source,
        proxy_models=proxy_models,
        api_key=api_key,
        task_spec=task_spec,
        concurrency=args.concurrency,
        label="rule_gen",
        n_evals=args.n_eval_seeds,
        eval_temperature=args.eval_temperature,
        tau_s=args.tau_s,
        tau_p=args.tau_p,
        tau_low=args.tau_low,
    )

    # ── 4. Skip guard ─────────────────────────────────────────────────────
    if rg_regions.skip_reason:
        print(f"\n[pipeline] SKIP — {rg_regions.skip_reason}")
        print("[pipeline] Writing anchor cheatsheet unchanged.")
        for mode in ("global", "routed"):
            out = output_dir / f"final_cheatsheet_{mode}.txt"
            out.write_text(anchor_text, encoding="utf-8")
        _write_summary(output_dir, args, accepted_rules=[], skipped=True,
                       skip_reason=rg_regions.skip_reason, regions=rg_regions)
        return

    # ── 5. Generate candidates ────────────────────────────────────────────
    # Quick-validate: test each candidate on its own subtype items before gate scoring.
    _sfcr_id_to_rg_item = {it["_sfcr_id"]: it for it in rg_regions.V_shared}

    def _quick_validate(candidate: dict) -> bool:
        ids = candidate.get("subtype_items") or []
        items_to_check = [_sfcr_id_to_rg_item[sid] for sid in ids if sid in _sfcr_id_to_rg_item]
        if not items_to_check:
            return True  # no subtype items to check, let it through
        test_cs = build_cheatsheet_with_rule(anchor_text, candidate)
        correct, _ = score_batch(
            items_to_check, test_cs, args.model_source, api_key,
            concurrency=min(args.concurrency, len(items_to_check)),
            temperature=0.0, task_spec=task_spec, cot_first=True,
        )
        n_fixed = len(correct)
        print(f"[gen] quick-validate: {n_fixed}/{len(items_to_check)} subtype items fixed")
        return n_fixed > 0

    if args.candidate_source == "manual":
        if not args.manual_rules_file:
            raise SystemExit("--manual-rules-file is required when --candidate-source manual")
        print(f"\n[pipeline] Loading manual candidates from {args.manual_rules_file}...")
        candidates = load_manual_rules(args.manual_rules_file, task=args.task)
    else:
        print(f"\n[pipeline] Generating candidates "
              f"(use_subtypes={use_subtypes}, n_candidates={args.n_candidates}, "
              f"quick_validate={not args.no_quick_validate})...")
        candidates = generate_candidates(
            V_shared=rg_regions.V_shared,
            V_private=rg_regions.V_private,
            anchor_cheatsheet=anchor_text,
            model=gen_model,
            api_key=api_key,
            n_candidates=args.n_candidates,
            temperatures=temperatures,
            oracle_mode=args.oracle_mode,
            max_rule_chars=args.max_rule_chars,
            use_subtypes=use_subtypes,
            candidates_per_subtype=args.candidates_per_subtype,
            memory_format=args.memory_format,
            quick_validate_fn=None if args.no_quick_validate else _quick_validate,
        )
        for c in candidates:
            c.setdefault("source_of_candidate", "generator")
            c.setdefault("generator_model", gen_model)

    if not candidates:
        print("[pipeline] No valid candidates generated — writing anchor unchanged.")
        for mode in ("global", "routed"):
            (output_dir / f"final_cheatsheet_{mode}.txt").write_text(anchor_text, encoding="utf-8")
        _write_summary(output_dir, args, accepted_rules=[], skipped=False,
                       skip_reason="generation produced no valid candidates",
                       regions=rg_regions)
        return

    # ── 6. Gate baseline ─────────────────────────────────────────────────
    print("\n[pipeline] Pre-computing gate baseline...")
    gate_baseline = compute_gate_baseline(
        gate_items=splits.gate,
        anchor_cheatsheet=anchor_text,
        source_model=args.model_source,
        proxy_models=proxy_models,
        api_key=api_key,
        task_spec=task_spec,
        concurrency=args.concurrency,
    )

    # ── 7. Build repair function (if requested) ───────────────────────────
    repair_fn = None
    if args.repair_attempts > 0:
        def repair_fn(rule: dict, failure_profile: dict) -> dict | None:
            return repair_candidate(
                rule=rule,
                failure_profile=failure_profile,
                anchor_cheatsheet=anchor_text,
                model=gen_model,
                api_key=api_key,
                max_rule_chars=args.max_rule_chars,
                max_attempts=args.repair_attempts,
            )

    # ── 8. Validate candidates ────────────────────────────────────────────
    print(f"\n[pipeline] Validating {len(candidates)} candidates "
          f"(repair_attempts={args.repair_attempts})...")
    def _run_validation(mode: str):
        return validate_candidates(
            candidates=candidates,
            gate_items=splits.gate,
            gate_baseline=gate_baseline,
            anchor_cheatsheet=anchor_text,
            source_model=args.model_source,
            proxy_models=proxy_models,
            held_out_target=args.held_out_target or None,
            api_key=api_key,
            task_spec=task_spec,
            concurrency=args.concurrency,
            lambda_w=args.lambda_w,
            mu_w=args.mu_w,
            nu_w=args.nu_w,
            max_accepted=args.max_accepted,
            private_activation_ceiling=args.private_act_ceil,
            reg_easy_ceiling=args.reg_easy_ceil,
            max_rule_chars=args.max_rule_chars,
            repair_fn=repair_fn,
            repair_attempts=args.repair_attempts,
            validation_routing_mode=mode,
            gate_mode=args.gate_mode,
            subtype_filter_mode="none",
            router_min_matches=args.router_min_matches,
            router_min_veto_matches=args.min_veto_matches,
            allow_empty_use_when_global=args.allow_empty_use_when_global,
            router_type=args.router_type,
            task=args.task,
            gate_profile=args.gate_profile,
        )

    if args.validation_routing_mode == "both":
        print("\n[pipeline] Validating candidates in global mode...")
        global_results = _run_validation("global")
        print("\n[pipeline] Validating candidates in routed mode...")
        routed_results = _run_validation("routed")
        val_results = routed_results
        comparison_rows = compare_global_routed_results(global_results, routed_results)
        _write_global_routed_comparison(output_dir, comparison_rows)
    else:
        val_results = _run_validation(args.validation_routing_mode)
        comparison_rows = []

    # ── 9. Collect accepted rules ─────────────────────────────────────────
    accepted_rules = [r.rule for r in val_results if r.accepted]
    print(f"\n[pipeline] Accepted {len(accepted_rules)} rule(s).")

    # ── 10. Write outputs ─────────────────────────────────────────────────
    # accepted_rules.json
    rules_path = output_dir / "accepted_rules.json"
    rules_path.write_text(
        json.dumps(
            [
                {
                    **r.rule,
                    "rule_id":                rule_id(r.rule),
                    "u_lcb":                  round(r.u_lcb, 6),
                    "private_activation_rate": round(r.private_activation_rate, 4),
                    "reg_easy_worst":         round(r.reg_easy_worst, 4),
                    "count_gate_used":        r.count_gate_used,
                    "validation_routing_mode": r.validation_routing_mode,
                    "gate_mode":              r.gate_mode,
                    "benefit_models":         r.benefit_models,
                    "safety_models":          r.safety_models,
                    "repaired":               r.rule.get("repaired", False),
                }
                for r in val_results if r.accepted
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    # validation_results.json (all candidates with stats)
    val_path = output_dir / "validation_results.json"
    val_path.write_text(
        json.dumps(
            [
                {
                    "rule_id":                rule_id(r.rule),
                    "rule":                   r.rule.get("rule", "")[:120],
                    "subtype_idx":            r.rule.get("subtype_idx"),
                    "repaired":               r.rule.get("repaired", False),
                    "accepted":               r.accepted,
                    "candidate_status":       r.candidate_status,
                    "u_lcb":                  round(r.u_lcb, 6),
                    "reject_reason":          r.reject_reason,
                    "reject_reasons":         r.reject_reasons,
                    "count_gate_used":        r.count_gate_used,
                    "validation_routing_mode": r.validation_routing_mode,
                    "gate_mode":              r.gate_mode,
                    "source_of_candidate":    r.rule.get("source_of_candidate", ""),
                    "generator_model":        r.rule.get("generator_model", ""),
                    "benefit_models":         r.benefit_models,
                    "safety_models":          r.safety_models,
                    "activation_debug":       r.activation_debug,
                    "private_activation_rate": round(r.private_activation_rate, 4),
                    "reg_easy_worst":         round(r.reg_easy_worst, 4),
                    "per_proxy": {
                        pm.split("/")[-1]: {
                            "delta_shared":            round(s.delta_shared, 4),
                            "shared_gain_mode":         s.shared_gain_mode,
                            "reg_private":             round(s.reg_private, 4),
                            "reg_easy":                round(s.reg_easy, 4),
                            "fixed_shared_count":      s.fixed_shared_count,
                            "reg_easy_count":          s.reg_easy_count,
                            "reg_private_count":       s.reg_private_count,
                            "private_activation_count": s.private_activation_count,
                            "easy_activation_count":    s.easy_activation_count,
                            "activated_shared_count":   s.activated_shared_count,
                            "n_shared_before_subtype_filter": s.n_shared_before_subtype_filter,
                            "n_shared_after_subtype_filter":  s.n_shared_after_subtype_filter,
                            "subtype_filter_mode":      s.subtype_filter_mode,
                            "activation_count":        s.activation_count,
                            "n_shared":                s.n_shared,
                            "n_private":               s.n_private,
                            "n_easy":                  s.n_easy,
                        }
                        for pm, s in r.per_proxy_stats.items()
                    },
                }
                for r in val_results
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_validation_csv(output_dir, val_results)
    _write_candidate_metrics(output_dir, val_results)
    _write_routing_audit(output_dir, val_results)
    _write_region_summary(output_dir, rg_regions)

    # Final cheatsheets
    for mode in ("global", "routed"):
        cs = build_cheatsheet(
            anchor_text,
            accepted_rules,
            mode=mode,
            router_min_matches=args.router_min_matches,
            allow_empty_use_when_global=args.allow_empty_use_when_global,
            router_type=args.router_type,
            task=args.task,
            memory_format=args.memory_format,
        )
        (output_dir / f"final_cheatsheet_{mode}.txt").write_text(cs, encoding="utf-8")

    # JSONL log
    logger = SFCRLogger(output_dir / "sfcr_log.jsonl", run_id=args.output_dir)

    rg_shared_ids  = {it["_sfcr_id"] for it in rg_regions.V_shared}
    rg_private_ids = {it["_sfcr_id"] for it in rg_regions.V_private}
    rg_easy_ids    = {it["_sfcr_id"] for it in rg_regions.V_easy}

    anchor_cb_by_model = {
        model: {iid: cb for iid, cb in cb_map.items()}
        for model, cb_map in gate_baseline.correct_by_model.items()
    }
    log_condition(
        logger=logger,
        items=splits.gate,
        scored_by_model=anchor_cb_by_model,
        cheatsheet_text=anchor_text,
        condition="anchor",
        accepted_rules=[],
        routing_mode=args.routing_mode,
        oracle_mode=args.oracle_mode,
        task=args.task,
        dataset=args.dataset,
        seed=args.seed,
        source_model=args.model_source,
        v_shared_ids=rg_shared_ids,
        v_private_ids=rg_private_ids,
        v_easy_ids=rg_easy_ids,
    )

    logger.close()

    act_summary = activation_summary(
        accepted_rules,
        splits.gate,
        router_min_matches=args.router_min_matches,
        allow_empty_use_when_global=args.allow_empty_use_when_global,
    )
    (output_dir / "activation_summary.json").write_text(
        json.dumps(act_summary, indent=2), encoding="utf-8"
    )

    _write_summary(
        output_dir, args,
        accepted_rules=accepted_rules,
        skipped=False,
        skip_reason=None,
        regions=rg_regions,
        val_results=val_results,
    )

    print(f"\n{'='*60}")
    print(f" SFCR complete — {len(accepted_rules)} rule(s) accepted")
    print(f" Output dir: {output_dir}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Summary file
# ---------------------------------------------------------------------------

def _write_global_routed_comparison(output_dir: Path, rows: list[dict]) -> None:
    (output_dir / "validation_global_vs_routed.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    csv_path = output_dir / "validation_global_vs_routed.csv"
    if rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")


def _write_validation_csv(output_dir: Path, val_results: list) -> None:
    rows = []
    for r in val_results:
        rows.append({
            "candidate_id": r.rule.get("id") or rule_id(r.rule),
            "task": r.rule.get("task", ""),
            "source_of_candidate": r.rule.get("source_of_candidate", ""),
            "validation_routing_mode": r.validation_routing_mode,
            "gate_mode": r.gate_mode,
            "accepted": r.accepted,
            "u_lcb": r.u_lcb,
            "reject_reason": r.reject_reason or "",
            "benefit_models": ",".join(r.benefit_models),
            "safety_models": ",".join(r.safety_models),
            "fixed_shared_count": max((s.fixed_shared_count for s in r.per_proxy_stats.values()), default=0),
            "reg_easy_count": max((s.reg_easy_count for s in r.per_proxy_stats.values()), default=0),
            "reg_private_count": max((s.reg_private_count for s in r.per_proxy_stats.values()), default=0),
            "private_activation_count": max((s.private_activation_count for s in r.per_proxy_stats.values()), default=0),
        })
    csv_path = output_dir / "validation_results.csv"
    if rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")


def _write_routing_audit(output_dir: Path, val_results: list) -> None:
    """Write per-item routing decisions to routing_audit.jsonl and routing_audit.csv."""
    all_entries: list[dict] = []
    for r in val_results:
        for entry in r.per_item_routing:
            all_entries.append({
                **entry,
                "candidate_status": r.candidate_status,
                "accepted": r.accepted,
            })

    jsonl_path = output_dir / "routing_audit.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for entry in all_entries:
            fh.write(json.dumps(entry) + "\n")

    csv_path = output_dir / "routing_audit.csv"
    if all_entries:
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_entries[0].keys()))
            writer.writeheader()
            writer.writerows(all_entries)
    else:
        csv_path.write_text("", encoding="utf-8")


def _write_candidate_metrics(output_dir: Path, val_results: list) -> None:
    metrics = []
    for r in val_results:
        use_when = r.rule.get("use_when", "")
        do_not = r.rule.get("do_not_use_when", "")
        metrics.append({
            "candidate_id": r.rule.get("id") or rule_id(r.rule),
            "source_of_candidate": r.rule.get("source_of_candidate", ""),
            "generator_model": r.rule.get("generator_model", ""),
            "mean_rule_length": len(r.rule.get("rule", "")),
            "mean_use_when_length": len(use_when),
            "mean_do_not_use_when_length": len(do_not),
            "activation_precision": max(
                (
                    s.activated_shared_count / s.activation_count
                    for s in r.per_proxy_stats.values()
                    if s.activation_count
                ),
                default=0.0,
            ),
            "activated_shared_count": max((s.activated_shared_count for s in r.per_proxy_stats.values()), default=0),
            "private_activation_count": max((s.private_activation_count for s in r.per_proxy_stats.values()), default=0),
            "easy_activation_count": max((s.easy_activation_count for s in r.per_proxy_stats.values()), default=0),
            "fixed_shared_count": max((s.fixed_shared_count for s in r.per_proxy_stats.values()), default=0),
            "reg_easy_count": max((s.reg_easy_count for s in r.per_proxy_stats.values()), default=0),
            "reg_private_count": max((s.reg_private_count for s in r.per_proxy_stats.values()), default=0),
        })
    (output_dir / "candidate_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


def _write_region_summary(output_dir: Path, regions) -> None:
    summary = {
        "source_accuracy": regions.source_accuracy,
        "F_s": len(regions.F_s),
        "V_shared": len(regions.V_shared),
        "V_private": len(regions.V_private),
        "V_easy": len(regions.V_easy),
        "skip_reason": regions.skip_reason,
        "per_model_correct": {
            model: len(ids) for model, ids in regions.per_model_correct.items()
        },
        "per_model_wrong": {
            model: len(ids) for model, ids in regions.per_model_wrong.items()
        },
    }
    (output_dir / "region_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def _write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    accepted_rules: list[dict],
    skipped: bool,
    skip_reason: str | None,
    regions,
    val_results: list | None = None,
) -> None:
    summary = {
        "task":            args.task,
        "seed":            args.seed,
        "oracle_mode":     args.oracle_mode,
        "routing_mode":    args.routing_mode,
        "router_type":     getattr(args, "router_type", "keyword"),
        "memory_format":   getattr(args, "memory_format", "rule"),
        "gate_profile":    getattr(args, "gate_profile", "auto"),
        "model_source":    args.model_source,
        "models_proxy":    args.models_proxy,
        "held_out_target": args.held_out_target,
        "n_eval_seeds":    args.n_eval_seeds,
        "use_subtypes":    not args.no_subtypes,
        "repair_attempts": args.repair_attempts,
        "skipped":         skipped,
        "skip_reason":     skip_reason,
        "source_accuracy": round(regions.source_accuracy, 4),
        # Region sizes and denominators
        "v_shared_size":   len(regions.V_shared),
        "v_private_size":  len(regions.V_private),
        "v_easy_size":     len(regions.V_easy),
        "f_s_size":        len(regions.F_s),
        "jaccard_matrix":  {f"{k[0]}↔{k[1]}": round(v, 4)
                            for k, v in regions.jaccard_matrix.items()},
        "n_accepted":      len(accepted_rules),
        "accepted_rules":  [r.get("rule", "")[:80] for r in accepted_rules],
    }
    if val_results is not None:
        summary["n_candidates"] = len(val_results)
        summary["n_rejected"]   = sum(1 for r in val_results if not r.accepted)
        summary["n_count_gate"] = sum(1 for r in val_results if r.count_gate_used)
        summary["n_repaired"]   = sum(1 for r in val_results if r.rule.get("repaired"))

    (output_dir / "sfcr_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
