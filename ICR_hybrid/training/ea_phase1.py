"""
ICR_hybrid/training/ea_phase1.py — Evolutionary Algorithm for Phase 1 PK refinement.

Replaces single-path _pk_patch_phase with a 3-member population search:
  - Each member holds a distinct mutation prompt slot:
      conservative  — minimal edits, clarify wording only
      generative    — free rewrite, new structure allowed
      structural    — explicit TIGHTEN / SPLIT / ADD_GUARD operation

  - Per generation:
      1. Generate candidates in parallel (one LLM call per member)
      2. Gate each: net-gain + curriculum-scaled λ + cross-member regression
      3. Score all members; update best_ever (elitism)
      4. Select top-n_survivors; crossover to refill population
      5. Convergence check

  - Overfitting protection:
      80/20 train/val split  — val is never seen by mutation prompts
      Cross-member regression — patch for member A is also checked against
                                sibling members' correct pools

  - Returns (n_total_patches, best_ever_val_acc, generations_done)
    and sets cheatsheet.prior_knowledge to the best PK found.
"""

from __future__ import annotations

import copy
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from utils.cheatsheet import Cheatsheet
from utils.llm_client import call_llm
from utils.scorer import score_batch
from utils.task_spec import TaskSpec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SLOTS = ("conservative", "generative", "structural")

_SLOT_TEMPERATURE: dict[str, float] = {
    "conservative": 0.2,
    "generative":   0.7,
    "structural":   0.3,
}


# ---------------------------------------------------------------------------
# Member dataclass
# ---------------------------------------------------------------------------

@dataclass
class Member:
    pk_text:     str
    slot:        str          # one of SLOTS
    accuracy:    float = 0.0  # on train split
    correct:     list  = field(default_factory=list)
    wrong:       list  = field(default_factory=list)
    n_accepted:  int   = 0
    static_gens: int   = 0
    prev_shown:  set   = field(default_factory=set)   # item ids shown last gen


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def _gate_accepts(
    n_fixed:          int,
    n_regressed:      int,
    n_correct_pool:   int,
    gen:              int,
    max_generations:  int,
    lambda_min:       float = 1.0,
    lambda_max:       float = 2.0,
    min_fix_count:    int   = 1,
    regress_hard_cap: int   = 3,
) -> tuple[bool, str]:
    """
    Combined gate: net-gain with curriculum λ + pool-size-aware absolute cap.

    λ ramps from lambda_min (gen 1, loose) to lambda_max (final gen, strict).
    This means early patches can break even on regressions; later patches
    must fix strictly more than they regress.
    """
    if n_fixed < min_fix_count:
        return False, f"n_fixed={n_fixed} < min_fix_count={min_fix_count}"

    progress = gen / max(1, max_generations - 1) if max_generations > 1 else 1.0
    lam = lambda_min + (lambda_max - lambda_min) * progress
    if n_regressed > 0 and n_fixed < lam * n_regressed:
        return False, f"net-gain: {n_fixed} < λ={lam:.2f} × {n_regressed}"

    cap = max(regress_hard_cap, int(0.20 * n_correct_pool)) if n_correct_pool > 0 else regress_hard_cap
    if n_regressed > cap:
        return False, f"regress cap: {n_regressed} > {cap} (pool={n_correct_pool})"

    return True, ""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _failure_block(items: list[dict], oracle, inject_oracle: bool) -> str:
    blocks = []
    for idx, item in enumerate(items, 1):
        inp = str(item.get("input", "")).strip()[:300]
        exp = str(item.get("expected", item.get("answer", "?"))).strip()
        got = str(item.get("predicted", "?"))
        pt  = (item.get("post_think") or item.get("thinking") or "").strip()
        lines = [
            f"[{idx}]",
            f"  Input:    {inp}",
            f"  Expected: {exp}  |  Got: {got}",
        ]
        if pt:
            lines.append(f"  Wrong reasoning:\n    {pt[:500]}")
        if inject_oracle:
            oracle_text = (
                item.get("_oracle_exact", "")
                or item.get("reason", "")
                or item.get("gold_reason", "")
            )
            if not oracle_text and oracle is not None:
                oracle_text = (oracle.get(str(item.get("id", ""))) or {}).get("explanation", "")
            if oracle_text:
                lines.append(f"  Correct reasoning:\n    {oracle_text[:600]}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _mutation_prompt(
    slot:          str,
    pk_text:       str,
    failures:      list[dict],
    oracle,
    inject_oracle: bool,
    size_budget:   int,
) -> str:
    fb = _failure_block(failures, oracle, inject_oracle)
    header = (
        f"You are refining a knowledge guide that helps a model answer questions correctly.\n\n"
        f"=== CURRENT KNOWLEDGE GUIDE ({len(pk_text)} chars) ===\n"
        f"{pk_text}\n"
        f"=== END KNOWLEDGE GUIDE ===\n\n"
        f"The model is making the following {len(failures)} errors:\n\n"
        f"{fb}\n\n---\n"
    )
    requirements = (
        f"\nRequirements:\n"
        f"  - Output must not exceed {size_budget} characters\n"
        f"  - Preserve rules that are working correctly\n"
        f"  - Focus on ABSTRACT REASONING PRINCIPLES: do not reproduce specific input\n"
        f"    text, entity names, or numbers from the failure examples above\n"
        f"  - Write principles that would help any capable model on new similar questions\n"
        f"  - Output ONLY the improved knowledge guide — no preamble, no commentary, no fences"
    )

    if slot == "conservative":
        instruction = (
            "Make MINIMAL changes to the knowledge guide.\n"
            "Only clarify ambiguous wording in existing rules.\n"
            "Do not add new rules unless absolutely necessary.\n"
            "Prefer editing one sentence over rewriting a paragraph."
        )
    elif slot == "generative":
        instruction = (
            "Freely REWRITE the failing section of the knowledge guide.\n"
            "You may introduce new rule structure, reorganise content, or add new principles.\n"
            "Prioritise correctness and generality over preserving original wording."
        )
    else:  # structural
        instruction = (
            "Apply exactly ONE of the following operations and name it on the first line:\n"
            "  TIGHTEN  — make an existing rule more specific to reduce false application\n"
            "  SPLIT    — divide one rule into two cases handled differently\n"
            "  ADD_GUARD — add an exception condition to an existing rule\n"
            "After naming the operation, output the full updated knowledge guide."
        )

    return header + instruction + requirements


def _merge_prompt(
    pk_a:            str,
    pk_b:            str,
    acc_a:           float,
    acc_b:           float,
    slot_a:          str,
    slot_b:          str,
    unique_wins_a:   list[dict],
    unique_wins_b:   list[dict],
    shared_failures: list[dict],
    size_budget:     int,
) -> str:
    def _fmt(items: list[dict], n: int = 3) -> str:
        out = []
        for item in items[:n]:
            inp = str(item.get("input", "")).strip()[:200]
            ans = str(item.get("answer", "?")).strip()
            out.append(f"  Q: {inp}\n  A: {ans}")
        return "\n".join(out) or "  (none)"

    return (
        f"You are merging two prior-knowledge guides for the same task.\n\n"
        f"=== PARENT A (train_acc={acc_a:.1%}, style={slot_a}) ===\n{pk_a}\n"
        f"=== END PARENT A ===\n\n"
        f"=== PARENT B (train_acc={acc_b:.1%}, style={slot_b}) ===\n{pk_b}\n"
        f"=== END PARENT B ===\n\n"
        f"Items Parent A uniquely gets right ({len(unique_wins_a)} total, showing up to 3):\n"
        f"{_fmt(unique_wins_a)}\n\n"
        f"Items Parent B uniquely gets right ({len(unique_wins_b)} total, showing up to 3):\n"
        f"{_fmt(unique_wins_b)}\n\n"
        f"Items both still get wrong ({len(shared_failures)} total, showing up to 3):\n"
        f"{_fmt(shared_failures)}\n\n"
        f"---\n"
        f"Merge the best insights from both parents into a single coherent knowledge guide.\n"
        f"Requirements:\n"
        f"  - Preserve what makes each parent uniquely correct on their respective items above\n"
        f"  - Address the shared failures if possible\n"
        f"  - Output must not exceed {size_budget} characters\n"
        f"  - Output ONLY the merged knowledge guide — no preamble, no commentary, no fences"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cs(base: Cheatsheet, pk_text: str) -> Cheatsheet:
    return Cheatsheet(
        roadmap=base.roadmap,
        case_studies=list(base.case_studies),
        prior_knowledge=pk_text,
        no_limit=getattr(base, "no_limit", False),
    )


def _item_key(item: dict) -> str:
    return str(item.get("id", item.get("input", id(item))))


def _partition_failures(wrong: list[dict], n_parts: int, rng: random.Random) -> list[list[dict]]:
    """
    Divide wrong items into n_parts disjoint slices of roughly equal size.
    Used in generation 1 so each member sees a different failure subset
    rather than all starting from the same random sample.
    """
    shuffled = list(wrong)
    rng.shuffle(shuffled)
    size = max(1, len(shuffled) // n_parts)
    parts = []
    for i in range(n_parts):
        chunk = shuffled[i * size: (i + 1) * size]
        if chunk:
            parts.append(chunk)
    # If fewer chunks than parts (tiny wrong set), duplicate last chunk
    while len(parts) < n_parts:
        parts.append(parts[-1])
    return parts


def _sample_failures(
    wrong:      list[dict],
    n_to_show:  int,
    frac:       float,
    prev_shown: set,
    rng:        random.Random,
) -> list[dict]:
    """
    Sample failures with rotation: prefer items not shown last generation.
    Only called from gen 2 onward; gen 1 uses _partition_failures instead.
    """
    n = min(n_to_show, max(1, int(len(wrong) * frac)))
    fresh = [it for it in wrong if _item_key(it) not in prev_shown]
    if len(fresh) >= n:
        return rng.sample(fresh, n)
    used = [it for it in wrong if _item_key(it) in prev_shown]
    combined = fresh + rng.sample(used, min(n - len(fresh), len(used)))
    return combined[:n]


def _sample_correct(correct: list[dict], n: int, rng: random.Random) -> list[dict]:
    if not correct:
        return []
    return rng.sample(correct, min(n, len(correct)))


def _split_train_val(
    items:        list[dict],
    val_fraction: float,
    seed:         int = 42,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]   # (train, val)


def _log_leaderboard(population: list[Member], log_fn: Callable) -> None:
    log_fn("  Leaderboard:")
    for m in sorted(population, key=lambda m: m.accuracy, reverse=True):
        log_fn(
            f"    [{m.slot:14s}]  train_acc={m.accuracy:.1%}  "
            f"accepted={m.n_accepted}  static={m.static_gens}"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ea_pk_phase(
    cheatsheet:          Cheatsheet,
    train_items:         list[dict],
    model_patch:         str,
    model_score:         str,
    api_key:             str,
    oracle,
    max_generations:     int,
    acc_goal:            float,
    static_gens_limit:   int,
    concurrency:         int,
    log_fn:              Callable,
    task_spec:           TaskSpec,
    reasoning_effort:    "str | None",
    cot_first:           bool,
    n_failures_to_show:  int   = 15,
    inject_oracle:       bool  = True,
    val_fraction:        float = 0.20,
    population_size:     int   = 3,
    n_survivors:         int   = 2,
    lambda_min:          float = 1.0,
    lambda_max:          float = 2.0,
    regress_hard_cap:    int   = 3,
    pk_size_budget:      int   = 12_000,
    failure_sample_frac: float = 0.60,
    seed:                int   = 42,
) -> tuple[int, float, int]:
    """
    EA-based Phase 1 PK refinement.

    Returns (n_total_patches, best_val_accuracy, generations_done).
    Writes best PK found to cheatsheet.prior_knowledge.
    """
    rng = random.Random(seed)

    def _do_score(items: list[dict], cs_text: str, label: str):
        return score_batch(
            items, cs_text, model_score, api_key,
            concurrency=min(concurrency, max(1, len(items))),
            reasoning_effort=reasoning_effort,
            cot_first=cot_first,
            progress_label=label,
            task_spec=task_spec,
        )

    # ── 1. Train / val split ─────────────────────────────────────────────────
    train_split, val_split = _split_train_val(train_items, val_fraction, seed)
    log_fn(
        f"\n[ea_phase1] train={len(train_split)}  val={len(val_split)}  "
        f"pop={population_size}  max_gen={max_generations}  "
        f"λ={lambda_min}→{lambda_max}  budget={pk_size_budget:,} chars"
    )

    # ── 2. Initialise population (all start from same PK) ───────────────────
    slots = list(SLOTS)[:population_size]
    initial_pk = cheatsheet.prior_knowledge.strip()
    population: list[Member] = [Member(pk_text=initial_pk, slot=s) for s in slots]

    # ── 3. Initial score ─────────────────────────────────────────────────────
    log_fn("\n[ea_phase1] Initial scoring ...")
    for m in population:
        m.correct, m.wrong = _do_score(
            train_split, _make_cs(cheatsheet, m.pk_text).render(), f"init/{m.slot}"
        )
        m.accuracy = len(m.correct) / len(train_split) if train_split else 0.0
        log_fn(f"  [{m.slot}]  train_acc={m.accuracy:.1%}  ({len(m.wrong)} failures)")

    # best_ever tracked by (pk_text, accuracy) to avoid aliasing bugs
    best_ever_pk   = max(population, key=lambda m: m.accuracy).pk_text
    best_ever_acc  = max(m.accuracy for m in population)
    best_ever_slot = next(m.slot for m in population if m.pk_text == best_ever_pk)

    n_total_patches = 0
    gens_done = 0

    # ── 4. Generation loop ───────────────────────────────────────────────────
    for gen in range(1, max_generations + 1):
        gens_done = gen
        log_fn(f"\n[ea_phase1] ══════ Generation {gen}/{max_generations} ══════")
        _log_leaderboard(population, log_fn)

        if all(m.accuracy >= acc_goal for m in population):
            log_fn(f"  All members ≥ acc_goal {acc_goal:.0%} — stopping early.")
            break

        # ── 4a. Generate candidates in parallel ──────────────────────────────
        # Gen 1: partition wrong set into disjoint slices (one per member) so
        # each member addresses a different failure mode from the start.
        # Gen 2+: normal rotation sampling (prefer items not shown last gen).
        per_member: list[tuple[list[dict], str | None]] = [([], None)] * len(population)

        is_gen1 = all(len(m.prev_shown) == 0 for m in population)
        if is_gen1:
            # All members share the same wrong set on gen 1 (same initial PK)
            shared_wrong = population[0].wrong
            partitions = _partition_failures(shared_wrong, len(population), rng)
            gen1_shown = {i: part[:n_failures_to_show] for i, part in enumerate(partitions)}
        else:
            gen1_shown = {}

        def _generate(idx: int, m: Member) -> tuple[int, list[dict], str | None]:
            shown = (
                gen1_shown[idx] if idx in gen1_shown
                else _sample_failures(m.wrong, n_failures_to_show, failure_sample_frac, m.prev_shown, rng)
            )
            if not shown:
                return idx, shown, None
            prompt = _mutation_prompt(
                slot=m.slot, pk_text=m.pk_text, failures=shown,
                oracle=oracle, inject_oracle=inject_oracle,
                size_budget=pk_size_budget,
            )
            max_tok = min(4000, max(800, int(len(m.pk_text) * 1.6)))
            resp = call_llm(
                prompt, model=model_patch, api_key=api_key,
                max_tokens=max_tok, temperature=_SLOT_TEMPERATURE[m.slot],
            )
            cand = resp.content.strip() if resp else None
            return idx, shown, cand

        with ThreadPoolExecutor(max_workers=len(population)) as ex:
            futures = [ex.submit(_generate, i, m) for i, m in enumerate(population)]
            for fut in futures:
                idx, shown, cand_pk = fut.result()
                per_member[idx] = (shown, cand_pk)

        # ── 4b. Gate each candidate ──────────────────────────────────────────
        for i, (m, (shown, cand_pk)) in enumerate(zip(population, per_member)):
            if not cand_pk:
                log_fn(f"  [{m.slot}] empty response — skipping")
                m.static_gens += 1
                continue
            if len(cand_pk) > pk_size_budget:
                log_fn(f"  [{m.slot}] oversized ({len(cand_pk)} chars > {pk_size_budget}) — skipping")
                m.static_gens += 1
                continue

            log_fn(f"  [{m.slot}] candidate {len(cand_pk)} chars — gating ...")
            cand_cs = _make_cs(cheatsheet, cand_pk)

            # Fix-rate on the same failures shown to the LLM
            new_correct, _ = _do_score(shown, cand_cs.render(), f"fix/{m.slot}")
            n_fixed = len(new_correct)

            # Own regression: sample from own correct pool
            own_sample = _sample_correct(m.correct, n=20, rng=rng)
            _, own_regressed = _do_score(own_sample, cand_cs.render(), f"own-reg/{m.slot}")

            # Cross-member regression: sample from each sibling's correct pool
            sibling_sample = [
                item
                for j, sib in enumerate(population) if j != i
                for item in _sample_correct(sib.correct, n=10, rng=rng)
            ]
            cross_regressed = []
            if sibling_sample:
                _, cross_regressed = _do_score(
                    sibling_sample, cand_cs.render(), f"cross-reg/{m.slot}"
                )

            n_regressed    = len(own_regressed) + len(cross_regressed)
            n_correct_pool = len(own_sample) + len(sibling_sample)

            ok, reason = _gate_accepts(
                n_fixed, n_regressed, n_correct_pool,
                gen, max_generations,
                lambda_min=lambda_min, lambda_max=lambda_max,
                regress_hard_cap=regress_hard_cap,
            )

            if ok:
                m.pk_text   = cand_pk
                m.n_accepted += 1
                m.static_gens = 0
                n_total_patches += 1
                log_fn(
                    f"  [{m.slot}] ACCEPTED  n_fixed={n_fixed}  "
                    f"n_regressed={n_regressed}  pk={len(cand_pk)} chars"
                )
                m.correct, m.wrong = _do_score(
                    train_split, cand_cs.render(), f"rescore/{m.slot}"
                )
                m.accuracy = len(m.correct) / len(train_split) if train_split else 0.0
                log_fn(f"  [{m.slot}] new train_acc={m.accuracy:.1%}")
            else:
                m.static_gens += 1
                log_fn(f"  [{m.slot}] REJECTED — {reason}")

            m.prev_shown = {_item_key(it) for it in shown}

        # ── 4c. Update best_ever (elitism) ──────────────────────────────────
        for m in population:
            if m.accuracy > best_ever_acc:
                best_ever_pk   = m.pk_text
                best_ever_acc  = m.accuracy
                best_ever_slot = m.slot
                log_fn(f"  New best_ever: [{m.slot}] train_acc={m.accuracy:.1%}")

        # ── 4d. Select survivors ─────────────────────────────────────────────
        population.sort(key=lambda m: m.accuracy, reverse=True)
        survivors = population[:n_survivors]

        # ── 4e. Crossover ────────────────────────────────────────────────────
        if len(survivors) >= 2:
            pa, pb = survivors[0], survivors[1]

            pb_inputs = {_item_key(it) for it in pb.correct}
            pa_inputs = {_item_key(it) for it in pa.correct}
            unique_a  = [it for it in pa.correct if _item_key(it) not in pb_inputs]
            unique_b  = [it for it in pb.correct if _item_key(it) not in pa_inputs]
            shared_f  = [
                it for it in pa.wrong
                if _item_key(it) in {_item_key(x) for x in pb.wrong}
            ]

            log_fn(
                f"\n  Crossover: [{pa.slot}] × [{pb.slot}]  "
                f"unique_a={len(unique_a)}  unique_b={len(unique_b)}  shared_fail={len(shared_f)}"
            )
            merge_p = _merge_prompt(
                pk_a=pa.pk_text, pk_b=pb.pk_text,
                acc_a=pa.accuracy, acc_b=pb.accuracy,
                slot_a=pa.slot, slot_b=pb.slot,
                unique_wins_a=unique_a, unique_wins_b=unique_b,
                shared_failures=shared_f,
                size_budget=pk_size_budget,
            )
            max_tok  = min(4000, max(800, pk_size_budget // 4))
            resp     = call_llm(merge_p, model=model_patch, api_key=api_key,
                                max_tokens=max_tok, temperature=0.3)
            child_pk = resp.content.strip() if resp else None

            if child_pk and len(child_pk) <= pk_size_budget:
                child = Member(pk_text=child_pk, slot=pb.slot)
                child.correct, child.wrong = _do_score(
                    train_split, _make_cs(cheatsheet, child_pk).render(),
                    f"crossover/{pb.slot}",
                )
                child.accuracy = len(child.correct) / len(train_split) if train_split else 0.0
                log_fn(f"  Crossover child [{pb.slot}] train_acc={child.accuracy:.1%}")
            else:
                log_fn("  Crossover produced oversized or empty result — cloning parent B")
                child = copy.deepcopy(pb)
                child.n_accepted = 0
                child.static_gens = 0

            # Rebuild population: survivors + child, padded to population_size
            population = list(survivors)
            while len(population) < population_size:
                population.append(child)

            # Elitism: ensure best_ever is represented
            pop_pks = {m.pk_text for m in population}
            if best_ever_pk not in pop_pks:
                elite = Member(
                    pk_text=best_ever_pk, slot=best_ever_slot,
                    accuracy=best_ever_acc,
                )
                elite.correct, elite.wrong = _do_score(
                    train_split, _make_cs(cheatsheet, best_ever_pk).render(), "elite-rescore"
                )
                population[-1] = elite   # replace weakest (last after sort)
                log_fn(f"  Elite member reinjected [{best_ever_slot}] acc={best_ever_acc:.1%}")

        # ── 4f. Convergence check ────────────────────────────────────────────
        if all(m.static_gens >= static_gens_limit for m in population):
            log_fn(f"  All members static for ≥{static_gens_limit} gens — stopping.")
            break

    # ── 5. Final val evaluation of best_ever ────────────────────────────────
    log_fn(f"\n[ea_phase1] Evaluating best_ever on val split ({len(val_split)} items) ...")
    val_correct, _ = _do_score(
        val_split, _make_cs(cheatsheet, best_ever_pk).render(), "final-val"
    )
    val_acc = len(val_correct) / len(val_split) if val_split else 0.0
    log_fn(
        f"[ea_phase1] Done — slot={best_ever_slot}  "
        f"train_acc={best_ever_acc:.1%}  val_acc={val_acc:.1%}  "
        f"total_patches={n_total_patches}  generations={gens_done}  "
        f"pk={len(best_ever_pk):,} chars"
    )

    cheatsheet.prior_knowledge = best_ever_pk
    return n_total_patches, val_acc, gens_done
