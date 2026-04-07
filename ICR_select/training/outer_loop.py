"""
ICR_select/training/outer_loop.py — Outer DT revision loop.

Wraps any ICR inner loop (ICR_naive, ICR_reasoning, ICR_select) in an outer
round structure:

    Round 1:  CS loop (DT frozen) → collect all failures with COT traces
              → DT revision pass → validate → accept/reject
    Round 2:  CS loop on revised DT (case studies reset) → ...
    ...

Stops when:
  - max_rounds reached, OR
  - DT revision is rejected (accuracy can't improve further), OR
  - accuracy improvement between rounds < plateau_threshold

The inner loop callable must match the signature of run_training_loop from
any ICR mode — it receives (cheatsheet, train_items, val_items, ...) and
returns a TrainingResult-compatible object with at least:
  .cheatsheet       Cheatsheet
  .train_accuracy   float
  .val_accuracy     float | None
  .update_log       list[dict]
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from utils.cheatsheet import Cheatsheet
from utils.scorer import score_batch
from .dt_reviser import run_dt_revision, DtRevisionResult


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class RoundResult:
    round_num: int
    train_accuracy: float
    val_accuracy: float | None
    n_case_studies: int
    dt_revised: bool
    dt_accuracy_before: float
    dt_accuracy_after: float


@dataclass
class OuterLoopResult:
    final_cheatsheet: Cheatsheet
    rounds: list[RoundResult]
    best_train_accuracy: float
    best_val_accuracy: float | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_all_failures(train_items: list[dict], cheatsheet: Cheatsheet,
                          model_score: str, api_key: str, concurrency: int,
                          reasoning_effort: str | None, cot_first: bool,
                          log_fn) -> list[dict]:
    """Re-score the full training set and return all wrong items with COT."""
    log_fn(f"  [outer] re-scoring {len(train_items)} items to collect failure COTs ...")
    _, wrong = score_batch(
        train_items, cheatsheet.render(), model_score, api_key,
        concurrency=concurrency, reasoning_effort=reasoning_effort, cot_first=cot_first,
    )
    log_fn(f"  [outer] {len(wrong)} failures collected.")
    return wrong


def _save_round(output_dir: Path, round_num: int,
                cheatsheet: Cheatsheet, dt_result: DtRevisionResult | None,
                round_result: RoundResult, all_rounds: list[RoundResult]) -> None:
    rd = output_dir / f"round_{round_num:02d}"
    rd.mkdir(parents=True, exist_ok=True)
    cheatsheet.save(rd / "cheatsheet_end_of_round")
    if dt_result:
        (rd / "dt_revision.json").write_text(
            json.dumps({
                "accepted": dt_result.accepted,
                "accuracy_before": dt_result.accuracy_before,
                "accuracy_after": dt_result.accuracy_after,
                "step_analysis": dt_result.step_analysis,
                "revised_dt": dt_result.revised_dt,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    (output_dir / "outer_loop_summary.json").write_text(
        json.dumps(
            [{"round": r.round_num, "train_acc": r.train_accuracy,
              "val_acc": r.val_accuracy, "dt_revised": r.dt_revised,
              "n_case_studies": r.n_case_studies}
             for r in all_rounds],
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Outer loop
# ---------------------------------------------------------------------------

def run_outer_loop(
    initial_cheatsheet: Cheatsheet,
    train_items: list[dict],
    val_items: list[dict] | None,
    inner_loop_fn: Callable,
    inner_loop_kwargs: dict,
    model_score: str,
    model_casestudy: str,
    api_key: str,
    # Outer loop config
    max_rounds: int = 3,
    plateau_threshold: float = 0.02,   # stop if improvement < 2%
    keep_case_studies: bool = False,   # whether to carry over CS between rounds
    min_failures_for_dt: int = 5,
    # Scoring config (for failure collection + DT validation)
    concurrency: int = 10,
    reasoning_effort: str | None = "low",
    cot_first: bool = True,
    output_dir: Path | None = None,
    log: bool = True,
) -> OuterLoopResult:

    def _log(msg: str) -> None:
        if log:
            print(msg, file=sys.stderr, flush=True)

    cheatsheet    = initial_cheatsheet
    rounds        : list[RoundResult] = []
    best_train    = 0.0
    best_val      = None
    best_cs       = cheatsheet

    _log(
        f"\n{'='*60}\n"
        f"Outer DT Revision Loop\n"
        f"  max_rounds={max_rounds}  plateau_threshold={plateau_threshold:.0%}\n"
        f"  keep_case_studies={keep_case_studies}\n"
        f"{'='*60}"
    )

    for round_num in range(1, max_rounds + 1):
        _log(f"\n{'─'*60}\nROUND {round_num}/{max_rounds}\n{'─'*60}")

        round_output = (output_dir / f"round_{round_num:02d}") if output_dir else None

        # ── Inner CS loop ────────────────────────────────────────────────────
        inner_result = inner_loop_fn(
            cheatsheet=cheatsheet,
            train_items=train_items,
            val_items=val_items,
            output_dir=round_output,
            **inner_loop_kwargs,
        )
        cheatsheet = inner_result.cheatsheet

        if inner_result.train_accuracy > best_train:
            best_train = inner_result.train_accuracy
            best_cs    = cheatsheet
        if inner_result.val_accuracy is not None:
            if best_val is None or inner_result.val_accuracy > best_val:
                best_val = inner_result.val_accuracy

        _log(
            f"\n[Round {round_num}] inner loop done — "
            f"train_acc={inner_result.train_accuracy:.1%}  "
            f"case_studies={len(cheatsheet.case_studies)}"
        )

        # ── Collect failures with COT for DT revision ────────────────────────
        failures = _collect_all_failures(
            train_items, cheatsheet, model_score, api_key,
            concurrency, reasoning_effort, cot_first, _log,
        )

        # ── DT revision pass ─────────────────────────────────────────────────
        dt_result = run_dt_revision(
            failures=failures,
            train_seen=train_items,   # use full train set for validation
            cheatsheet=cheatsheet,
            model_casestudy=model_casestudy,
            model_score=model_score,
            api_key=api_key,
            concurrency=concurrency,
            reasoning_effort=reasoning_effort,
            cot_first=cot_first,
            min_failures=min_failures_for_dt,
            log=log,
        )

        round_result = RoundResult(
            round_num=round_num,
            train_accuracy=inner_result.train_accuracy,
            val_accuracy=inner_result.val_accuracy,
            n_case_studies=len(cheatsheet.case_studies),
            dt_revised=dt_result.accepted,
            dt_accuracy_before=dt_result.accuracy_before,
            dt_accuracy_after=dt_result.accuracy_after,
        )
        rounds.append(round_result)

        if output_dir:
            _save_round(output_dir, round_num, cheatsheet, dt_result, round_result, rounds)

        # ── Apply DT revision (if accepted) → prepare next round ─────────────
        if dt_result.accepted:
            _log(
                f"  [outer] DT revised: "
                f"{dt_result.accuracy_before:.1%} → {dt_result.accuracy_after:.1%}"
            )
            new_case_studies = cheatsheet.case_studies if keep_case_studies else []
            cheatsheet = Cheatsheet(
                decision_tree=dt_result.revised_dt,
                case_studies=new_case_studies,
            )
        else:
            _log("  [outer] DT revision not accepted — stopping outer loop.")
            break

        # ── Plateau check ─────────────────────────────────────────────────────
        if round_num >= 2:
            improvement = rounds[-1].train_accuracy - rounds[-2].train_accuracy
            _log(f"  [outer] accuracy improvement this round: {improvement:+.1%}")
            if improvement < plateau_threshold:
                _log(
                    f"  [outer] improvement {improvement:.1%} < "
                    f"plateau_threshold {plateau_threshold:.1%} — stopping."
                )
                break

    _log(
        f"\n{'='*60}\n"
        f"Outer loop complete.\n"
        f"  rounds_run        : {len(rounds)}\n"
        f"  best_train_acc    : {best_train:.1%}\n"
        f"  best_val_acc      : {f'{best_val:.1%}' if best_val is not None else 'N/A'}\n"
        f"  {best_cs.summary()}\n"
        f"{'='*60}"
    )

    return OuterLoopResult(
        final_cheatsheet=best_cs,
        rounds=rounds,
        best_train_accuracy=best_train,
        best_val_accuracy=best_val,
    )
