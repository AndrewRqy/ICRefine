"""
training/loop.py — HypoGenic-style training loop.

Process training items in mini-batches:
  1. Score the batch against the current cheatsheet.
  2. Collect failures into the failure bin.
  3. When the bin reaches bin_threshold, flush it:
       a. Generate a new case study from the failures.
       b. Append it to the cheatsheet.
       c. Save a checkpoint.
       d. Clear the bin.
  4. Continue until all items are processed.
  5. Optionally flush any remainder at the end.
  6. Score the final cheatsheet on the validation set.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from ..core.cheatsheet import Cheatsheet
from ..core.data import FailureBin
from ..generators.case_study import generate_case_study
from .scorer import score_batch, test_cheatsheet


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    cheatsheet: Cheatsheet
    n_case_studies_added: int
    train_accuracy: float
    val_accuracy: float | None
    update_log: list[dict]


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
    bin_threshold: int = 5,
    batch_size: int = 20,
    concurrency: int = 10,
    casestudy_temperature: float = 0.3,
    flush_remainder: bool = True,
    output_dir: Path | None = None,
    log: bool = True,
    reasoning_effort: str | None = "low",
) -> TrainingResult:
    """
    Run the HypoGenic-style training loop.

    Parameters
    ----------
    cheatsheet           : initial Cheatsheet (mutated in-place)
    train_items          : labeled training examples
    val_items            : held-out validation examples (None to skip)
    model_score          : model used for scoring each mini-batch
    model_casestudy      : model used for generating case studies from the bin
    api_key              : OpenRouter API key
    bin_threshold        : failures that trigger a new case study
    batch_size           : items scored per mini-batch
    concurrency          : parallel API requests during scoring
    casestudy_temperature: generation temperature for case studies
    flush_remainder      : flush leftover bin at end even if < threshold
    output_dir           : save checkpoints here (None = no saving)
    log                  : print progress to stderr
    """
    def _log(msg: str) -> None:
        if log:
            print(msg, file=sys.stderr, flush=True)

    bin_        = FailureBin(threshold=bin_threshold)
    update_log  : list[dict] = []
    n_added     = 0
    total_correct = 0
    total_scored  = 0

    _log(
        f"\n{'='*60}\n"
        f"Training loop\n"
        f"  items={len(train_items)}  batch_size={batch_size}\n"
        f"  bin_threshold={bin_threshold}\n"
        f"  model_score={model_score}\n"
        f"  model_casestudy={model_casestudy}\n"
        f"{'='*60}"
    )

    total_batches = (len(train_items) + batch_size - 1) // batch_size

    for batch_start in range(0, len(train_items), batch_size):
        batch     = train_items[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1

        _log(
            f"\n[batch {batch_num}/{total_batches}]  "
            f"items {batch_start+1}–{min(batch_start+len(batch), len(train_items))}  "
            f"bin={len(bin_)}/{bin_threshold}"
        )

        correct, wrong = score_batch(
            batch, cheatsheet.render(), model_score, api_key, concurrency,
            reasoning_effort=reasoning_effort,
        )
        total_correct += len(correct)
        total_scored  += len(batch)
        running_acc = total_correct / total_scored

        _log(
            f"  correct={len(correct)}  wrong={len(wrong)}  "
            f"running_accuracy={running_acc:.1%}"
        )

        for item in wrong:
            bin_.add(item)

        while bin_.is_full():
            failures = bin_.flush()
            _log(f"  [bin full] {len(failures)} failures → new case study")
            new_cs = generate_case_study(
                failures=failures,
                cheatsheet_text=cheatsheet.render(),
                model=model_casestudy,
                api_key=api_key,
                temperature=casestudy_temperature,
            )
            cheatsheet.add_case_study(new_cs)
            n_added += 1
            _log(f"  → cheatsheet now has {len(cheatsheet.case_studies)} case study(ies).")

            update_log.append({
                "event": "bin_flush",
                "batch": batch_num,
                "items_processed": total_scored,
                "n_failures": len(failures),
                "n_case_studies_total": len(cheatsheet.case_studies),
                "running_train_accuracy": running_acc,
            })
            if output_dir:
                _save_checkpoint(cheatsheet, update_log, output_dir, n_added)

    # Remainder flush
    if flush_remainder and len(bin_) > 0:
        failures = bin_.flush()
        _log(f"\n[remainder] {len(failures)} leftover failures → final case study")
        new_cs = generate_case_study(
            failures=failures,
            cheatsheet_text=cheatsheet.render(),
            model=model_casestudy,
            api_key=api_key,
            temperature=casestudy_temperature,
        )
        cheatsheet.add_case_study(new_cs)
        n_added += 1
        update_log.append({
            "event": "remainder_flush",
            "n_failures": len(failures),
            "n_case_studies_total": len(cheatsheet.case_studies),
        })
        if output_dir:
            _save_checkpoint(cheatsheet, update_log, output_dir, "final_remainder")

    train_accuracy = total_correct / total_scored if total_scored > 0 else 0.0
    _log(
        f"\n{'='*60}\n"
        f"Training complete.\n"
        f"  train_accuracy={train_accuracy:.1%}  case_studies_added={n_added}\n"
        f"  {cheatsheet.summary()}"
    )

    # Validation
    val_accuracy: float | None = None
    if val_items:
        _log(f"\nValidating on {len(val_items)} held-out items ...")
        result = test_cheatsheet(
            cheatsheet_text=cheatsheet.render(),
            val_items=val_items,
            model=model_score,
            api_key=api_key,
            concurrency=concurrency,
            reasoning_effort=reasoning_effort,
        )
        val_accuracy = result.accuracy
        _log(f"  {result.summary()}")

    if output_dir:
        _save_checkpoint(cheatsheet, update_log, output_dir, "final")

    return TrainingResult(
        cheatsheet=cheatsheet,
        n_case_studies_added=n_added,
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
        update_log=update_log,
    )
