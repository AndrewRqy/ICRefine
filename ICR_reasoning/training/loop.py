"""
ICR_reasoning/training/loop.py — Training loop that passes post-think to the bin.

Identical structure to ICR_naive/training/loop.py with one key difference:
failures stored in the bin carry their post_think field, which is forwarded to
generate_case_study_with_reasoning() so the case study LLM can see WHY the model
failed, not just that it failed.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from utils.cheatsheet import Cheatsheet
from utils.data import FailureBin
from ..generators.case_study import generate_case_study_with_reasoning
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
    cot_first: bool = False,
    output_dir: Path | None = None,
    log: bool = True,
    reasoning_effort: str | None = "low",
) -> TrainingResult:
    def _log(msg: str) -> None:
        if log:
            print(msg, file=sys.stderr, flush=True)

    bin_          = FailureBin(threshold=bin_threshold)
    update_log    : list[dict] = []
    n_added       = 0
    total_correct = 0
    total_scored  = 0
    total_batches = (len(train_items) + batch_size - 1) // batch_size

    _log(
        f"\n{'='*60}\n"
        f"ICR_reasoning Training loop\n"
        f"  items={len(train_items)}  batch_size={batch_size}\n"
        f"  bin_threshold={bin_threshold}\n"
        f"  model_score={model_score}\n"
        f"  model_casestudy={model_casestudy}\n"
        f"{'='*60}"
    )

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
            reasoning_effort=reasoning_effort, cot_first=cot_first,
        )
        total_correct += len(correct)
        total_scored  += len(batch)
        running_acc = total_correct / total_scored

        _log(
            f"  correct={len(correct)}  wrong={len(wrong)}  "
            f"running_accuracy={running_acc:.1%}"
        )

        for item in wrong:
            bin_.add(item)   # item carries post_think from scorer

        while bin_.is_full():
            failures = bin_.flush()
            _log(f"  [bin full] {len(failures)} failures → reasoning-aware case study + roadmap patch")
            result = generate_case_study_with_reasoning(
                failures=failures,
                cheatsheet=cheatsheet,
                model=model_casestudy,
                api_key=api_key,
                temperature=casestudy_temperature,
            )
            cheatsheet.add_case_study(result.case_study)
            n_added += 1
            patched = False
            if result.roadmap_patch:
                cheatsheet.patch_roadmap(result.roadmap_patch)
                patched = True
                _log(
                    f"  → roadmap patched ({len(result.roadmap_patch)} chars).  "
                    f"cheatsheet now has {len(cheatsheet.case_studies)} case study(ies)."
                )
            else:
                _log(f"  → cheatsheet now has {len(cheatsheet.case_studies)} case study(ies) (no roadmap patch).")

            update_log.append({
                "event": "bin_flush",
                "batch": batch_num,
                "items_processed": total_scored,
                "n_failures": len(failures),
                "n_case_studies_total": len(cheatsheet.case_studies),
                "roadmap_patch_applied": patched,
                "running_train_accuracy": running_acc,
            })
            if output_dir:
                _save_checkpoint(cheatsheet, update_log, output_dir, n_added)

    if flush_remainder and len(bin_) > 0:
        failures = bin_.flush()
        _log(f"\n[remainder] {len(failures)} failures → reasoning-aware case study + roadmap patch")
        result = generate_case_study_with_reasoning(
            failures=failures,
            cheatsheet=cheatsheet,
            model=model_casestudy,
            api_key=api_key,
            temperature=casestudy_temperature,
        )
        cheatsheet.add_case_study(result.case_study)
        n_added += 1
        patched = False
        if result.roadmap_patch:
            cheatsheet.patch_roadmap(result.roadmap_patch)
            patched = True
            _log(f"  → roadmap patched ({len(result.roadmap_patch)} chars).")
        update_log.append({
            "event": "remainder_flush",
            "n_failures": len(failures),
            "n_case_studies_total": len(cheatsheet.case_studies),
            "roadmap_patch_applied": patched,
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
            cot_first=cot_first,
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
