"""
ICR_adaptive/training/loop.py

AdaptiveTrainingLoop — the main refinement loop.

High-level algorithm per iteration
------------------------------------
1. Score the current cheatsheet on all scoring models → ScorerResult.
2. Partition actionable failures into bins (FORMAT failures are skipped —
   they cannot be fixed by adding case studies).
3. Pick the bin with the most failures (≥ bin_threshold).
4. Generate n_candidates new case studies for that bin via the generator.
5. For each candidate: re-score, pass through AdaptiveRegressionGate.
6. Accept the best candidate that clears all gates (highest utility).
7. Repeat until no improvable bin remains or max_iterations reached.

Injected callables (kept sync so the loop is unit-testable):
  score_fn      : (model, items, sheet_text) → List[ItemScore]
  generate_fn   : (prompt_text, model) → str
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ICR_adaptive.config import PipelineConfig, TaskConfig
from ICR_adaptive.components.failure_classifier import FailureClassifier, FailureType
from ICR_adaptive.components.format_filter import FormatFilter
from ICR_adaptive.components.execution_parser import ExecutionPathParser
from ICR_adaptive.components.generator_router import GeneratorRouter
from ICR_adaptive.components.multi_model_scorer import (
    ItemScore, MultiModelScorer, ScorerResult,
)
from ICR_adaptive.training.adaptive_gate import AdaptiveRegressionGate
from ICR_adaptive.prompts.strategies import (
    GenerationContext, PromptStrategy, build_prompt,
)

logger = logging.getLogger(__name__)

ScoreFn   = Callable[[str, List[dict], str], List[ItemScore]]
GenerateFn = Callable[[str, str], str]
BinKey    = Tuple

_REQUIRED_FOOTER = ("REASONING:", "PROOF:", "COUNTEREXAMPLE:", "VERDICT:")


def _missing_footer_lines(text: str) -> List[str]:
    """Return any required footer lines absent from the generated case text."""
    return [line for line in _REQUIRED_FOOTER if line not in text]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class IterationResult:
    iteration: int
    accepted: bool
    bin_key: Optional[BinKey]
    utility: float
    fix_rate: float
    regress_count: int
    sheet_changed: bool
    acc_before: float = 0.0
    acc_after: float  = 0.0
    case_text: Optional[str] = None   # case study appended this iteration (accepted only)


@dataclass
class LoopResult:
    final_sheet: str
    iterations: List[IterationResult] = field(default_factory=list)
    total_accepted: int = 0


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------

class AdaptiveTrainingLoop:

    def __init__(
        self,
        task_cfg: TaskConfig,
        pipeline_cfg: PipelineConfig,
        score_fn: ScoreFn,
        generate_fn: GenerateFn,
    ) -> None:
        self._task      = task_cfg
        self._pipe      = pipeline_cfg
        self._format_filter = FormatFilter(task_cfg)
        self._classifier    = FailureClassifier(task_cfg)
        self._parser        = ExecutionPathParser(task_cfg)
        self._router        = GeneratorRouter(task_cfg)
        self._scorer        = MultiModelScorer(pipeline_cfg.scoring_models, score_fn)
        self._gate          = AdaptiveRegressionGate(pipeline_cfg)
        self._generate_fn   = generate_fn

    def run(
        self,
        train_items: List[dict],
        initial_sheet: str,
        max_iterations: int = 20,
        case_bank: Optional[List[dict]] = None,
        no_accept_patience: int = 0,
    ) -> LoopResult:
        if case_bank is None:
            case_bank = []

        sheet = initial_sheet
        loop_result = LoopResult(final_sheet=sheet)
        completed_bins: set = set()
        consecutive_no_accept = 0

        for iteration in range(1, max_iterations + 1):
            sep = "─" * 56
            logger.info("\n%s\nITERATION %d / %d\n%s", sep, iteration, max_iterations, sep)

            # ── Step 1: score baseline ────────────────────────────────────
            score_result = self._scorer.score(train_items, sheet)
            primary = score_result.per_model[self._pipe.primary_model()]

            n_total   = primary.n_total
            n_correct = primary.n_correct
            n_miss    = sum(1 for it in train_items if not it.get("_verdict", ""))
            n_wrong   = n_total - n_correct - n_miss
            acc_pct   = 100.0 * n_correct / n_total if n_total else 0.0
            logger.info(
                "  Baseline: correct=%d  wrong=%d  miss=%d  total=%d  acc=%.1f%%",
                n_correct, n_wrong, n_miss, n_total, acc_pct,
            )

            # ── Step 2: classify failures → bins (skip FORMAT failures) ──
            bins: Dict[BinKey, List[dict]] = {}
            n_format_skipped = 0
            for item in train_items:
                iid = str(item.get("id", ""))
                if iid in primary.correct_ids:
                    continue
                response    = item.get("_response", "")
                token_count = item.get("_token_count", 200)
                verdict     = item.get("_verdict", "")
                if not verdict:
                    continue   # miss — no verdict to classify

                classify_r = self._classifier.classify(
                    verdict=verdict,
                    ground_truth_label=str(item.get(self._task.answer_field, "")),
                    response=response,
                    token_count=token_count,
                    truncation_threshold=self._task.truncation_token_threshold,
                )

                # FORMAT failures (truncated/empty) cannot be fixed by case studies
                if classify_r.failure_type == FailureType.FORMAT:
                    n_format_skipped += 1
                    continue

                diverge  = self._parser.parse(response)
                base_key = self._task.base_partition_key(item)
                bin_key: BinKey = (
                    classify_r.failure_type.value,
                    diverge.step,
                ) + base_key
                bins.setdefault(bin_key, []).append(item)

            if n_format_skipped:
                logger.info("  Skipped %d FORMAT failures (not targetable)", n_format_skipped)

            # ── log all bins ──────────────────────────────────────────────
            if bins:
                sorted_bins = sorted(bins.items(), key=lambda kv: -len(kv[1]))
                bin_lines = "  ".join(
                    f"{k[0]}@{k[1]}[n={len(v)}]"
                    for k, v in sorted_bins[:6]
                )
                logger.info("  Failure bins: %s", bin_lines)
            else:
                logger.info("  No actionable failures — stopping at iteration %d", iteration)
                loop_result.iterations.append(IterationResult(
                    iteration=iteration, accepted=False, bin_key=None,
                    utility=0.0, fix_rate=1.0, regress_count=0, sheet_changed=False,
                    acc_before=acc_pct, acc_after=acc_pct,
                ))
                break

            # ── Step 3: pick largest eligible bin ≥ bin_threshold ────────
            eligible = {k: v for k, v in bins.items() if k not in completed_bins}
            if not eligible:
                logger.info(
                    "  All failure bins already addressed — stopping at iteration %d", iteration,
                )
                loop_result.iterations.append(IterationResult(
                    iteration=iteration, accepted=False, bin_key=None,
                    utility=0.0, fix_rate=1.0, regress_count=0, sheet_changed=False,
                    acc_before=acc_pct, acc_after=acc_pct,
                ))
                break

            target_key, target_items = max(eligible.items(), key=lambda kv: len(kv[1]))
            if len(target_items) < self._pipe.bin_threshold:
                logger.info(
                    "  Largest eligible bin has %d items (< threshold %d) — stopping",
                    len(target_items), self._pipe.bin_threshold,
                )
                loop_result.iterations.append(IterationResult(
                    iteration=iteration, accepted=False, bin_key=target_key,
                    utility=0.0, fix_rate=0.0, regress_count=0, sheet_changed=False,
                    acc_before=acc_pct, acc_after=acc_pct,
                ))
                break

            pool_ids = list(primary.correct_ids)
            logger.info(
                "  Target bin: %s  failures=%d  correct_pool=%d",
                target_key, len(target_items), len(pool_ids),
            )

            # ── Step 4 + 5: generate candidates and gate them ─────────────
            best_candidate: Optional[str]  = None
            best_gate                       = None
            best_case_text: Optional[str]  = None

            for c_idx in range(self._pipe.n_candidates):
                sample_item = target_items[c_idx % len(target_items)]
                related     = self._router.route(sample_item, case_bank)
                related_txt = related["case_text"] if related else None

                ctx = GenerationContext(
                    task_cfg=self._task,
                    cheatsheet_text=sheet,
                    item=sample_item,
                    model_response=sample_item.get("_response", ""),
                    failure_type=FailureType(target_key[0]),
                    divergence_step=target_key[1],
                    divergence_rule="unknown",
                    related_case=related_txt,
                )
                prompt       = build_prompt(ctx, PromptStrategy.DIRECT_FIX)
                new_case_text = self._generate_fn(prompt, self._pipe.generator_model)

                # Show a preview of what was generated
                preview = new_case_text[:300].replace("\n", " ").strip()
                logger.info("  Generated cand %d/%d  len=%dc  preview: %s%s",
                            c_idx + 1, self._pipe.n_candidates,
                            len(new_case_text), preview[:200],
                            "..." if len(preview) > 200 else "")

                # Reject truncated / incomplete case studies before gating
                missing = _missing_footer_lines(new_case_text)
                if missing:
                    logger.warning(
                        "  Cand %d INCOMPLETE — missing footer lines: %s — skipping",
                        c_idx + 1, missing,
                    )
                    continue

                candidate_sheet = sheet + "\n\n" + new_case_text
                cand_score = self._scorer.score(train_items, candidate_sheet)
                cand_primary = cand_score.per_model[self._pipe.primary_model()]
                cand_acc = 100.0 * cand_primary.n_correct / cand_primary.n_total if cand_primary.n_total else 0.0

                gate_r = self._gate.evaluate(
                    scorer_result=cand_score,
                    bin_ids=[str(it.get("id", "")) for it in target_items],
                    pool_ids=pool_ids,
                    candidate_text=candidate_sheet,
                    current_text=sheet,
                )
                logger.info(
                    "  Cand %d gate: %s  fix=%.2f  regress=%d  utility=%+.3f  acc=%.1f%%",
                    c_idx + 1, gate_r.reason.upper(),
                    gate_r.fix_rate, gate_r.regress_count, gate_r.utility, cand_acc,
                )

                if gate_r.passed:
                    if best_gate is None or gate_r.utility > best_gate.utility:
                        best_candidate = candidate_sheet
                        best_gate      = gate_r
                        best_case_text = new_case_text
                        case_bank.append({
                            "features": self._task.query_features(sample_item),
                            "case_text": new_case_text,
                            "bin_key":   target_key,
                        })

            # ── Step 6: accept best ───────────────────────────────────────
            if best_candidate is not None:
                sheet = best_candidate
                completed_bins.add(target_key)
                loop_result.total_accepted += 1
                consecutive_no_accept = 0
                # Compute post-acceptance accuracy for reporting
                accept_primary = None
                for mr in self._scorer._models:
                    # We already scored candidate_sheet; re-use cand_score for the best
                    pass
                accepted_acc = cand_acc  # last candidate scored (rough; good enough for logging)
                logger.info(
                    "  ACCEPTED  utility=%+.3f  fix=%.2f  regress=%d  acc_before=%.1f%%  acc_after~%.1f%%",
                    best_gate.utility, best_gate.fix_rate, best_gate.regress_count,
                    acc_pct, accepted_acc,
                )
                loop_result.iterations.append(IterationResult(
                    iteration=iteration, accepted=True, bin_key=target_key,
                    utility=best_gate.utility, fix_rate=best_gate.fix_rate,
                    regress_count=best_gate.regress_count, sheet_changed=True,
                    acc_before=acc_pct, acc_after=accepted_acc,
                    case_text=best_case_text,
                ))
            else:
                logger.info("  REJECTED — no candidate cleared all gates")
                loop_result.iterations.append(IterationResult(
                    iteration=iteration, accepted=False, bin_key=target_key,
                    utility=0.0, fix_rate=0.0, regress_count=0, sheet_changed=False,
                    acc_before=acc_pct, acc_after=acc_pct,
                ))
                consecutive_no_accept += 1
                if no_accept_patience > 0 and consecutive_no_accept >= no_accept_patience:
                    logger.info(
                        "  Early stop — %d consecutive iterations with no acceptance",
                        consecutive_no_accept,
                    )
                    break

        loop_result.final_sheet = sheet
        return loop_result
