"""
ICR_adaptive/pipeline.py

AdaptivePipeline — convenience facade that wires all ICR_adaptive components
together and exposes a single run() method.

This is NOT a CLI itself — it is the programmatic entry point that downstream
scripts or notebooks call.  See the module-level usage example below.

Usage
-----
    from ICR_adaptive.pipeline import AdaptivePipeline
    from ICR_adaptive.config import TaskConfig, PipelineConfig

    task = TaskConfig(
        domain_description="...",
        input_fields=["eq1", "eq2"],
        answer_field="answer",
        verdict_pattern=r"(?i)\\bVERDICT\\s*:\\s*(TRUE|FALSE)\\b",
        answer_map={"True": "TRUE", "False": "FALSE"},
    )
    pipe_cfg = PipelineConfig(
        scoring_models=["openai/gpt-4o"],
        generator_model="openai/gpt-4o",
    )

    # Provide your own score_fn and generate_fn that call your LLM backend.
    pipeline = AdaptivePipeline(task, pipe_cfg, score_fn=my_scorer, generate_fn=my_gen)
    result = pipeline.run(train_items, initial_sheet="...")
    print(result.final_sheet)
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from ICR_adaptive.config import PipelineConfig, TaskConfig
from ICR_adaptive.components.multi_model_scorer import ItemScore
from ICR_adaptive.training.loop import AdaptiveTrainingLoop, LoopResult

logger = logging.getLogger(__name__)

ScoreFn = Callable[[str, List[dict], str], List[ItemScore]]
GenerateFn = Callable[[str, str], str]


class AdaptivePipeline:
    """
    Thin facade over AdaptiveTrainingLoop.

    Parameters
    ----------
    task_cfg        : TaskConfig — domain-specific settings
    pipeline_cfg    : PipelineConfig — training loop settings
    score_fn        : callable(model, items, sheet_text) → List[ItemScore]
    generate_fn     : callable(prompt_text, model_id) → str
    """

    def __init__(
        self,
        task_cfg: TaskConfig,
        pipeline_cfg: PipelineConfig,
        score_fn: ScoreFn,
        generate_fn: GenerateFn,
    ) -> None:
        task_cfg.validate()
        pipeline_cfg.validate()
        self._task = task_cfg
        self._pipe = pipeline_cfg
        self._loop = AdaptiveTrainingLoop(task_cfg, pipeline_cfg, score_fn, generate_fn)

    def run(
        self,
        train_items: List[dict],
        initial_sheet: str,
        max_iterations: int = 20,
        case_bank: Optional[List[dict]] = None,
        no_accept_patience: int = 0,
    ) -> LoopResult:
        """
        Run the adaptive refinement loop.

        Parameters
        ----------
        train_items     : list of problem dicts
        initial_sheet   : starting cheatsheet text
        max_iterations  : hard cap on loop iterations
        case_bank       : optional mutable list; new case studies are appended

        Returns
        -------
        LoopResult with final_sheet and per-iteration metadata
        """
        logger.info(
            "AdaptivePipeline.run — models=%s generator=%s max_iter=%d",
            self._pipe.scoring_models,
            self._pipe.generator_model,
            max_iterations,
        )
        return self._loop.run(
            train_items=train_items,
            initial_sheet=initial_sheet,
            max_iterations=max_iterations,
            case_bank=case_bank,
            no_accept_patience=no_accept_patience,
        )
