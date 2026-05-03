"""
ICR_select/generators/case_study.py — Multi-candidate case study generator.

Generates N candidates per bin flush in parallel at different temperatures,
so the selection loop has real diversity to choose from rather than running
the same prompt twice and hoping the output differs.
"""

from __future__ import annotations

import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.cheatsheet import Cheatsheet, extract_query_features
from utils.case_study import CaseStudy
from utils.llm_client import call_llm, _OPENAI_REASONING, _OPENAI_PREFIXES
from ICR_reasoning.core.oracle import OracleDict
from ICR_reasoning.generators.case_study import _format_failures_with_reasoning, _parse_response, _render_case_studies_text
from ..prompts.templates import (
    CASE_STUDY_WITH_REASONING_PROMPT,
    CROSSOVER_PROMPT,
    CROSSOVER_MAX_TOKENS,
    RETRY_CONTEXT_TEMPLATE,
    FLUSH_MAX_TOKENS,
    N_CANDIDATES,
    CANDIDATE_TEMPS,
)

# ---------------------------------------------------------------------------
# Failure analysis helpers (ported from ICR_adaptive)
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(r'\[STEP:\s*([^\]]+)\]')

_ABANDONMENT_PHRASES = [
    "given the complexity",
    "without explicit calculation",
    "requires further analysis",
    "the process requires",
    "a detailed analysis would",
    "due to the format",
    "without completing each step",
    "this requires careful consideration",
    "a comprehensive analysis",
    "without more information",
]


def _detect_failure_type(failures: list[dict]) -> str:
    """Return 'ABANDONMENT' if the majority of failures show bail-out signals, else 'WRONG_ANSWER'."""
    n_abandon = sum(
        1 for item in failures
        if any(phrase in item.get("_response", "").lower() for phrase in _ABANDONMENT_PHRASES)
    )
    return "ABANDONMENT" if n_abandon > len(failures) / 2 else "WRONG_ANSWER"


def _parse_divergence_step(failures: list[dict]) -> str:
    """
    Return the most common divergence step across failures.
    When oracle_nearest is present, divergence = first oracle step absent from model response.
    Otherwise, divergence = last [STEP:] tag in the model response.
    """
    step_counts: dict[str, int] = {}
    for item in failures:
        resp = item.get("_response", "")
        oracle_trace = (item.get("oracle_nearest") or {}).get("reasoning", "")
        if oracle_trace:
            oracle_steps = _STEP_RE.findall(oracle_trace)
            model_steps = {s.strip() for s in _STEP_RE.findall(resp)}
            for step in oracle_steps:
                if step.strip() not in model_steps:
                    key = step.strip()
                    step_counts[key] = step_counts.get(key, 0) + 1
                    break
        else:
            steps = _STEP_RE.findall(resp)
            if steps:
                key = steps[-1].strip()
                step_counts[key] = step_counts.get(key, 0) + 1
    return max(step_counts, key=step_counts.__getitem__) if step_counts else "unknown"


def _find_related_case(failures: list[dict], cheatsheet: Cheatsheet) -> str | None:
    """
    Jaccard-retrieve the most structurally similar existing case study for the
    given failure batch.  Returns the rendered case study text or None.
    """
    if not cheatsheet.case_studies or not failures:
        return None
    try:
        sample_sig = set(extract_query_features(failures[0]).signature().split("_"))
    except Exception:
        return None
    best_cs, best_score = None, -1.0
    for cs in cheatsheet.case_studies:
        cs_sig = set(cs.feature_signature.split("_")) if cs.feature_signature else set()
        union = sample_sig | cs_sig
        score = len(sample_sig & cs_sig) / len(union) if union else 0.0
        if score > best_score:
            best_score, best_cs = score, cs
    if best_cs is None or best_score == 0.0:
        return None
    return best_cs.render()


def _format_already_covered(cheatsheet: Cheatsheet) -> str:
    """
    Summarise patterns already covered by the roadmap and existing case studies
    so the generation prompt can explicitly tell the LLM not to restate them.
    """
    lines: list[str] = []

    # Extract key roadmap rules as bullet points (first sentence of each ASPECT line)
    for line in cheatsheet.roadmap.splitlines():
        line = line.strip()
        if line.upper().startswith("ASPECT") or line.upper().startswith("RULE") or line.upper().startswith("STEP"):
            # Keep only the label + first clause
            short = line[:120] + ("..." if len(line) > 120 else "")
            lines.append(f"  [roadmap] {short}")

    # Summarise ACTIVATE IF conditions from existing case studies
    for cs in cheatsheet.case_studies:
        if cs.activate_if:
            conditions = "; ".join(cs.activate_if[:2])  # first 2 conditions
            lines.append(f"  [case study: {cs.title}] {conditions}")
        elif cs.title:
            lines.append(f"  [case study: {cs.title}]")

    return "\n".join(lines) if lines else "  (none yet — this is the first case study)"


def _build_failure_lines(
    failures: list[dict],
    format_fn,
    oracle: OracleDict | None = None,
    inject_gold_oracle: bool = True,
) -> str:
    """
    Format a failure list into the failure_lines block for the generation prompt.

    Pre-bakes exact oracle matches (keyed by equation pair) into each item as
    `_oracle_exact` before passing to the per-item format_fn.  The format_fn
    (task_spec.format_failure) reads _oracle_exact and oracle_nearest from the
    item dict — no oracle reference inside format_fn itself.

    inject_gold_oracle: when True (default), falls back to item["reason"] as a
    correct-reasoning contrast signal for datasets that carry gold CoT.  Set to
    False to disable this injection for ablation studies.
    """
    lines = []
    for i, item in enumerate(failures, 1):
        # Bake exact oracle reasoning when available (magma-specific keys gracefully ignored for other tasks)
        if oracle:
            try:
                key = (item.get("equation1", "").strip(), item.get("equation2", "").strip())
                exact = oracle.get(key, "")
                if exact:
                    item = {**item, "_oracle_exact": exact}
            except Exception:
                pass
        # Fallback: use item["reason"] (gold CoT present in BBH and similar datasets)
        if inject_gold_oracle and "_oracle_exact" not in item and item.get("reason"):
            item = {**item, "_oracle_exact": item["reason"]}
        lines.append(f"--- Failure {i} ---")
        lines.append(format_fn(item))
    return "\n".join(lines)


def generate_candidates(
    failures: list[dict],
    cheatsheet: Cheatsheet,
    model: str,
    api_key: str,
    n: int = N_CANDIDATES,
    temperatures: list[float] | None = None,
    oracle: OracleDict | None = None,
    prev_attempt: dict | None = None,
    polarity: str = "",
    failure_type_hint: str = "",    # "ABANDONMENT" or "WRONG_ANSWER"; auto-detected if empty
    divergence_step_hint: str = "", # last step before divergence; auto-detected if empty
    task_spec=None,                 # TaskSpec | None — defaults to MAGMA_TASK
    inject_gold_oracle: bool = True,
) -> list[CaseStudy]:
    """
    Generate *n* candidate case study strings in parallel at different temperatures.

    Returns a list of case study strings (the CASE STUDY section only, without
    the DT patch), ordered by temperature ascending.  Failures are silently
    dropped — a failed generation produces an empty string, which the caller
    filters out.

    oracle: optional (eq1, eq2) -> correct_reasoning dict; when provided, each
            failure that has a matching oracle entry will show the correct
            reasoning as a contrast signal alongside the wrong model reasoning.

    prev_attempt: optional dict with keys "candidate" (str), "still_wrong" (list[dict]),
            and "reason" (str — "fix_rate" or "regression"). When provided (retry
            flush strategy), the previous candidate and its still-wrong items are
            appended to the prompt so the model knows what was tried and what failed.
    """
    # Resolve task_spec — default to MAGMA_TASK for backward compat
    if task_spec is None:
        from tasks.magma import MAGMA_TASK
        task_spec = MAGMA_TASK

    temps = (temperatures or CANDIDATE_TEMPS)[:n]

    # ── Polarity instruction ──────────────────────────────────────────────────
    ftype    = failure_type_hint    or _detect_failure_type(failures)
    div_step = divergence_step_hint or _parse_divergence_step(failures)

    if task_spec.build_polarity_instruction is not None:
        polarity_instruction = task_spec.build_polarity_instruction(polarity, ftype, div_step)
    else:
        # Built-in magma-specific polarity directives
        _p = polarity.strip().upper()
        if _p == "TRUE":
            polarity_instruction = (
                "POLARITY DIRECTIVE — FALSE NEGATIVE bin (model said FALSE, correct answer is TRUE):\n"
                "Prioritize TYPE A (MISSING KNOWLEDGE). These failures happen because the weaker model "
                "lacks an algebraic fact that would let it see WHY E1 forces E2. Your goal is to distill "
                "a missing lemma, identity, or structural property from the oracle traces above — something "
                "the weaker model never considers even when following a plausible path. A good TYPE A case "
                "study gives the model a concrete shortcut: IF [lemma condition holds] THEN verdict is TRUE "
                "immediately, no further analysis needed.\n"
                "Do NOT generate a FALSE-counterexample case study for these failures."
            )
        elif _p == "FALSE":
            polarity_instruction = (
                "POLARITY DIRECTIVE — FALSE POSITIVE bin (model said TRUE, correct answer is FALSE):\n"
                "Prioritize TYPE B (WRONG/MISSING REASONING PATTERN). These failures happen because the "
                "weaker model applies a flawed heuristic, stops too early, or skips a necessary "
                "counterexample check. Identify the exact wrong move and the correct structural test "
                "that produces a counterexample. A good TYPE B case study names the trap and gives a "
                "mechanical check: IF [structural condition] THEN try building a counterexample magma.\n"
                "Do NOT generate a missing-lemma/proof case study for these failures."
            )
        else:
            polarity_instruction = (
                "Diagnose whether these failures are TYPE A (missing algebraic knowledge) or TYPE B "
                "(wrong reasoning pattern), choosing the type that best explains the majority of cases."
            )

        # ABANDONMENT / divergence addenda (built-in — override via build_polarity_instruction)
        if ftype == "ABANDONMENT":
            step_note = f" at step [{div_step}]" if div_step != "unknown" else ""
            polarity_instruction = (
                f"STRATEGY — ABANDONMENT bin (model gave up{step_note} instead of completing the protocol):\n"
                "These failures are TYPE B (WRONG REASONING PATTERN): the model stopped reasoning "
                "before reaching a verdict. Your case study must:\n"
                "  1. Show the WRONG PATH — quote where the model abandoned reasoning.\n"
                "  2. Show the CORRECT PATH — demonstrate the specific step the model should have "
                "taken instead of stopping.\n"
                "  3. Write NEXT CHECK as a direct mechanical instruction that prevents bail-out: "
                "e.g. 'If no counterexample found in size-2/3 magmas, proceed to affine probe — "
                "do not conclude TRUE without exhausting the checklist.'\n"
                + polarity_instruction
            )
        elif div_step != "unknown":
            polarity_instruction = (
                f"DIVERGENCE POINT — model response last completed step [{div_step}] before going wrong.\n"
                "Your case study NEXT CHECK must specifically address what to do at or after this step.\n\n"
                + polarity_instruction
            )

    # ── Adaptive: retrieve most structurally similar existing case study ──────
    related_case = _find_related_case(failures, cheatsheet)
    related_section = (
        f"\n=== MOST RELEVANT EXISTING CASE STUDY (for reference — do NOT duplicate) ===\n"
        f"{related_case}\n"
        f"=== END RELATED CASE ===\n"
        if related_case else ""
    )

    # ── Format failure lines using task_spec ──────────────────────────────────
    failure_lines = _build_failure_lines(failures, task_spec.format_failure, oracle=oracle,
                                         inject_gold_oracle=inject_gold_oracle)

    if prev_attempt:
        reason_desc = (
            "it fixed too few failures (fix-rate gate)"
            if prev_attempt["reason"] == "fix_rate"
            else "it broke too many previously-correct items (regression gate)"
        )
        still_wrong = prev_attempt["still_wrong"]
        still_wrong_lines = _build_failure_lines(still_wrong, task_spec.format_failure,
                                                 inject_gold_oracle=inject_gold_oracle)
        prev_cand = prev_attempt["candidate"]
        prev_cand_text = prev_cand.render() if isinstance(prev_cand, CaseStudy) else str(prev_cand).strip()
        prev_section = RETRY_CONTEXT_TEMPLATE.format(
            reason_desc=reason_desc,
            prev_candidate=prev_cand_text,
            n_still_wrong=len(still_wrong),
            still_wrong_lines=still_wrong_lines,
        )
        failure_lines = failure_lines + "\n\n" + prev_section

    # ── Build generation prompt using task_spec template ─────────────────────
    prompt = task_spec.generation_prompt_template.format(
        roadmap=cheatsheet.roadmap.strip(),
        case_studies=_render_case_studies_text(cheatsheet),
        failure_lines=failure_lines + related_section,
        already_covered=_format_already_covered(cheatsheet),
        polarity_instruction=polarity_instruction,
        retry_context="",   # placeholder consumed by templates that include it
    )

    # Prepend prior_knowledge so the generator sees the full cheatsheet context,
    # not just roadmap + case_studies.  This is critical when prior_knowledge
    # holds a CS-ICL bootstrap — without it the generator works blind and may
    # duplicate or contradict existing principles.
    _pk_text = cheatsheet._render_prior_knowledge().strip()
    if _pk_text:
        prompt = (
            "=== PRIOR KNOWLEDGE (existing cheat sheet — read before deciding) ===\n"
            f"{_pk_text}\n"
            "=== END PRIOR KNOWLEDGE ===\n\n"
            + prompt
        )

    # Append MODIFY/ADD choice instruction.  The model must choose on its first
    # output line whether to refine an existing entry or write a new one.
    _has_existing = bool(cheatsheet.case_studies) or bool(_pk_text)
    if _has_existing:
        prompt += (
            "\n\n---\n"
            "Before writing, state your CHOICE on the very first line of your response:\n\n"
            "  CHOICE: MODIFY \"<exact title of the rule or case study to improve>\"\n"
            "  — or —\n"
            "  CHOICE: ADD NEW\n\n"
            "Choose MODIFY when an existing entry is partially relevant but incomplete, "
            "mis-scoped, or wrong for this failure pattern — the rewrite should replace "
            "that entry entirely, not append to it.\n"
            "Choose ADD NEW when no existing entry addresses this failure pattern.\n\n"
            "Write your CHOICE line first, then the full case study in the standard format."
        )

    _COMPLETION_RETRY_PROMPT = """\
The case study below is INCOMPLETE — it is missing the following required fields: {missing}.

Please rewrite it in full, adding the missing fields. Keep all existing content intact.
Required format:
=== CASE STUDY: [title] ===
FAILURE_TYPE: A or B
ACTIVATE IF:
  - [condition 1]
  - [condition 2]
DO NOT ACTIVATE IF: [boundary case where this should not fire]
COMMON WRONG MOVE: [what the weaker model does wrong]
NEXT CHECK: [mechanical check to perform instead — end with "If yes → TRUE/FALSE."]
WHY THIS WORKS: [1–2 sentence justification]
SUPPORT:
  • E1 = ...  |  E2 = ...  |  Answer: TRUE/FALSE  — [brief note]
  • E1 = ...  |  E2 = ...  |  Answer: TRUE/FALSE  — [brief note]
TARGET_STEP: [roadmap aspect this corrects]

=== INCOMPLETE CASE STUDY ===
{incomplete_text}
=== END ===

Output ONLY the completed case study starting with === CASE STUDY: ..."""

    # Detect reasoning models (o1/o3/o4 series) — they don't support temperature
    # and need a much larger token budget because reasoning tokens eat into max_completion_tokens.
    _model_name = model.removeprefix("openai/")
    _is_reasoning = (
        any(_model_name.startswith(p) for p in _OPENAI_PREFIXES)
        and any(_model_name.startswith(p) for p in _OPENAI_REASONING)
    )
    _gen_max_tokens   = 4000 if _is_reasoning else FLUSH_MAX_TOKENS
    _gen_effort       = "low" if _is_reasoning else None
    _gen_temps        = [temps[0]] if _is_reasoning else temps  # reasoning models ignore temp anyway

    _CHOICE_RE = re.compile(
        r'CHOICE:\s*(?:MODIFY\s+"([^"]+)"|ADD\s+NEW)',
        re.IGNORECASE,
    )

    def _extract_modification_target(text: str) -> str:
        m = _CHOICE_RE.search(text)
        return m.group(1).strip() if (m and m.group(1)) else ""

    def _strip_choice_line(text: str) -> str:
        return re.sub(r"^CHOICE:.*\n?", "", text, count=1, flags=re.IGNORECASE | re.MULTILINE).strip()

    def _call(temp: float) -> CaseStudy | None:
        try:
            resp = call_llm(
                prompt, model, api_key,
                temperature=temp,
                max_tokens=_gen_max_tokens,
                reasoning_effort=_gen_effort,
            )
            modification_target = _extract_modification_target(resp.content)
            clean_content = _strip_choice_line(resp.content)
            result = _parse_response(clean_content)
            cs = result.case_study
            if cs is None:
                return None

            # Preserve the roadmap patch and modification choice on the CaseStudy
            # so the training loop can apply both when this candidate is accepted.
            cs.roadmap_patch = result.roadmap_patch
            cs.modification_target = modification_target

            ok, missing = cs.is_complete()
            if ok:
                return cs

            # Retry once with the incomplete output fed back to the LLM
            print(
                f"  [candidate gen] temp={temp} incomplete (missing: {', '.join(missing)}) — retrying",
                file=sys.stderr,
            )
            retry_prompt = _COMPLETION_RETRY_PROMPT.format(
                missing=", ".join(missing),
                incomplete_text=cs.raw_text.strip(),
            )
            retry_resp = call_llm(
                retry_prompt, model, api_key,
                temperature=0.3,
                max_tokens=_gen_max_tokens,
                reasoning_effort=_gen_effort,
            )
            retry_result = _parse_response(retry_resp.content)
            retry_cs = retry_result.case_study
            if retry_cs is None:
                return None
            retry_cs.roadmap_patch = retry_result.roadmap_patch or cs.roadmap_patch
            ok2, missing2 = retry_cs.is_complete()
            if not ok2:
                print(
                    f"  [candidate gen] temp={temp} still incomplete after retry "
                    f"(missing: {', '.join(missing2)}) — dropping",
                    file=sys.stderr,
                )
                return None
            return retry_cs

        except Exception as exc:
            print(f"  [candidate gen] temp={temp} failed: {exc}", file=sys.stderr)
            return None

    _active_temps = _gen_temps
    candidates: list[CaseStudy | None] = [None] * len(_active_temps)
    with ThreadPoolExecutor(max_workers=len(_active_temps)) as pool:
        futures = {pool.submit(_call, t): i for i, t in enumerate(_active_temps)}
        for fut in as_completed(futures):
            candidates[futures[fut]] = fut.result()

    # Auto-compute feature_signature from failure structural features when the
    # LLM didn't write a FEATURE_SIGNATURE: line.  Without this, build_vmatch
    # returns [] for every candidate → utility gate always falls back.
    #
    # TYPE A (missing knowledge): the lemma applies to an entire E1 form class,
    # not a specific structural pair.  Use only E1's form token so the case study
    # routes broadly to every query where that lemma condition holds.
    # TYPE B (wrong reasoning pattern): mistake is configuration-specific;
    # keep the full pair signature so routing stays narrow and precise.
    failure_qfs = []
    for item in failures:
        try:
            failure_qfs.append(extract_query_features(item))
        except Exception:
            pass

    full_pair_sig = failure_qfs[0].signature() if failure_qfs else ""
    e1_form_sig   = failure_qfs[0].form_e1.lower() if failure_qfs else ""

    valid = [c for c in candidates if c is not None]
    if not valid:
        raise RuntimeError("All candidate generations failed or were incomplete after retry.")

    for c in valid:
        if not c.feature_signature:
            # TYPE A → broad scope (E1 form only); TYPE B or unknown → full pair
            c.feature_signature = e1_form_sig if c.failure_type == "A" else full_pair_sig

    print(
        f"  [candidates] {len(valid)}/{n} candidates complete and valid",
        file=sys.stderr,
    )
    return valid


def generate_crossover(
    cs1: CaseStudy,
    fr1: float,
    cs2: CaseStudy,
    fr2: float,
    model: str,
    api_key: str,
) -> CaseStudy | None:
    """
    Evolve two archived (failed) candidates into a single crossover child.

    The child inherits the best ACTIVATE IF conditions and structural checks
    from both parents, targeting the union of failure cases each parent caught.
    Returns None if generation or parsing fails — caller treats this as a
    no-op (falls back to regular candidate generation without crossover).
    """
    prompt = CROSSOVER_PROMPT.format(
        fr1=fr1,
        cs_a=cs1.render(),
        fr2=fr2,
        cs_b=cs2.render(),
    )
    try:
        resp = call_llm(
            prompt, model, api_key,
            temperature=0.5,
            max_tokens=CROSSOVER_MAX_TOKENS,
            reasoning_effort=None,
        )
        result = _parse_response(resp.content)
        child = result.case_study
        if child is None:
            print("  [crossover] parse returned no case study", file=sys.stderr)
            return None
        ok, missing = child.is_complete()
        if not ok:
            print(
                f"  [crossover] incomplete case study (missing: {', '.join(missing)}) — dropping",
                file=sys.stderr,
            )
            return None
        return child
    except Exception as exc:
        print(f"  [crossover] failed: {exc}", file=sys.stderr)
        return None
