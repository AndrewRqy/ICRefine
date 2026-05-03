"""
tools/prompt_revision.py — Meta-optimization of ICRefine generation prompts.

Pipeline:
  1. Score N items from a dataset with a weak scoring model to collect failures
     (each failure has the model's wrong reasoning in post_think).
  2. Feed: reference cheatsheet + current prompt + sample failures →
     ask a revision LLM to propose K improved prompt variants.
  3. For each variant, call it on the same failures to produce a sample output.
  4. Write a report (prompt_revision_report.md) comparing all variants.

Usage:
  cd ICRefine
  python tools/prompt_revision.py \\
      --dataset datasets/mixed_n100_h1_h2.jsonl \\
      --cheatsheet runs/partition_mixed_n100h1h2/cheatsheet_current.txt \\
      --reference-pk NeuriCo_v2_prior_knowledge.txt \\
      --scoring-model openai/gpt-4o-mini \\
      --revision-model openai/gpt-4o \\
      --n-sample 60 \\
      --n-failures 6 \\
      --n-variants 3 \\
      --prompt-type case_study \\
      --out tools/prompt_revision_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---- project root on sys.path ----
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from utils.llm_client import call_llm, get_api_key
from utils.scorer import score_batch
from utils.cheatsheet import Cheatsheet
from utils.equation_features import compute_pair_features


# ---------------------------------------------------------------------------
# The meta-revision prompts
# ---------------------------------------------------------------------------

CASE_STUDY_REVISION_META_PROMPT = """\
You are a prompt engineer specializing in mathematical reasoning systems.
Your job is to improve a GENERATION PROMPT used to write teaching notes for a weaker model.

=== CONTEXT ===

The system decides whether E1 (a magma equation) implies E2.
A new structural feature vector is NOW pre-computed for every query:
  E1: size, vars, imb, bare, LP, RP, SET, XOR, AB
  E2: size, vars, imb, bare, LP, RP, SET, XOR, AB
  SEPARATOR: which separator invariant fires (LP/RP/SET/XOR/AB) or "none"
  COLLAPSE: left_proj / right_proj / none (canonical source-collapse class)

These features are already injected into the SCORING prompt that the weak model reads.
So the weak model CAN reference them, but only if case studies tell it when to.

=== REFERENCE CHEATSHEET (ground-truth decision algorithm) ===

{reference_pk}

=== CURRENT CASE STUDY GENERATION PROMPT ===

{current_prompt}

=== SAMPLE FAILURES (the weak model got these wrong) ===

{failure_lines}

=== YOUR TASK ===

The current prompt produces correct case studies but they sometimes:
  (a) Use vague prose conditions in ACTIVATE IF instead of feature-vector predicates
  (b) Ignore separator and collapse information that the scoring prompt already provides
  (c) Give NEXT CHECK steps that are hard to execute by direct inspection
  (d) Miss the tight TYPE A / TYPE B diagnosis when both are plausible

Produce {n_variants} IMPROVED VARIANTS of the current prompt.

Each variant should be a complete, self-contained replacement for the section between
the first line ("You are an expert...") and the last line ("Output ONLY these two...").

Prioritize improvements to:
  1. ACTIVATE IF — encourage feature-vector terms (bare(E1)=TRUE, vars(E1)>=4, etc.)
  2. NEXT CHECK — must reference PRECOMPUTED FEATURES block when available
  3. TYPE A diagnosis — better signal: "oracle invokes a fact model never mentions"
  4. SUPPORT — require the structural condition the ACTIVATE IF caught, not just "answer is X"

Format your response using markdown section headers exactly like this:

### VARIANT 1
CHANGE: [one line — what this variant changes vs the original]
[complete replacement prompt text]

---

### VARIANT 2
CHANGE: [one line — what this variant changes]
[complete replacement prompt text]

---

... and so on for all {n_variants} variants.

Make each variant meaningfully different — try different emphasis, instruction styles,
or structural changes. Each CHANGE line must be on the very first line after the heading.
"""


PK_SECTION_REVISION_META_PROMPT = """\
You are a prompt engineer specializing in mathematical reasoning systems.
Your job is to improve a GENERATION PROMPT that writes prior-knowledge sections for
a decision guide on magma equational implication.

=== CONTEXT ===

A structural feature vector is pre-computed for every (E1, E2) query:
  size, vars, imb, bare, LP, RP, SET, XOR, AB (for each of E1 and E2)
  SEPARATOR: which separator invariant fires first, or "none"
  COLLAPSE: left_proj / right_proj / none

=== REFERENCE CHEATSHEET (the ideal decision algorithm) ===

{reference_pk}

=== CURRENT PK SECTION GENERATION PROMPT ===

{current_prompt}

=== SAMPLE FAILURES (structural class: {partition_label}) ===

{failure_lines}

=== YOUR TASK ===

The current prompt generates rules in a generic style. Improve it by:
  1. Instructing the model to name rules using the feature vector (RULE: bare(E1)=T, vars>=4 → TRUE)
  2. Referencing separator and collapse conditions explicitly
  3. Asking for step-ordering (STOP after rule fires)
  4. Encouraging worked examples that show feature values alongside equations

Produce {n_variants} IMPROVED VARIANTS. Format using markdown:

### VARIANT 1
CHANGE: [one line — what this variant changes]
[complete prompt text]

---

### VARIANT 2
CHANGE: ...
[complete prompt text]

---
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_failures_from_oracle_csv(csv_path: str, n: int) -> list[dict]:
    """Load failures from a gpt5.4 oracle CSV (weak model wrong reasoning)."""
    import csv as csv_mod
    rows = []
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            if row.get("correct", "True") == "False":
                rows.append({
                    "equation1":  row["equation1"],
                    "equation2":  row["equation2"],
                    "answer":     row["answer"],
                    "post_think": row.get("response", ""),
                    "predicted":  None,
                    "expected":   row["answer"],
                })
    return rows[:n]


def _format_failures_for_meta(failures: list[dict]) -> str:
    parts = []
    for i, f in enumerate(failures, 1):
        answer_str = ("TRUE" if str(f.get("answer","")).lower() in ("true","1") else "FALSE")
        predicted_str = f.get("predicted") or "?"
        think = (f.get("post_think") or f.get("thinking") or "").strip()
        parts.append(
            f"[{i}] E1 = {f['equation1']}  |  E2 = {f['equation2']}\n"
            f"    Expected: {answer_str}  Predicted: {predicted_str}\n"
            f"    Weak model reasoning (excerpt):\n"
            + textwrap.indent(think[:600] + ("..." if len(think)>600 else ""), "      ")
        )
    return "\n\n".join(parts)


def _parse_variants(response_text: str) -> list[tuple[str, str]]:
    """
    Extract (change_note, body) pairs from the meta-response.
    Handles three delimiter styles the LLM might use:
      - === VARIANT N === ... === END VARIANT N ===
      - ### VARIANT N ... ### VARIANT N+1 (markdown heading sections)
      - --- with VARIANT N: heading
    """
    import re

    # Style 1: explicit === VARIANT N === delimiters
    pattern1 = re.compile(
        r"===\s*VARIANT\s*\d+\s*===\s*\n(.*?)(?====\s*(?:END\s*VARIANT\s*\d+|VARIANT\s*\d+)\s*===)",
        re.DOTALL | re.IGNORECASE,
    )
    results = list(pattern1.finditer(response_text))

    if not results:
        # Style 2: markdown ### VARIANT N headings — split on them
        parts = re.split(r"(?m)^#{1,4}\s*VARIANT\s*\d+\b", response_text)
        if len(parts) > 1:
            chunks = parts[1:]  # first part is preamble
            for chunk in chunks:
                # Stop at next ---+whitespace divider if present
                chunk = re.split(r"\n---+\s*\n", chunk)[0].strip()
                lines = chunk.splitlines()
                change = ""
                if lines and lines[0].upper().startswith("CHANGE:"):
                    change = lines[0][7:].strip()
                    chunk = "\n".join(lines[1:]).strip()
                if chunk:
                    results_pairs = getattr(_parse_variants, "_pairs", None)
                    return_list = [(change, chunk) for change, chunk in
                                   [(change, chunk)] + []]
            # rebuild properly
            pairs = []
            for chunk in parts[1:]:
                chunk = re.split(r"\n---+\s*\n", chunk)[0].strip()
                lines = chunk.splitlines()
                change = ""
                if lines and lines[0].upper().startswith("CHANGE:"):
                    change = lines[0][7:].strip()
                    chunk = "\n".join(lines[1:]).strip()
                if chunk:
                    pairs.append((change, chunk))
            return pairs

        # Style 3: numbered blocks — try splitting on blank line + "VARIANT N"
        parts3 = re.split(r"\n+(?=VARIANT\s*\d+\s*[\n:])", response_text)
        if len(parts3) > 1:
            pairs = []
            for chunk in parts3:
                chunk = chunk.strip()
                if not chunk:
                    continue
                m = re.match(r"VARIANT\s*\d+\s*:?\s*\n?", chunk, re.IGNORECASE)
                if m:
                    chunk = chunk[m.end():].strip()
                lines = chunk.splitlines()
                change = ""
                if lines and lines[0].upper().startswith("CHANGE:"):
                    change = lines[0][7:].strip()
                    chunk = "\n".join(lines[1:]).strip()
                if chunk:
                    pairs.append((change, chunk))
            return pairs

        return []

    # Style 1 matched
    pairs = []
    for m in results:
        body = m.group(1).strip()
        change = ""
        lines = body.splitlines()
        if lines and lines[0].upper().startswith("CHANGE:"):
            change = lines[0][7:].strip()
            body = "\n".join(lines[1:]).strip()
        pairs.append((change, body))
    return pairs


def _generate_case_study_with_prompt(
    prompt_template: str,
    failures: list[dict],
    cheatsheet: Cheatsheet,
    model: str,
    api_key: str,
) -> str:
    """
    Call prompt_template (a generation prompt) with failures and return the raw response.
    The template must accept {roadmap}, {case_studies}, {failure_lines}, {already_covered},
    {polarity_instruction} placeholders (same as CASE_STUDY_WITH_REASONING_PROMPT).
    """
    from ICR_reasoning.generators.case_study import (
        _format_failures_with_reasoning, _render_case_studies_text,
    )
    from ICR_select.generators.case_study import _format_already_covered

    failure_lines = _format_failures_with_reasoning(failures, oracle=None)
    case_studies_text = _render_case_studies_text(cheatsheet)
    already_covered  = _format_already_covered(cheatsheet)
    polarity_instruction = (
        "Diagnose whether these failures are TYPE A (missing algebraic knowledge) or TYPE B "
        "(wrong reasoning pattern), choosing the type that best explains the majority of cases."
    )
    try:
        prompt = prompt_template.format(
            roadmap=cheatsheet.roadmap.strip(),
            case_studies=case_studies_text,
            failure_lines=failure_lines,
            already_covered=already_covered,
            polarity_instruction=polarity_instruction,
        )
    except KeyError as e:
        return f"[ERROR: prompt template missing placeholder {e}]"

    try:
        resp = call_llm(prompt, model, api_key, temperature=0.4, max_tokens=1200)
        return resp.content.strip()
    except Exception as exc:
        return f"[ERROR calling LLM: {exc}]"


def _generate_pk_section_with_prompt(
    prompt_template: str,
    partition_label: str,
    form_e1: str,
    form_e2: str,
    polarity: str,
    failures: list[dict],
    model: str,
    api_key: str,
) -> str:
    """Generate a PK section using prompt_template (must accept _GEN_PROMPT placeholders)."""
    failure_text = "\n".join(
        f"  E1 = {item['equation1']}  |  E2 = {item['equation2']}  |  answer = {item.get('answer','?')}"
        for item in failures[:8]
    )
    try:
        prompt = prompt_template.format(
            partition_label=partition_label,
            form_e1=form_e1,
            form_e2=form_e2,
            polarity=polarity,
            depth_desc="depth-2+ (two or more * operators)",
            n_examples=min(len(failures), 8),
            failure_examples=failure_text,
            existing_titles="  (none yet)",
        )
    except KeyError as e:
        return f"[ERROR: prompt template missing placeholder {e}]"

    try:
        resp = call_llm(prompt, model, api_key, temperature=0.4, max_tokens=1200)
        return resp.content.strip()
    except Exception as exc:
        return f"[ERROR calling LLM: {exc}]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt revision meta-optimizer")
    parser.add_argument("--dataset",        default="datasets/mixed_n100_h1_h2.jsonl")
    parser.add_argument("--cheatsheet",     default="runs/partition_mixed_n100h1h2/cheatsheet_current.txt")
    parser.add_argument("--reference-pk",   default="NeuriCo_v2_prior_knowledge.txt")
    parser.add_argument("--oracle-csv",     default="gpt5.4_mixed_oracle.csv",
                        help="Optional: pre-collected oracle failures CSV (skips inline scoring)")
    parser.add_argument("--scoring-model",  default="openai/gpt-4o-mini")
    parser.add_argument("--revision-model", default="openai/gpt-4o",
                        help="Model used for meta-revision (proposing improved prompts)")
    parser.add_argument("--test-model",     default=None,
                        help="Model used to test revised prompts (default: same as scoring-model)")
    parser.add_argument("--n-sample",       type=int, default=80,
                        help="Items to score inline if oracle-csv not used")
    parser.add_argument("--n-failures",     type=int, default=6,
                        help="Number of failures to feed into the meta-revision prompt")
    parser.add_argument("--n-variants",     type=int, default=3,
                        help="Number of prompt variants to request")
    parser.add_argument("--prompt-type",    choices=["case_study", "pk_section", "both"],
                        default="case_study")
    parser.add_argument("--out",            default="tools/prompt_revision_report.md")
    parser.add_argument("--api-key",        default=None)
    args = parser.parse_args()

    api_key = args.api_key or get_api_key()
    test_model = args.test_model or args.scoring_model

    # ---- 1. Load reference PK ----
    ref_pk_path = Path(args.reference_pk)
    if not ref_pk_path.is_absolute():
        ref_pk_path = _ROOT / ref_pk_path
    reference_pk = ref_pk_path.read_text()
    print(f"[step 1] Loaded reference PK: {ref_pk_path} ({len(reference_pk)} chars)")

    # ---- 2. Load cheatsheet ----
    cs_path = Path(args.cheatsheet)
    if not cs_path.is_absolute():
        cs_path = _ROOT / cs_path
    # Cheatsheet.load() reads from .json sidecar; pass path without suffix
    cheatsheet = Cheatsheet.load(cs_path.with_suffix(""))
    print(f"[step 2] Loaded cheatsheet: {cs_path} — {len(cheatsheet.case_studies)} case studies")

    # ---- 3. Get failures ----
    oracle_csv_path = _ROOT / args.oracle_csv
    if oracle_csv_path.exists():
        print(f"[step 3] Loading failures from oracle CSV: {oracle_csv_path}")
        failures = _load_failures_from_oracle_csv(str(oracle_csv_path), args.n_failures)
        print(f"         Found {len(failures)} failures in oracle CSV")
    else:
        print(f"[step 3] Scoring {args.n_sample} items from {args.dataset} with {args.scoring_model} ...")
        dataset_path = Path(args.dataset)
        if not dataset_path.is_absolute():
            dataset_path = _ROOT / dataset_path
        with open(dataset_path) as f:
            all_items = [json.loads(l) for l in f]
        sample = all_items[:args.n_sample]
        _, wrong = score_batch(
            sample, cheatsheet.render(), args.scoring_model, api_key,
            concurrency=20, temperature=0.0, reasoning_effort=None,
            progress_label="scoring for failures",
        )
        failures = wrong[:args.n_failures]
        print(f"         Found {len(wrong)} failures; using first {len(failures)}")

    if not failures:
        print("[ERROR] No failures found — cannot proceed with prompt revision.")
        sys.exit(1)

    failure_lines_text = _format_failures_for_meta(failures[:args.n_failures])

    # ---- 4. Load current prompts ----
    from ICR_reasoning.prompts.templates import CASE_STUDY_WITH_REASONING_PROMPT
    from ICR_pk.training.generator import _GEN_PROMPT

    print(f"[step 4] Loaded current prompts.")

    # ---- 5. Meta-revision calls ----
    report_sections: list[str] = []

    def _section_header(title: str) -> str:
        return f"\n\n{'='*80}\n# {title}\n{'='*80}\n"

    def _do_case_study_revision() -> None:
        print(f"\n[step 5a] Requesting {args.n_variants} case study prompt variants "
              f"from {args.revision_model} ...")
        meta_prompt = CASE_STUDY_REVISION_META_PROMPT.format(
            reference_pk=reference_pk,
            current_prompt=CASE_STUDY_WITH_REASONING_PROMPT,
            failure_lines=failure_lines_text,
            n_variants=args.n_variants,
        )
        try:
            meta_resp = call_llm(
                meta_prompt, args.revision_model, api_key,
                temperature=0.7, max_tokens=6000,
            )
            meta_text = meta_resp.content.strip()
        except Exception as exc:
            report_sections.append(f"\n[ERROR getting variants: {exc}]")
            return

        variants = _parse_variants(meta_text)
        print(f"         Parsed {len(variants)} variants from meta-response.")

        report_sections.append(_section_header("CASE STUDY PROMPT REVISION"))
        report_sections.append(f"\n## Failures used ({len(failures[:args.n_failures])} items)\n")
        report_sections.append("```\n" + failure_lines_text + "\n```\n")
        report_sections.append("\n## Original Prompt\n")
        report_sections.append("```\n" + CASE_STUDY_WITH_REASONING_PROMPT + "\n```\n")
        report_sections.append("\n## Meta-revision raw response\n")
        report_sections.append("```\n" + meta_text[:3000] + ("..." if len(meta_text)>3000 else "") + "\n```\n")

        # Test each variant
        print(f"[step 5b] Testing {len(variants)} variant prompts on failures ...")
        with ThreadPoolExecutor(max_workers=len(variants) or 1) as pool:
            futs = {}
            for i, (change, body) in enumerate(variants):
                if not body:
                    continue
                f = pool.submit(
                    _generate_case_study_with_prompt,
                    body, failures[:args.n_failures], cheatsheet,
                    test_model, api_key,
                )
                futs[f] = (i+1, change, body)

            results = {}
            for f in as_completed(futs):
                idx, change, body = futs[f]
                out = f.result()
                results[idx] = (change, body, out)

        for idx in sorted(results):
            change, body, out = results[idx]
            report_sections.append(f"\n## Variant {idx}: {change or '(no change label)'}\n")
            report_sections.append("### Proposed Prompt\n```\n" + body[:2000] + ("..." if len(body)>2000 else "") + "\n```\n")
            report_sections.append("### Sample Output\n```\n" + out + "\n```\n")

    def _do_pk_section_revision() -> None:
        # Use a representative partition label
        partition_label = "STANDARD→GENERAL/depth2/FALSE"
        form_e1 = "STANDARD"
        form_e2 = "GENERAL"
        polarity = "FALSE"

        print(f"\n[step 5c] Requesting {args.n_variants} PK section prompt variants "
              f"from {args.revision_model} ...")
        meta_prompt = PK_SECTION_REVISION_META_PROMPT.format(
            reference_pk=reference_pk,
            current_prompt=_GEN_PROMPT,
            failure_lines=failure_lines_text,
            partition_label=partition_label,
            n_variants=args.n_variants,
        )
        try:
            meta_resp = call_llm(
                meta_prompt, args.revision_model, api_key,
                temperature=0.7, max_tokens=6000,
            )
            meta_text = meta_resp.content.strip()
        except Exception as exc:
            report_sections.append(f"\n[ERROR getting PK variants: {exc}]")
            return

        variants = _parse_variants(meta_text)
        print(f"         Parsed {len(variants)} PK variants.")

        report_sections.append(_section_header("PK SECTION PROMPT REVISION"))
        report_sections.append("\n## Original Prompt\n")
        report_sections.append("```\n" + _GEN_PROMPT + "\n```\n")
        report_sections.append("\n## Meta-revision raw response\n")
        report_sections.append("```\n" + meta_text[:3000] + ("..." if len(meta_text)>3000 else "") + "\n```\n")

        print(f"[step 5d] Testing {len(variants)} PK variant prompts ...")
        with ThreadPoolExecutor(max_workers=len(variants) or 1) as pool:
            futs = {}
            for i, (change, body) in enumerate(variants):
                if not body:
                    continue
                f = pool.submit(
                    _generate_pk_section_with_prompt,
                    body, partition_label, form_e1, form_e2, polarity,
                    failures[:args.n_failures], test_model, api_key,
                )
                futs[f] = (i+1, change, body)

            results = {}
            for f in as_completed(futs):
                idx, change, body = futs[f]
                out = f.result()
                results[idx] = (change, body, out)

        for idx in sorted(results):
            change, body, out = results[idx]
            report_sections.append(f"\n## Variant {idx}: {change or '(no change label)'}\n")
            report_sections.append("### Proposed Prompt\n```\n" + body[:2000] + ("..." if len(body)>2000 else "") + "\n```\n")
            report_sections.append("### Sample Output\n```\n" + out + "\n```\n")

    if args.prompt_type in ("case_study", "both"):
        _do_case_study_revision()
    if args.prompt_type in ("pk_section", "both"):
        _do_pk_section_revision()

    # ---- 6. Write report ----
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _ROOT / out_path

    header = (
        f"# Prompt Revision Report\n\n"
        f"**Dataset**: {args.dataset}  \n"
        f"**Cheatsheet**: {args.cheatsheet}  \n"
        f"**Scoring model**: {args.scoring_model}  \n"
        f"**Revision model**: {args.revision_model}  \n"
        f"**Test model**: {test_model}  \n"
        f"**Failures used**: {min(len(failures), args.n_failures)}  \n"
        f"**Variants requested**: {args.n_variants}  \n"
    )
    report = header + "\n".join(report_sections)
    out_path.write_text(report)
    print(f"\n[done] Report written to {out_path}")


if __name__ == "__main__":
    main()
