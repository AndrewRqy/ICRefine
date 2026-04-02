"""
llm_client.py — Shared LLM call utility for the ICRefine pipeline.

Provides a synchronous call_llm() for single calls and a parallel
call_llm_batch() backed by a ThreadPoolExecutor for scoring many
examples at once.

All modules in this package import from here so API settings stay
in one place.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent.parent.parent / "SAIR_evaluation_pipeline" / ".env")
load_dotenv(Path(__file__).parent.parent / ".env")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOKENS = 16_000  # match SAIR pipeline — reasoning models need space
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds; doubles on each retry


def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("Error: OPENROUTER_API_KEY environment variable is not set.")
    return key


# ---------------------------------------------------------------------------
# Core call
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.3,
    max_tokens: int = MAX_TOKENS,
    reasoning_effort: str | None = "low",
) -> str:
    """
    Send a single prompt to the model via OpenRouter.
    Retries up to MAX_RETRIES times on 429 / 5xx.
    Returns the response text.

    reasoning_effort: "low" | "medium" | "high" | None.
      Sent as {"reasoning": {"effort": ...}} for models that support it
      (e.g. gpt-oss-120b, Claude 3.7+). Set to None to omit.
    """
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/sair-evaluation",
        "X-Title": "SAIR ICRefine",
    }

    delay = RETRY_BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                OPENROUTER_URL, json=payload, headers=headers, timeout=120
            )
            if resp.status_code in (429,) or resp.status_code >= 500:
                if attempt < MAX_RETRIES:
                    time.sleep(delay)
                    delay *= 2
                    continue
                resp.raise_for_status()
            resp.raise_for_status()
            message = resp.json()["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning") or ""
            return content.strip()
        except requests.RequestException as exc:
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= 2
            else:
                raise RuntimeError(
                    f"LLM call failed after {MAX_RETRIES} retries: {exc}"
                ) from exc

    raise RuntimeError("Unexpected exit from retry loop.")


# ---------------------------------------------------------------------------
# Parallel batch call
# ---------------------------------------------------------------------------

def call_llm_batch(
    prompts: list[str],
    model: str,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    concurrency: int = 10,
    progress_label: str = "",
    reasoning_effort: str | None = "low",
) -> list[str | None]:
    """
    Call the LLM for each prompt in parallel using a thread pool.

    Returns a list of response strings in the same order as prompts.
    Entries are None where the call failed after all retries.
    """
    results: list[str | None] = [None] * len(prompts)
    total = len(prompts)
    done = 0

    def _call(idx: int, prompt: str) -> tuple[int, str | None]:
        try:
            return idx, call_llm(prompt, model, api_key, temperature, max_tokens, reasoning_effort)
        except Exception as exc:
            print(f"\n  [batch] item {idx} error: {exc}", file=sys.stderr)
            return idx, None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_call, i, p): i for i, p in enumerate(prompts)}
        for future in as_completed(futures):
            idx, text = future.result()
            results[idx] = text
            done += 1
            label = f"  {progress_label} " if progress_label else "  "
            print(f"\r{label}{done}/{total}", end="", flush=True, file=sys.stderr)

    print(file=sys.stderr)  # newline after progress
    return results
