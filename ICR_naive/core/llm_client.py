"""
llm_client.py — Unified LLM call utility for all ICRefine pipelines.

Returns LLMResponse(content, thinking) for every call so post-think
is always available. content = structured output after reasoning;
thinking = full internal CoT trace (empty for non-reasoning models).

Per Heddaya et al. (ACL 2026), post-think preserves deductive markers
at 25× higher density than externally prompted summaries.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

OPENROUTER_URL    = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL        = "https://api.openai.com/v1/chat/completions"
MAX_TOKENS        = 16_000
MAX_RETRIES       = 3
RETRY_BASE_DELAY  = 2.0
VLLM_READ_TIMEOUT = 600   # local inference can be slow — 10 min per request

_OPENAI_PREFIXES = ("gpt-4", "gpt-3", "o1", "o3", "o4")


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    content:  str   # post-think: structured output after internal reasoning
    thinking: str   # full CoT trace from the reasoning field (empty if unavailable)


# ---------------------------------------------------------------------------
# API key + endpoint resolution
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("Error: OPENROUTER_API_KEY environment variable is not set.")
    return key


def _resolve_endpoint(model: str) -> tuple[str, str, bool, bool]:
    """
    Return (url, api_key, is_openai, is_vllm) for the given model.

    Routing priority:
      1. vLLM — when VLLM_BASE_URL + VLLM_MODEL are set and model matches.
      2. OpenAI direct — when OPENAI_API_KEY is set and model starts with a
         known OpenAI prefix.
      3. OpenRouter — fallback for everything else.
    """
    vllm_url   = os.environ.get("VLLM_BASE_URL", "")
    vllm_model = os.environ.get("VLLM_MODEL", "")
    if vllm_url and vllm_model and model == vllm_model:
        return vllm_url, os.environ.get("VLLM_API_KEY", ""), False, True

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    bare = model.removeprefix("openai/")
    if openai_key and bare.startswith(_OPENAI_PREFIXES):
        return OPENAI_URL, openai_key, True, False

    return OPENROUTER_URL, os.environ.get("OPENROUTER_API_KEY", ""), False, False


# ---------------------------------------------------------------------------
# Core call — returns LLMResponse
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.3,
    max_tokens: int = MAX_TOKENS,
    reasoning_effort: str | None = "low",
) -> LLMResponse:
    """
    Send a single prompt and return LLMResponse(content, thinking).

    content  = message["content"]            — post-think structured output
    thinking = message["reasoning_content"]  — full internal CoT (vLLM)
             = message["reasoning"]          — full internal CoT (OpenRouter)

    For models that do not expose reasoning (e.g. gpt-4o), thinking will be "".
    reasoning_effort: "low" | "medium" | "high" | None. Sent as
      {"reasoning": {"effort": ...}} for OpenRouter reasoning models. Omitted
      for vLLM and OpenAI (not supported).
    """
    url, resolved_key, is_openai, is_vllm = _resolve_endpoint(model)
    model_name = model.removeprefix("openai/") if is_openai else model

    payload: dict = {
        "model":       model_name,
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }
    if reasoning_effort is not None and not is_openai and not is_vllm:
        payload["reasoning"] = {"effort": reasoning_effort}

    headers = {"Content-Type": "application/json"}
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"
    if not is_openai and not is_vllm:
        headers["HTTP-Referer"] = "https://github.com/sair-evaluation"
        headers["X-Title"]      = "SAIR ICRefine"

    read_timeout = VLLM_READ_TIMEOUT if is_vllm else 300
    delay = RETRY_BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                url, json=payload, headers=headers, timeout=(10, read_timeout)
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < MAX_RETRIES:
                    print(
                        f"\n  [retry] HTTP {resp.status_code} — backing off {delay:.0f}s "
                        f"(attempt {attempt}/{MAX_RETRIES})",
                        file=sys.stderr, flush=True,
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                resp.raise_for_status()
            resp.raise_for_status()

            message = resp.json()["choices"][0]["message"]
            content = (message.get("content") or "").strip()

            if is_vllm:
                thinking = (message.get("reasoning_content") or "").strip()
                # If content is empty (reasoning used all tokens), fall back to thinking
                if not content and thinking:
                    content  = thinking
                    thinking = ""
            else:
                thinking = (message.get("reasoning") or "").strip()
                # gpt-oss-120b via OpenRouter sometimes puts everything in "reasoning"
                if not content and thinking:
                    content  = thinking
                    thinking = ""

            return LLMResponse(content=content, thinking=thinking)

        except requests.RequestException as exc:
            if attempt < MAX_RETRIES:
                print(
                    f"\n  [retry] {type(exc).__name__} — backing off {delay:.0f}s "
                    f"(attempt {attempt}/{MAX_RETRIES}): {exc}",
                    file=sys.stderr, flush=True,
                )
                time.sleep(delay)
                delay *= 2
            else:
                raise RuntimeError(
                    f"LLM call failed after {MAX_RETRIES} retries: {exc}"
                ) from exc

    raise RuntimeError("Unexpected exit from retry loop.")


# ---------------------------------------------------------------------------
# Parallel batch call — returns list[LLMResponse | None]
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
) -> list[LLMResponse | None]:
    """
    Call the LLM for each prompt in parallel using a thread pool.

    Returns a list of LLMResponse in the same order as prompts.
    Entries are None where the call failed after all retries.
    """
    results: list[LLMResponse | None] = [None] * len(prompts)
    done = 0

    def _call(idx: int, prompt: str) -> tuple[int, LLMResponse | None]:
        try:
            return idx, call_llm(prompt, model, api_key, temperature, max_tokens, reasoning_effort)
        except Exception as exc:
            print(f"\n  [batch] item {idx} error: {exc}", file=sys.stderr)
            return idx, None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_call, i, p): i for i, p in enumerate(prompts)}
        for future in as_completed(futures):
            idx, resp = future.result()
            results[idx] = resp
            done += 1
            label = f"  {progress_label} " if progress_label else "  "
            print(f"\r{label}{done}/{len(prompts)}", end="", flush=True, file=sys.stderr)

    print(file=sys.stderr)
    return results
