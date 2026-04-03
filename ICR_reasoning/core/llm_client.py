"""
ICR_reasoning/core/llm_client.py — LLM client that preserves both response fields.

Key difference from ICR_naive: returns LLMResponse(content, thinking) instead of
a plain string, keeping the separation between:
  - thinking  : full internal CoT trace (the "think" block, ~thousands of tokens)
  - content   : the structured output the model writes after reasoning (the "post-think")

Per Heddaya et al. (ACL 2026), post-think naturally preserves deductive markers
and logical scaffolding at 25× higher density than externally prompted summaries.
We use content (post-think) as the distillation signal for case studies, not the
raw thinking trace.
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

load_dotenv(Path(__file__).parent.parent.parent.parent / "SAIR_evaluation_pipeline" / ".env")
load_dotenv(Path(__file__).parent.parent.parent / ".env")

OPENROUTER_URL   = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL       = "https://api.openai.com/v1/chat/completions"
MAX_TOKENS       = 16_000
MAX_RETRIES      = 3
RETRY_BASE_DELAY = 2.0

_OPENAI_PREFIXES = ("gpt-4", "gpt-3", "o1", "o3", "o4")


def _resolve_endpoint(model: str) -> tuple[str, str, bool]:
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    bare = model.removeprefix("openai/")
    if openai_key and bare.startswith(_OPENAI_PREFIXES):
        return OPENAI_URL, openai_key, True
    return OPENROUTER_URL, os.environ.get("OPENROUTER_API_KEY", ""), False


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    content:  str   # post-think: structured output after internal reasoning
    thinking: str   # full CoT trace from the reasoning field


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("Error: OPENROUTER_API_KEY not set.")
    return key


# ---------------------------------------------------------------------------
# Core call — returns LLMResponse
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float = 1.0,
    max_tokens: int = MAX_TOKENS,
    reasoning_effort: str | None = "low",
) -> LLMResponse:
    """
    Send a prompt and return LLMResponse(content, thinking).

    content  = message["content"]  — the post-think structured output
    thinking = message["reasoning"] — the full internal CoT trace

    For models that don't expose reasoning (e.g. gpt-4o), thinking will be "".
    """
    url, resolved_key, is_openai = _resolve_endpoint(model)
    model_name = model.removeprefix("openai/") if is_openai else model

    payload: dict = {
        "model":      model_name,
        "messages":   [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if reasoning_effort is not None and not is_openai:
        payload["reasoning"] = {"effort": reasoning_effort}

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {resolved_key}"}
    if not is_openai:
        headers["HTTP-Referer"] = "https://github.com/sair-evaluation"
        headers["X-Title"]      = "SAIR ICR-Reasoning"

    delay = RETRY_BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=180)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < MAX_RETRIES:
                    print(
                        f"\n  [retry] HTTP {resp.status_code} — backing off {delay:.0f}s "
                        f"(attempt {attempt}/{MAX_RETRIES})",
                        file=sys.stderr, flush=True,
                    )
                    time.sleep(delay); delay *= 2; continue
                resp.raise_for_status()
            resp.raise_for_status()

            message  = resp.json()["choices"][0]["message"]
            content  = (message.get("content")   or "").strip()
            thinking = (message.get("reasoning") or "").strip()

            # gpt-oss-120b via OpenRouter sometimes puts everything in "reasoning"
            # when content is null — treat that whole response as content (post-think)
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
                time.sleep(delay); delay *= 2
            else:
                raise RuntimeError(f"LLM call failed after {MAX_RETRIES} retries: {exc}") from exc

    raise RuntimeError("Unexpected exit from retry loop.")


# ---------------------------------------------------------------------------
# Batch call — returns list[LLMResponse | None]
# ---------------------------------------------------------------------------

def call_llm_batch(
    prompts: list[str],
    model: str,
    api_key: str,
    temperature: float = 1.0,
    max_tokens: int = MAX_TOKENS,
    concurrency: int = 10,
    progress_label: str = "",
    reasoning_effort: str | None = "low",
) -> list[LLMResponse | None]:
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
