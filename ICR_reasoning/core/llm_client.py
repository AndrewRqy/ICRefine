"""
ICR_reasoning/core/llm_client.py — Re-exports from the unified client in ICR_naive.

The unified client lives in ICR_naive/core/llm_client.py and returns
LLMResponse(content, thinking) for all calls. Import from there directly
in new code; this shim exists for backward compatibility.
"""

from ICR_naive.core.llm_client import (  # noqa: F401
    LLMResponse,
    call_llm,
    call_llm_batch,
    get_api_key,
    _resolve_endpoint,
    MAX_TOKENS,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    VLLM_READ_TIMEOUT,
    OPENROUTER_URL,
    OPENAI_URL,
)
