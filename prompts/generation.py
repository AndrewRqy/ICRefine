"""
prompts/generation.py — Versioned case-study generation prompts.

Prompts now live in their respective task files:
  tasks/web_of_lies.py       — WOL_PROMPTS  (v3, v4)
  tasks/date_understanding.py — DU_PROMPTS   (v3, v4)
  tasks/geometric_shapes.py   — GEO_PROMPTS  (v3, v4)

The active version is selected at import time via the ICR_GEN_PROMPT_VERSION
environment variable (default: v3). This shim re-exports get_prompt() for
backward compatibility and ablation access.

Usage
-----
    from prompts.generation import get_prompt
    v4_prompt = get_prompt("web_of_lies", "v4")
"""

from __future__ import annotations

import os

from tasks.web_of_lies import WOL_PROMPTS
from tasks.date_understanding import DU_PROMPTS
from tasks.geometric_shapes import GEO_PROMPTS

_REGISTRY: dict[str, dict[str, str]] = {
    "web_of_lies":        WOL_PROMPTS,
    "date_understanding": DU_PROMPTS,
    "geometric_shapes":   GEO_PROMPTS,
}

_LATEST: dict[str, str] = {
    "web_of_lies":        "v3",
    "date_understanding": "v3",
    "geometric_shapes":   "v3",
}

VERSIONS: dict[str, list[str]] = {
    task: list(versions) for task, versions in _REGISTRY.items()
}


def get_prompt(task: str, version: str = "latest") -> str:
    """Return the generation prompt for *task* at *version*.

    Parameters
    ----------
    task:
        One of 'web_of_lies', 'geometric_shapes', 'date_understanding'.
    version:
        A version string ('v3', 'v4') or 'latest' to use the current default.
        When 'latest', the env var ``ICR_GEN_PROMPT_VERSION`` is checked first.
    """
    if task not in _REGISTRY:
        raise KeyError(
            f"No versioned generation prompt for task '{task}'. "
            f"Available: {list(_REGISTRY)}"
        )
    if version == "latest":
        env_ver = os.environ.get("ICR_GEN_PROMPT_VERSION", "").strip()
        version = env_ver if env_ver in _REGISTRY[task] else _LATEST[task]
    if version not in _REGISTRY[task]:
        raise KeyError(
            f"Unknown version '{version}' for task '{task}'. "
            f"Available: {list(_REGISTRY[task])}"
        )
    return _REGISTRY[task][version]
