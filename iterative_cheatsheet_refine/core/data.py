"""
core/data.py — Dataset loading, sampling, and splitting utilities.

All functions that touch .jsonl files or partition items live here.
Import _is_true from here rather than defining it locally in each module.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------

def _is_true(answer) -> bool:
    """Normalise a JSON bool or string answer to Python bool."""
    if isinstance(answer, bool):
        return answer
    return str(answer).strip().upper() == "TRUE"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Load all non-empty lines from a .jsonl file."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_instances(
    items: list[dict],
    n_true: int,
    n_false: int,
    seed: int,
) -> list[dict]:
    """
    Return up to n_true TRUE and n_false FALSE examples, shuffled together.
    Prints a warning to stderr if the dataset has fewer examples than requested.
    """
    rng = random.Random(seed)
    true_items  = [it for it in items if _is_true(it["answer"])]
    false_items = [it for it in items if not _is_true(it["answer"])]

    actual_true  = min(n_true,  len(true_items))
    actual_false = min(n_false, len(false_items))

    if actual_true < n_true:
        print(
            f"Warning: only {len(true_items)} TRUE examples available, requested {n_true}.",
            file=sys.stderr,
        )
    if actual_false < n_false:
        print(
            f"Warning: only {len(false_items)} FALSE examples available, requested {n_false}.",
            file=sys.stderr,
        )

    combined = rng.sample(true_items, actual_true) + rng.sample(false_items, actual_false)
    rng.shuffle(combined)
    return combined


# ---------------------------------------------------------------------------
# Train / val / seed splitting
# ---------------------------------------------------------------------------

def split_dataset(
    items: list[dict],
    val_fraction: float,
    seed_examples: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Stratified split into (seed, train, val).

    seed_examples items (balanced TRUE/FALSE) are reserved for initial
    cheatsheet generation.  val_fraction of the remainder become the
    validation set.  The rest become the training set for the loop.

    Returns (seed_set, train_set, val_set).
    """
    rng = random.Random(seed)

    true_items  = [it for it in items if _is_true(it["answer"])]
    false_items = [it for it in items if not _is_true(it["answer"])]

    def _split_one(lst: list) -> tuple[list, list, list]:
        shuffled = lst[:]
        rng.shuffle(shuffled)
        n_seed = min(seed_examples // 2, len(shuffled))
        seed_part = shuffled[:n_seed]
        rest = shuffled[n_seed:]
        n_val = max(1, round(len(rest) * val_fraction)) if val_fraction > 0 else 0
        return seed_part, rest[n_val:], rest[:n_val]

    s_t, tr_t, v_t = _split_one(true_items)
    s_f, tr_f, v_f = _split_one(false_items)

    seed_set  = s_t + s_f
    train_set = tr_t + tr_f
    val_set   = v_t + v_f

    rng.shuffle(seed_set)
    rng.shuffle(train_set)
    rng.shuffle(val_set)

    return seed_set, train_set, val_set
