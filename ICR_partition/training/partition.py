"""
ICR_partition/training/partition.py — Structural partitioning of items into bins.

Each item is assigned a PartitionKey computed entirely from equation syntax —
no LLM calls.  Items with the same key share a structurally homogeneous failure
mode and are solved independently.

PartitionKey = (form_e1, form_e2, depth_bucket_e1, expected_answer)
  form_*       : TRIVIAL | SINGLETON | ABSORBING | STANDARD | GENERAL
  depth_bucket : 0 (no * operators), 1 (exactly one), 2 (two or more)
  expected_answer : "TRUE" | "FALSE"

This yields up to 5 × 5 × 3 × 2 = 150 possible keys.  In practice a
2000-item dataset will populate 25–50 keys.

Design rationale
----------------
* Including expected_answer separates false-positive failures (model says TRUE
  when answer is FALSE) from false-negative failures (model says FALSE when
  answer is TRUE).  These have structurally different root causes.

* depth_bucket_e1 separates trivially-shallow equations (d0: "x = y") from
  one-level expressions (d1: "x = y * z") from deeply nested ones (d2+).
  Failures at different depths often require different diagnostic reasoning.

* form_e2 is included so the case study generator always receives a batch
  where both E1 and E2 are structurally uniform, which tightens the ACTIVATE IF
  conditions it can write.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field

from utils.cheatsheet import QueryFeatures, extract_query_features
from utils.data import is_true


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

PartitionKey = tuple[str, str, int, str]
# (form_e1, form_e2, depth_bucket_e1, expected_answer)


# ---------------------------------------------------------------------------
# Key computation — pure structural, O(1), no LLM
# ---------------------------------------------------------------------------

def _depth_bucket(depth: int) -> int:
    """Coarsen raw depth into 0 / 1 / 2+ bucket."""
    return min(depth, 2)


def item_partition_key(item: dict) -> PartitionKey:
    """
    Compute the partition key for one item.  Pure structural — no LLM calls.
    Falls back to ("GENERAL", "GENERAL", 2, polarity) on any parse error.
    """
    polarity = "TRUE" if is_true(item.get("answer", False)) else "FALSE"
    try:
        qf: QueryFeatures = extract_query_features(item)
        return (qf.form_e1, qf.form_e2, _depth_bucket(qf.depth_e1), polarity)
    except Exception:
        return ("GENERAL", "GENERAL", 2, polarity)


def partition_label(key: PartitionKey) -> str:
    """Human-readable label for a partition key."""
    form_e1, form_e2, depth_b, polarity = key
    depth_str = ("d0", "d1", "d2+")[depth_b]
    return f"{form_e1}→{form_e2}_{depth_str}_{polarity}"


_FORM_DESC = {
    "TRIVIAL":   "x = x (tautology)",
    "SINGLETON": "x = y (forces all elements equal)",
    "ABSORBING": "one side is a variable absent from the other side",
    "STANDARD":  "lone variable appears on both sides",
    "GENERAL":   "both sides contain * operations",
}

_DEPTH_DESC = {
    0: "no * operators (depth 0)",
    1: "exactly one * operator (depth 1)",
    2: "two or more * operators (depth 2+)",
}


def partition_key_to_conditions(key: PartitionKey) -> list[str]:
    """
    Convert a PartitionKey to a list of structural ACTIVATE IF conditions
    expressed in the same plain-English format the LLM uses.

    These are injected as the first conditions in every generated case study,
    guaranteeing that the case study only fires on items from this partition.
    """
    form_e1, form_e2, depth_b, polarity = key
    return [
        f"E1 is {form_e1} form ({_FORM_DESC.get(form_e1, form_e1.lower())})",
        f"E2 is {form_e2} form ({_FORM_DESC.get(form_e2, form_e2.lower())})",
        f"E1 has {_DEPTH_DESC.get(depth_b, 'unknown depth')}",
        f"Expected answer is {polarity}",
    ]


# ---------------------------------------------------------------------------
# PartitionBin
# ---------------------------------------------------------------------------

# Reservoir cap for the designated correct pool per partition.
# Smaller than the global CORRECT_POOL_MAX (40) used in ICR_select because
# items are already structurally homogeneous — fewer are needed for a
# representative regression check.
CORRECT_POOL_PER_PARTITION_MAX = 50


@dataclass
class PartitionBin:
    """
    All failures sharing the same structural partition key, paired with a
    designated correct pool drawn exclusively from the same structural class.

    correct_pool is used for regression checks: a case study generated for
    this partition should not regress items in the same structural class.
    Because the pool is structurally matched to the candidate's ACTIVATE IF
    conditions, this check is both tighter (fewer false passes) and cheaper
    (smaller pool) than a global reservoir.
    """
    key:          PartitionKey
    failures:     list[dict] = field(default_factory=list)
    correct_pool: list[dict] = field(default_factory=list)
    solved:       bool       = False   # retired — skip in future iterations
    n_flushes:    int        = 0       # case studies accepted for this partition

    @property
    def label(self) -> str:
        return partition_label(self.key)

    def add_correct(self, item: dict) -> None:
        """Reservoir-sample item into correct_pool (bounded at CORRECT_POOL_PER_PARTITION_MAX)."""
        n = len(self.correct_pool)
        if n < CORRECT_POOL_PER_PARTITION_MAX:
            self.correct_pool.append(item)
        else:
            # Reservoir sampling: replace a random existing slot
            j = random.randrange(n + 1)
            if j < CORRECT_POOL_PER_PARTITION_MAX:
                self.correct_pool[j] = item

    def __len__(self) -> int:
        return len(self.failures)


# ---------------------------------------------------------------------------
# Build partitions from a scored pass
# ---------------------------------------------------------------------------

def build_partitions(
    wrong_items:   list[dict],
    correct_items: list[dict],
    bin_threshold: int = 3,
) -> dict[PartitionKey, PartitionBin]:
    """
    Route wrong and correct items into PartitionBins keyed by structural class.

    A bin is only created when its failure count >= bin_threshold (not enough
    failures to form a meaningful teaching batch → skip).  Correct items are
    always routed to their partition's correct_pool regardless of whether a bin
    exists for that key.

    Returns the dict of active (failure-count >= bin_threshold) bins only.
    """
    # Build correct pools for all structural classes first
    correct_pools: dict[PartitionKey, list[dict]] = {}
    for item in correct_items:
        key = item_partition_key(item)
        correct_pools.setdefault(key, []).append(item)

    # Group failures by partition key
    failure_groups: dict[PartitionKey, list[dict]] = {}
    for item in wrong_items:
        key = item_partition_key(item)
        failure_groups.setdefault(key, []).append(item)

    bins: dict[PartitionKey, PartitionBin] = {}
    for key, failures in sorted(failure_groups.items(), key=lambda kv: -len(kv[1])):
        if len(failures) < bin_threshold:
            continue
        pb = PartitionBin(key=key, failures=list(failures))
        for item in correct_pools.get(key, []):
            pb.add_correct(item)
        bins[key] = pb

    return bins


# ---------------------------------------------------------------------------
# Refresh partitions after a re-score pass
# ---------------------------------------------------------------------------

def refresh_partitions(
    bins:                 dict[PartitionKey, PartitionBin],
    new_wrong:            list[dict],
    new_correct:          list[dict],
    retirement_threshold: int = 2,
    log_fn=None,
) -> None:
    """
    Update bin.failures from a fresh scoring pass and retire bins whose
    remaining failure count fell below retirement_threshold.

    Called after each outer iteration.  Mutates bins in-place.
    """
    # Re-index fresh wrong items by partition key
    fresh_wrong: dict[PartitionKey, list[dict]] = {}
    for item in new_wrong:
        key = item_partition_key(item)
        fresh_wrong.setdefault(key, []).append(item)

    # Absorb new correct items into partition correct pools
    for item in new_correct:
        key = item_partition_key(item)
        if key in bins:
            bins[key].add_correct(item)

    # Update failure lists and check retirement
    for key, pb in bins.items():
        if pb.solved:
            continue
        old_n = len(pb.failures)
        pb.failures = fresh_wrong.get(key, [])
        new_n = len(pb.failures)
        if new_n < retirement_threshold:
            pb.solved = True
            if log_fn:
                log_fn(
                    f"  [partition:{pb.label}] retired — "
                    f"failures {old_n} → {new_n} < threshold={retirement_threshold}"
                )
        elif log_fn and new_n != old_n:
            log_fn(f"  [partition:{pb.label}] failures {old_n} → {new_n}")


# ---------------------------------------------------------------------------
# Diagnostic summary
# ---------------------------------------------------------------------------

def partition_summary(bins: dict[PartitionKey, PartitionBin]) -> list[dict]:
    """Return a serialisable per-bin summary for logging."""
    rows = []
    for pb in sorted(bins.values(), key=lambda b: -len(b)):
        rows.append({
            "partition":    pb.label,
            "failures":     len(pb.failures),
            "correct_pool": len(pb.correct_pool),
            "n_flushes":    pb.n_flushes,
            "solved":       pb.solved,
        })
    return rows


def print_partition_table(
    bins:    dict[PartitionKey, PartitionBin],
    title:   str = "PARTITION SUMMARY",
    file=sys.stderr,
) -> None:
    print(f"\n{'='*65}", file=file)
    print(title, file=file)
    print(f"{'='*65}", file=file)
    print(f"  {'Partition':<38} {'Fail':>4} {'Corr':>5} {'Flush':>5} {'Status'}", file=file)
    print(f"  {'-'*63}", file=file)
    for pb in sorted(bins.values(), key=lambda b: -len(b)):
        status = "retired" if pb.solved else "active"
        print(
            f"  {pb.label:<38} {len(pb.failures):>4} "
            f"{len(pb.correct_pool):>5} {pb.n_flushes:>5}  {status}",
            file=file,
        )
    active = sum(1 for pb in bins.values() if not pb.solved)
    print(f"\n  active={active}  total={len(bins)}", file=file)
