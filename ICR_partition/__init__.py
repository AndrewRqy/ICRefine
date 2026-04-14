"""
ICR_partition — Partition-parallel iterative cheatsheet refinement.

Each item is assigned to a structural partition based on its (E1, E2) equation
forms and expected-answer polarity.  Failures within the same partition share a
structurally homogeneous failure mode and are solved independently and
concurrently.  Case studies generated for one partition should not fire on items
in a different partition (enforced by strict ACTIVATE IF conditions).

Key differences from ICR_select
--------------------------------
* Oracle is ON by default — failure items are always enriched with the
  nearest oracle reasoning trace when available.
* Roadmap is OFF by default — the cheatsheet starts with an empty roadmap
  and grows purely through case studies.
* Per-partition correct pools — regression checks use only correct items
  from the same structural class as the candidate, not a global reservoir.
* Concurrent bin solving — all active partitions are solved in parallel
  (bounded by --partition-concurrency).
* Bin retirement — partitions whose failure count falls below
  --retirement-threshold are retired and excluded from future iterations,
  focusing compute on the hard tail.
"""
