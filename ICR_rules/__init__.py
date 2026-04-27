"""
ICR_rules — Rule-patch variant of ICRefine.

Instead of generating case studies, this pipeline proposes minimal patches to
individual cheatsheet rules (tighten a condition, split a rule, add a guard).

Pipeline is structurally identical to ICR_partition:
  - Same partition key logic
  - Same oracle (GPT-5.4) for correct reasoning traces
  - Same bin grouping and per-bin testing
  - Changed: generation output is a RulePatch, not a CaseStudy
  - Changed: scoring uses SAIR-style prompt (no pre-computed features) to
    accurately measure real deployment performance for Gemma
"""
