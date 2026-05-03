# Summary Tables

_Generated from ICRefine runs. Test eval for ablation_size2 pending._

## Table 1: Phase Contribution — gpt-4.1-mini (RF test accuracy)

| Task | CS-ICL | Phase 0 | Phase 1 (PK) | Phase 2 (full) | Δ P1→P2 |
|------|--------|---------|-------------|----------------|---------|
| Bool | 100.0% | 100.0% | 100.0% | 100.0% | +0.0pp |
| CJ | 66.7% | 73.6% | 71.3% | 73.6% | +2.3pp |
| DU | 95.0% | 94.0% | 92.0% | 94.0% | +2.0pp |
| DQ | 91.0% | 84.0% | 85.0% | 84.0% | -1.0pp |
| FF | 95.0% | 96.0% | 96.0% | 96.0% | +0.0pp |
| GS | 79.0% | 70.0% | 77.0% | 70.0% | -7.0pp |
| LD3 | 100.0% | 100.0% | 100.0% | 100.0% | +0.0pp |
| Nav | 100.0% | 100.0% | 100.0% | 100.0% | +0.0pp |
| Snarks | 95.8% | 95.8% | 95.8% | 95.8% | +0.0pp |
| Sports | 98.0% | 100.0% | 99.0% | 100.0% | +1.0pp |
| WOL | 100.0% | 100.0% | 100.0% | 100.0% | +0.0pp |

## Table 2: Non-Train Transfer — Avg RF Accuracy (5 non-ceiling tasks: GS/FF/Snarks/DQ/CJ)

| Model | CS-ICL | PK only | Full | Δ full vs CS-ICL |
|-------|--------|---------|------|-----------------|
| GPT-4.1 | 87.6% | 85.3% | 83.6% | -4.0pp |
| Claude | 86.5% | 87.8% | 83.7% | -2.8pp |
| Gemini | 80.1% | 83.1% | 82.9% | +2.8pp |
| Llama | 78.1% | 76.1% | 73.9% | -4.3pp |

## Table 3: Phase 2 CS Count Ablation — Train Accuracy

_(ablation_size run; test eval for ablation_size2 pending)_

| Task | Unlimited | Best-of-1 | Best-of-3 | Best CS fix-rate (pool) |
|------|-----------|-----------|-----------|------------------------|
| GS | 83.3% | 78.0% | 89.3% | pool=4, best=100% |
| FF | 71.3% | 69.3% | 69.3% | pool=5, best=50% |
| Snarks | 87.9% | 91.6% | 87.9% | – |
| DQ | 86.7% | 84.0% | 86.0% | pool=4, best=57% |

## Table 4: Phase 1 PK Char Limit Ablation — Train Accuracy + Final PK Size

_(ablation_size run, sequential patching; ablation_size2 with Phase 0 cap in progress)_

| Task | 3K acc / PK | 6K acc / PK | 12K acc / PK | Unlim acc / PK |
|------|------|------|------|------|
| GS | 86.0% / 6K | 88.0% / 4K | 89.3% / 9K | 83.3% / 5K |
| FF | 70.7% / 10K | 71.3% / 8K | 72.7% / 7K | 71.3% / 8K |
| Snarks | 87.9% / 6K | 88.8% / 8K | 88.8% / 6K | 87.9% / 8K |
| DQ | 84.7% / 8K | 82.0% / 7K | 82.7% / 11K | 86.7% / 10K |

## Table 5: EA Phase 1 vs Standard Phase 1 — Train Accuracy

| Task | Std train acc | Std CS | EA train acc | EA CS | EA patches | Δ train |
|------|--------------|--------|-------------|-------|------------|---------|
| CJ | – | – | 64.0% | 4 | 0 | – |
| GS | 83.3% | 2 | 89.3% | 0 | 3 | +6.0pp |
| Snarks | 87.9% | 2 | 90.7% | 1 | 0 | +2.8pp |
| DQ | 86.7% | 0 | 88.7% | 3 | 1 | +2.0pp |
