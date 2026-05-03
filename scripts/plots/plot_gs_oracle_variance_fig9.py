"""
plot_gs_oracle_variance_fig9.py — Figure 9 (NEW): GS Oracle Gradient — Per-Seed Variance

Three-panel figure: E3 / v3 / v5 full CS accuracy on Geometric Shapes.
Each panel shows per-seed dots + 3-seed mean bar for all 5 models.
Directly backs the claim that GS oracle effects are inconsistent and
the variation is cross-seed noise, not oracle signal.

Run from ICRefine root:
    python3 scripts/plots/plot_gs_oracle_variance_fig9.py
"""

import json
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

MODELS = [
    ("openai/gpt-4.1-mini",                 "mini*"),
    ("openai/gpt-4.1",                       "GPT-4.1"),
    ("anthropic/claude-3-7-sonnet-20250219", "Claude"),
    ("google/gemini-2.0-flash-001",          "Gemini"),
    ("meta-llama/llama-3.3-70b-instruct",    "Llama"),
]
MODEL_IDS  = [m for m, _ in MODELS]
MODEL_LBLS = [l for _, l in MODELS]


def load(path):
    return json.loads((ROOT / path).read_text())


# Seed-level files for GS
e3_seeds = [
    load("runs/e3_no_oracle_rf.json"),
    load("runs/variance/eval_results/e3_seed2_gs_rf.json"),
    load("runs/variance/eval_results/e3_seed3_gs_rf.json"),
]
v3_seeds = [
    load("runs/rf_transfer_5tasks_v3.json"),
    load("runs/variance/eval_results/v3_seed2_rf.json"),
    load("runs/variance/eval_results/v3_seed3_rf.json"),
]
v5_seeds = [
    load("runs/v5_full_oracle_rf.json"),
    load("runs/variance/eval_results/v5_seed2_gs_rf.json"),
    load("runs/variance/eval_results/v5_seed3_gs_rf.json"),
]

TASK = "geometric_shapes"
METRIC = "full"


def collect_seeds(seed_files):
    """Returns {model_label: [seed1_val, seed2_val, seed3_val]}."""
    out = {}
    for mid, lbl in MODELS:
        vals = []
        for sf in seed_files:
            if TASK in sf and mid in sf[TASK]:
                v = sf[TASK][mid].get(METRIC)
                if v is not None:
                    vals.append(v)
        if vals:
            out[lbl] = vals
    return out


CONDITIONS = [
    ("E3\n(no oracle)",    collect_seeds(e3_seeds), "#4C72B0", "#9DB8D2"),
    ("v3\n(Phase 2 oracle)", collect_seeds(v3_seeds), "#55A868", "#A8D4B0"),
    ("v5\n(both oracles)", collect_seeds(v5_seeds), "#C44E52", "#E8A0A2"),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
fig.subplots_adjust(wspace=0.12)

rng = np.random.default_rng(42)
x = np.arange(len(MODEL_LBLS))
w = 0.55


def draw_panel(ax, data, label, bar_color, dot_color, is_first):
    means = [statistics.mean(data[l]) if l in data else 0 for l in MODEL_LBLS]
    ax.bar(x, means, width=w, color=bar_color, alpha=0.80, zorder=2)

    for i, lbl in enumerate(MODEL_LBLS):
        for v in data.get(lbl, []):
            ax.scatter(i + rng.uniform(-0.12, 0.12), v,
                       color=dot_color, s=28, zorder=4, alpha=0.9,
                       edgecolors=bar_color, linewidths=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LBLS, fontsize=9)
    ax.set_ylim(0.36, 0.92)
    if is_first:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
        ax.set_ylabel("RF accuracy (full CS)", fontsize=9)
    else:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_title(label, fontsize=10, fontweight="bold", pad=6)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Std annotation
    for i, lbl in enumerate(MODEL_LBLS):
        vals = data.get(lbl, [])
        if len(vals) > 1:
            sd = statistics.stdev(vals)
            ax.text(i, 0.375, f"σ={sd:.0%}", ha="center", va="bottom",
                    fontsize=6.5, color="#555")


for ax, (lbl, data, bc, dc) in zip(axes, CONDITIONS):
    draw_panel(ax, data, f"GS — {lbl}", bc, dc, ax is axes[0])

fig.suptitle("Geometric Shapes: Oracle Gradient Per-Seed Variance (3 seeds each)",
             fontsize=11, fontweight="bold", y=1.01)

fig.text(0.5, -0.07,
         "Figure 9: Per-seed (dots) and 3-seed mean (bars) full CS accuracy on Geometric Shapes "
         "under three oracle conditions.\n"
         "Unlike CJ (where E3 > v3 > v5 consistently), GS shows no monotone oracle ordering. "
         "High within-condition variance (σ labels) confirms oracle signal is weak relative to\n"
         "pipeline randomness on this task. GS case study harm is structural "
         "(SVG path parsing difficulty), not oracle-contamination driven.",
         ha="center", va="top", fontsize=7.8, color="#333")

out = ROOT / "ICR_paper_prep/figures/fig9_gs_oracle_variance.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved → {out}")
