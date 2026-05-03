"""
plot_oracle_gradient_fig8.py — Figure 8: Oracle Contamination Gradient

Two-panel grouped bar chart: 5 models × 3 oracle conditions (E3, v3, v5).
Left panel: Causal Judgement (full CS accuracy) — 3-seed means.
Right panel: Geometric Shapes (full CS accuracy) — 3-seed means.

All six condition×task cells now use 3-seed means.

Run from ICRefine root:
    python3 scripts/plots/plot_oracle_gradient_fig8.py
"""

import json
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
MODEL_IDS = [m for m, _ in MODELS]
MODEL_LABELS = [l for _, l in MODELS]


def load(path):
    return json.loads((ROOT / path).read_text())


# CJ: all 3-seed means from their respective aggregation files
cj_e3 = load("runs/variance/e3_3seed_mean.json")["causal_judgement"]
cj_v3 = load("runs/variance/v3_3seed_mean.json")["causal_judgement"]
cj_v5 = load("runs/variance/v5_3seed_mean.json")["causal_judgement"]

# GS: all 3-seed means
gs_e3 = load("runs/variance/e3_gs_3seed_mean.json")["geometric_shapes"]
gs_v3 = load("runs/variance/v3_3seed_mean.json")["geometric_shapes"]
gs_v5 = load("runs/variance/v5_gs_3seed_mean.json")["geometric_shapes"]


def get_vals(models_dict, metric="full"):
    return [models_dict.get(mid, {}).get(metric) for mid in MODEL_IDS]


CONDITIONS = [
    ("E3\n(no oracle)",   get_vals(cj_e3), get_vals(gs_e3), "#4C72B0"),
    ("v3\n(Phase 2 oracle)", get_vals(cj_v3), get_vals(gs_v3), "#55A868"),
    ("v5\n(both oracles)", get_vals(cj_v5), get_vals(gs_v5), "#C44E52"),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=False)
fig.subplots_adjust(wspace=0.34)

n_models = len(MODEL_LABELS)
n_conds = len(CONDITIONS)
x = np.arange(n_models)
w = 0.22
offsets = np.linspace(-(n_conds - 1) / 2, (n_conds - 1) / 2, n_conds) * w


def draw_panel(ax, cj_or_gs_idx, title, ylim):
    for i, (cond_label, cj_vals, gs_vals, color) in enumerate(CONDITIONS):
        vals = cj_vals if cj_or_gs_idx == 0 else gs_vals
        safe_vals = [v if v is not None else 0 for v in vals]
        bars = ax.bar(x + offsets[i], safe_vals, width=w, color=color,
                      alpha=0.85, label=cond_label, zorder=2)
        # Delta vs v3 annotation (skip v3 itself)
        if i != 1:
            v3_vals = cj_vals if cj_or_gs_idx == 0 else gs_vals
            ref = CONDITIONS[1][1] if cj_or_gs_idx == 0 else CONDITIONS[1][2]
            for j, (bar, v) in enumerate(zip(bars, vals)):
                if v is not None and ref[j] is not None:
                    delta = v - ref[j]
                    sign = "+" if delta >= 0 else ""
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.004,
                            f"{sign}{delta:.0%}",
                            ha="center", va="bottom", fontsize=6.5, color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=9)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("RF accuracy (full CS, 3-seed mean)", fontsize=8.5)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.legend(fontsize=8, framealpha=0.9, loc="lower right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


draw_panel(axes[0], 0, "Causal Judgement — Oracle Gradient (3-seed means)",
           ylim=(0.54, 0.78))
draw_panel(axes[1], 1, "Geometric Shapes — Oracle Gradient (3-seed means)",
           ylim=(0.40, 0.90))

fig.text(0.5, -0.06,
         "Figure 8: Oracle contamination gradient — full CS accuracy under three oracle conditions "
         "(E3: no oracle, v3: Phase 2 oracle only, v5: both oracles). All values are 3-seed means.\n"
         "Left (CJ): Removing oracle access consistently improves all 5 models (E3 > v3); "
         "full oracle (v5) degrades accuracy further. Δ labels show delta vs v3 baseline.\n"
         "Right (GS): Oracle effects are inconsistent across models, confirming GS case study harm "
         "is structural (SVG parsing difficulty) rather than oracle-contamination driven.",
         ha="center", va="top", fontsize=7.8, color="#333")

out = ROOT / "ICR_paper_prep/figures/fig8_oracle_gradient.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved → {out}")
