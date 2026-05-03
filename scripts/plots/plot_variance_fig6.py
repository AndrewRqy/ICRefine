"""
plot_variance_fig6.py — Figure 6: Seed Variance for Key Conditions

Two-panel figure:
  Left:  v3 GS pk_only vs EA GS pk_only (all 3 seeds + 3-seed mean)
  Right: v3 CJ full    vs E3 CJ full    (all 3 seeds + 3-seed mean)

Run from ICRefine root:
    python3 scripts/plots/plot_variance_fig6.py
"""

import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent


def load(path):
    return json.loads((ROOT / path).read_text())


v3_s1 = load("runs/rf_transfer_5tasks_v3.json")
v3_s2 = load("runs/variance/eval_results/v3_seed2_rf.json")
v3_s3 = load("runs/variance/eval_results/v3_seed3_rf.json")

ea_s1 = load("runs/bbh_ea_phase1_rf.json")
ea_s2 = load("runs/variance/eval_results/ea_seed2_rf.json")
ea_s3 = load("runs/variance/eval_results/ea_seed3_rf.json")

e3_s1 = load("runs/e3_no_oracle_rf.json")
e3_s2 = load("runs/variance/eval_results/e3_seed2_rf.json")
e3_s3 = load("runs/variance/eval_results/e3_seed3_rf.json")

MODELS = [
    ("openai/gpt-4.1-mini",                  "mini*"),
    ("openai/gpt-4.1",                        "GPT-4.1"),
    ("anthropic/claude-3-7-sonnet-20250219",  "Claude"),
    ("google/gemini-2.0-flash-001",           "Gemini"),
    ("meta-llama/llama-3.3-70b-instruct",     "Llama"),
]


def collect(task, metric, *seeds):
    out = {}
    for mid, label in MODELS:
        vals = [s[task][mid][metric]
                for s in seeds if task in s and mid in s[task]]
        if vals:
            out[label] = vals
    return out


gs_v3 = collect("geometric_shapes", "pk_only", v3_s1, v3_s2, v3_s3)
gs_ea = collect("geometric_shapes", "pk_only", ea_s1, ea_s2, ea_s3)
cj_v3 = collect("causal_judgement", "full",    v3_s1, v3_s2, v3_s3)
cj_e3 = collect("causal_judgement", "full",    e3_s1, e3_s2, e3_s3)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
fig.subplots_adjust(wspace=0.32)

COL_A,  DOT_A  = "#4C72B0", "#9DB8D2"   # blue  — v3 baseline
COL_B,  DOT_B  = "#DD8452", "#F2C4A0"   # orange — comparison


def draw_panel(ax, data_a, data_b, label_a, label_b, title, ylim=(0.48, 0.92)):
    labels = [lbl for _, lbl in MODELS if lbl in data_a or lbl in data_b]
    x_idx  = np.arange(len(labels))
    w = 0.35

    means_a = [statistics.mean(data_a[l]) if l in data_a else None for l in labels]
    means_b = [statistics.mean(data_b[l]) if l in data_b else None for l in labels]

    ax.bar(x_idx - w / 2, [m or 0 for m in means_a],
           width=w, color=COL_A, alpha=0.85, label=label_a, zorder=2)
    ax.bar(x_idx + w / 2, [m or 0 for m in means_b],
           width=w, color=COL_B, alpha=0.85, label=label_b, zorder=2)

    rng = np.random.default_rng(42)
    for i, lbl in enumerate(labels):
        for v in data_a.get(lbl, []):
            ax.scatter(i - w / 2 + rng.uniform(-0.06, 0.06), v,
                       color=DOT_A, s=22, zorder=4, alpha=0.75)
        for v in data_b.get(lbl, []):
            ax.scatter(i + w / 2 + rng.uniform(-0.06, 0.06), v,
                       color=DOT_B, s=22, zorder=4, alpha=0.75)

    # Δ annotation above comparison bar
    for i, (ma, mb) in enumerate(zip(means_a, means_b)):
        if ma is not None and mb is not None:
            delta = mb - ma
            sign  = "+" if delta >= 0 else ""
            ax.text(i + w / 2, mb + 0.007, f"{sign}{delta:.0%}",
                    ha="center", va="bottom", fontsize=7.5, color="#444")

    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("RF accuracy", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


draw_panel(axes[0], gs_v3, gs_ea,
           "v3 Std Phase 1", "EA Phase 1",
           "Geometric Shapes — PK-only (3 seeds)",
           ylim=(0.48, 0.90))

draw_panel(axes[1], cj_v3, cj_e3,
           "v3 (oracle CS)", "E3 (no-oracle CS)",
           "Causal Judgement — Full (3 seeds)",
           ylim=(0.52, 0.80))

fig.text(0.5, -0.04,
         "Figure 6: Per-seed (dots) and 3-seed mean (bars) RF accuracy for the two key ablation comparisons.\n"
         "Left: EA Phase 1 raises GS PK-only by +2–6 pp across all models; EA results are more stable "
         "(Llama σ=4.7% vs v3 σ=10.4%).\n"
         "Right: Removing oracle access (E3) improves CJ full accuracy for all five models (+1.9–3.8 pp), "
         "consistent with oracle contamination as a contributing factor; heterogeneous failure distribution "
         "is a plausible co-contributor (see Appendix B).",
         ha="center", va="top", fontsize=8, color="#333", wrap=True)

out = ROOT / "ICR_paper_prep/figures/fig6_variance_analysis.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved → {out}")
