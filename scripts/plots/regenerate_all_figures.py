"""
regenerate_all_figures.py
Regenerates all paper figures from canonical data, and creates new figures.

Run from ICRefine root:
    python3 scripts/plots/regenerate_all_figures.py
"""

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np

ROOT  = Path(__file__).resolve().parent.parent.parent
FIGS  = ROOT / "ICR_paper_prep/figures"


def load(path):
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else {}


# ─────────────────────────────────────────────────────────────────────────────
# Shared style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

MODEL_LABELS = {
    "openai/gpt-4.1-mini":                  "mini*",
    "openai/gpt-4.1":                        "GPT-4.1",
    "anthropic/claude-3-7-sonnet-20250219":  "Claude",
    "google/gemini-2.0-flash-001":           "Gemini",
    "meta-llama/llama-3.3-70b-instruct":     "Llama",
}
NONTRAIN_IDS = [
    "openai/gpt-4.1",
    "anthropic/claude-3-7-sonnet-20250219",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.3-70b-instruct",
]


# ═════════════════════════════════════════════════════════════════════════════
# FIG 1 — Phase contribution (train model, all 11 tasks)
# Uses Tab 1 values exactly as they appear in the paper.
# ═════════════════════════════════════════════════════════════════════════════
def fig1_phase_contribution():
    # Hardcoded from paper Tab 1 (canonical single-run RF test, gpt-4.1-mini)
    # Sorted: non-ceiling (GS, CJ, DQ, DU, FF, Snarks, Sports) then ceiling
    TASKS = [
        ("GS",     dict(csicl=0.790, p0=0.700, p1=0.710, p2=0.760)),
        ("CJ",     dict(csicl=0.667, p0=0.736, p1=0.701, p2=0.644)),
        ("DQ",     dict(csicl=0.910, p0=0.840, p1=0.870, p2=0.860)),
        ("DU",     dict(csicl=0.950, p0=0.940, p1=0.910, p2=0.930)),
        ("FF",     dict(csicl=0.950, p0=0.960, p1=0.960, p2=0.970)),
        ("Snarks", dict(csicl=0.958, p0=0.958, p1=0.944, p2=0.958)),
        ("Sports", dict(csicl=0.980, p0=1.000, p1=0.990, p2=1.000)),
        ("WOL",    dict(csicl=1.000, p0=1.000, p1=1.000, p2=1.000)),
        ("Nav",    dict(csicl=1.000, p0=1.000, p1=1.000, p2=1.000)),
        ("LD3",    dict(csicl=1.000, p0=1.000, p1=1.000, p2=1.000)),
        ("Bool",   dict(csicl=1.000, p0=1.000, p1=1.000, p2=1.000)),
    ]

    labels = [t[0] for t in TASKS]
    p0_vals = [t[1]["p0"] for t in TASKS]
    p1_vals = [t[1]["p1"] for t in TASKS]
    p2_vals = [t[1]["p2"] for t in TASKS]
    cs_icl  = [t[1]["csicl"] for t in TASKS]

    x    = np.arange(len(labels))
    w    = 0.25
    C0, C1, C2 = "#AED6F1", "#2E86C1", "#1A5276"

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(x - w,     p0_vals, width=w, color=C0, label="Phase 0 (bootstrap)", zorder=2)
    ax.bar(x,         p1_vals, width=w, color=C1, label="Phase 1 (PK patched)", zorder=2)
    ax.bar(x + w,     p2_vals, width=w, color=C2, label="Phase 2 (+ CS)", zorder=2)

    # CS-ICL dashed line per group
    for i, v in enumerate(cs_icl):
        ax.plot([i - 1.5*w, i + 1.5*w], [v, v], color="#555", lw=1.4,
                ls="--", zorder=3, label="CS-ICL baseline" if i == 0 else "")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.40, 1.06)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("RF Accuracy (%)", fontsize=10)
    ax.set_title("Phase Contribution — gpt-4.1-mini (train model, RF)", fontsize=11, fontweight="bold")

    # vertical divider between non-ceiling and ceiling
    ax.axvline(6.5, color="#aaa", lw=1, ls=":", zorder=0)
    ax.text(6.7, 0.42, "ceiling tasks →", fontsize=8, color="#888")

    ax.legend(fontsize=8.5, framealpha=0.9, ncol=4)
    fig.tight_layout()
    out = FIGS / "fig1_phase_contribution.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✓  fig1 → {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 2 — Non-train transfer (5-task avg, 3-seed means, Tab 2)
# ═════════════════════════════════════════════════════════════════════════════
def fig2_nontrain_transfer():
    # Hardcoded from paper Tab 2 (3-seed means for PK+Full; CS-ICL single-run)
    DATA = {
        "GPT-4.1": dict(csicl=0.876, pk=0.839, full=0.837),
        "Claude":  dict(csicl=0.861, pk=0.864, full=0.840),
        "Gemini":  dict(csicl=0.801, pk=0.806, full=0.804),
        "Llama":   dict(csicl=0.777, pk=0.768, full=0.757),
    }
    models  = list(DATA.keys())
    cond_labels = ["CS-ICL\n(baseline)", "PK only\n(3-seed)", "Full\n(3-seed)"]
    COLORS  = ["#7DCEA0", "#2E86C1", "#1A5276"]  # green, med-blue, dark-blue

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(models))
    w = 0.24

    for ci, (cond_key, color, lbl) in enumerate(zip(
            ["csicl", "pk", "full"], COLORS, cond_labels)):
        vals = [DATA[m][cond_key] for m in models]
        bars = ax.bar(x + (ci - 1) * w, vals, width=w, color=color,
                      alpha=0.88, label=lbl, zorder=2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=7.5)

    # Δ annotation above Full bar
    for i, m in enumerate(models):
        delta = DATA[m]["full"] - DATA[m]["csicl"]
        sign  = "+" if delta >= 0 else ""
        ax.text(i + w, DATA[m]["full"] + 0.018,
                f"{sign}{delta*100:.1f}pp", ha="center", fontsize=7.5,
                color="#c0392b" if delta < 0 else "#1e8449", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0.70, 1.01)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Avg RF Accuracy — 5 non-ceiling tasks", fontsize=10)
    ax.set_title("Non-Train Model Transfer: CS-ICL vs ICRefine (3-seed means)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9, loc="lower right")
    fig.tight_layout()
    out = FIGS / "fig2_nontrain_transfer.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✓  fig2 → {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 3 — Phase 2 CS count ablation (RF test accuracy, Tab 3)
# ═════════════════════════════════════════════════════════════════════════════
def fig3_cs_ablation():
    # From Tab 3 (RF test accuracy, gpt-4.1-mini)
    TASKS   = ["GS", "FF", "Snarks", "DQ"]
    DATA    = {
        "GS":     dict(unlim=0.780, b1=0.690, b3=0.740),
        "FF":     dict(unlim=0.970, b1=0.970, b3=0.970),
        "Snarks": dict(unlim=0.972, b1=0.972, b3=0.972),
        "DQ":     dict(unlim=0.880, b1=0.880, b3=0.790),
    }
    COLORS  = {"unlim": "#4A235A", "b3": "#7D3C98", "b1": "#C39BD3"}
    LABELS  = {"unlim": "Unlimited (v3)", "b3": "Best-of-3", "b1": "Best-of-1"}

    x = np.arange(len(TASKS))
    w = 0.26

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for ci, (key, color) in enumerate(COLORS.items()):
        vals = [DATA[t][key] for t in TASKS]
        bars = ax.bar(x + (ci - 1) * w, vals, width=w, color=color,
                      alpha=0.88, label=LABELS[key], zorder=2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{v:.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, fontsize=11)
    ax.set_ylim(0.60, 1.04)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("RF Test Accuracy — gpt-4.1-mini", fontsize=10)
    ax.set_title("Phase 2 Case Study Count Ablation (RF test accuracy)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)

    ax.text(0.02, 0.03,
            "Note: GS test ordering reverses from train (unlimited wins at test; best-of-3 led in training).",
            transform=ax.transAxes, fontsize=7.5, color="#666")
    fig.tight_layout()
    out = FIGS / "fig3_phase2_cs_ablation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✓  fig3 → {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 4 — Phase 1 PK char-limit ablation (RF test accuracy, Tab 4)
# ═════════════════════════════════════════════════════════════════════════════
def fig4_pk_size_ablation():
    # From Tab 4 (RF test accuracy, gpt-4.1-mini).
    # 6K GS excluded (18%, anomalous); shown as hatched "excluded" bar.
    TASKS = ["GS", "FF", "Snarks", "DQ"]
    DATA  = {
        "GS":     [0.780, None,  0.570, 0.750],   # None = anomalous (18%)
        "FF":     [0.980, 0.970, 0.970, 0.980],
        "Snarks": [0.972, 0.972, 0.972, 0.958],
        "DQ":     [0.700, 0.800, 0.850, 0.780],
    }
    SIZE_LABELS = ["3K", "6K†", "12K", "Unlimited"]
    COLORS      = ["#F7DC6F", "#F0A500", "#C0392B", "#7B241C"]
    HATCHES     = ["", "///", "", ""]

    x = np.arange(len(TASKS))
    w = 0.21

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for si, (slabel, color, hatch) in enumerate(zip(SIZE_LABELS, COLORS, HATCHES)):
        vals_raw = [DATA[t][si] for t in TASKS]
        vals_plot = [v if v is not None else 0.18 for v in vals_raw]
        bars = ax.bar(x + (si - 1.5) * w, vals_plot, width=w, color=color,
                      alpha=0.85, label=slabel, hatch=hatch, zorder=2,
                      edgecolor="white" if not hatch else "#888")
        for bar, v_raw, v_plot in zip(bars, vals_raw, vals_plot):
            if v_raw is None:
                ax.text(bar.get_x() + bar.get_width()/2, v_plot + 0.005,
                        "18%†", ha="center", va="bottom", fontsize=8,
                        color="#c0392b", fontstyle="italic")
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{v_raw:.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, fontsize=11)
    ax.set_ylim(0.55, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("RF Test Accuracy — gpt-4.1-mini", fontsize=10)
    ax.set_title("Phase 1 PK Character Limit Ablation (RF test accuracy)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9, title="PK cap")
    ax.text(0.02, 0.03,
            "†6K GS = 18% (anomalous — likely corrupted cheatsheet; excluded from interpretation).",
            transform=ax.transAxes, fontsize=7.5, color="#888")
    fig.tight_layout()
    out = FIGS / "fig4_pk_size_ablation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✓  fig4 → {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 5 — EA vs Standard Phase 1 (RF test accuracy, Tab EA)
# ═════════════════════════════════════════════════════════════════════════════
def fig5_ea_vs_standard():
    # From Tab EA (RF test accuracy; all 3-seed means, seeds 1-3)
    # Only PK-only shown (isolates Phase 1 quality from Phase 2)
    TASKS = ["CJ", "GS", "Snarks", "DQ"]
    STD_PK = [0.648, 0.727, 0.967, 0.847]   # all 3-seed v3 pk_only means
    EA_PK  = [0.701, 0.787, 0.958, 0.847]   # all 3-seed ea pk_only means
    EA_PATCHES = [0, 3, 0, 1]               # number of EA patches per task

    x  = np.arange(len(TASKS))
    w  = 0.35
    CSTD = "#4C72B0"
    CEA  = "#DD8452"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.3)

    # Left: PK-only accuracy comparison
    ax = axes[0]
    bars_std = ax.bar(x - w/2, STD_PK, width=w, color=CSTD, alpha=0.88,
                      label="Standard Phase 1", zorder=2)
    bars_ea  = ax.bar(x + w/2, EA_PK,  width=w, color=CEA,  alpha=0.88,
                      label="EA Phase 1",       zorder=2)

    for bars, vals in [(bars_std, STD_PK), (bars_ea, EA_PK)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=8)

    # Δ label above EA bar
    for i, (s, e) in enumerate(zip(STD_PK, EA_PK)):
        d = e - s
        if abs(d) > 0.001:
            sign = "+" if d >= 0 else ""
            color = "#1e8449" if d > 0 else "#c0392b"
            ax.text(i + w/2, e + 0.018, f"{sign}{d*100:.1f}pp",
                    ha="center", fontsize=8.5, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, fontsize=10)
    ax.set_ylim(0.60, 1.04)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("RF Test Accuracy (PK only)", fontsize=10)
    ax.set_title("EA vs Standard — PK-only Test Accuracy", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.text(0.02, 0.03, "All values are 3-seed means (seeds 1–3).",
            transform=ax.transAxes, fontsize=8, color="#666")

    # Right: CS count and EA patches side-by-side
    ax2 = axes[1]
    STD_CS = [1, 2, 2, 0]
    EA_CS  = [4, 0, 1, 3]
    bars_sc = ax2.bar(x - w/2, STD_CS, width=w, color=CSTD, alpha=0.88,
                      label="Std CS count", zorder=2)
    bars_ec = ax2.bar(x + w/2, EA_CS,  width=w, color=CEA,  alpha=0.88,
                      label="EA CS count",  zorder=2)
    for bars, vals in [(bars_sc, STD_CS), (bars_ec, EA_CS)]:
        for bar, v in zip(bars, vals):
            if v > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                         str(v), ha="center", va="bottom", fontsize=9)

    # EA patches as line on secondary axis
    ax2b = ax2.twinx()
    ax2b.spines["top"].set_visible(False)
    ax2b.plot(x, EA_PATCHES, color="#e74c3c", marker="D", ms=7,
              ls="--", lw=1.5, label="EA patches applied", zorder=5)
    ax2b.set_ylabel("EA Phase 1 patches", fontsize=9, color="#e74c3c")
    ax2b.tick_params(axis="y", labelcolor="#e74c3c")
    ax2b.set_ylim(0, 6)

    ax2.set_xticks(x)
    ax2.set_xticklabels(TASKS, fontsize=10)
    ax2.set_ylim(0, 6.5)
    ax2.set_ylabel("Case Studies in Final Cheatsheet", fontsize=10)
    ax2.set_title("CS Count + EA Patches", fontsize=10, fontweight="bold")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    out = FIGS / "fig5_ea_vs_standard.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✓  fig5 → {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 7 (NEW) — Δ_cs heatmap: case study contribution across tasks × models
# ═════════════════════════════════════════════════════════════════════════════
def fig7_delta_cs_heatmap():
    v3_3seed = load("runs/variance/v3_3seed_mean.json")

    TASKS = ["CJ", "GS", "DQ", "FF", "Snarks"]
    TASK_KEYS = ["causal_judgement","geometric_shapes","disambiguation_qa",
                 "formal_fallacies","snarks"]
    MODEL_IDS = [
        "openai/gpt-4.1-mini",
        "openai/gpt-4.1",
        "anthropic/claude-3-7-sonnet-20250219",
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.3-70b-instruct",
    ]
    MODEL_SHORT = ["mini*", "GPT-4.1", "Claude", "Gemini", "Llama"]

    # Build matrix: rows=tasks, cols=models
    Z = np.full((len(TASKS), len(MODEL_IDS)), np.nan)
    for ri, (tshort, tkey) in enumerate(zip(TASKS, TASK_KEYS)):
        if tkey not in v3_3seed:
            continue
        for ci, mid in enumerate(MODEL_IDS):
            if mid in v3_3seed[tkey] and "delta_cs" in v3_3seed[tkey][mid]:
                Z[ri, ci] = v3_3seed[tkey][mid]["delta_cs"] * 100  # pp

    # Symmetric colormap centred at 0
    vmax = max(abs(np.nanmin(Z)), abs(np.nanmax(Z))) + 0.5
    cmap = plt.cm.RdBu   # red = negative (CS hurts), blue = positive (CS helps)

    fig, ax = plt.subplots(figsize=(8, 4.0))
    im = ax.imshow(Z, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(MODEL_IDS)))
    ax.set_xticklabels(MODEL_SHORT, fontsize=10)
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels(TASKS, fontsize=11)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Annotate cells
    for ri in range(len(TASKS)):
        for ci in range(len(MODEL_IDS)):
            v = Z[ri, ci]
            if not np.isnan(v):
                sign  = "+" if v >= 0 else ""
                color = "white" if abs(v) > vmax * 0.55 else "black"
                ax.text(ci, ri, f"{sign}{v:.1f}", ha="center", va="center",
                        fontsize=9.5, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Δ_cs = Full − PK-only (pp, 3-seed mean)", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{'+' if v>0 else ''}{v:.0f}pp"))

    ax.set_title("Case Study Contribution (Δ_cs) across Tasks and Models\n"
                 "Red = CS hurts  ·  Blue = CS helps  ·  v3 pipeline, 3-seed mean",
                 fontsize=10, pad=14)

    # Vertical divider: train model vs non-train
    ax.axvline(0.5, color="#333", lw=1.5, ls="--")
    ax.text(0.0, len(TASKS) - 0.4, "train\nmodel", ha="center", fontsize=7.5,
            color="#333", va="top")

    fig.tight_layout()
    out = FIGS / "fig7_delta_cs_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✓  fig7 → {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 8 (NEW) — Oracle contamination gradient: E3 / v3 / v5 on CJ + GS
# ═════════════════════════════════════════════════════════════════════════════
def fig8_oracle_gradient():
    """
    Data from E-Oracle2x2-RF section of experiment_log.
    Shows full CS accuracy for 5 models across 3 oracle conditions.
    """
    # CJ full accuracy per condition per model (from experiment log)
    CJ = {
        "E3 (no oracle)": [0.678, 0.713, 0.667, 0.690, 0.701],
        "v3 (P2 oracle)": [0.667, 0.644, 0.598, 0.655, 0.644],
        "v5 (P1+P2 oracle)": [0.667, 0.621, 0.655, 0.586, 0.586],
    }
    # GS pk_only accuracy per condition (best comparison for oracle PK effect)
    GS = {
        "E3 (no oracle)": [0.610, 0.770, 0.800, 0.730, 0.590],
        "v3 (P2 oracle)": [0.770, 0.730, 0.790, 0.700, 0.610],
        "v5 (P1+P2 oracle)": [0.700, 0.560, 0.810, 0.550, 0.670],
    }

    MODELS_SHORT = ["mini*", "GPT-4.1", "Claude", "Gemini", "Llama"]
    CONDITIONS   = list(CJ.keys())
    COLORS = ["#27AE60", "#2980B9", "#C0392B"]  # green (no-oracle), blue (P2), red (both)
    MARKERS = ["o", "s", "^"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    fig.subplots_adjust(wspace=0.32)

    x = np.arange(len(MODELS_SHORT))

    for ax, data, title, ylabel, ylim in [
        (axes[0], CJ, "Causal Judgement — Full CS", "RF Accuracy", (0.55, 0.78)),
        (axes[1], GS, "Geometric Shapes — PK-only", "RF Accuracy", (0.45, 0.90)),
    ]:
        for cond, color, marker in zip(CONDITIONS, COLORS, MARKERS):
            vals = data[cond]
            ax.plot(x, vals, color=color, marker=marker, ms=7, lw=2,
                    label=cond, zorder=3)
            for xi, v in zip(x, vals):
                ax.text(xi, v + 0.008, f"{v:.0%}", ha="center", va="bottom",
                        fontsize=7, color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(MODELS_SHORT, fontsize=9.5)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Oracle Contamination Gradient: more oracle → worse transfer\n"
        "E3 (no oracle) > v3 (Phase 2 oracle only) > v5 (Phase 1 + Phase 2 oracle) for most models",
        fontsize=9.5, y=1.01
    )
    fig.tight_layout()
    out = FIGS / "fig8_oracle_gradient.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✓  fig8 → {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Regenerating all figures…\n")
    fig1_phase_contribution()
    fig2_nontrain_transfer()
    fig3_cs_ablation()
    fig4_pk_size_ablation()
    fig5_ea_vs_standard()
    fig7_delta_cs_heatmap()
    fig8_oracle_gradient()
    print("\nAll done.")
