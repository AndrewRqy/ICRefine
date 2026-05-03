"""
plot_pk_size_fig4.py — Figure 4: Phase 1 PK Character Limit Ablation

Grouped bar chart: 4 tasks × 4 character-limit conditions.
GS 6K is now filled in with the corrected rerun value (75.0%).

Run from ICRefine root:
    python3 scripts/plots/plot_pk_size_fig4.py
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MINI = "openai/gpt-4.1-mini"

CONDITIONS = [
    ("3K",        "runs/ablation_size2_p1_3000chars_rf.json",   "#4C72B0"),
    ("6K",        "runs/ablation_size2_p1_6000chars_rf.json",   "#DD8452"),
    ("12K",       "runs/ablation_size2_p1_12000chars_rf.json",  "#55A868"),
    ("Unlimited", "runs/ablation_size2_p1_unlimited_rf.json",   "#C44E52"),
]

TASKS = [
    ("geometric_shapes",  "GS"),
    ("disambiguation_qa", "DQ"),
    ("formal_fallacies",  "FF"),
    ("snarks",            "Snarks"),
]


def load_acc(fpath):
    d = json.loads((ROOT / fpath).read_text())
    out = {}
    for task, task_key in TASKS:
        if task in d and MINI in d[task]:
            out[task_key] = d[task][MINI].get("full")
    return out


data = {label: load_acc(fpath) for label, fpath, _ in CONDITIONS}

fig, ax = plt.subplots(figsize=(8, 4.2))

task_labels = [tk for _, tk in TASKS]
n_tasks = len(task_labels)
n_conds = len(CONDITIONS)
x = np.arange(n_tasks)
w = 0.18
offsets = np.linspace(-(n_conds - 1) / 2, (n_conds - 1) / 2, n_conds) * w

for i, (label, fpath, color) in enumerate(CONDITIONS):
    vals = [data[label].get(tk, 0) for tk in task_labels]
    bars = ax.bar(x + offsets[i], vals, width=w, color=color, alpha=0.87,
                  label=label, zorder=2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{v:.0%}", ha="center", va="bottom", fontsize=7, color="#333")

ax.set_xticks(x)
ax.set_xticklabels(task_labels, fontsize=10)
ax.set_ylim(0.50, 1.05)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
ax.set_ylabel("RF test accuracy (gpt-4.1-mini)", fontsize=9)
ax.set_title("Phase 1 PK Character Limit Ablation", fontsize=11, fontweight="bold", pad=8)
ax.legend(title="PK cap", fontsize=8.5, title_fontsize=9, framealpha=0.9, loc="lower right")
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.text(0.5, -0.05,
         "Figure 4: RF test accuracy under four Phase 1 PK character-limit conditions (single-run, gpt-4.1-mini).\n"
         "No cap consistently dominates; the ordering is non-monotone across all four tasks.\n"
         "The 3K cap matches or exceeds larger limits on GS (78%) and FF/Snarks (ceiling tasks).",
         ha="center", va="top", fontsize=7.8, color="#333")

out = ROOT / "ICR_paper_prep/figures/fig4_pk_size_ablation.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved → {out}")
