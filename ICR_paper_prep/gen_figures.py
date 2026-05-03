"""
gen_figures.py — Generate all paper figures and summary tables.

Run from ICRefine/ root:
    python3 ICR_paper_prep/gen_figures.py

Outputs (all under ICR_paper_prep/figures/):
    fig1_phase_contribution.png   — Phase 0/1/2 accuracy by task (train model)
    fig2_nontrain_transfer.png    — Non-train avg accuracy across conditions
    fig3_phase2_ablation.png      — CS count ablation (train accuracy proxy)
    fig4_pk_size_ablation.png     — PK char limit ablation (train accuracy proxy)
    fig5_ea_vs_v3.png             — EA Phase 1 vs standard Phase 1 (train acc)
    tables.md                     — All summary tables in markdown
"""

import json, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT   = Path(__file__).parent.parent
OUTDIR = Path(__file__).parent / "figures"
OUTDIR.mkdir(exist_ok=True)

def load(fp):
    return json.load(open(ROOT / fp))

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "phase0":    "#9ecae1",
    "phase1":    "#4292c6",
    "phase2":    "#08519c",
    "csicl":     "#d9d9d9",
    "ea":        "#e6550d",
    "v3":        "#08519c",
    "p1_3k":     "#fdcc8a",
    "p1_6k":     "#fc8d59",
    "p1_12k":    "#e34a33",
    "p1_unlim":  "#b30000",
    "p2_1cs":    "#cbc9e2",
    "p2_3cs":    "#756bb1",
    "p2_unlim":  "#54278f",
}

TASK_LABELS = {
    "geometric_shapes":    "GS",
    "formal_fallacies":    "FF",
    "snarks":              "Snarks",
    "disambiguation_qa":   "DQ",
    "causal_judgement":    "CJ",
    "web_of_lies":         "WOL",
    "date_understanding":  "DU",
    "navigate":            "Nav",
    "boolean_expressions": "Bool",
    "sports_understanding":"Sports",
    "logical_deduction_three": "LD3",
}

MODEL_SHORT = {
    "openai/gpt-4.1-mini":                        "mini (train)",
    "openai/gpt-4.1":                             "GPT-4.1",
    "anthropic/claude-3-7-sonnet-20250219":        "Claude",
    "google/gemini-2.0-flash-001":                 "Gemini",
    "meta-llama/llama-3.3-70b-instruct":           "Llama",
}

# ── Merged result loader ──────────────────────────────────────────────────────
def load_merged_rf():
    """Merge all RF result files into one combined dict: task → model → {cond: val}."""
    combined = {}
    for fp in [
        "runs/rf_transfer_5tasks_v3.json",
        "runs/rf_transfer_6tasks_e9.json",
        "runs/oracle_fix_rf_gpt41mini.json",
        "runs/all_tasks_rf_gpt41mini_fix.json",
        "runs/phase0_mini_rf.json",
    ]:
        try:
            d = load(fp)
        except FileNotFoundError:
            continue
        for task, models in d.items():
            if task not in combined:
                combined[task] = {}
            if not isinstance(models, dict):
                continue
            for m, vals in models.items():
                if m not in combined[task]:
                    combined[task][m] = {}
                if isinstance(vals, dict):
                    combined[task][m].update(vals)
    return combined

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Phase contribution (train model, RF): Phase0 / Phase1(pk_only) / Phase2(full)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_phase_contribution():
    p0  = load("runs/phase0_mini_rf.json")
    rf  = load_merged_rf()

    MODEL = "openai/gpt-4.1-mini"
    tasks = [t for t in rf if t in p0 and MODEL in rf[t] and MODEL in p0[t]
             and "full" in rf[t][MODEL]]
    tasks.sort(key=lambda t: rf[t][MODEL].get("full", 0))

    ph0  = [p0[t][MODEL]["full"] * 100 for t in tasks]
    ph1  = [rf[t][MODEL].get("pk_only", p0[t][MODEL]["full"]) * 100 for t in tasks]
    ph2  = [rf[t][MODEL]["full"] * 100 for t in tasks]
    csicl = [rf[t][MODEL].get("cs_icl", None) for t in tasks]
    csicl_vals = [c * 100 if c is not None else None for c in csicl]

    x = np.arange(len(tasks))
    w = 0.22
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x - w, ph0,  w, label="Phase 0 (bootstrap)", color=C["phase0"])
    ax.bar(x,     ph1,  w, label="Phase 1 (PK patched)", color=C["phase1"])
    ax.bar(x + w, ph2,  w, label="Phase 2 (+ CS)",       color=C["phase2"])
    for i, cv in enumerate(csicl_vals):
        if cv is not None:
            ax.hlines(cv, x[i] - 1.5*w, x[i] + 1.5*w, colors="#555", linewidths=1.2, linestyles="--")

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=9)
    ax.set_ylabel("RF Accuracy (%)")
    ax.set_title("Phase Contribution — gpt-4.1-mini (train model, RF)")
    ax.legend(fontsize=8)
    ax.set_ylim(40, 105)
    ax.axhline(100, color="#aaa", lw=0.5, ls=":")

    # CS-ICL legend entry
    csicl_line = plt.Line2D([0],[0], color="#555", lw=1.2, ls="--", label="CS-ICL baseline")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [csicl_line], labels + ["CS-ICL baseline"], fontsize=8)

    fig.tight_layout()
    fp = OUTDIR / "fig1_phase_contribution.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"  saved {fp.name}")
    return tasks, ph0, ph1, ph2

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Non-train transfer: cs_icl / pk_only / full across models
# Uses merged RF (5 non-ceiling tasks: GS, FF, Snarks, DQ, CJ)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_nontrain_transfer():
    rf = load_merged_rf()
    MODELS = [m for m in MODEL_SHORT if m != "openai/gpt-4.1-mini"]  # non-train only
    NON_CEILING = ["geometric_shapes","formal_fallacies","snarks","disambiguation_qa","causal_judgement"]
    tasks  = [t for t in NON_CEILING if t in rf]

    # Average over tasks for each model × condition
    conds = ["cs_icl", "pk_only", "full"]
    cond_labels = ["CS-ICL", "Phase 1\n(PK only)", "Phase 2\n(full)"]

    model_avgs = {}
    for m in MODELS:
        avgs = []
        for c in conds:
            vals = [rf[t][m][c] for t in tasks if m in rf.get(t,{}) and c in rf[t].get(m,{})]
            avgs.append(np.mean(vals) * 100 if vals else None)
        model_avgs[m] = avgs

    x = np.arange(len(conds))
    w = 0.15
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#1b7837","#762a83","#e08214","#2166ac"]
    for i, (m, clr) in enumerate(zip(MODELS, colors)):
        offset = (i - len(MODELS)/2 + 0.5) * w
        vals = model_avgs[m]
        ax.bar(x + offset, [v if v else 0 for v in vals], w,
               label=MODEL_SHORT[m], color=clr, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=10)
    ax.set_ylabel("Avg RF Accuracy (%) — 6 tasks")
    ax.set_title("Non-Train Model Transfer Across Conditions")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(50, 100)

    fig.tight_layout()
    fp = OUTDIR / "fig2_nontrain_transfer.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"  saved {fp.name}")
    return model_avgs

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Phase 2 CS count ablation (train accuracy from ablation_size run)
# ─────────────────────────────────────────────────────────────────────────────
def fig3_phase2_ablation():
    TASKS = ["geometric_shapes","formal_fallacies","snarks","disambiguation_qa"]
    CONDITIONS = {
        "p1_unlimited": ("Unlimited\n(baseline)", C["p2_unlim"]),
        "p2_1cs":       ("Best-of-1",             C["p2_1cs"]),
        "p2_3cs":       ("Best-of-3",             C["p2_3cs"]),
    }

    # Read from ablation_size run logs
    def get_train_acc(cond, task):
        log = ROOT / f"runs/ablation_size/{cond}/logs/{task}.log"
        if not log.exists():
            return None
        for line in reversed(log.read_text().splitlines()):
            if "final train accuracy=" in line:
                try: return float(line.split("final train accuracy=")[1].strip().rstrip("%"))
                except: pass
        return None

    x = np.arange(len(TASKS))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (cond, (lbl, clr)) in enumerate(CONDITIONS.items()):
        vals = [get_train_acc(cond, t) or 0 for t in TASKS]
        offset = (i - 1) * w
        ax.bar(x + offset, vals, w, label=lbl, color=clr)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t,t) for t in TASKS])
    ax.set_ylabel("Train Accuracy (%)")
    ax.set_title("Phase 2 CS Count Ablation — Train Accuracy\n(best-of-N with loosened pool threshold 0.20)")
    ax.legend()
    ax.set_ylim(60, 100)
    ax.axhline(90, color="#aaa", lw=0.5, ls=":")

    note = "Note: train accuracy only — test eval pending (ablation_size2 run in progress)"
    ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=7, color="#888")

    fig.tight_layout()
    fp = OUTDIR / "fig3_phase2_cs_ablation.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"  saved {fp.name}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Phase 1 PK char limit ablation (train accuracy)
# ─────────────────────────────────────────────────────────────────────────────
def fig4_pk_size_ablation():
    TASKS = ["geometric_shapes","formal_fallacies","snarks","disambiguation_qa"]
    CONDITIONS = {
        "p1_3000chars":  ("3K chars",   C["p1_3k"]),
        "p1_6000chars":  ("6K chars",   C["p1_6k"]),
        "p1_12000chars": ("12K chars",  C["p1_12k"]),
        "p1_unlimited":  ("Unlimited",  C["p1_unlim"]),
    }

    def get_train_acc(cond, task):
        log = ROOT / f"runs/ablation_size/{cond}/logs/{task}.log"
        if not log.exists():
            return None
        for line in reversed(log.read_text().splitlines()):
            if "final train accuracy=" in line:
                try: return float(line.split("final train accuracy=")[1].strip().rstrip("%"))
                except: pass
        return None

    def get_pk_size(cond, task):
        jf = ROOT / f"runs/ablation_size/{cond}/{task}/cheatsheet_final.json"
        if not jf.exists():
            return None
        d = json.load(open(jf))
        return len(d.get("prior_knowledge",""))

    x = np.arange(len(TASKS))
    w = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: train accuracy by condition
    ax = axes[0]
    for i, (cond, (lbl, clr)) in enumerate(CONDITIONS.items()):
        vals = [get_train_acc(cond, t) or 0 for t in TASKS]
        offset = (i - 1.5) * w
        ax.bar(x + offset, vals, w, label=lbl, color=clr)
    ax.set_xticks(x); ax.set_xticklabels([TASK_LABELS.get(t,t) for t in TASKS])
    ax.set_ylabel("Train Accuracy (%)"); ax.set_ylim(60, 100)
    ax.set_title("Phase 1 PK Char Limit — Train Accuracy")
    ax.legend(fontsize=8)

    # Right: actual PK sizes achieved
    ax2 = axes[1]
    for i, (cond, (lbl, clr)) in enumerate(CONDITIONS.items()):
        vals = [(get_pk_size(cond, t) or 0) / 1000 for t in TASKS]
        offset = (i - 1.5) * w
        ax2.bar(x + offset, vals, w, label=lbl, color=clr)
    # v3 baseline sizes
    v3_pk = {"geometric_shapes":9.0,"formal_fallacies":5.9,"snarks":6.8,"disambiguation_qa":16.1}
    for i, t in enumerate(TASKS):
        ax2.hlines(v3_pk[t], x[i]-2*w, x[i]+2*w, colors="#333", lw=1.2, ls="--")
    ax2.set_xticks(x); ax2.set_xticklabels([TASK_LABELS.get(t,t) for t in TASKS])
    ax2.set_ylabel("PK Size (K chars)"); ax2.set_title("Actual PK Size After Phase 1")
    baseline_line = plt.Line2D([0],[0], color="#333", lw=1.2, ls="--", label="v3 baseline")
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles+[baseline_line], labels+["v3 baseline"], fontsize=8)

    note = "Note: train accuracy only — ablation_size2 with Phase 0 cap in progress"
    fig.text(0.01, 0.01, note, fontsize=7, color="#888")
    fig.tight_layout()
    fp = OUTDIR / "fig4_pk_size_ablation.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"  saved {fp.name}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — EA Phase 1 vs standard Phase 1 (train accuracy + PK/CS structure)
# ─────────────────────────────────────────────────────────────────────────────
def fig5_ea_vs_v3():
    TASKS = ["causal_judgement","geometric_shapes","snarks","disambiguation_qa"]

    def get_ea_stats(task):
        log = ROOT / f"runs/bbh_ea_phase1/logs/{task}.log"
        if not log.exists(): return None, None, None
        acc = None
        patches = 0
        for line in log.read_text().splitlines():
            if "final train accuracy=" in line:
                try: acc = float(line.split("=")[1].strip().rstrip("%"))
                except: pass
            if "total_patches=" in line:
                try: patches = int(line.split("total_patches=")[1].split()[0])
                except: pass
        jf = ROOT / f"runs/bbh_ea_phase1/{task}/cheatsheet_final.json"
        cs = 0
        if jf.exists():
            d = json.load(open(jf))
            cs = len(d.get("case_studies",[]))
        return acc, patches, cs

    def get_v3_stats(task):
        # Use p1_unlimited as the sequential-patching baseline (same Phase 2)
        log = ROOT / f"runs/ablation_size/p1_unlimited/logs/{task}.log"
        if not log.exists(): return None, None
        acc = None
        for line in reversed(log.read_text().splitlines()):
            if "final train accuracy=" in line:
                try: acc = float(line.split("=")[1].strip().rstrip("%"))
                except: pass
                break
        jf = ROOT / f"runs/ablation_size/p1_unlimited/{task}/cheatsheet_final.json"
        cs = 0
        if jf.exists():
            d = json.load(open(jf))
            cs = len(d.get("case_studies",[]))
        return acc, cs

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy comparison
    ax = axes[0]
    ea_accs  = []
    v3_accs  = []
    for t in TASKS:
        ea_acc, _, _ = get_ea_stats(t)
        v3_acc, _    = get_v3_stats(t)
        ea_accs.append(ea_acc or 0)
        v3_accs.append(v3_acc or 0)

    x = np.arange(len(TASKS))
    w = 0.35
    ax.bar(x - w/2, v3_accs, w, label="Standard Phase 1", color=C["v3"], alpha=0.85)
    ax.bar(x + w/2, ea_accs, w, label="EA Phase 1",        color=C["ea"], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([TASK_LABELS.get(t,t) for t in TASKS])
    ax.set_ylabel("Final Train Accuracy (%)"); ax.set_ylim(55, 100)
    ax.set_title("EA vs Standard Phase 1 — Train Accuracy")
    ax.legend()
    for i, (v, e) in enumerate(zip(v3_accs, ea_accs)):
        delta = e - v
        clr = "#c00" if delta < 0 else "#080"
        ax.text(i, max(v,e)+0.5, f"{delta:+.1f}", ha="center", fontsize=8, color=clr, fontweight="bold")

    # CS count comparison
    ax2 = axes[1]
    ea_cs = []
    v3_cs = []
    ea_patches = []
    for t in TASKS:
        _, patches, cs = get_ea_stats(t)
        _, vcs = get_v3_stats(t)
        ea_cs.append(cs or 0)
        v3_cs.append(vcs or 0)
        ea_patches.append(patches or 0)

    ax2.bar(x - w/2, v3_cs, w, label="Standard CS count", color=C["v3"], alpha=0.85)
    ax2.bar(x + w/2, ea_cs, w, label="EA CS count",       color=C["ea"], alpha=0.85)
    ax2r = ax2.twinx()
    ax2r.plot(x, ea_patches, "D--", color="#e6550d", markersize=6, lw=1.5, label="EA patches applied")
    ax2r.set_ylabel("EA Phase 1 patches applied", color="#e6550d")
    ax2r.tick_params(axis="y", colors="#e6550d")
    ax2r.set_ylim(0, 6)
    ax2.set_xticks(x); ax2.set_xticklabels([TASK_LABELS.get(t,t) for t in TASKS])
    ax2.set_ylabel("Case Studies in Final Cheatsheet"); ax2.set_ylim(0, 8)
    ax2.set_title("EA vs Standard — CS Count + EA Patches")
    ax2.legend(loc="upper left", fontsize=8)
    ax2r.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fp = OUTDIR / "fig5_ea_vs_standard.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"  saved {fp.name}")

    return TASKS, ea_accs, v3_accs, ea_cs, v3_cs

# ─────────────────────────────────────────────────────────────────────────────
# Tables — markdown summary
# ─────────────────────────────────────────────────────────────────────────────
def make_tables():
    lines = []
    lines.append("# Summary Tables\n")
    lines.append(f"_Generated from ICRefine runs. Test eval for ablation_size2 pending._\n")

    # Table 1: Phase contribution (train model, RF test accuracy)
    lines.append("## Table 1: Phase Contribution — gpt-4.1-mini (RF test accuracy)\n")
    lines.append("| Task | CS-ICL | Phase 0 | Phase 1 (PK) | Phase 2 (full) | Δ P1→P2 |")
    lines.append("|------|--------|---------|-------------|----------------|---------|")
    p0  = load("runs/phase0_mini_rf.json")
    rf  = load_merged_rf()
    MODEL = "openai/gpt-4.1-mini"
    for t in sorted(rf):
        if MODEL not in rf[t] or "full" not in rf[t][MODEL]: continue
        csicl = rf[t][MODEL].get("cs_icl")
        ph0   = p0.get(t,{}).get(MODEL,{}).get("full")
        ph1   = rf[t][MODEL].get("pk_only")
        ph2   = rf[t][MODEL]["full"]
        delta = (ph2 - ph1) * 100 if ph1 is not None else None
        v_csicl = f"{csicl*100:.1f}%" if csicl else "–"
        v_ph0   = f"{ph0*100:.1f}%"   if ph0   else "–"
        v_ph1   = f"{ph1*100:.1f}%"   if ph1 is not None else "–"
        v_ph2   = f"{ph2*100:.1f}%"
        v_delta = f"{delta:+.1f}pp"   if delta is not None else "–"
        lines.append(f"| {TASK_LABELS.get(t,t)} | {v_csicl} | {v_ph0} | {v_ph1} | {v_ph2} | {v_delta} |")
    lines.append("")

    # Table 2: Non-train transfer summary (5 non-ceiling tasks)
    lines.append("## Table 2: Non-Train Transfer — Avg RF Accuracy (5 non-ceiling tasks: GS/FF/Snarks/DQ/CJ)\n")
    rf6 = load_merged_rf()
    MODELS_NT = [m for m in MODEL_SHORT if m != "openai/gpt-4.1-mini"]
    tasks6 = [t for t in ["geometric_shapes","formal_fallacies","snarks","disambiguation_qa","causal_judgement"] if t in rf6]
    lines.append("| Model | CS-ICL | PK only | Full | Δ full vs CS-ICL |")
    lines.append("|-------|--------|---------|------|-----------------|")
    for m in MODELS_NT:
        vals = {}
        for c in ["cs_icl","pk_only","full"]:
            v = [rf6[t][m][c] for t in tasks6 if m in rf6.get(t,{}) and c in rf6[t].get(m,{})]
            vals[c] = np.mean(v)*100 if v else None
        delta = (vals["full"] - vals["cs_icl"]) if vals.get("full") and vals.get("cs_icl") else None
        v_csicl  = f'{vals["cs_icl"]:.1f}%'  if vals.get("cs_icl")  else "–"
        v_pk     = f'{vals["pk_only"]:.1f}%' if vals.get("pk_only") else "–"
        v_full   = f'{vals["full"]:.1f}%'    if vals.get("full")    else "–"
        v_delta  = f'{delta:+.1f}pp'         if delta is not None   else "–"
        lines.append(f"| {MODEL_SHORT[m]} | {v_csicl} | {v_pk} | {v_full} | {v_delta} |")
    lines.append("")

    # Table 3: Phase 2 CS ablation (train accuracy, ablation_size run)
    lines.append("## Table 3: Phase 2 CS Count Ablation — Train Accuracy\n")
    lines.append("_(ablation_size run; test eval for ablation_size2 pending)_\n")
    ABTASKS = ["geometric_shapes","formal_fallacies","snarks","disambiguation_qa"]
    ABCONDS = [("p1_unlimited","Unlimited"),("p2_1cs","Best-of-1"),("p2_3cs","Best-of-3")]
    lines.append("| Task | Unlimited | Best-of-1 | Best-of-3 | Best CS fix-rate (pool) |")
    lines.append("|------|-----------|-----------|-----------|------------------------|")
    for t in ABTASKS:
        row = []
        for cond, _ in ABCONDS:
            log = ROOT / f"runs/ablation_size/{cond}/logs/{t}.log"
            acc = None
            if log.exists():
                for line in reversed(log.read_text().splitlines()):
                    if "final train accuracy=" in line:
                        try: acc = float(line.split("=")[1].strip().rstrip("%"))
                        except: pass
                        break
            row.append(f"{acc:.1f}%" if acc else "–")
        # pool info for p2_1cs
        p1log = ROOT / f"runs/ablation_size/p2_1cs/logs/{t}.log"
        pool_size, best_fr = None, None
        if p1log.exists():
            for line in p1log.read_text().splitlines():
                if "Pool has" in line:
                    try: pool_size = int(line.split("Pool has")[1].split("CS")[0].strip())
                    except: pass
                if "KEEP" in line and "fix_rate=" in line:
                    try: best_fr = line.split("fix_rate=")[1].strip()
                    except: pass
                    break
        pool_info = f"pool={pool_size}, best={best_fr}" if pool_size else "–"
        lines.append(f"| {TASK_LABELS.get(t,t)} | {' | '.join(row)} | {pool_info} |")
    lines.append("")

    # Table 4: Phase 1 PK size ablation (train accuracy + actual PK sizes)
    lines.append("## Table 4: Phase 1 PK Char Limit Ablation — Train Accuracy + Final PK Size\n")
    lines.append("_(ablation_size run, sequential patching; ablation_size2 with Phase 0 cap in progress)_\n")
    P1CONDS = [("p1_3000chars","3K"),("p1_6000chars","6K"),("p1_12000chars","12K"),("p1_unlimited","Unlim")]
    lines.append("| Task | " + " | ".join(f"{l} acc / PK" for _,l in P1CONDS) + " |")
    lines.append("|------|" + "|".join(["------"]*len(P1CONDS)) + "|")
    for t in ABTASKS:
        row = []
        for cond, _ in P1CONDS:
            log = ROOT / f"runs/ablation_size/{cond}/logs/{t}.log"
            jf  = ROOT / f"runs/ablation_size/{cond}/{t}/cheatsheet_final.json"
            acc = None
            if log.exists():
                for line in reversed(log.read_text().splitlines()):
                    if "final train accuracy=" in line:
                        try: acc = float(line.split("=")[1].strip().rstrip("%"))
                        except: pass
                        break
            pk = None
            if jf.exists():
                d = json.load(open(jf))
                pk = len(d.get("prior_knowledge","")) // 1000
            row.append(f"{acc:.1f}% / {pk}K" if acc and pk else "–")
        lines.append(f"| {TASK_LABELS.get(t,t)} | " + " | ".join(row) + " |")
    lines.append("")

    # Table 5: EA Phase 1 vs Standard
    lines.append("## Table 5: EA Phase 1 vs Standard Phase 1 — Train Accuracy\n")
    lines.append("| Task | Std train acc | Std CS | EA train acc | EA CS | EA patches | Δ train |")
    lines.append("|------|--------------|--------|-------------|-------|------------|---------|")
    EA_TASKS = ["causal_judgement","geometric_shapes","snarks","disambiguation_qa"]
    for t in EA_TASKS:
        ea_log = ROOT / f"runs/bbh_ea_phase1/logs/{t}.log"
        std_log = ROOT / f"runs/ablation_size/p1_unlimited/logs/{t}.log"
        ea_acc=ea_patches=ea_cs=std_acc=std_cs = None
        if ea_log.exists():
            for line in ea_log.read_text().splitlines():
                if "final train accuracy=" in line:
                    try: ea_acc = float(line.split("=")[1].strip().rstrip("%"))
                    except: pass
                if "total_patches=" in line:
                    try: ea_patches = int(line.split("total_patches=")[1].split()[0])
                    except: pass
            jf = ROOT / f"runs/bbh_ea_phase1/{t}/cheatsheet_final.json"
            if jf.exists():
                ea_cs = len(json.load(open(jf)).get("case_studies",[]))
        if std_log.exists():
            for line in reversed(std_log.read_text().splitlines()):
                if "final train accuracy=" in line:
                    try: std_acc = float(line.split("=")[1].strip().rstrip("%"))
                    except: pass
                    break
            jf2 = ROOT / f"runs/ablation_size/p1_unlimited/{t}/cheatsheet_final.json"
            if jf2.exists():
                std_cs = len(json.load(open(jf2)).get("case_studies",[]))
        delta = (ea_acc - std_acc) if ea_acc and std_acc else None
        lines.append(
            f"| {TASK_LABELS.get(t,t)} "
            f"| {f'{std_acc:.1f}%' if std_acc else '–'} "
            f"| {std_cs if std_cs is not None else '–'} "
            f"| {f'{ea_acc:.1f}%' if ea_acc else '–'} "
            f"| {ea_cs if ea_cs is not None else '–'} "
            f"| {ea_patches if ea_patches is not None else '–'} "
            f"| {f'{delta:+.1f}pp' if delta is not None else '–'} |"
        )
    lines.append("")

    out = OUTDIR / "tables.md"
    out.write_text("\n".join(lines))
    print(f"  saved {out.name}")

# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures ...")
    fig1_phase_contribution()
    fig2_nontrain_transfer()
    fig3_phase2_ablation()
    fig4_pk_size_ablation()
    fig5_ea_vs_v3()
    make_tables()
    print(f"\nAll outputs in {OUTDIR}/")
