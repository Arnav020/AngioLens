"""Render report figures from models/metrics.json  ->  docs/figures/*.png

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
M = json.loads((ROOT / "models" / "metrics.json").read_text())
TARGETS = ["CAD", "LAD", "LCX", "RCA"]
NAME = {"CAD": "Overall CAD", "LAD": "LAD", "LCX": "LCX", "RCA": "RCA"}
# categorical identity (validated palette, light mode, fixed order) + emphasis colour for "ours"
SERIES = {"CAD": "#2a78d6", "LAD": "#eb6834", "LCX": "#1baf7a", "RCA": "#eda100"}
OURS = "#c2410c"
GRAY = "#b9bdc2"
INK = "#1d2329"
MUTED = "#6b7480"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": "#c9ced4", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#eceef1", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200, "savefig.bbox": "tight",
    "legend.frameon": False, "axes.axisbelow": True,
})


def roc_grid():
    fig, axs = plt.subplots(1, 4, figsize=(9.2, 2.45))
    models = list(M["models"])
    for ax, t in zip(axs, TARGETS):
        for k in models:
            if k == "angiolens":
                continue
            xs, ys = zip(*M["targets"][t]["curves"][k]["roc"])
            ax.plot(xs, ys, color=GRAY, lw=0.9)
        xs, ys = zip(*M["targets"][t]["curves"]["angiolens"]["roc"])
        auc = M["targets"][t]["summary"]["angiolens"]["roc_auc"]
        ax.plot(xs, ys, color=OURS, lw=1.8)
        ax.plot([0, 1], [0, 1], color="#d9dde2", lw=0.7)
        ax.set_title(f"{NAME[t]}  ·  AUC {auc['mean']:.3f}", fontsize=8.5, color=INK, loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel("False-positive rate")
        if t == "CAD":
            ax.set_ylabel("True-positive rate")
    axs[0].plot([], [], color=OURS, lw=1.8, label="AngioLens")
    axs[0].plot([], [], color=GRAY, lw=0.9, label="baselines")
    axs[0].legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "roc.png")
    plt.close(fig)


def calibration_and_dca():
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 2.9))
    for t in TARGETS:
        rel = M["targets"][t]["curves"]["angiolens"]["reliability"]
        a.plot([r["p_mean"] for r in rel], [r["y_rate"] for r in rel], "-o", color=SERIES[t], lw=1.3, ms=3.5,
               label=f"{t} (slope {M['targets'][t]['calibration']['slope']:.2f})")
    a.plot([0, 1], [0, 1], color="#c9ced4", lw=0.8)
    a.set_xlabel("Predicted probability")
    a.set_ylabel("Observed frequency")
    a.set_title("Calibration (out-of-fold)", loc="left", fontsize=8.5)
    a.legend(fontsize=7, loc="upper left")
    d = M["targets"]["CAD"]["curves"]["angiolens"]["dca"]
    th = [x["threshold"] for x in d]
    b.plot(th, [x["model"] for x in d], color=OURS, lw=1.8, label="AngioLens")
    b.plot(th, [x["treat_all"] for x in d], color="#2a78d6", lw=1.2, label="Refer all")
    b.plot(th, [0] * len(th), color=MUTED, lw=1.0, label="Refer none")
    b.set_ylim(-0.1, 0.75)
    b.set_xlabel("Threshold probability")
    b.set_ylabel("Net benefit")
    b.set_title("Decision-curve analysis, overall CAD", loc="left", fontsize=8.5)
    b.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "calibration_dca.png")
    plt.close(fig)


def importance():
    imp = M["importance"]
    fig, axs = plt.subplots(1, 4, figsize=(9.2, 2.6), gridspec_kw={"width_ratios": [1.25, 1, 1, 1]})
    for ax, (stage, n) in zip(axs, [("CAD", 10), ("LAD|CAD", 6), ("LCX|CAD", 6), ("RCA|CAD", 6)]):
        feats = imp[stage]["features"][:n][::-1]
        ax.barh([f["label"][:26] for f in feats], [f["mean_abs_shap"] for f in feats], color=OURS if stage == "CAD" else "#8a94a0",
                height=0.62)
        ax.set_title("P(CAD)" if stage == "CAD" else f"P({stage.split('|')[0]} | CAD)", loc="left", fontsize=8.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("mean |SHAP| (log-odds)", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "importance.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    roc_grid()
    calibration_and_dca()
    importance()
    print("figures ->", OUT)
