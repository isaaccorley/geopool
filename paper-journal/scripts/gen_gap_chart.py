"""Generalization gap bar chart — replaces random-vs-spatial scatter."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

REPO = Path("/u/isaaccorley/github/geopool")
FIG_OUT = REPO / "paper-journal/figures/gap_chart.pdf"

METHOD_NAMES = {
    "mean":                     "Mean",
    "std":                      "Std",
    "mean_std":                 "Mean+Std",
    "stats":                    "Stats",
    "max":                      "Max",
    "gem":                      "GeM",
    "mean_max":                 "Mean+Max",
    "percentiles":              "Percentiles",
    "center_weighted_mean":     "Center-Weighted",
    "median_iqr":               "Median+IQR",
    "flattened_cov":            "Covariance",
    "pca_64":                   "PCA-64",
    "bovw_128":                 "BoVW-128",
}

# Method family colors
FAMILY_COLOR = {
    "Mean": "#7f7f7f",
    "Std": "#aaaaaa",
    "Max": "#aaaaaa",
    "GeM": "#aaaaaa",
    "Mean+Std": "#2ECC71",
    "Mean+Max": "#27AE60",
    "Stats": "#E74C3C",
    "Percentiles": "#E67E22",
    "Center-Weighted": "#F39C12",
    "Median+IQR": "#E67E22",
    "Covariance": "#9B59B6",
    "PCA-64": "#BDC3C7",
    "BoVW-128": "#BDC3C7",
}

# Load and compute gaps (avg across 3 datasets)
dfs = []
for ds in ["aef", "olmoearth", "tessera"]:
    d = pd.read_csv(REPO / f"results/eurosat-{ds}/linear_results.csv")
    dfs.append(d)
df = pd.concat(dfs)

# Use k=5 accuracy
df = df[df.split_type.isin(["standard", "spatial"])]
piv = df.groupby(["pool", "split_type"])["accuracy"].mean().unstack()
piv["gap"] = piv["standard"] - piv["spatial"]
piv["spatial_acc"] = piv["spatial"]
piv = piv[piv.index.isin(METHOD_NAMES)]
piv.index = piv.index.map(METHOD_NAMES)
piv = piv.sort_values("gap", ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                gridspec_kw={"width_ratios": [1.6, 1]})

# ── Left: gap bar chart ────────────────────────────────────────────────────
methods = piv.index.tolist()
gaps = piv["gap"].values
colors = [FAMILY_COLOR.get(m, "#cccccc") for m in methods]

bars = ax1.barh(range(len(methods)), gaps * 100, color=colors,
                edgecolor="white", linewidth=0.5, height=0.7)

# Annotate values
for i, (bar, gap) in enumerate(zip(bars, gaps)):
    ax1.text(gap * 100 + 0.15, i, f"{gap*100:.1f}pp",
             va="center", ha="left", fontsize=7.5, color="#444444")

# Reference: mean gap
mean_gap = piv.loc["Mean", "gap"] * 100
ax1.axvline(mean_gap, color="#7f7f7f", lw=1.2, ls="--", alpha=0.7)
ax1.text(mean_gap + 0.1, len(methods) - 0.5, "Mean\nbaseline",
         fontsize=7, color="#7f7f7f", va="top")

ax1.set_yticks(range(len(methods)))
ax1.set_yticklabels(methods, fontsize=8.5)
ax1.set_xlabel("Random → Spatial accuracy drop (pp ↓ lower = more robust)", fontsize=9)
ax1.set_title("(a) Generalization gap", fontsize=10, fontweight="bold")
ax1.invert_yaxis()
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_xlim(0, gaps.max() * 100 * 1.18)
ax1.grid(axis="x", alpha=0.2, lw=0.6)

# Highlight Mean+Std (recommended)
if "Mean+Std" in methods:
    idx = methods.index("Mean+Std")
    ax1.get_yticklabels()[idx].set_color("#2ECC71")
    ax1.get_yticklabels()[idx].set_fontweight("bold")

# ── Right: spatial accuracy scatter (method vs spatial acc) ───────────────
ax2.scatter(piv["gap"] * 100, piv["spatial_acc"] * 100,
            c=[FAMILY_COLOR.get(m, "#cccccc") for m in piv.index],
            s=80, edgecolors="white", linewidths=0.8, zorder=3)

# Label key methods
label_methods = {"Mean", "Stats", "Covariance", "Mean+Std", "Center-Weighted"}
for method, row in piv.iterrows():
    if method in label_methods:
        offset = (0.15, 0.3)
        ax2.annotate(method, (row["gap"] * 100, row["spatial_acc"] * 100),
                     xytext=(row["gap"] * 100 + offset[0], row["spatial_acc"] * 100 + offset[1]),
                     fontsize=7.5, color=FAMILY_COLOR.get(method, "#555"),
                     arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.8))

ax2.set_xlabel("Generalization gap (pp)", fontsize=9)
ax2.set_ylabel("Spatial accuracy (%)", fontsize=9)
ax2.set_title("(b) Accuracy vs.\ gap", fontsize=10, fontweight="bold")
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(alpha=0.2, lw=0.6)

# Annotate quadrant
ax2_xlim = ax2.get_xlim()
ax2_ylim = ax2.get_ylim()
ax2.text(ax2_xlim[0] + 0.2, ax2_ylim[1] - 0.5,
         "← ideal\n(accurate + robust)", fontsize=7, color="#27AE60", va="top")

plt.tight_layout()
plt.savefig(FIG_OUT, bbox_inches="tight", dpi=200)
print(f"Saved: {FIG_OUT}")
