"""Method Ranking Heatmap — replace the area breakdown table."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

REPO = Path("/u/isaaccorley/github/geopool")
FIG_OUT = REPO / "paper-journal/figures/ranking_heatmap.pdf"

df = pd.read_csv(REPO / "results/area_breakdown.csv")
df["bin"] = df["bin"].str.strip()

# Clean bin order
BIN_ORDER = ["tiny\n(<50)", "small\n(50-150)", "medium\n(150-500)", "large\n(>500)", "overall"]
BIN_LABELS = ["Tiny\n< 50 px", "Small\n50–150 px", "Medium\n150–500 px", "Large\n> 500 px", "Overall"]

# Method display names
METHOD_NAMES = {
    "mean":                     "Mean",
    "std":                      "Std",
    "mean_std":                 "Mean+Std",
    "stats":                    "Stats",
    "max":                      "Max",
    "gem":                      "GeM",
    "signed_non_cancelling_gem":"Signed NC-GeM ★",
    "mean_max":                 "Mean+Max",
    "percentiles":              "Percentiles",
    "center_weighted_mean":     "Center-Weighted",
    "median_iqr":               "Median+IQR",
    "flattened_cov":            "Covariance",
    "pca_64":                   "PCA-64",
    "bovw_128":                 "BoVW-128",
}

# Compute avg rank across datasets for each (method, bin)
rows = []
for ds in ["aef", "olmoearth", "tessera"]:
    dds = df[df.dataset == ds].copy()
    for bin_name in BIN_ORDER:
        sub = dds[dds.bin == bin_name].copy()
        sub = sub.sort_values("f1", ascending=False).reset_index(drop=True)
        sub["rank"] = sub.index + 1
        rows.append(sub[["method", "rank"]].assign(bin=bin_name, dataset=ds))

rank_df = pd.concat(rows)
pivot = rank_df.groupby(["method", "bin"])["rank"].mean().unstack()[BIN_ORDER]
pivot = pivot.loc[pivot.index.isin(METHOD_NAMES)]
pivot = pivot.rename(index=METHOD_NAMES)

# Sort by overall rank
pivot = pivot.sort_values("overall")

n_methods = len(pivot)
n_bins = len(BIN_ORDER)

fig, ax = plt.subplots(figsize=(8, 5.5))

# Custom colormap: green (rank 1) → white → red (rank last)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "rank_cmap", ["#27AE60", "#F9F9F9", "#E74C3C"])

# Draw heatmap
data = pivot.values.astype(float)
vmin, vmax = 1, n_methods
im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

# Annotate cells
for i in range(n_methods):
    for j in range(n_bins):
        val = data[i, j]
        # White text on dark cells
        brightness = 1 - (val - vmin) / (vmax - vmin)
        color = "white" if brightness < 0.45 or brightness > 0.85 else "#333333"
        weight = "bold" if val <= 2 else "normal"
        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                fontsize=8.5, color=color, fontweight=weight)

# Axes labels
ax.set_xticks(range(n_bins))
ax.set_xticklabels(BIN_LABELS, fontsize=9)
ax.set_yticks(range(n_methods))
ax.set_yticklabels(pivot.index, fontsize=9)
ax.set_xlabel("Parcel size bin  →  PASTIS", fontsize=9.5, labelpad=8)
ax.set_title("", pad=10)  # no title — caption carries the narrative

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Avg rank", fontsize=8)
cbar.ax.invert_yaxis()

# Highlight Signed NC-GeM row
sgem_idx = list(pivot.index).index("Signed NC-GeM ★")
ax.add_patch(plt.Rectangle((-0.5, sgem_idx - 0.5), n_bins, 1,
             fill=False, edgecolor="#3498DB", lw=2.5, zorder=5))

# Divider between training-free and parametric (pca, bovw are last)
fitted_start = list(pivot.index).index("PCA-64") if "PCA-64" in pivot.index else None
if fitted_start is not None:
    ax.axhline(fitted_start - 0.5, color="#aaaaaa", lw=1.2, ls="--")
    ax.text(n_bins - 0.5, fitted_start - 0.55, "fitted →",
            fontsize=7, color="#aaaaaa", ha="right", va="bottom")

# Vertical divider before Overall
ax.axvline(3.5, color="#999999", lw=1.2, ls="--")

plt.tight_layout()
plt.savefig(FIG_OUT, bbox_inches="tight", dpi=200)
print(f"Saved: {FIG_OUT}")
