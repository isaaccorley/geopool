"""N/D Phase Transition Plot — the hero insight figure."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

REPO = Path("/u/isaaccorley/github/geopool")
FIG_OUT = REPO / "paper-journal/figures/nd_phase_transition.pdf"

# ── Load per-bin PASTIS data ────────────────────────────────────────────────
df = pd.read_csv(REPO / "results/area_breakdown.csv")
df["bin"] = df["bin"].str.strip()

# Bin midpoint N values (representative pixel counts)
BIN_N = {
    "tiny\n(<50)": 25,
    "small\n(50-150)": 90,
    "medium\n(150-500)": 275,
    "large\n(>500)": 800,
}

# For EuroSAT we load linear spatial results per dataset
def load_eurosat_spatial(ds):
    path = REPO / f"results/eurosat-{ds}/linear_results.csv"
    d = pd.read_csv(path)
    return d[d.split_type == "spatial"].set_index("pool")["accuracy"]

eurosat = {ds: load_eurosat_spatial(ds) for ds in ["aef", "olmoearth", "tessera"]}
EUROSAT_N = 4096
D = {"aef": 64, "olmoearth": 128, "tessera": 128}

# Key methods to plot (pick 4-5 with diverging stories)
METHODS = {
    "mean":                    {"label": "Mean",            "color": "#7f7f7f", "ls": "--",  "lw": 1.8},
    "mean_std":                {"label": "Mean+Std",        "color": "#2ECC71", "ls": "-.",  "lw": 2.0},
    "signed_non_cancelling_gem":{"label": "Signed NC-GeM",  "color": "#3498DB", "ls": "-",   "lw": 2.8},
    "stats":                   {"label": "Stats",           "color": "#E74C3C", "ls": "-",   "lw": 2.0},
    "flattened_cov":           {"label": "Covariance",      "color": "#9B59B6", "ls": ":",   "lw": 2.2},
}

# Average across datasets, compute N/D for each point
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, sharex=True)
YLIM = (0.1, 1.0)
YTICKS = [0.2, 0.4, 0.6, 0.8, 1.0]
# Match rightmost (Tessera, D=128) x-range: 25/128 ≈ 0.2 to 4096/128 = 32
XLIM = (0.15, 40.0)
XTICKS = [0.1, 1.0, 10.0]
datasets = [("aef", "AEF  ($D{=}64$)"), ("olmoearth", "OlmoEarth  ($D{=}128$)"), ("tessera", "Tessera  ($D{=}128$)")]

for ax, (ds, ds_label) in zip(axes, datasets):
    d_dim = D[ds]
    df_ds = df[df.dataset == ds]

    for method, style in METHODS.items():
        xs, ys = [], []

        # PASTIS bins
        for bin_name, n_mid in BIN_N.items():
            row = df_ds[(df_ds.method == method) & (df_ds.bin == bin_name)]
            if row.empty:
                continue
            xs.append(n_mid / d_dim)
            ys.append(row.f1.values[0])

        # EuroSAT point
        if method in eurosat[ds].index:
            xs.append(EUROSAT_N / d_dim)
            ys.append(eurosat[ds][method])

        if not xs:
            continue

        xs, ys = zip(*sorted(zip(xs, ys)))
        ax.plot(xs, ys,
                color=style["color"], ls=style["ls"], lw=style["lw"],
                marker="o", markersize=5, markerfacecolor="white",
                markeredgecolor=style["color"], markeredgewidth=1.5,
                label=style["label"], zorder=3)

    ax.set_ylim(*YLIM)
    ax.set_yticks(YTICKS)

    # N/D=1 reference line
    ax.axvline(1.0, color="#cccccc", lw=1.2, ls="--", zorder=1)
    ax.text(1.05, YLIM[0], "$N{=}D$", fontsize=7.5, color="#999999", va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([str(t) for t in XTICKS])
    ax.set_xlabel("$N / D$  (pixels per region / embedding dim)", fontsize=9)
    ax.set_ylabel("Performance", fontsize=9)
    ax.set_title(ds_label, fontsize=10, fontweight="bold", pad=6)  # panel label kept
    ax.grid(True, which="both", alpha=0.2, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    # Label EuroSAT landmark
    ax.axvline(EUROSAT_N / d_dim, color="#eeeeee", lw=1.0, ls=":", zorder=1)
    ax.text(EUROSAT_N / d_dim * 0.88, YLIM[0],
            "EuroSAT", fontsize=6.5, color="#bbbbbb", va="bottom", ha="right")

    # PASTIS region shading
    ax.axvspan(0, 4, alpha=0.04, color="#3498DB")
    ax.axvspan(4, EUROSAT_N / d_dim * 2, alpha=0.04, color="#E74C3C")

# Legend on first axis
handles = [mpatches.Patch(color=v["color"], label=v["label"]) for v in METHODS.values()]
axes[0].legend(handles=handles, fontsize=8, framealpha=0.9, loc="upper left",
               handlelength=1.6, borderpad=0.6)

# Shared annotations
fig.text(0.25, 0.01, "← PASTIS parcels ($N \\approx D$)", ha="center", fontsize=8, color="#3498DB")
fig.text(0.75, 0.01, "EuroSAT patches ($N \\gg D$) →", ha="center", fontsize=8, color="#E74C3C")

plt.tight_layout()
plt.savefig(FIG_OUT, bbox_inches="tight", dpi=200)
print(f"Saved: {FIG_OUT}")
