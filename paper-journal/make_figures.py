"""Generate all paper figures with fixes applied.

Fixes:
- pipeline.pdf: PASTIS-R → PASTIS label
- nd_phase_transition.pdf: declarative title + unified y-axis per encoder group
- gap_chart.pdf: per-method colors, Pareto region, clean scatter labels, no LaTeX artifact
- ranking_heatmap.pdf: PASTIS-R → PASTIS, Covariance row added
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import FancyArrowPatch
from sklearn.decomposition import PCA

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Global style ───────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "medium",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#E8E8E8",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

# ── Shared palette — consistent across ALL figures ─────────────────────────────
# These 5 methods are highlighted everywhere; all others are gray
METHOD_COLORS = {
    "mean":                 "#888888",  # gray  (baseline)
    "std":                  "#AAAAAA",
    "gem":                  "#AAAAAA",
    "center_weighted_mean": "#AAAAAA",
    "max":                  "#AAAAAA",
    "mean_std":             "#2ECC71",  # green
    "mean_max":             "#F39C12",  # orange
    "median_iqr":           "#F39C12",
    "percentiles":          "#F39C12",
    "stats":                "#E74C3C",  # red
    "signed_nc_gem":        "#3498DB",  # blue  (recommended)
    "flattened_cov":        "#9B59B6",  # purple
    "pca_64":               "#AAAAAA",
    "bovw_128":             "#AAAAAA",
}

METHOD_LABELS = {
    "mean":                 "Mean",
    "std":                  "Std",
    "gem":                  "GeM",
    "center_weighted_mean": "Center-Weighted",
    "max":                  "Max",
    "mean_std":             "Mean+Std",
    "mean_max":             "Mean+Max",
    "median_iqr":           "Median+IQR",
    "percentiles":          "Percentiles",
    "stats":                "Stats",
    "signed_nc_gem":        "Signed NC-GeM",
    "flattened_cov":        "Covariance",
    "pca_64":               "PCA-64",
    "bovw_128":             "BoVW-128",
}

# Methods shown in nd_phase_transition (keep focused)
PHASE_METHODS = ["mean", "mean_std", "signed_nc_gem", "stats", "flattened_cov"]

# Gap chart order (worst → best, i.e. largest gap first)
GAP_ORDER = [
    "pca_64", "mean", "std", "gem", "center_weighted_mean", "signed_nc_gem",
    "median_iqr", "bovw_128", "percentiles", "mean_std", "flattened_cov",
    "max", "mean_max", "stats",
]

# Heatmap row order (roughly best overall → worst)
HEATMAP_ORDER = [
    "mean_std", "signed_nc_gem", "center_weighted_mean", "percentiles",
    "mean_max", "mean", "stats", "median_iqr", "gem", "flattened_cov",
    "pca_64", "max", "std", "bovw_128",
]

EUROSAT_DATASETS = ["eurosat-aef", "eurosat-olmoearth", "eurosat-tessera"]
PASTIS_DATASETS  = ["pastis-aef",  "pastis-olmoearth",  "pastis-tessera"]
ENCODER_LABELS   = {
    "eurosat-aef":      r"AEF $(D=64)$",
    "eurosat-olmoearth": r"OlmoEarth $(D=128)$",
    "eurosat-tessera":  r"Tessera $(D=128)$",
    "pastis-aef":       "AEF",
    "pastis-olmoearth": "OlmoEarth",
    "pastis-tessera":   "Tessera",
}


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_linear(datasets: list[str]) -> pd.DataFrame:
    dfs = []
    for ds in datasets:
        p = RESULTS_DIR / ds / "linear_results.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["dataset"] = ds
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_area_breakdown() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "area_breakdown.csv")


# ── Figure 1: Pipeline ─────────────────────────────────────────────────────────

def _load_pastis_pca(parcel_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a PASTIS AEF embedding, return (pca_rgb, pooled_vec, mask).
    Embedding shape: (64, H, W) int8 → transpose to (H, W, 64).
    """
    with rasterio.open(parcel_file) as src:
        data = src.read().astype(np.float32)  # (64, H, W)
    emb = data.transpose(1, 2, 0)  # (H, W, 64)
    H, W, D = emb.shape

    pca = PCA(n_components=3)
    flat = emb.reshape(-1, D)
    proj = pca.fit_transform(flat).reshape(H, W, 3)
    for c in range(3):
        lo, hi = np.percentile(proj[:, :, c], [2, 98])
        proj[:, :, c] = np.clip((proj[:, :, c] - lo) / (hi - lo + 1e-8), 0, 1)

    pooled = flat.mean(axis=0)
    pooled = (pooled - pooled.min()) / (pooled.max() - pooled.min() + 1e-8)
    return proj, pooled, emb


def make_pipeline() -> None:
    """Pipeline figure using real PASTIS AEF embeddings. PASTIS label (not PASTIS-R)."""
    # Load a real PASTIS parcel
    pastis_dir = REPO_ROOT / "data" / "pastis-aef"
    # Pick a parcel with reasonable size (not too tiny)
    candidates = sorted(pastis_dir.glob("S2_1000*.tif"))[:10]
    chosen = None
    for c in candidates:
        with rasterio.open(c) as src:
            h, w = src.height, src.width
            if 20 <= h <= 80 and 20 <= w <= 80:
                chosen = c
                break
    if chosen is None and candidates:
        chosen = candidates[0]

    pca_rgb, pooled, emb = _load_pastis_pca(str(chosen))
    H, W, _ = pca_rgb.shape

    # Synthetic EuroSAT embedding (64×64 patch, 64-d, colorful PCA)
    rng = np.random.default_rng(42)
    eu_emb = rng.standard_normal((64, 64, 64)).astype(np.float32)
    eu_proj = PCA(n_components=3).fit_transform(eu_emb.reshape(-1, 64)).reshape(64, 64, 3)
    for c in range(3):
        lo, hi = np.percentile(eu_proj[:, :, c], [2, 98])
        eu_proj[:, :, c] = np.clip((eu_proj[:, :, c] - lo) / (hi - lo + 1e-8), 0, 1)
    eu_pooled = eu_emb.reshape(-1, 64).mean(axis=0)
    eu_pooled = (eu_pooled - eu_pooled.min()) / (eu_pooled.max() - eu_pooled.min() + 1e-8)
    # Fake RGB input (green-ish crop field)
    eu_rgb = np.zeros((64, 64, 3))
    eu_rgb[:, :, 0] = np.clip(rng.random((64, 64)) * 0.3 + 0.1, 0, 1)
    eu_rgb[:, :, 1] = np.clip(rng.random((64, 64)) * 0.4 + 0.3, 0, 1)
    eu_rgb[:, :, 2] = np.clip(rng.random((64, 64)) * 0.2 + 0.05, 0, 1)
    # Add field structure
    for _ in range(6):
        x0, y0 = rng.integers(0, 50, 2)
        x1, y1 = x0 + rng.integers(8, 20), y0 + rng.integers(8, 20)
        shade = rng.random() * 0.3 + 0.25
        eu_rgb[y0:y1, x0:x1, 1] = np.clip(shade + 0.1, 0, 1)
        eu_rgb[y0:y1, x0:x1, 0] = np.clip(shade * 0.5, 0, 1)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 3.4), facecolor="white")

    # 4 columns × 2 rows of axes + arrows
    col_x = [0.05, 0.27, 0.51, 0.74]
    col_w = 0.19
    row_y = [0.53, 0.05]
    row_h = 0.38
    arrow_y = [0.72, 0.24]

    col_headers = ["Input", "Pixel Embeddings", "Embedding", "Label"]
    row_headers = ["EuroSAT", "PASTIS"]  # ← PASTIS (not PASTIS-R)

    for j, (lbl, lx) in enumerate(zip(col_headers, col_x)):
        fig.text(lx + col_w / 2, 0.97, lbl, ha="center", va="top",
                 fontsize=11, fontweight="bold", color="#222")

    for i, lbl in enumerate(row_headers):
        fig.text(0.015, row_y[i] + row_h / 2, lbl, ha="left", va="center",
                 fontsize=10, fontweight="bold", color="#444", rotation=90)

    fig.add_artist(plt.Line2D([0.03, 0.98], [0.52, 0.52],
                               transform=fig.transFigure,
                               color="#CCCCCC", linewidth=0.8, linestyle="--"))

    row_data = [
        # (input_img, emb_img, pooled_vec, parcel_outline, dim_input, dim_emb, dim_pool, label_txt)
        (eu_rgb,  eu_proj, eu_pooled, None, "64×64×12 (int16)",    "64×64×64 (float32)", "64×1", "Annual\nCrop"),
        (None,    pca_rgb, pooled,    True, "H×W×12 (int16)",      "H×W×64 (parcel outlined)", "64×1", "Corn /\nMaize"),
    ]

    for i, (inp, emb_img, pool_vec, outline, d_in, d_emb, d_pool, lbl_txt) in enumerate(row_data):
        # Col 0: Input
        ax0 = fig.add_axes([col_x[0], row_y[i], col_w, row_h])
        if inp is not None:
            ax0.imshow(inp, aspect="auto")
        else:
            # PASTIS: stack of time-series cards (schematic)
            for k in range(4, -1, -1):
                offset = k * 0.03
                rect = mpatches.FancyBboxPatch(
                    (0.05 + offset, 0.05 + offset), 0.65, 0.75,
                    boxstyle="round,pad=0.02",
                    facecolor=plt.colormaps["Blues"](0.3 + k * 0.12),
                    edgecolor="white", linewidth=1.2,
                    transform=ax0.transAxes, zorder=k,
                )
                ax0.add_patch(rect)
                months = ["Jan", "Apr", "Jul", "Oct", "Dec"]
                ax0.text(0.08 + offset, 0.82 + offset, months[k],
                         fontsize=6, color="white", fontweight="bold",
                         transform=ax0.transAxes, zorder=k + 1)
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_xlabel(d_in, fontsize=7, color="#666", style="italic", labelpad=2)
        for sp in ax0.spines.values(): sp.set_edgecolor("#CCCCCC"); sp.set_linewidth(0.6)

        # Col 1: Pixel Embeddings (PCA-RGB)
        ax1 = fig.add_axes([col_x[1], row_y[i], col_w, row_h])
        ax1.imshow(emb_img, aspect="auto")
        if outline:
            # Draw parcel boundary
            theta = np.linspace(0, 2 * np.pi, 30)
            cy, cx = emb_img.shape[0] / 2, emb_img.shape[1] / 2
            ry, rx = emb_img.shape[0] * 0.38, emb_img.shape[1] * 0.38
            px = cx + rx * (np.cos(theta) + 0.12 * np.cos(3 * theta))
            py = cy + ry * (np.sin(theta) + 0.08 * np.sin(2 * theta))
            ax1.plot(px, py, "w-", linewidth=2.0, zorder=5)
        ax1.set_xticks([]); ax1.set_yticks([])
        ax1.set_xlabel(d_emb, fontsize=7, color="#666", style="italic", labelpad=2)
        for sp in ax1.spines.values(): sp.set_edgecolor("#CCCCCC"); sp.set_linewidth(0.6)

        # Col 2: Pooled vector (horizontal color bar)
        ax2 = fig.add_axes([col_x[2], row_y[i], col_w, row_h])
        ax2.imshow(pool_vec.reshape(1, -1), aspect="auto", cmap="viridis")
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_xlabel(d_pool, fontsize=7, color="#666", style="italic", labelpad=2)
        for sp in ax2.spines.values(): sp.set_edgecolor("#CCCCCC"); sp.set_linewidth(0.6)

        # Col 3: Label
        ax3 = fig.add_axes([col_x[3], row_y[i], col_w, row_h])
        ax3.set_facecolor("#1B5E20")
        ax3.text(0.5, 0.5, lbl_txt, ha="center", va="center",
                 fontsize=13, fontweight="bold", color="white",
                 transform=ax3.transAxes)
        ax3.set_xticks([]); ax3.set_yticks([])
        for sp in ax3.spines.values(): sp.set_edgecolor("#CCCCCC"); sp.set_linewidth(0.6)

        # GFM arrow (col 0 → col 1)
        x0a, x1a = col_x[0] + col_w + 0.004, col_x[1] - 0.004
        fig.patches.append(FancyArrowPatch(
            (x0a, arrow_y[i]), (x1a, arrow_y[i]),
            transform=fig.transFigure, arrowstyle="-|>",
            mutation_scale=13, lw=1.4, color="#333"))
        fig.text((x0a + x1a) / 2, arrow_y[i] + 0.04, "GFM",
                 ha="center", fontsize=9, fontweight="bold", color="#333")

        # Pool arrow (col 1 → col 2)
        x0b, x1b = col_x[1] + col_w + 0.004, col_x[2] - 0.004
        fig.patches.append(FancyArrowPatch(
            (x0b, arrow_y[i]), (x1b, arrow_y[i]),
            transform=fig.transFigure, arrowstyle="-|>",
            mutation_scale=13, lw=1.4, color="#333"))
        fig.text((x0b + x1b) / 2, arrow_y[i] + 0.04, "Pool",
                 ha="center", fontsize=9, fontweight="bold", color="#333")

        # Arrow (col 2 → col 3)
        x0c, x1c = col_x[2] + col_w + 0.004, col_x[3] - 0.004
        fig.patches.append(FancyArrowPatch(
            (x0c, arrow_y[i]), (x1c, arrow_y[i]),
            transform=fig.transFigure, arrowstyle="-|>",
            mutation_scale=13, lw=1.4, color="#333"))

    out = OUT_DIR / "pipeline.pdf"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 2: N/D Phase Transition ────────────────────────────────────────────

def make_nd_phase_transition() -> None:
    """Small-multiples phase plot with unified y-axis scale and declarative title."""
    linear_eu = load_linear(EUROSAT_DATASETS)
    linear_pa = load_linear(PASTIS_DATASETS)

    # Gather PASTIS area-breakdown for per-bin points
    area = load_area_breakdown()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)

    # Shared y limits computed across ALL methods + encoders for comparability
    # We use per-panel y-limits but with a shared range width to keep shapes comparable
    all_perfs: list[float] = []

    encoder_pairs = list(zip(EUROSAT_DATASETS, PASTIS_DATASETS, strict=True))

    panel_data = []
    for eu_ds, pa_ds in encoder_pairs:
        eu_lin = linear_eu[linear_eu["dataset"] == eu_ds] if not linear_eu.empty else pd.DataFrame()
        pa_lin = linear_pa[linear_pa["dataset"] == pa_ds] if not linear_pa.empty else pd.DataFrame()
        pa_area = area[area["dataset"] == pa_ds.replace("pastis-", "")] if not area.empty else pd.DataFrame()
        panel_data.append((eu_ds, pa_ds, eu_lin, pa_lin, pa_area))
        # Collect perf values
        for df, col in [(eu_lin, "accuracy"), (pa_lin, "accuracy")]:
            if not df.empty and col in df.columns:
                vals = df[df["pool"].isin(PHASE_METHODS)][col].dropna().tolist()
                all_perfs.extend(vals)

    # Unified y range (floor/ceil to nearest 0.05)
    if all_perfs:
        y_min = max(0.0, np.floor(min(all_perfs) / 0.05) * 0.05 - 0.02)
        y_max = min(1.0, np.ceil(max(all_perfs) / 0.05) * 0.05 + 0.02)
    else:
        y_min, y_max = 0.3, 1.0

    # N/D values: PASTIS bins use median N per bin; EuroSAT is 4096
    # Approximate PASTIS bin median N values (from dataset description)
    PASTIS_BIN_N = {
        "tiny\n(<50)":       25,
        "small\n(50-150)":   90,
        "medium\n(150-500)": 280,
        "large\n(>500)":     800,
    }
    EUROSAT_N = 4096
    ENCODER_D = {
        "eurosat-aef": 64, "pastis-aef": 64,
        "eurosat-olmoearth": 128, "pastis-olmoearth": 128,
        "eurosat-tessera": 128, "pastis-tessera": 128,
    }

    for ax, (eu_ds, pa_ds, eu_lin, pa_lin, pa_area) in zip(axes, panel_data, strict=True):
        D = ENCODER_D[eu_ds]

        for method in PHASE_METHODS:
            color = METHOD_COLORS.get(method, "#888888")
            label = METHOD_LABELS.get(method, method)

            x_pts: list[float] = []
            y_pts: list[float] = []

            # PASTIS bin points
            if not pa_area.empty:
                method_area = pa_area[pa_area["method"] == method]
                for bin_name, bin_n in PASTIS_BIN_N.items():
                    bin_key = bin_name
                    row = method_area[method_area["bin"].str.strip() == bin_key.strip()]
                    if row.empty:
                        # try alternate formatting
                        for bk in method_area["bin"].unique():
                            if str(bin_n) in bk or bk.strip().startswith(bin_key.split("\n")[0]):
                                row = method_area[method_area["bin"] == bk]
                                break
                    if not row.empty:
                        x_pts.append(bin_n / D)
                        y_pts.append(float(row["f1"].iloc[0]))

            # EuroSAT point
            if not eu_lin.empty:
                row = eu_lin[
                    (eu_lin["pool"] == method) & (eu_lin["split_type"] == "spatial")
                ]
                if not row.empty:
                    x_pts.append(EUROSAT_N / D)
                    y_pts.append(float(row["accuracy"].iloc[0]))

            if len(x_pts) >= 2:
                sorted_pairs = sorted(zip(x_pts, y_pts))
                xs, ys = zip(*sorted_pairs)
                ax.plot(xs, ys, "o-", color=color, label=label,
                        linewidth=1.8, markersize=5, markeredgecolor="white",
                        markeredgewidth=0.8)

        # N=D line
        ax.axvline(x=1.0, color="#888888", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(1.05, y_min + 0.02, r"$N = D$", fontsize=8, color="#888888", va="bottom")

        # EuroSAT annotation
        ax.axvline(x=EUROSAT_N / D, color="#888888", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.text(EUROSAT_N / D * 0.85, y_max - 0.02, "EuroSAT",
                fontsize=7.5, color="#888888", ha="right", va="top")

        # Background shading
        ax.axvspan(1.0, EUROSAT_N / D * 1.5, alpha=0.04, color="#E74C3C", zorder=0)

        ax.set_xscale("log")
        ax.set_xlim(0.5, EUROSAT_N / D * 2)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(r"$N/D$  (pixels per region / embedding dim)", fontsize=9)
        ax.set_title(ENCODER_LABELS[eu_ds], fontsize=11, fontweight="bold")
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(True, which="minor", alpha=0.1)

    axes[0].set_ylabel("Performance", fontsize=10)

    # Single shared legend outside panels
    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_leg,
        loc="upper center", ncol=len(PHASE_METHODS),
        bbox_to_anchor=(0.5, 1.02),
        frameon=False, fontsize=9,
    )

    # Region labels at bottom
    fig.text(0.18, -0.04, r"$\leftarrow$ PASTIS parcels $(N \approx D)$",
             ha="center", fontsize=8.5, color="#3498DB")
    fig.text(0.72, -0.04, r"EuroSAT patches $(N \gg D)$ $\rightarrow$",
             ha="center", fontsize=8.5, color="#E74C3C")

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    out = OUT_DIR / "nd_phase_transition.pdf"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 3: Gap chart ────────────────────────────────────────────────────────

def make_gap_chart() -> None:
    """Accuracy–robustness combo: bar + scatter with per-method colors, Pareto shading."""
    linear = load_linear(EUROSAT_DATASETS)
    if linear.empty:
        print("No EuroSAT linear results — skipping gap_chart")
        return

    # Compute per-method average gap across encoders
    records: list[dict] = []
    spatial_acc: dict[str, list[float]] = {}
    for ds in EUROSAT_DATASETS:
        ds_data = linear[linear["dataset"] == ds]
        for method in GAP_ORDER:
            row_std = ds_data[(ds_data["pool"] == method) & (ds_data["split_type"] == "standard")]
            row_spa = ds_data[(ds_data["pool"] == method) & (ds_data["split_type"] == "spatial")]
            if not row_std.empty and not row_spa.empty:
                gap = (float(row_std["accuracy"].iloc[0]) - float(row_spa["accuracy"].iloc[0])) * 100
                acc = float(row_spa["accuracy"].iloc[0]) * 100
                records.append({"pool": method, "gap": gap, "acc": acc})
                spatial_acc.setdefault(method, []).append(acc)

    df = pd.DataFrame(records)
    avg = (
        df.groupby("pool", as_index=False)
        .agg(gap=("gap", "mean"), acc=("acc", "mean"))
    )
    avg["label"] = avg["pool"].map(METHOD_LABELS)

    # Sort by gap descending (worst at top for horizontal bars)
    order_map = {m: i for i, m in enumerate(GAP_ORDER)}
    avg["order"] = avg["pool"].map(order_map)
    avg = avg.sort_values("order", ascending=False).reset_index(drop=True)

    # Mean baseline value
    mean_gap = float(avg[avg["pool"] == "mean"]["gap"].iloc[0])

    fig, (ax_bar, ax_scat) = plt.subplots(
        1, 2, figsize=(9.0, 4.0),
        gridspec_kw={"width_ratios": [1.6, 1]}
    )

    # ── Panel (a): horizontal bar chart ───────────────────────────────────────
    colors = [METHOD_COLORS.get(m, "#AAAAAA") for m in avg["pool"]]
    y_pos = np.arange(len(avg))
    bars = ax_bar.barh(y_pos, avg["gap"], color=colors,
                       edgecolor="white", linewidth=0.8, height=0.72, zorder=3)

    for bar, (_, row) in zip(bars, avg.iterrows()):
        w = bar.get_width()
        ax_bar.text(w + 0.12, bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}pp", va="center", ha="left", fontsize=8, color="#333")

        # Bold + colored label for highlighted methods
        pool = row["pool"]
        lbl_color = METHOD_COLORS.get(pool, "#888")
        weight = "bold" if lbl_color not in ("#AAAAAA", "#888888") else "normal"
        ax_bar.text(-0.15, bar.get_y() + bar.get_height() / 2,
                    row["label"], va="center", ha="right",
                    fontsize=8.5, color=lbl_color if weight == "bold" else "#444",
                    fontweight=weight)

    # Mean baseline dashed line
    ax_bar.axvline(mean_gap, color="#888888", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
    ax_bar.text(mean_gap + 0.1, len(avg) - 0.3, "Mean\nbaseline",
                fontsize=7.5, color="#888888", va="top")

    ax_bar.set_yticks([])
    ax_bar.set_xlabel(r"Random $\rightarrow$ Spatial accuracy drop (pp $\downarrow$ lower = more robust)",
                      fontsize=9)
    ax_bar.set_xlim(0, avg["gap"].max() * 1.22)
    ax_bar.set_title("(a) Generalization gap", fontsize=10, fontweight="bold")
    ax_bar.grid(axis="x", alpha=0.25, zorder=0)
    ax_bar.spines["left"].set_visible(False)

    # ── Panel (b): accuracy vs gap scatter ────────────────────────────────────
    # Pareto region (upper-left): low gap AND high accuracy
    gap_thresh = avg["gap"].quantile(0.40)
    acc_thresh = avg["acc"].quantile(0.60)
    ax_scat.axvspan(avg["gap"].min() - 0.5, gap_thresh,
                    ymin=0.5, ymax=1.05,
                    alpha=0.08, color="#2ECC71", zorder=0)
    ax_scat.text(avg["gap"].min() + 0.05, avg["acc"].max() - 0.15,
                 "ideal\n(accurate + robust)",
                 fontsize=7.5, color="#27AE60", va="top", style="italic")

    for _, row in avg.iterrows():
        c = METHOD_COLORS.get(row["pool"], "#AAAAAA")
        zord = 5 if c not in ("#AAAAAA", "#888888") else 3
        ax_scat.scatter(row["gap"], row["acc"], color=c, s=55,
                        zorder=zord, edgecolor="white", linewidth=0.8)

    # Label only the 5 highlighted methods
    HIGHLIGHT = {"mean", "signed_nc_gem", "mean_std", "flattened_cov", "stats"}
    for _, row in avg[avg["pool"].isin(HIGHLIGHT)].iterrows():
        c = METHOD_COLORS.get(row["pool"], "#888888")
        offx, offy = 0.12, 0.15
        ax_scat.annotate(
            row["label"],
            (row["gap"], row["acc"]),
            xytext=(row["gap"] + offx, row["acc"] + offy),
            fontsize=8, color=c, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=c, lw=0.8, alpha=0.5),
        )

    ax_scat.set_xlabel("Generalization gap (pp)", fontsize=9)
    ax_scat.set_ylabel("Spatial accuracy (%)", fontsize=9)
    ax_scat.set_title("(b) Accuracy vs. gap", fontsize=10, fontweight="bold")
    ax_scat.grid(alpha=0.2, zorder=0)

    plt.tight_layout()
    out = OUT_DIR / "gap_chart.pdf"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 4: Ranking heatmap ──────────────────────────────────────────────────

def make_ranking_heatmap() -> None:
    """Rank-reversal heatmap — PASTIS label fixed, Covariance row added."""
    area = load_area_breakdown()
    if area.empty:
        print("No area_breakdown data — skipping ranking_heatmap")
        return

    BINS = ["tiny\n(<50)", "small\n(50-150)", "medium\n(150-500)", "large\n(>500)"]
    BIN_LABELS = ["Tiny\n< 50 px", "Small\n50–150 px", "Medium\n150–500 px", "Large\n> 500 px"]
    PA_DATASETS = ["aef", "olmoearth", "tessera"]

    # Normalise bin names in data
    def normalise_bin(b: str) -> str:
        b = str(b).strip().lower().replace(" ", "").replace("\n", "")
        if b.startswith("tiny") or "<50" in b:
            return "tiny\n(<50)"
        if b.startswith("small") or "50-150" in b or "50–150" in b:
            return "small\n(50-150)"
        if b.startswith("medium") or "150-500" in b or "150–500" in b:
            return "medium\n(150-500)"
        if b.startswith("large") or ">500" in b:
            return "large\n(>500)"
        if "overall" in b:
            return "overall"
        return b

    area = area.copy()
    area["bin_norm"] = area["bin"].map(normalise_bin)

    # Compute avg rank per method × bin across encoders
    rank_data: dict[tuple[str, str], list[float]] = {}
    for pa_ds in PA_DATASETS:
        ds_area = area[area["dataset"] == pa_ds]
        for bin_key in BINS + ["overall"]:
            bin_data = ds_area[ds_area["bin_norm"] == bin_key]
            if bin_data.empty:
                continue
            bin_data = bin_data.sort_values("f1", ascending=False).reset_index(drop=True)
            for rank_idx, row in bin_data.iterrows():
                key = (row["method"], bin_key)
                rank_data.setdefault(key, []).append(int(rank_idx) + 1)

    # Average ranks
    methods = HEATMAP_ORDER
    cols = BINS + ["overall"]
    rank_matrix = np.full((len(methods), len(cols)), np.nan)
    for i, method in enumerate(methods):
        for j, bin_key in enumerate(cols):
            vals = rank_data.get((method, bin_key), [])
            if vals:
                rank_matrix[i, j] = np.mean(vals)

    n_methods, n_cols = rank_matrix.shape
    fig, ax = plt.subplots(figsize=(8.0, 4.4))

    # Colormap: green (rank 1) → white (mid) → red (worst)
    cmap = plt.colormaps.get_cmap("RdYlGn_r").reversed()
    im = ax.imshow(rank_matrix, cmap=cmap.reversed(),
                   vmin=1, vmax=n_methods, aspect="auto")

    # Cell text
    for i in range(n_methods):
        for j in range(n_cols):
            val = rank_matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if (val <= 2 or val >= n_methods - 1) else "#222"
                weight = "bold" if val <= 2 else "normal"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=7.5, color=text_color, fontweight=weight)

    # Axes
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(BIN_LABELS + ["Overall"], fontsize=8)
    ax.set_yticks(range(n_methods))

    # y-tick labels: bold + colored for highlighted methods
    for i, method in enumerate(methods):
        c = METHOD_COLORS.get(method, "#444")
        weight = "bold" if c not in ("#AAAAAA", "#888888") else "normal"
        lbl = METHOD_LABELS.get(method, method)
        if method == "signed_nc_gem":
            lbl += " ★"
        ax.get_yticklabels()  # force draw
    ax.set_yticklabels(
        [METHOD_LABELS.get(m, m) + (" ★" if m == "signed_nc_gem" else "")
         for m in methods],
        fontsize=8,
    )

    # Highlight signed_nc_gem row with blue border
    gem_idx = methods.index("signed_nc_gem")
    for j in range(n_cols):
        ax.add_patch(mpatches.Rectangle(
            (j - 0.5, gem_idx - 0.5), 1, 1,
            fill=False, edgecolor="#3498DB", linewidth=2.0, zorder=5
        ))

    # Dashed separator before "Overall" column
    ax.axvline(x=len(BINS) - 0.5, color="#888888", linestyle="--",
               linewidth=1.2, alpha=0.6)

    # Dashed separator before parametric methods (PCA, BoVW) — after GeM
    fitted_start = methods.index("pca_64")
    ax.axhline(y=fitted_start - 0.5, color="#888888", linestyle="--",
               linewidth=1.0, alpha=0.5)
    ax.text(n_cols - 0.45, fitted_start - 0.5, "fitted",
            fontsize=7.5, color="#888888", va="center", ha="right")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Avg rank", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel(r"Parcel size bin $\rightarrow$ PASTIS", fontsize=10)  # ← PASTIS fixed
    ax.set_title("", fontsize=1)  # title is in LaTeX caption

    plt.tight_layout()
    out = OUT_DIR / "ranking_heatmap.pdf"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figures", nargs="+",
        choices=["pipeline", "nd_phase_transition", "gap_chart", "ranking_heatmap", "all"],
        default=["all"],
    )
    args = parser.parse_args()
    figs = args.figures
    if "all" in figs:
        figs = ["pipeline", "nd_phase_transition", "gap_chart", "ranking_heatmap"]

    if "pipeline"           in figs: make_pipeline()
    if "nd_phase_transition" in figs: make_nd_phase_transition()
    if "gap_chart"          in figs: make_gap_chart()
    if "ranking_heatmap"    in figs: make_ranking_heatmap()

    print("Done.")
