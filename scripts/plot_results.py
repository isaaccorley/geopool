"""Generate figures for paper with multi-model support."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer

from geopool.pool import get_output_dim

# Modern plot style (inspired by high-quality ML papers)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "medium",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": "#E5E5E5",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.7,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)

# Method display names and order
METHOD_ORDER = [
    "mean",
    "std",
    "mean_std",
    "stats",
    "max",
    "gem",
    "mean_max",
    "percentiles",
    "flattened_cov",
    "center_weighted_mean",
    "median_iqr",
    "pca_64",
    "bovw_128",
]
METHOD_LABELS = {
    "mean": "Mean",
    "std": "Std",
    "mean_std": "Mean+Std",
    "stats": "Stats",
    "max": "Max",
    "gem": "GeM",
    "mean_max": "Mean+Max",
    "percentiles": "Percentiles",
    "flattened_cov": "Covariance",
    "center_weighted_mean": "Center-Weighted",
    "median_iqr": "Median+IQR",
    "pca_64": "PCA",
    "bovw_128": "BoVW",
}

# Dataset display names (full names, no abbreviations)
DATASET_LABELS = {
    "aef": "AEF",
    "olmoearth-nano": "OlmoEarth",
    "tessera": "Tessera",
}


def build_method_dim(input_dim: int) -> dict[str, int]:
    """Build method dimensionalities based on input embedding dimension."""
    return {method: get_output_dim(method, input_dim=input_dim) for method in METHOD_ORDER}


# Modern color palette (vibrant yet professional)
PALETTE = sns.color_palette(
    [
        "#4C9A7E",  # Teal green
        "#2E7D5C",  # Deep green
        "#1F5C42",  # Dark green
        "#154433",  # Very dark green
    ]
)

# Split colors (high contrast)
COLORS = {
    "standard": "#3498DB",  # Bright blue
    "spatial": "#E74C3C",  # Coral red
}

# Dataset colors for multi-model plots (vibrant, distinguishable)
DATASET_COLORS = {
    "aef": "#2ECC71",  # Emerald green
    "olmoearth-nano": "#3498DB",  # Bright blue
    "tessera": "#F39C12",  # Orange
}

# Dataset markers
DATASET_MARKERS = {
    "aef": "o",
    "olmoearth-nano": "s",
    "tessera": "D",
}


def load_all_results(results_dir: Path, datasets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load KNN and linear results for all datasets."""
    knn_dfs = []
    linear_dfs = []
    for dataset in datasets:
        dataset_dir = results_dir / dataset
        knn_path = dataset_dir / "knn_results.csv"
        linear_path = dataset_dir / "linear_results.csv"
        if knn_path.exists():
            df = pd.read_csv(knn_path)
            df["dataset"] = dataset
            knn_dfs.append(df)
        if linear_path.exists():
            df = pd.read_csv(linear_path)
            df["dataset"] = dataset
            linear_dfs.append(df)
    knn = pd.concat(knn_dfs, ignore_index=True) if knn_dfs else pd.DataFrame()
    linear = pd.concat(linear_dfs, ignore_index=True) if linear_dfs else pd.DataFrame()
    return knn, linear


def plot_gap_analysis(linear: pd.DataFrame, paper_dir: Path, datasets: list[str]) -> None:
    """Show average random-spatial gap per method across datasets with modern styling."""
    # Filter to methods of interest
    pools = cast("pd.Series", linear["pool"])
    data = linear[pools.isin(METHOD_ORDER)].copy()

    # Compute gap per dataset/method, then average across datasets
    gaps: list[dict[str, float | str]] = []
    for dataset in datasets:
        ds_data = data[data["dataset"] == dataset]
        pivoted = ds_data.pivot(index="pool", columns="split_type", values="accuracy")
        pivoted = pivoted.reindex(METHOD_ORDER).dropna()
        for method in pivoted.index:
            gap = (pivoted.loc[method, "standard"] - pivoted.loc[method, "spatial"]) * 100
            gaps.append(
                {
                    "pool": method,
                    "gap": float(gap),
                }
            )

    gap_df = pd.DataFrame(gaps)
    avg_gap = gap_df.groupby("pool", as_index=False)["gap"].mean()
    # Sort descending so lowest gap is at top (best)
    avg_gap = avg_gap.sort_values("gap", ascending=False)
    avg_gap["label"] = avg_gap["pool"].map(lambda m: METHOD_LABELS.get(m, m))

    # Create figure - transposed (horizontal bars), compact for inline placement
    n_methods = len(avg_gap)
    fig, ax = plt.subplots(figsize=(3.2, 0.32 * n_methods + 0.6))

    # Use a sequential colormap: darker = higher gap (worse)
    gap_values = avg_gap["gap"].values
    norm_gaps = (gap_values - gap_values.min()) / (gap_values.max() - gap_values.min() + 1e-8)
    cmap = plt.colormaps.get_cmap("Blues")
    colors = cmap(0.35 + 0.5 * norm_gaps)  # Light to dark blue

    # Draw horizontal bars
    y_pos = range(len(avg_gap))
    bars = ax.barh(
        y_pos,
        avg_gap["gap"],
        color=colors,
        edgecolor="white",
        linewidth=1.0,
        height=0.7,
        zorder=3,
    )

    # Add value labels at end of each bar
    for bar in bars:
        bar.set_alpha(0.9)
        width = bar.get_width()
        ax.annotate(
            f"{width:.1f}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#333333",
        )

    # Set y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(avg_gap["label"], fontsize=9)

    # Styling
    ax.set_xlabel("Gap (pp)", fontsize=10, fontweight="medium", labelpad=5)
    ax.set_xlim(0, max(gap_values) * 1.18)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=9)

    # Enhanced grid
    ax.xaxis.grid(True, linestyle="-", alpha=0.3, color="#CCCCCC", zorder=0)
    ax.yaxis.grid(False)

    # Remove top/right spines (already done via rcParams, but explicit)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    fig.savefig(paper_dir / "gap_analysis.pdf", bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(paper_dir / "gap_analysis.png", bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {paper_dir / 'gap_analysis.pdf'}")


def plot_random_vs_spatial_combined(
    knn_df: pd.DataFrame,
    linear_df: pd.DataFrame,
    paper_dir: Path,
    datasets: list[str],
) -> None:
    """Plot random vs spatial accuracy for KNN and Linear probes side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.8))
    fig.patch.set_facecolor("white")

    for ax, (df, probe_name) in zip(
        axes, [(knn_df, "KNN Probe (k=5)"), (linear_df, "Linear Probe")], strict=False
    ):
        pools = cast("pd.Series", df["pool"])
        data = df[pools.isin(METHOD_ORDER)].copy()
        ax.set_facecolor("white")

        # Collect all data points for axis limits
        all_random = []
        all_spatial = []

        for dataset in datasets:
            ds_data = data[data["dataset"] == dataset]
            pivoted = ds_data.pivot(index="pool", columns="split_type", values="accuracy")
            pivoted = pivoted.reindex(METHOD_ORDER).dropna().reset_index()

            random_vals = pivoted["standard"] * 100
            spatial_vals = pivoted["spatial"] * 100
            all_random.extend(random_vals.tolist())
            all_spatial.extend(spatial_vals.tolist())

            # Main scatter
            ax.scatter(
                random_vals,
                spatial_vals,
                s=40,
                c=DATASET_COLORS[dataset],
                marker=DATASET_MARKERS[dataset],
                edgecolor="white",
                linewidth=1.0,
                alpha=0.85,
                label=DATASET_LABELS[dataset],
                zorder=5,
            )

        # Calculate symmetric axis limits to show diagonal well
        all_vals = all_random + all_spatial
        val_min = min(all_vals) - 5
        val_max = max(all_vals) + 5
        axis_min = min(val_min, 60)
        axis_max = max(val_max, 100)

        # Diagonal reference line (y=x)
        ax.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            linestyle="--",
            color="#666666",
            linewidth=1.5,
            alpha=0.7,
            zorder=2,
        )

        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_aspect("equal", adjustable="box")

        # Labels and title
        ax.set_xlabel("Random Split Accuracy (%)", fontsize=10, fontweight="medium", labelpad=6)
        ax.set_ylabel("Spatial Split Accuracy (%)", fontsize=10, fontweight="medium", labelpad=6)
        ax.set_title(probe_name, fontsize=11, fontweight="bold", pad=8)

        # Minimal grid
        ax.grid(True, linestyle="-", alpha=0.2, color="#CCCCCC", zorder=0)

        # Reduce number of ticks
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(axis="both", labelsize=9)

        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_color("#CCCCCC")

    # Single legend for right panel
    axes[1].legend(
        loc="lower right", frameon=True, framealpha=0.95, edgecolor="#CCCCCC", fontsize=8
    )

    plt.tight_layout()
    fig.savefig(
        paper_dir / "random_vs_spatial.pdf", bbox_inches="tight", dpi=300, facecolor="white"
    )
    fig.savefig(
        paper_dir / "random_vs_spatial.png", bbox_inches="tight", dpi=300, facecolor="white"
    )
    plt.close(fig)
    print(f"Saved: {paper_dir / 'random_vs_spatial.pdf'}")


def plot_random_vs_spatial_scatter(
    df: pd.DataFrame, paper_dir: Path, datasets: list[str], probe_name: str, filename: str
) -> None:
    """Plot random vs spatial accuracy for a given probe type with clean styling."""
    pools = cast("pd.Series", df["pool"])
    data = df[pools.isin(METHOD_ORDER)].copy()

    # Create figure with white background
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Collect all data points for axis limits
    all_random = []
    all_spatial = []

    for dataset in datasets:
        ds_data = data[data["dataset"] == dataset]
        pivoted = ds_data.pivot(index="pool", columns="split_type", values="accuracy")
        pivoted = pivoted.reindex(METHOD_ORDER).dropna().reset_index()

        random_vals = pivoted["standard"] * 100
        spatial_vals = pivoted["spatial"] * 100
        all_random.extend(random_vals.tolist())
        all_spatial.extend(spatial_vals.tolist())

        # Main scatter
        ax.scatter(
            random_vals,
            spatial_vals,
            s=120,
            c=DATASET_COLORS[dataset],
            marker=DATASET_MARKERS[dataset],
            edgecolor="white",
            linewidth=1.5,
            alpha=0.85,
            label=DATASET_LABELS[dataset],
            zorder=5,
        )

    # Calculate symmetric axis limits to show diagonal well
    all_vals = all_random + all_spatial
    val_min = min(all_vals) - 5
    val_max = max(all_vals) + 5
    # Make symmetric range to ensure diagonal is visible
    axis_min = min(val_min, 60)  # Ensure we can see lower values
    axis_max = max(val_max, 100)

    # Diagonal reference line (y=x)
    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        linestyle="--",
        color="#666666",
        linewidth=1.5,
        alpha=0.7,
        zorder=2,
        label="y = x (no gap)",
    )

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)

    # Set equal aspect ratio so diagonal is at 45 degrees
    ax.set_aspect("equal", adjustable="box")

    # Labels and title
    ax.set_xlabel("Random Split Accuracy (%)", fontsize=11, fontweight="medium", labelpad=8)
    ax.set_ylabel("Spatial Split Accuracy (%)", fontsize=11, fontweight="medium", labelpad=8)
    ax.set_title(probe_name, fontsize=12, fontweight="bold", pad=10)

    # Minimal grid
    ax.grid(True, linestyle="-", alpha=0.2, color="#CCCCCC", zorder=0)

    # Reduce number of ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.tick_params(axis="both", labelsize=10)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")

    # Legend
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        fontsize=10,
    )

    plt.tight_layout()
    fig.savefig(paper_dir / f"{filename}.pdf", bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(paper_dir / f"{filename}.png", bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {paper_dir / f'{filename}.pdf'}")


def plot_comparison_bars(
    knn: pd.DataFrame, linear: pd.DataFrame, paper_dir: Path, datasets: list[str]
) -> None:
    """Create grouped bar chart comparing methods across splits/probes."""
    knn5 = knn[knn["probe"] == "knn_5"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)

    for ax, (name, df) in zip(axes, [("KNN (k=5)", knn5), ("Linear", linear)], strict=False):
        pools = cast("pd.Series", df["pool"])
        data = df[pools.isin(METHOD_ORDER)].copy()
        data["pool"] = pd.Categorical(data["pool"], categories=METHOD_ORDER, ordered=True)
        data["accuracy_pct"] = (data["accuracy"] * 100).astype(float)

        sns.barplot(
            data=cast("pd.DataFrame", data),
            x="pool",
            y="accuracy_pct",
            hue="split_type",
            hue_order=["standard", "spatial"],
            palette={"standard": COLORS["standard"], "spatial": COLORS["spatial"]},
            edgecolor="#3a3a3a",
            linewidth=0.3,
            ax=ax,
        )

        ax.set_ylabel("Accuracy (%)" if ax == axes[0] else "")
        ax.set_title(name, fontweight="semibold")
        ax.set_xlabel("")
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(70, 100)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.45)
        ax.legend(title="Split", loc="lower right", frameon=True, framealpha=0.95)

        for container in ax.containers:
            if isinstance(container, BarContainer):
                ax.bar_label(container, fmt="%.1f", padding=2, fontsize=8)

    sns.despine(fig)
    fig.tight_layout(pad=0.4)
    fig.savefig(paper_dir / "performance_comparison.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(paper_dir / "performance_comparison.png", bbox_inches="tight", dpi=300)
    print(f"Saved: {paper_dir / 'performance_comparison.pdf'}")


def plot_acc_vs_dim(
    knn: pd.DataFrame,
    linear: pd.DataFrame,
    paper_dir: Path,
    method_dim: dict[str, int],
) -> None:
    """Plot accuracy vs output dimensionality."""
    knn5 = knn[knn["probe"] == "knn_5"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

    for ax, (name, df) in zip(axes, [("KNN (k=5)", knn5), ("Linear", linear)], strict=False):
        for split, color in [("standard", COLORS["standard"]), ("spatial", COLORS["spatial"])]:
            pools = cast("pd.Series", df["pool"])
            split_data = df[(df["split_type"] == split) & (pools.isin(METHOD_ORDER))]
            dims = [method_dim.get(m, 64) for m in split_data["pool"]]
            accs = (split_data["accuracy"] * 100).tolist()
            labels = [METHOD_LABELS.get(m, m) for m in split_data["pool"]]

            ax.scatter(
                dims,
                accs,
                s=70,
                color=color,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
                label=split.title(),
            )

            for d, a, lbl in zip(dims, accs, labels, strict=False):
                offset_x = 5 if d < 200 else -5
                ha = "left" if d < 200 else "right"
                ax.annotate(
                    lbl,
                    (d, a),
                    xytext=(offset_x, 0),
                    textcoords="offset points",
                    fontsize=7,
                    ha=ha,
                    va="center",
                )

        ax.set_xlabel("Output Dimensionality")
        ax.set_ylabel("Accuracy (%)" if ax == axes[0] else "")
        ax.set_title(name, fontweight="semibold")
        ax.legend(title="Split", loc="lower right", frameon=True, framealpha=0.95)
        ax.grid(axis="both", linestyle=":", alpha=0.3)
        ax.set_xlim(0, 350)

    axes[0].set_ylim(70, 100)
    sns.despine(fig)
    fig.tight_layout(pad=0.4)
    fig.savefig(paper_dir / "acc_vs_dim.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(paper_dir / "acc_vs_dim.png", bbox_inches="tight", dpi=300)
    print(f"Saved: {paper_dir / 'acc_vs_dim.pdf'}")


def plot_knn_vs_linear(knn: pd.DataFrame, linear: pd.DataFrame, paper_dir: Path) -> None:
    """Plot KNN vs Linear accuracy scatter."""
    knn5 = knn[knn["probe"] == "knn_5"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    for ax, split in zip(axes, ["standard", "spatial"], strict=False):
        knn_split = cast("pd.DataFrame", knn5[knn5["split_type"] == split]).set_index("pool")
        lin_split = cast("pd.DataFrame", linear[linear["split_type"] == split]).set_index("pool")

        methods = [m for m in METHOD_ORDER if m in knn_split.index and m in lin_split.index]

        knn_acc = [float(knn_split.loc[m, "accuracy"]) * 100 for m in methods]
        lin_acc = [float(lin_split.loc[m, "accuracy"]) * 100 for m in methods]
        labels = [METHOD_LABELS.get(m, m) for m in methods]

        ax.scatter(
            knn_acc,
            lin_acc,
            s=70,
            color=COLORS[split],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

        for k, ln, lbl in zip(knn_acc, lin_acc, labels, strict=False):
            ax.annotate(lbl, (k, ln), xytext=(4, 4), textcoords="offset points", fontsize=8)

        # Diagonal line
        lim_min = min(min(knn_acc), min(lin_acc)) - 2
        lim_max = max(max(knn_acc), max(lin_acc)) + 2
        ax.plot(
            [lim_min, lim_max], [lim_min, lim_max], linestyle="--", color="#6b6b6b", linewidth=1
        )

        ax.set_xlabel("KNN (k=5) Accuracy (%)")
        ax.set_ylabel("Linear Accuracy (%)")
        ax.set_title(f"{split.title()} Split", fontweight="semibold")
        ax.grid(axis="both", linestyle=":", alpha=0.3)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect("equal")

    sns.despine(fig)
    fig.tight_layout(pad=0.4)
    fig.savefig(paper_dir / "knn_vs_linear.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(paper_dir / "knn_vs_linear.png", bbox_inches="tight", dpi=300)
    print(f"Saved: {paper_dir / 'knn_vs_linear.pdf'}")


def plot_knn_sensitivity(knn: pd.DataFrame, paper_dir: Path) -> None:
    """Plot KNN accuracy vs k for top methods."""
    top_methods = ["stats", "mean_max", "std", "mean_std"]
    palette = sns.color_palette(["#6B8F7E", "#4E7C69", "#3D6B59", "#2F5E4E"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

    for ax, split in zip(axes, ["standard", "spatial"], strict=False):
        for method, color in zip(top_methods, palette, strict=False):
            data = knn[(knn["pool"] == method) & (knn["split_type"] == split)]
            ks = [int(p.split("_")[1]) for p in data["probe"]]
            accs = list(data["accuracy"].values * 100)

            ax.plot(
                ks,
                accs,
                marker="o",
                markersize=6,
                linewidth=1.8,
                color=color,
                label=METHOD_LABELS.get(method, method),
            )

        ax.set_xlabel("k")
        ax.set_ylabel("Accuracy (%)" if ax == axes[0] else "")
        ax.set_title(f"{split.title()} Split", fontweight="semibold")
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.set_xticks([1, 3, 5, 7, 9])
        ax.legend(loc="lower right", frameon=True, framealpha=0.95)

    axes[0].set_ylim(82, 96)
    sns.despine(fig)
    fig.tight_layout(pad=0.4)
    fig.savefig(paper_dir / "knn_sensitivity.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(paper_dir / "knn_sensitivity.png", bbox_inches="tight", dpi=300)
    print(f"Saved: {paper_dir / 'knn_sensitivity.pdf'}")


def plot_feature_pareto(linear: pd.DataFrame, paper_dir: Path, method_dim: dict[str, int]) -> None:
    """Plot feature length vs accuracy (Pareto-style)."""
    # Focus on spatial split (harder task)
    pools = cast("pd.Series", linear["pool"])
    data = linear[(linear["split_type"] == "spatial") & (pools.isin(METHOD_ORDER))].copy()

    fig, ax = plt.subplots(figsize=(7, 4.8))

    dims = [method_dim.get(m, 64) for m in data["pool"]]
    accs = list(data["accuracy"].values * 100)
    labels = [METHOD_LABELS.get(m, m) for m in data["pool"]]

    scatter = ax.scatter(
        dims,
        accs,
        s=90,
        c=accs,
        cmap="Greens",
        edgecolor="white",
        linewidth=0.8,
    )

    for d, a, lbl in zip(dims, accs, labels, strict=False):
        offset_y = 0.8 if a < 85 else -1.2
        va = "bottom" if offset_y > 0 else "top"
        ax.annotate(
            lbl,
            (d, a),
            xytext=(0, offset_y),
            textcoords="offset points",
            fontsize=9,
            ha="center",
            va=va,
        )

    ax.set_xlabel("Feature Dimensionality")
    ax.set_ylabel("Spatial Split Accuracy (%)")
    ax.set_title("Efficiency vs Performance (Linear Probe)", fontweight="semibold")
    ax.set_xlim(0, 350)
    ax.set_ylim(74, 92)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.82)
    cbar.set_label("Accuracy (%)")

    sns.despine(fig)
    fig.tight_layout(pad=0.4)
    fig.savefig(paper_dir / "feature_pareto.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(paper_dir / "feature_pareto.png", bbox_inches="tight", dpi=300)
    print(f"Saved: {paper_dir / 'feature_pareto.pdf'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures for paper")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing dataset result folders (default: results)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["aef", "olmoearth-nano", "tessera"],
        help="Dataset names to include (default: aef olmoearth-nano tessera)",
    )
    parser.add_argument(
        "--paper-dir",
        type=str,
        default="paper/figures",
        help="Directory to write figure outputs",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=64,
        help="Input embedding dimension used to annotate dimensionality plots",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    paper_dir = Path(args.paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    method_dim = build_method_dim(args.input_dim)

    # Load results from all datasets
    knn, linear = load_all_results(results_dir, args.datasets)

    if linear.empty:
        print("No linear results found!")
        return

    # Generate multi-model figures (main paper)
    # Generate combined scatter plot for KNN and linear probes
    knn5 = knn[knn["probe"] == "knn_5"].copy() if not knn.empty else knn
    if not knn5.empty:
        plot_random_vs_spatial_combined(knn5, linear, paper_dir, args.datasets)

    # Generate single-dataset figures (supplementary) using first dataset
    first_dataset = args.datasets[0]
    knn_single = knn[knn["dataset"] == first_dataset] if not knn.empty else knn
    linear_single = linear[linear["dataset"] == first_dataset]

    if not knn_single.empty:
        plot_comparison_bars(knn_single, linear_single, paper_dir, args.datasets)
        plot_acc_vs_dim(knn_single, linear_single, paper_dir, method_dim)
        plot_knn_vs_linear(knn_single, linear_single, paper_dir)
        plot_knn_sensitivity(knn_single, paper_dir)
    plot_feature_pareto(linear_single, paper_dir, method_dim)

    print("Done!")


if __name__ == "__main__":
    main()
