"""Generate presentation figures using actual EuroSAT-AEF embeddings."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import rasterio
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from sklearn.decomposition import PCA

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA = SCRIPT_DIR.parent / "data" / "eurosat-aef"
OUT = SCRIPT_DIR / "figures"
OUT.mkdir(exist_ok=True)

OCEAN = "#1B4F72"
ACCENT = "#E67E22"
GRAY = "#5D6D7E"


def load_embedding(cls: str, idx: int = 1) -> np.ndarray:
    path = DATA / cls / f"{cls}_{idx}.tif"
    with rasterio.open(path) as f:
        return f.read().transpose(1, 2, 0).astype(np.float32)  # H x W x D


def pca_rgb(emb: np.ndarray, pca_model: PCA | None = None) -> tuple[np.ndarray, PCA]:
    """Project embedding to 3-channel pseudo-RGB via PCA."""
    H, W, D = emb.shape
    flat = emb.reshape(-1, D)
    if pca_model is None:
        pca_model = PCA(n_components=3)
        proj = pca_model.fit_transform(flat)
    else:
        proj = pca_model.transform(flat)
    # Normalize to [0, 1]
    proj = proj.reshape(H, W, 3)
    for c in range(3):
        lo, hi = np.percentile(proj[:, :, c], [2, 98])
        proj[:, :, c] = np.clip((proj[:, :, c] - lo) / (hi - lo + 1e-8), 0, 1)
    return proj, pca_model


# ================================================================
# Figure 1: Patch vs Pixel Embeddings
# ================================================================
def make_patch_vs_pixel() -> None:
    print("Making patch vs pixel figure...")

    # Load a few classes for visual variety
    residential = load_embedding("Residential", 100)
    forest = load_embedding("Forest", 50)

    # Fit PCA on residential for consistent coloring
    res_rgb, pca = pca_rgb(residential)
    _for_rgb, _ = pca_rgb(forest, pca)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # --- Left: PATCH embedding ---
    ax = axes[0]
    ax.imshow(res_rgb)
    # Overlay a semi-transparent blue rectangle to represent single patch vector
    overlay = np.ones_like(res_rgb) * np.array([0.106, 0.310, 0.447])
    ax.imshow(overlay, alpha=0.55)
    # Crosshair
    ax.axhline(32, color="white", lw=0.5, alpha=0.4)
    ax.axvline(32, color="white", lw=0.5, alpha=0.4)
    # Center dot
    ax.plot(
        32, 32, "o", color="#8BC34A", markersize=10, markeredgecolor="white", markeredgewidth=1.5
    )
    ax.set_title("PATCH Embedding", fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("One vector per patch", fontsize=11, color=GRAY)
    # Border
    for spine in ax.spines.values():
        spine.set_edgecolor(OCEAN)
        spine.set_linewidth(2)
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Right: PIXEL embedding ---
    ax = axes[1]
    ax.imshow(res_rgb)
    # Draw grid of small squares to represent per-pixel vectors
    step = 2
    for i in range(0, 64, step):
        for j in range(0, 64, step):
            rect = plt.Rectangle(
                (j - 0.3, i - 0.3),
                step - 0.4,
                step - 0.4,
                fill=False,
                edgecolor="white",
                lw=0.3,
                alpha=0.5,
            )
            ax.add_patch(rect)
            # Small dot in center
            ax.plot(
                j + step / 2 - 0.5, i + step / 2 - 0.5, "s", color=OCEAN, markersize=1.5, alpha=0.6
            )

    ax.set_title("PIXEL Embedding", fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("One vector per pixel", fontsize=11, color=GRAY)
    for spine in ax.spines.values():
        spine.set_edgecolor(OCEAN)
        spine.set_linewidth(2)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout(w_pad=3)
    fig.savefig(OUT / "patch_vs_pixel.pdf")
    fig.savefig(OUT / "patch_vs_pixel.png")
    plt.close()
    print("  -> patch_vs_pixel.pdf")


# ================================================================
# Figure 2: Heterogeneity — why mean pooling loses signal
# ================================================================
def make_heterogeneity() -> None:
    print("Making heterogeneity figure...")

    classes = {
        "Forest": (50, "#2E7D32"),
        "Residential": (100, "#1565C0"),
        "Industrial": (30, "#6A1B9A"),
        "AnnualCrop": (20, "#E65100"),
    }

    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    pca = PCA(n_components=3)
    # Fit PCA on combined data
    all_embs = []
    for cls in classes:
        emb = load_embedding(cls, classes[cls][0])
        all_embs.append(emb.reshape(-1, 64))
    combined = np.vstack(all_embs)
    pca.fit(combined[::4])  # subsample for speed

    for col, (cls, (idx, color)) in enumerate(classes.items()):
        emb = load_embedding(cls, idx)
        _H, _W, D = emb.shape
        flat = emb.reshape(-1, D)

        # Top row: PCA pseudo-RGB
        ax_img = fig.add_subplot(gs[0, col])
        rgb, _ = pca_rgb(emb, pca)
        ax_img.imshow(rgb)
        ax_img.set_title(cls, fontsize=13, fontweight="bold", color=color)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

        # Compute stats
        pixel_std = flat.std(axis=0)
        mean_std = pixel_std.mean()

        ax_img.text(
            0.5,
            -0.08,
            f"Avg pixel std: {mean_std:.1f}",
            transform=ax_img.transAxes,
            ha="center",
            fontsize=9,
            color=GRAY,
        )

        # Bottom row: distribution of pixel embeddings (first 3 PCA dims)
        ax_dist = fig.add_subplot(gs[1, col])
        proj = pca.transform(flat)
        for dim, (dname, alpha) in enumerate(
            zip(["PC1", "PC2", "PC3"], [0.8, 0.5, 0.3], strict=False)
        ):
            vals = proj[:, dim]
            ax_dist.hist(
                vals, bins=40, alpha=alpha, color=color, label=dname, density=True, edgecolor="none"
            )
        ax_dist.set_xlabel("Embedding value", fontsize=9)
        if col == 0:
            ax_dist.set_ylabel("Density", fontsize=9)
        ax_dist.tick_params(labelsize=8)
        ax_dist.legend(fontsize=7, loc="upper right", framealpha=0.7)
        ax_dist.set_title("Pixel distribution", fontsize=10, color=GRAY)

        # Mean line
        mean_val = proj[:, 0].mean()
        ax_dist.axvline(mean_val, color="red", lw=1.5, ls="--", alpha=0.8)
        ax_dist.text(
            mean_val,
            ax_dist.get_ylim()[1] * 0.9,
            "μ",
            color="red",
            fontsize=10,
            fontweight="bold",
            ha="center",
        )

    fig.suptitle(
        "Within-patch heterogeneity across land cover classes",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    fig.savefig(OUT / "heterogeneity.pdf")
    fig.savefig(OUT / "heterogeneity.png")
    plt.close()
    print("  -> heterogeneity.pdf")


# ================================================================
# Figure 3: What different pooling methods capture
# ================================================================
def make_pooling_visual() -> None:
    print("Making pooling methods visual...")

    # Load a heterogeneous patch
    emb = load_embedding("Residential", 100)
    _H, _W, D = emb.shape
    flat = emb.reshape(-1, D)

    # Compute different pooling results
    mean_pool = flat.mean(axis=0)
    max_pool = flat.max(axis=0)
    std_pool = flat.std(axis=0)
    min_pool = flat.min(axis=0)

    p = 3
    gem_pool = np.power(np.mean(np.power(flat.astype(np.float64), p), axis=0), 1.0 / p)

    pca = PCA(n_components=3)
    pca.fit(flat)
    rgb, _ = pca_rgb(emb, pca)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))

    # Top-left: the patch (PCA pseudo-RGB)
    ax = axes[0, 0]
    ax.imshow(rgb)
    ax.set_title("Residential patch\n(PCA pseudo-RGB)", fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    # Show each pooling method as a bar chart of first 20 dims
    dims = np.arange(20)
    methods = [
        ("Mean pooling", mean_pool[:20], "#E74C3C", axes[0, 1]),
        ("Max pooling", max_pool[:20], "#2E86C1", axes[0, 2]),
        ("GeM pooling (p=3)", gem_pool[:20], "#27AE60", axes[1, 0]),
        ("Std pooling", std_pool[:20], "#8E44AD", axes[1, 1]),
        ("Stats pooling", None, "#E67E22", axes[1, 2]),
    ]

    for name, vals, color, ax in methods:
        if name == "Stats pooling":
            # Show all 4 concatenated
            stats = np.concatenate([min_pool[:5], max_pool[:5], mean_pool[:5], std_pool[:5]])
            x = np.arange(20)
            colors = ["#3498DB"] * 5 + ["#E74C3C"] * 5 + ["#2ECC71"] * 5 + ["#9B59B6"] * 5
            ax.bar(x, stats, color=colors, width=0.7, edgecolor="none")
            # Separators
            for sep in [5, 10, 15]:
                ax.axvline(sep - 0.5, color="gray", lw=0.8, ls=":")
            ax.set_xticks([2.5, 7.5, 12.5, 17.5])
            ax.set_xticklabels(["min", "max", "mean", "std"], fontsize=8)
            ax.set_title(
                f"{name}\n(4× dim — concat all)",  # noqa: RUF001
                fontsize=11,
                fontweight="bold",
                color=color,
            )
        else:
            ax.bar(dims, vals, color=color, width=0.7, edgecolor="none", alpha=0.85)
            ax.set_xlabel("Dimension", fontsize=9)
            ax.set_title(f"{name}\n(1× dim)", fontsize=11, fontweight="bold", color=color)  # noqa: RUF001

        ax.tick_params(labelsize=8)
        ax.set_ylabel("Value", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(h_pad=2, w_pad=2)
    fig.savefig(OUT / "pooling_visual.pdf")
    fig.savefig(OUT / "pooling_visual.png")
    plt.close()
    print("  -> pooling_visual.pdf")


# ================================================================
# Figure 4: Resolution mismatch diagram
# ================================================================
def make_resolution_mismatch() -> None:
    print("Making resolution mismatch figure...")

    # Load diverse patches
    forest = load_embedding("Forest", 50)
    residential = load_embedding("Residential", 100)

    pca = PCA(n_components=3)
    combined = np.vstack([forest.reshape(-1, 64), residential.reshape(-1, 64)])
    pca.fit(combined[::8])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    # Panel 1: Dense pixel embeddings (PCA pseudo-RGB)
    ax = axes[0]
    rgb, _ = pca_rgb(residential, pca)
    ax.imshow(rgb)
    ax.set_title("Pixel embeddings\n(64 × 64 × 64-d)", fontsize=11, fontweight="bold")  # noqa: RUF001
    ax.set_xlabel("4,096 vectors", fontsize=9, color=GRAY)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(OCEAN)
        spine.set_linewidth(1.5)

    # Panel 2: Arrow showing pooling
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.annotate(
        "",
        xy=(8, 5),
        xytext=(2, 5),
        arrowprops={"arrowstyle": "-|>", "color": ACCENT, "lw": 3, "mutation_scale": 25},
    )
    ax.text(5, 7, "Pool?", fontsize=16, ha="center", fontweight="bold", color=OCEAN)
    ax.text(5, 3, "mean / GeM /\nstats / ...", fontsize=10, ha="center", color=GRAY)

    # Panel 3: Single pooled vector
    ax = axes[2]
    # Represent as a color bar
    pooled = residential.reshape(-1, 64).mean(axis=0)
    pooled_norm = (pooled - pooled.min()) / (pooled.max() - pooled.min())
    bar_img = pooled_norm.reshape(1, -1)
    ax.imshow(bar_img, aspect="auto", cmap="viridis", extent=[0, 64, 0, 8])
    ax.set_title("Pooled vector\n(1 × 64-d)", fontsize=11, fontweight="bold")  # noqa: RUF001
    ax.set_xlabel("1 vector", fontsize=9, color=GRAY)
    ax.set_xticks([0, 32, 64])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(OCEAN)
        spine.set_linewidth(1.5)

    # Panel 4: Arrow to label
    ax = axes[3]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    # Label box
    rect = FancyBboxPatch(
        (1.5, 2.5), 7, 5, boxstyle="round,pad=0.3", facecolor=OCEAN, edgecolor=OCEAN, alpha=0.9
    )
    ax.add_patch(rect)
    ax.text(
        5, 5, "Residential", fontsize=16, ha="center", va="center", color="white", fontweight="bold"
    )
    ax.set_title("Patch label\n(1 class)", fontsize=11, fontweight="bold")
    ax.set_xlabel("kNN / linear probe", fontsize=9, color=GRAY)

    plt.tight_layout(w_pad=1)
    fig.savefig(OUT / "resolution_mismatch.pdf")
    fig.savefig(OUT / "resolution_mismatch.png")
    plt.close()
    print("  -> resolution_mismatch.pdf")


if __name__ == "__main__":
    make_patch_vs_pixel()
    make_heterogeneity()
    make_pooling_visual()
    make_resolution_mismatch()
    print("\nAll figures generated!")
