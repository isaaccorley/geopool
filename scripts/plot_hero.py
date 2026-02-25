"""Generate hero figure showing pixel-to-patch pooling concept."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import FancyArrowPatch
from sklearn.decomposition import PCA

# Paths - use absolute paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
EUROSAT_RGB_DIR = DATA_DIR / "ds/images/remote_sensing/otherDatasets/sentinel_2/tif"
EUROSAT_AEF_DIR = DATA_DIR / "eurosat-aef"
OUTPUT_DIR = REPO_ROOT / "paper/figures"

# Sample classes and IDs to display (diverse set)
SAMPLES = [
    ("Forest", "Forest_1000"),
    ("Residential", "Residential_1000"),
    ("AnnualCrop", "AnnualCrop_1000"),
]

CLASS_COLORS = {
    "Forest": "#8B4513",  # Brown (matching reference)
    "Residential": "#FFD700",  # Yellow/Gold (Urban)
    "AnnualCrop": "#228B22",  # Green (Crop)
    "HerbaceousVegetation": "#90EE90",
    "Highway": "#808080",
    "Industrial": "#A9A9A9",
    "Pasture": "#98FB98",
    "PermanentCrop": "#006400",
    "River": "#4169E1",
    "SeaLake": "#1E90FF",
}


def load_rgb_image(class_name: str, sample_id: str) -> np.ndarray:
    """Load and normalize RGB bands from EuroSAT TIF."""
    path = EUROSAT_RGB_DIR / class_name / f"{sample_id}.tif"
    with rasterio.open(path) as src:
        # EuroSAT has 13 bands, RGB are bands 4, 3, 2 (1-indexed) or 3, 2, 1 (0-indexed)
        # Actually for Sentinel-2 in EuroSAT: B4=Red, B3=Green, B2=Blue
        r = src.read(4).astype(np.float32)  # Red (Band 4)
        g = src.read(3).astype(np.float32)  # Green (Band 3)
        b = src.read(2).astype(np.float32)  # Blue (Band 2)
    rgb = np.stack([r, g, b], axis=-1)
    # Flip horizontally
    rgb = np.fliplr(rgb)
    # Simple normalization: divide by 2500 and clip
    return np.clip(rgb / 2500.0, 0, 1)


def load_embedding(class_name: str, sample_id: str) -> np.ndarray:
    """Load AEF embedding tensor."""
    path = EUROSAT_AEF_DIR / class_name / f"{sample_id}.tif"
    with rasterio.open(path) as src:
        emb = src.read()  # (C, H, W)
    return emb.transpose(1, 2, 0).astype(np.float32)  # (H, W, C)


def embedding_to_pca_rgb(emb: np.ndarray, pca: PCA | None = None) -> tuple[np.ndarray, PCA]:
    """Convert embedding to pseudo-RGB via PCA."""
    h, w, c = emb.shape
    flat = emb.reshape(-1, c)
    if pca is None:
        pca = PCA(n_components=3)
        pca.fit(flat)
    rgb = pca.transform(flat).reshape(h, w, 3)
    # Invert channel order (reverse PC1, PC2, PC3 -> PC3, PC2, PC1)
    # rgb = rgb[:, :, ::-1]
    # Normalize each channel to [0, 1]
    for i in range(3):
        ch = rgb[:, :, i]
        p2, p98 = np.percentile(ch, [2, 98])
        rgb[:, :, i] = np.clip((ch - p2) / (p98 - p2 + 1e-8), 0, 1)
    return rgb, pca


def pool_stats(emb: np.ndarray) -> np.ndarray:
    """Stats pooling: min, max, mean, std."""
    flat = emb.reshape(-1, emb.shape[-1])
    return np.concatenate(
        [
            flat.min(axis=0),
            flat.max(axis=0),
            flat.mean(axis=0),
            flat.std(axis=0),
        ]
    )


def create_hero_figure() -> None:
    """Create the hero figure showing pixel-to-patch pooling."""
    fig = plt.figure(figsize=(3.2, 3.0))

    # Load one good sample
    class_name, sample_id = "AnnualCrop", "AnnualCrop_1002"

    rgb = load_rgb_image(class_name, sample_id)
    emb = load_embedding(class_name, sample_id)
    pca_rgb, _ = embedding_to_pca_rgb(emb)
    # Use mean pooling for 64-d vector
    pooled = emb.reshape(-1, emb.shape[-1]).mean(axis=0)

    # Custom layout using axes positions directly
    # Top row: [Image] -> [Embeddings]
    # Bottom row: [thin Vector] -> [Label]
    # Diagonal arrow from embeddings to vector

    # Panel 1: Input RGB Image (top left)
    ax1 = fig.add_axes((0.05, 0.52, 0.38, 0.42))
    ax1.imshow(rgb)
    ax1.set_title("Input Image (10m)", fontsize=8, fontweight="bold")
    ax1.set_xlabel(r"$\mathbf{64{\times}64{\times}12}$", fontsize=6)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_linewidth(1)

    # Panel 2: Pixel Embeddings (top right)
    ax2 = fig.add_axes((0.57, 0.52, 0.38, 0.42))
    ax2.imshow(pca_rgb)
    ax2.set_title("Pixel Embeddings (10m)", fontsize=8, fontweight="bold")
    ax2.set_xlabel(r"$\mathbf{64{\times}64{\times}64}$", fontsize=6)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_linewidth(1)

    # Arrow 1->2 (top row, horizontal)
    fig.patches.append(
        FancyArrowPatch(
            (0.44, 0.73),
            (0.55, 0.73),
            transform=fig.transFigure,
            arrowstyle="->",
            mutation_scale=12,
            lw=1.2,
            color="#333",
        )
    )

    # Panel 3: Pooled Vector (bottom left) - THIN horizontal bar
    ax3 = fig.add_axes((0.05, 0.20, 0.38, 0.05))  # Very short height = thin
    pooled_norm = (pooled - pooled.min()) / (pooled.max() - pooled.min() + 1e-8)
    pooled_2d = pooled_norm.reshape(1, -1)
    ax3.imshow(pooled_2d, aspect="auto", cmap="viridis")
    ax3.set_title("Pooled Vector", fontsize=8, fontweight="bold")
    ax3.set_xlabel(r"$\mathbf{64{\times}1}$", fontsize=6)
    ax3.set_xticks([])
    ax3.set_yticks([])
    for spine in ax3.spines.values():
        spine.set_linewidth(1)

    # Diagonal arrow from embeddings (top right) to vector (bottom left)
    fig.patches.append(
        FancyArrowPatch(
            (0.57, 0.52),
            (0.43, 0.32),
            transform=fig.transFigure,
            arrowstyle="->",
            mutation_scale=12,
            lw=1.2,
            color="#333",
            connectionstyle="arc3,rad=-0.1",
        )
    )

    # Panel 4: Class Label (bottom right)
    ax4 = fig.add_axes((0.57, 0.05, 0.38, 0.35))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    rect = plt.Rectangle(
        (0.0, 0.0), 1.0, 1.0, facecolor=CLASS_COLORS[class_name], edgecolor="black", linewidth=1
    )
    ax4.add_patch(rect)
    ax4.text(
        0.5,
        0.5,
        class_name.replace("Annual", "Annual\n"),
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="white",
    )
    ax4.set_title("Patch Label (640m)", fontsize=8, fontweight="bold")
    ax4.set_xticks([])
    ax4.set_yticks([])

    # Arrow vector -> label (bottom row, horizontal)
    fig.patches.append(
        FancyArrowPatch(
            (0.44, 0.225),
            (0.55, 0.225),
            transform=fig.transFigure,
            arrowstyle="->",
            mutation_scale=12,
            lw=1.2,
            color="#333",
        )
    )

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "hero.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    create_hero_figure()
