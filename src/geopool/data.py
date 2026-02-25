"""Data loading utilities for EuroSAT embeddings."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio

CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}


def parse_split_file(path: str | Path) -> list[tuple[str, str]]:
    """Parse EuroSAT split file. Returns list of (rel_path, class_name) tuples."""
    path = Path(path)
    samples = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            basename = line.replace(".jpg", "")
            parts = basename.rsplit("_", 1)
            if len(parts) != 2:
                raise ValueError(f"Unexpected filename format: {line}")
            class_name = parts[0]
            rel_path = f"{class_name}/{basename}"
            samples.append((rel_path, class_name))

    return samples


def load_embedding(rel_path: str, data_dir: str | Path) -> np.ndarray:
    """Load embedding from GeoTIFF. Returns (H, W, D) array."""
    data_dir = Path(data_dir)
    tif_path = data_dir / f"{rel_path}.tif"

    with rasterio.open(tif_path) as src:
        data = src.read(out_dtype="float32")

    return np.moveaxis(data, 0, -1)


def load_dataset_with_pool(
    split_file: str | Path,
    data_dir: str | Path,
    pool_fn: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load dataset and apply pooling on-the-fly. Returns (X, y, filenames)."""
    samples = parse_split_file(split_file)

    embeddings = []
    labels = []
    filenames = []

    for rel_path, class_name in samples:
        emb = load_embedding(rel_path, data_dir)
        pool_emb = pool_fn(emb)
        embeddings.append(pool_emb)
        labels.append(CLASS_TO_IDX[class_name])
        filenames.append(rel_path)

    X = np.stack(embeddings, axis=0)
    y = np.array(labels, dtype=np.int64)

    return X, y, filenames


def load_dataset_raw(
    split_file: str | Path,
    data_dir: str | Path,
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Load raw embeddings as list (variable spatial sizes). Returns (X_list, y, filenames)."""
    samples = parse_split_file(split_file)

    embeddings = []
    labels = []
    filenames = []

    for rel_path, class_name in samples:
        emb = load_embedding(rel_path, data_dir)
        embeddings.append(emb)
        labels.append(CLASS_TO_IDX[class_name])
        filenames.append(rel_path)

    y = np.array(labels, dtype=np.int64)

    return embeddings, y, filenames


def get_split_paths(
    data_dir: str | Path,
    split_type: Literal["standard", "spatial"] = "standard",
) -> tuple[Path, Path, Path]:
    """Get paths to train/val/test split files."""
    data_dir = Path(data_dir)

    prefix = "eurosat" if split_type == "standard" else "eurosat-spatial"

    train = data_dir / f"{prefix}-train.txt"
    val = data_dir / f"{prefix}-val.txt"
    test = data_dir / f"{prefix}-test.txt"

    return train, val, test
