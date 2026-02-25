"""PyTorch Dataset for EuroSAT embedding GeoTIFFs."""

from pathlib import Path

import numpy as np
import rasterio
from torch.utils.data import Dataset

from .data import CLASS_TO_IDX, parse_split_file


class EuroSATEmbeddingDataset(Dataset):
    """Loads embedding GeoTIFFs. Returns (embedding, label) where embedding is (H, W, D)."""

    def __init__(
        self,
        split_file: str | Path,
        data_dir: str | Path,
        size: int | None = 64,
        dtype: str = "float32",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.samples = parse_split_file(split_file)
        self.size = (size, size) if size is not None else None
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        rel_path, class_name = self.samples[index]
        tif_path = self.data_dir / f"{rel_path}.tif"

        with rasterio.open(tif_path) as src:
            data = src.read(out_dtype=self.dtype, out_shape=self.size)
            data = np.moveaxis(data, 0, -1)

        return data, CLASS_TO_IDX[class_name]
