import multiprocessing as mp
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

H = W = 64
EMBED_DIM = 64
NUM_WORKERS = 16


def write(paths: tuple[Path, Path]) -> None:
    input_path, output_path = paths
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(input_path) as src:
        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "width": 64,
            "height": 64,
            "count": EMBED_DIM,
            "crs": src.crs,
            "transform": src.transform,
            "compress": "zstd",
            "predictor": 2,
            "interleave": "band",
        }
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(src.read().astype(np.uint8))


if __name__ == "__main__":
    root = "data/eurosat-aef"
    output_dir = "data/eurosat-aef-converted"
    filepaths = list(Path(root).rglob("*.tif"))
    output_paths = [Path(output_dir) / filepath.relative_to(root) for filepath in filepaths]
    paths = zip(filepaths, output_paths, strict=False)
    with mp.Pool(NUM_WORKERS) as pool:
        pbar = tqdm(pool.imap_unordered(write, paths), total=len(filepaths))
        for _ in pbar:
            pass
