from __future__ import annotations

import argparse
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
import rasterio.transform
from geotessera import GeoTessera
from geotessera.registry import tile_from_world
from pyproj import Transformer
from torchgeo.datasets import EuroSAT
from tqdm import tqdm

H = W = 64
EMBED_DIM = 128

error_lock = threading.Lock()


def extract_patch(
    filepath: Path,
    quantized: np.ndarray,
    tile_crs: rasterio.crs.CRS,
    tile_transform: rasterio.transform.Affine,
    output: str,
) -> tuple[Path, str | None]:
    """Extract a 64x64 quantized tessera embedding patch aligned with a EuroSAT image."""
    try:
        with rasterio.open(filepath) as src:
            src_crs = src.crs
            src_transform = src.transform

        proj = Transformer.from_crs(src_crs, tile_crs, always_xy=True)

        # Generate all pixel center coordinates in the EuroSAT grid
        rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        xs, ys = rasterio.transform.xy(src_transform, rows.ravel(), cols.ravel())
        xs, ys = np.array(xs), np.array(ys)

        # Transform to tessera tile CRS
        txs, tys = proj.transform(xs, ys)

        # Map to pixel coords in tessera tile (nearest neighbor)
        trows, tcols = rasterio.transform.rowcol(tile_transform, txs, tys)
        trows, tcols = np.array(trows), np.array(tcols)

        tile_h, tile_w = quantized.shape[:2]
        valid = (trows >= 0) & (trows < tile_h) & (tcols >= 0) & (tcols < tile_w)

        patch = np.zeros((H * W, EMBED_DIM), dtype=np.int8)
        patch[valid] = quantized[trows[valid], tcols[valid], :]
        # Shift int8 (-128..127) -> uint8 (0..255) since GeoTIFF doesn't support signed int8
        patch = patch.astype(np.int16) + 128
        patch = patch.astype(np.uint8).reshape(H, W, EMBED_DIM).transpose(2, 0, 1)  # (C, H, W)

        output_path = os.path.join(output, filepath.stem + ".tif")
        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "width": W,
            "height": H,
            "count": EMBED_DIM,
            "crs": src_crs,
            "transform": src_transform,
            "compress": "zstd",
            "predictor": 2,
            "interleave": "band",
        }
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(patch)
    except Exception as e:
        return filepath, str(e)
    return filepath, None


def process_tile(
    tile_lon: float,
    tile_lat: float,
    group_paths: list[Path],
    gt: GeoTessera,
    year: int,
    output: str,
) -> list[tuple[Path, str]]:
    """Download one tessera tile and extract all patches from it."""
    errors: list[tuple[Path, str]] = []
    try:
        emb_path = gt.registry.fetch(
            year=year, lon=tile_lon, lat=tile_lat, is_scales=False, progressbar=False
        )
        landmask_path = gt.registry.fetch_landmask(lon=tile_lon, lat=tile_lat, progressbar=False)
        quantized = np.load(emb_path)  # (H, W, 128), int8
        with rasterio.open(landmask_path) as lm:
            tile_crs = lm.crs
            tile_transform = lm.transform
    except Exception as e:
        return [(fp, str(e)) for fp in group_paths]

    for filepath in group_paths:
        _, error = extract_patch(filepath, quantized, tile_crs, tile_transform, output)
        if error:
            errors.append((filepath, error))
    return errors


def main(args: argparse.Namespace) -> None:
    gt = GeoTessera()

    filepaths: list[Path] = []
    for split in ["train", "val", "test"]:
        ds = EuroSAT(root=args.root, split=split, download=True, checksum=True)
        filepaths.extend([Path(img) for img, _ in ds.imgs])

    # Skip already-completed patches
    existing = {p.stem for p in Path(args.output).glob("**/*.tif")}
    before = len(filepaths)
    filepaths = [fp for fp in filepaths if fp.stem not in existing]
    print(
        f"Skipping {before - len(filepaths)} already-completed patches, {len(filepaths)} remaining"
    )

    if not filepaths:
        print("All patches already extracted.")
        return

    # Group images by tessera tile to avoid redundant downloads
    print(f"Grouping {len(filepaths)} images by tessera tile...")
    tile_groups: dict[tuple[float, float], list[Path]] = {}
    for filepath in tqdm(filepaths, desc="Reading metadata"):
        with rasterio.open(filepath) as src:
            proj = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            cx = src.transform.c + src.width / 2 * src.transform.a
            cy = src.transform.f + src.height / 2 * src.transform.e
            lon, lat = proj.transform(cx, cy)

        tile_lon, tile_lat = tile_from_world(lon, lat)
        key = (tile_lon, tile_lat)
        if key not in tile_groups:
            tile_groups[key] = []
        tile_groups[key].append(filepath)

    print(f"Found {len(tile_groups)} unique tessera tiles")

    os.makedirs(args.output, exist_ok=True)

    if not Path("errors.csv").exists():
        with open("errors.csv", "w") as f:
            f.write("filepath,error\n")

    # Process tiles in parallel: each worker downloads + extracts patches for one tile
    tile_items = list(tile_groups.items())
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_tile, tile_lon, tile_lat, group_paths, gt, args.year, args.output
            ): (tile_lon, tile_lat)
            for (tile_lon, tile_lat), group_paths in tile_items
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tiles"):
            tile_errors = future.result()
            if tile_errors:
                with error_lock, open("errors.csv", "a") as f:
                    f.writelines(f"{fp},{err}\n" for fp, err in tile_errors)

    # Reorganize images into class folders
    images = list(Path(args.output).glob("*.tif"))
    print(f"Reorganizing {len(images)} images into class folders...")
    for image in tqdm(images):
        folder = image.parent / image.stem.split("_")[0]
        folder.mkdir(parents=True, exist_ok=True)
        shutil.move(image, folder / image.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--output", type=str, default="eurosat-tessera")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--root", type=str, default="data")
    args = parser.parse_args()
    main(args)
