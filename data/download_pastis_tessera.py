"""Download Tessera embeddings for PASTIS tiles via the geotessera library.

Tessera ships pre-computed annual embeddings (10m, 128-d int8) on a global
0.1-deg grid. Per PASTIS tile (128x128 @ 10m in native UTM):
  1. find the geotessera tile(s) covering the PASTIS bbox
  2. fetch embeddings + per-pixel scales + landmask
  3. dequantize: float32 = int8 * scale[h, w] (per-pixel scalar)
  4. reproject + resample to 128x128 in PASTIS tile UTM
  5. write a 128-band float32 GeoTIFF

Many PASTIS tiles share a tessera 0.1-deg tile, so we group by tessera tile
and fetch each only once.
"""

import argparse
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from geotessera import GeoTessera
from geotessera.registry import tile_from_world
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from tqdm import tqdm

EMBED_DIM = 128
H = W = 128
PIXEL_M = 10.0

error_lock = threading.Lock()


def write_patch(output_path: Path, patch: np.ndarray, dst_crs: str, dst_transform) -> None:
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": W,
        "height": H,
        "count": EMBED_DIM,
        "crs": dst_crs,
        "transform": dst_transform,
        "compress": "zstd",
        "predictor": 3,  # floating-point predictor
        "interleave": "band",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(patch.astype(np.float32, copy=False))


def tessera_tiles_for_bbox(lon_min, lat_min, lon_max, lat_max) -> list[tuple[float, float]]:
    """Tessera grid is 0.1-deg. Enumerate all tiles intersecting the bbox."""
    lo_lon, lo_lat = tile_from_world(lon_min, lat_min)
    hi_lon, hi_lat = tile_from_world(lon_max, lat_max)
    lons = np.round(np.arange(lo_lon, hi_lon + 0.1, 0.1), 2)
    lats = np.round(np.arange(lo_lat, hi_lat + 0.1, 0.1), 2)
    return [(float(lon), float(lat)) for lon in lons for lat in lats]


def fetch_tessera_tile(gt: GeoTessera, tile_lon: float, tile_lat: float, year: int):
    """Return (array (H,W,128) float32 dequantized, crs, transform)."""
    emb_path = gt.registry.fetch(
        year=year, lon=tile_lon, lat=tile_lat, is_scales=False, progressbar=False
    )
    scales_path = gt.registry.fetch(
        year=year, lon=tile_lon, lat=tile_lat, is_scales=True, progressbar=False
    )
    lm_path = gt.registry.fetch_landmask(lon=tile_lon, lat=tile_lat, progressbar=False)

    arr = np.load(emb_path)          # (H, W, 128) int8
    scales = np.load(scales_path)    # (H, W) float32 — per-pixel scalar

    # Dequantize: broadcast scale across embedding dim
    arr_f32 = arr.astype(np.float32) * scales[:, :, None]  # (H, W, 128) float32

    with rasterio.open(lm_path) as lm:
        crs = lm.crs
        transform = lm.transform
    return arr_f32, crs, transform


def process_tile_group(
    tile_lon: float,
    tile_lat: float,
    rows: list[pd.Series],
    gt: GeoTessera,
    year: int,
    output_dir: Path,
) -> list[tuple[int, str]]:
    errors: list[tuple[int, str]] = []
    try:
        src_arr, src_crs, src_transform = fetch_tessera_tile(gt, tile_lon, tile_lat, year)
        # (H, W, 128) float32 -> (128, H, W) for rasterio's band-first convention
        src_arr = np.moveaxis(src_arr, -1, 0)
    except Exception as e:
        return [(int(r.id_patch), f"fetch_tessera_tile({tile_lon},{tile_lat}): {e}") for r in rows]

    for row in rows:
        id_patch = int(row.id_patch)
        output_path = output_dir / f"S2_{id_patch}.tif"
        if output_path.exists():
            continue
        try:
            dst_crs = f"EPSG:{int(row.utm_epsg)}"
            dst_transform = from_origin(row.utm_minx, row.utm_maxy, PIXEL_M, PIXEL_M)
            dst = np.zeros((EMBED_DIM, H, W), dtype=np.float32)
            reproject(
                source=src_arr,
                destination=dst,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
            write_patch(output_path, dst, dst_crs, dst_transform)
        except Exception as e:
            errors.append((id_patch, str(e)))
    return errors


def main(args: argparse.Namespace) -> None:
    tiles = pd.read_parquet(args.tiles)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip already-done tiles
    done = {int(p.stem.split("_")[1]) for p in output_dir.glob("S2_*.tif")}
    before = len(tiles)
    tiles = tiles[~tiles.id_patch.isin(done)].reset_index(drop=True)
    print(f"skip {before - len(tiles)} already-done tiles, {len(tiles)} remaining")
    if len(tiles) == 0:
        return

    # Group PASTIS tiles by tessera 0.1-deg tile (each PASTIS tile typically
    # touches 1-4 tessera tiles -- we still let each PASTIS tile see all its
    # source tessera tiles via reproject, but we cache by source).
    print(f"grouping {len(tiles)} PASTIS tiles by tessera 0.1-deg tile...")
    groups: dict[tuple[float, float], list[pd.Series]] = defaultdict(list)
    for _, row in tiles.iterrows():
        for tlon, tlat in tessera_tiles_for_bbox(row.lon_min, row.lat_min, row.lon_max, row.lat_max):
            groups[(tlon, tlat)].append(row)
    print(f"found {len(groups)} unique tessera tiles")

    gt = GeoTessera()
    errors_csv = Path("errors_pastis_tessera.csv")
    if not errors_csv.exists():
        errors_csv.write_text("id_patch,error\n")

    items = list(groups.items())
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {
            ex.submit(process_tile_group, tlon, tlat, rows, gt, args.year, output_dir): (tlon, tlat)
            for (tlon, tlat), rows in items
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="tessera groups"):
            errs = fut.result()
            if errs:
                with error_lock, open(errors_csv, "a") as f:
                    f.writelines(f"{i},{e}\n" for i, e in errs)

    n_done = len(list(output_dir.glob("S2_*.tif")))
    print(f"done: {n_done} tiles -> {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tiles", default="data/pastis_tiles.parquet")
    p.add_argument("--output", default="data/pastis-tessera")
    p.add_argument("--year", type=int, default=2019)
    p.add_argument("--num-workers", type=int, default=16)
    main(p.parse_args())
