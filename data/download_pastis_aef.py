"""Download AlphaEarth (AEF) embeddings for PASTIS tiles from source.coop.

AEF COGs live at s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/<year>/<utm_zone>/<id>.tiff
indexed by https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet

For each PASTIS tile (128x128 @ 10m in native UTM):
  1. find AEF COG(s) whose WGS84 bbox intersects the tile
  2. reproject + resample to 128x128 in the PASTIS tile's UTM CRS
  3. write a 64-band int8 GeoTIFF

NB: AEF COGs are stored south-up (transform y-scale is +10, not -10), so a
naive `read(window=...)` fails. rasterio.warp.reproject handles this.
"""

import argparse
import os
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from tqdm import tqdm

AEF_INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
EMBED_DIM = 64
H = W = 128
PIXEL_M = 10.0

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("AWS_REGION", "us-west-2")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "200000000")

error_lock = threading.Lock()


def fetch_aef_index(cache_path: Path, year: int) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        print(f"downloading AEF index -> {cache_path}")
        subprocess.run(["curl", "-fsSL", AEF_INDEX_URL, "-o", str(cache_path)], check=True)
    df = pd.read_parquet(cache_path)
    df = df[df["year"] == year].reset_index(drop=True)
    print(f"AEF index for year={year}: {len(df)} COGs")
    return df


def find_intersecting_cogs(idx: pd.DataFrame, lon_min, lat_min, lon_max, lat_max) -> pd.DataFrame:
    return idx[
        (idx.wgs84_west < lon_max)
        & (idx.wgs84_east > lon_min)
        & (idx.wgs84_south < lat_max)
        & (idx.wgs84_north > lat_min)
    ]


def write_patch(output_path: Path, patch: np.ndarray, dst_crs: str, dst_transform) -> None:
    profile = {
        "driver": "GTiff",
        "dtype": "int8",
        "width": W,
        "height": H,
        "count": EMBED_DIM,
        "crs": dst_crs,
        "transform": dst_transform,
        "compress": "zstd",
        "predictor": 2,
        "interleave": "band",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(patch)


def process_tile(
    row: pd.Series, idx: pd.DataFrame, output_dir: Path
) -> tuple[int, str | None]:
    id_patch = int(row.id_patch)
    output_path = output_dir / f"S2_{id_patch}.tif"
    if output_path.exists():
        return id_patch, None
    try:
        hits = find_intersecting_cogs(idx, row.lon_min, row.lat_min, row.lon_max, row.lat_max)
        if len(hits) == 0:
            return id_patch, "no AEF COG covers this tile"

        dst_crs = f"EPSG:{int(row.utm_epsg)}"
        dst_transform = from_origin(row.utm_minx, row.utm_maxy, PIXEL_M, PIXEL_M)
        dst = np.zeros((EMBED_DIM, H, W), dtype=np.int8)

        for _, h in hits.iterrows():
            vsipath = "/vsis3/" + h.path.replace("s3://", "")
            with rasterio.open(vsipath) as src:
                reproject(
                    source=rasterio.band(src, list(range(1, EMBED_DIM + 1))),
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
        write_patch(output_path, dst, dst_crs, dst_transform)
    except Exception as e:
        return id_patch, str(e)
    return id_patch, None


def main(args: argparse.Namespace) -> None:
    tiles = pd.read_parquet(args.tiles)
    idx = fetch_aef_index(Path(args.index_cache), args.year)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    errors_csv = Path("errors_pastis_aef.csv")
    if not errors_csv.exists():
        errors_csv.write_text("id_patch,error\n")

    rows = list(tiles.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(process_tile, r, idx, output_dir): r.id_patch for r in rows}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="AEF tiles"):
            id_patch, err = fut.result()
            if err:
                with error_lock, open(errors_csv, "a") as f:
                    f.write(f"{id_patch},{err}\n")

    n_done = len(list(output_dir.glob("S2_*.tif")))
    print(f"done: {n_done} / {len(tiles)} tiles -> {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tiles", default="data/pastis_tiles.parquet")
    p.add_argument("--output", default="data/pastis-aef")
    p.add_argument("--year", type=int, default=2019)
    p.add_argument("--index-cache", default="data/aef_index.parquet")
    p.add_argument("--num-workers", type=int, default=16)
    main(p.parse_args())
