"""Extract per-tile bboxes from PASTIS metadata.geojson.

Emits data/pastis_tiles.parquet with one row per PASTIS patch:
  id_patch, fold, tile (MGRS), n_parcel, parcel_cover,
  utm_epsg, utm_minx, utm_miny, utm_maxx, utm_maxy,
  lon_min, lat_min, lon_max, lat_max,
  geom_l93 (Lambert-93 rotated polygon, WKB)

The native PASTIS rasters are axis-aligned in their parent UTM zone (T30 or T31).
Reprojecting from EPSG:2154 (Lambert-93) to native UTM recovers exact 128*10m bounds.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def mgrs_to_utm_epsg(tile: str) -> int:
    """`t30uxv` -> 32630, `t31tcj` -> 32631 (UTM north zones, EPSG:326NN)."""
    # PASTIS tiles look like 't<zone><band><col><row>'; e.g. t30uxv, t31tcj
    zone = int(tile[1:3])
    # All PASTIS tiles are in northern hemisphere
    return 32600 + zone


def main(args: argparse.Namespace) -> None:
    src = Path(args.metadata)
    gdf = gpd.read_file(src)
    assert gdf.crs.to_epsg() == 2154, f"expected EPSG:2154, got {gdf.crs}"

    gdf["utm_epsg"] = gdf["TILE"].map(mgrs_to_utm_epsg)

    # Reproject per UTM zone (typically zones 30 and 31)
    pieces = []
    for utm_epsg, sub in gdf.groupby("utm_epsg"):
        sub_utm = sub.to_crs(f"EPSG:{utm_epsg}")
        b = sub_utm.geometry.bounds
        sub = sub.copy()
        sub["utm_minx"] = b["minx"].values
        sub["utm_miny"] = b["miny"].values
        sub["utm_maxx"] = b["maxx"].values
        sub["utm_maxy"] = b["maxy"].values
        pieces.append(sub)
    gdf = pd.concat(pieces, ignore_index=True)

    wgs = gdf.to_crs("EPSG:4326")
    wb = wgs.geometry.bounds
    gdf["lon_min"] = wb["minx"].values
    gdf["lat_min"] = wb["miny"].values
    gdf["lon_max"] = wb["maxx"].values
    gdf["lat_max"] = wb["maxy"].values

    gdf["geom_l93_wkb"] = gdf.geometry.to_wkb()

    out_cols = [
        "ID_PATCH", "Fold", "TILE", "N_Parcel", "Parcel_Cover",
        "utm_epsg", "utm_minx", "utm_miny", "utm_maxx", "utm_maxy",
        "lon_min", "lat_min", "lon_max", "lat_max",
        "geom_l93_wkb",
    ]
    df = gdf[out_cols].rename(
        columns={
            "ID_PATCH": "id_patch",
            "Fold": "fold",
            "TILE": "tile",
            "N_Parcel": "n_parcel",
            "Parcel_Cover": "parcel_cover",
        }
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    width_m = df["utm_maxx"] - df["utm_minx"]
    height_m = df["utm_maxy"] - df["utm_miny"]
    print(f"wrote {len(df)} rows -> {out}")
    print(f"  UTM widths  (m): min={width_m.min():.2f} max={width_m.max():.2f} mean={width_m.mean():.2f}")
    print(f"  UTM heights (m): min={height_m.min():.2f} max={height_m.max():.2f} mean={height_m.mean():.2f}")
    print(f"  UTM zones: {sorted(df.utm_epsg.unique().tolist())}")
    print(f"  Folds: {dict(df.fold.value_counts().sort_index())}")
    print(f"  WGS84 bounds: lon [{df.lon_min.min():.4f}, {df.lon_max.max():.4f}], "
          f"lat [{df.lat_min.min():.4f}, {df.lat_max.max():.4f}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", default="data/pastis-r/PASTIS-R/metadata.geojson")
    p.add_argument("--output", default="data/pastis_tiles.parquet")
    main(p.parse_args())
