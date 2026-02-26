from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import ee
import rioxarray as rio
import xarray as xr
from torchgeo.datasets import EuroSAT
from tqdm import tqdm


def quantize_aef(image: ee.Image) -> ee.Image:
    """quantize float64 -> uint8"""
    power = 2.0
    scale = 127.5
    min_value = -127
    max_value = 127

    sat = image.abs().pow(ee.Number(1.0).divide(power)).multiply(image.signum())
    snapped = sat.multiply(ee.Number(scale)).round()
    return snapped.clamp(min_value, max_value).add(ee.Number(127)).uint8()


def download_aef(
    filepath: Path,
    output: str,
    year: int,
    collection: ee.ImageCollection,
) -> tuple[Path, str | None]:
    try:
        raster = rio.open_rasterio(filepath)
        raster = raster.rio.reproject("EPSG:4326")  # type: ignore[union-attr]
        bounds = raster.rio.bounds()  # must be in EPSG:4326
        startDate = ee.Date.fromYMD(year, 1, 1)
        endDate = startDate.advance(1, "year")
        geometry = ee.Geometry.BBox(*bounds)
        image = (
            collection.filter(ee.Filter.date(startDate, endDate))
            .filter(ee.Filter.bounds(geometry))
            .first()
        )
        ds = xr.open_dataset(
            quantize_aef(image),  # type: ignore[arg-type]
            engine="ee",
            geometry=bounds,
            projection=image.select(0).projection(),
        )
        ds = ds.isel(time=0)
        ds = ds.rename({"X": "x", "Y": "y"})
        ds = ds.to_array(dim="band").transpose("band", "y", "x")
        output_path = os.path.join(output, filepath.stem + ".tif")
        ds.rio.to_raster(output_path, driver="COG", compress="deflate", dtype="uint8")
    except Exception as e:
        return filepath, str(e)
    return filepath, None


def main(args: argparse.Namespace) -> None:
    ee.Authenticate()
    ee.Initialize(
        project=args.project,
        opt_url="https://earthengine-highvolume.googleapis.com",
    )
    ee.data.setWorkloadTag(args.workload_tag)
    collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

    filepaths = []
    for split in ["train", "val", "test"]:
        ds = EuroSAT(root=args.root, split=split, download=True, checksum=True)
        filepaths.extend([Path(img) for img, _ in ds.imgs])

    with open("errors.csv", "w") as f:
        f.write("filepath,error\n")

    os.makedirs(args.output, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        func = partial(download_aef, output=args.output, year=args.year, collection=collection)
        futures = [executor.submit(func, filepath) for filepath in filepaths]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(filepaths)):
            filepath, error = future.result()
            if error:
                with open("errors.csv", "a") as f:
                    f.write(f"{filepath},{error}\n")

    # Reorganize images into folders (optional)
    images = list(Path(args.output).glob("*.tif"))
    print(len(images))
    for image in tqdm(images):
        folder = image.parent / image.stem.split("_")[0]
        folder.mkdir(parents=True, exist_ok=True)
        shutil.move(image, folder / image.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--output", type=str, default="eurosat-aef")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--project", type=str)
    parser.add_argument("--workload_tag", type=str, default="aef-eurosat")
    args = parser.parse_args()
    main(args)
