"""Pool PASTIS parcel embeddings, saving per-parcel fold IDs for k-fold CV.

Unlike pool_pastis.py which pre-splits into train (folds 1-4) and test (fold 5),
this script extracts ALL parcels in one pass and saves a flat npz per method:
  embeddings_cv/<dataset_name>/<method>/pastis_cv.npz
with arrays: x (N, D'), y (N,), n_pixels (N,), fold (N,) where fold ∈ {1..5}.

Downstream probe_pastis_cv.py rotates the test fold to compute CV mean ± std.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for scripts.*

from scripts.pool_pastis import (
    PARCEL_POOL_FNS,
    PASTIS_CLASSES,
    _extract_patch_worker,
)


def load_all_parcels(
    emb_dir: Path,
    ann_dir: Path,
    fold_by_patch: dict[int, int],
    min_pixels: int,
    num_workers: int,
) -> list[tuple[int, np.ndarray, int, int]]:
    """Return [(cls, pixels, n_pixels, fold)] across all patches."""
    sorted_ids = sorted(fold_by_patch.keys())
    worker_args = [(pid, str(emb_dir), str(ann_dir), min_pixels) for pid in sorted_ids]
    results: dict[int, list] = {}
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futs = {ex.submit(_extract_patch_worker, a): a[0] for a in worker_args}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="extract patches"):
            pid, patch_parcels = fut.result()
            results[pid] = patch_parcels
    parcels: list[tuple[int, np.ndarray, int, int]] = []
    for pid in sorted_ids:
        fold = fold_by_patch[pid]
        for cls, pixels, n in results[pid]:
            parcels.append((cls, pixels, n, fold))
    return parcels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", required=True, help="e.g. aef, olmoearth, tessera")
    p.add_argument("--emb-dir", default=None)
    p.add_argument("--ann-dir", default="data/pastis-r/PASTIS-R/ANNOTATIONS")
    p.add_argument("--meta", default="data/pastis-r/PASTIS-R/metadata.geojson")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--min-pixels", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    emb_dir = Path(args.emb_dir) if args.emb_dir else Path(f"data/pastis-{args.dataset_name}")
    ann_dir = Path(args.ann_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"embeddings_cv/pastis-{args.dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Paper's training-free methods (NC-GeM removed)
    DEFAULT_METHODS = [
        "mean", "std", "mean_std", "stats", "max", "gem",
        "mean_max", "percentiles", "center_weighted_mean",
        "median_iqr", "flattened_cov",
    ]
    methods = args.methods or DEFAULT_METHODS
    methods = [m for m in methods if m in PARCEL_POOL_FNS]

    todo = [m for m in methods if args.overwrite or not (output_dir / m / "pastis_cv.npz").exists()]
    if not todo:
        print("All methods already pooled; nothing to do.")
        return

    with open(args.meta) as f:
        meta = json.load(f)
    fold_by_patch = {
        feat["properties"]["ID_PATCH"]: int(feat["properties"]["Fold"])
        for feat in meta["features"]
    }
    print(f"Patches: {len(fold_by_patch)}  Fold dist: {sorted({(f, sum(1 for ff in fold_by_patch.values() if ff == f)) for f in set(fold_by_patch.values())})}")

    print("\nExtracting all parcels (one pass)...")
    parcels = load_all_parcels(emb_dir, ann_dir, fold_by_patch, args.min_pixels, args.num_workers)
    print(f"  {len(parcels)} parcels total")

    y = np.array([c for c, _, _, _ in parcels], dtype=np.int32)
    n_pix = np.array([n for _, _, n, _ in parcels], dtype=np.int32)
    fold = np.array([f for _, _, _, f in parcels], dtype=np.int8)

    print("\nFold parcel counts:")
    for f in sorted(set(fold)):
        print(f"  fold {f}: {(fold == f).sum()} parcels")

    for method in todo:
        fn = PARCEL_POOL_FNS[method]
        try:
            x = np.stack([
                fn(px.astype(np.float32))
                for _, px, _, _ in tqdm(parcels, desc=f"{method}", leave=False)
            ]).astype(np.float32)
        except Exception as e:
            print(f"  {method} FAILED: {e}")
            continue
        out = output_dir / method / "pastis_cv.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, x=x, y=y, n_pixels=n_pix, fold=fold)
        print(f"  {method}: x={x.shape} -> {out}")

    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
