"""Pool PASTIS parcel embeddings per instance mask.

For each PASTIS patch:
  - Load the per-pixel embedding GeoTIFF (128-band int8 from AEF or Tessera)
  - Load the instance mask (ParcelIDs_<id>.npy) and semantic label (TARGET_<id>.npy[0])
  - For each unique parcel (skip background=0, void=19, < min_pixels):
      * Extract pixel embeddings -> (N_pixels, D) float32
      * Apply each pooling method -> (D',) vector
  - Split by PASTIS folds: train=1-4, test=5
  - Save embeddings/<dataset_name>/<method>/pastis.npz

The probe scripts (knnprobe.py / linearprobe.py) accept --splits pastis to
load these files.
"""

import argparse
import json
import threading
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

from geopool.pool import BOVW_VARIANTS, POOL_METHODS, PCA_VARIANTS, VLAD_VARIANTS, WHITENED_MEAN_VARIANTS, BoVWPooler, VLADPooler, WhitenedMeanPooler

# ── PASTIS label constants ────────────────────────────────────────────────────
BACKGROUND = 0
VOID = 19
PASTIS_CLASSES = {
    1: "Meadow",
    2: "Soft winter wheat",
    3: "Corn",
    4: "Winter barley",
    5: "Winter rapeseed",
    6: "Spring barley",
    7: "Sunflower",
    8: "Grapevine",
    9: "Beet",
    10: "Winter triticale",
    11: "Winter durum wheat",
    12: "Fruits/vegetables/flowers",
    13: "Potatoes",
    14: "Leguminous fodder",
    15: "Soybeans",
    16: "Orchard",
    17: "Mixed cereal",
    18: "Sorghum",
}
VALID_CLASSES = set(PASTIS_CLASSES.keys())

# ── Parcel-level pool functions (N_pixels, D) -> (D',) ───────────────────────


def _parcel_mean(px: np.ndarray) -> np.ndarray:
    return px.mean(axis=0)


def _parcel_std(px: np.ndarray) -> np.ndarray:
    return px.std(axis=0)


def _parcel_max(px: np.ndarray) -> np.ndarray:
    return px.max(axis=0)


def _parcel_mean_std(px: np.ndarray) -> np.ndarray:
    return np.concatenate([px.mean(axis=0), px.std(axis=0)])


def _parcel_stats(px: np.ndarray) -> np.ndarray:
    return np.concatenate([px.min(axis=0), px.max(axis=0), px.mean(axis=0), px.std(axis=0)])


def _parcel_mean_max(px: np.ndarray) -> np.ndarray:
    return np.concatenate([px.mean(axis=0), px.max(axis=0)])


def _parcel_gem(px: np.ndarray, p: float = 3.0) -> np.ndarray:
    powered = np.sign(px) * np.abs(px) ** p
    pooled = powered.mean(axis=0)
    return np.sign(pooled) * np.abs(pooled) ** (1.0 / p)


def _parcel_signed_non_cancelling_gem(px: np.ndarray, p: float = 3.0, eps: float = 1e-6) -> np.ndarray:
    pos = np.maximum(px, 0.0)
    neg = np.maximum(-px, 0.0)
    pos_pooled = np.mean(np.maximum(pos, eps) ** p, axis=0) ** (1.0 / p)
    neg_pooled = np.mean(np.maximum(neg, eps) ** p, axis=0) ** (1.0 / p)
    return np.concatenate([pos_pooled, neg_pooled])


def _parcel_adaptive_nc_gem(px: np.ndarray, p_max: float = 3.0, eps: float = 1e-6) -> np.ndarray:
    """NC-GeM with p that scales with N/D: p→1 (mean) when N≪D, p→p_max when N≫D."""
    N, D = px.shape
    p = 1.0 + (p_max - 1.0) * min(1.0, N / float(D))
    pos = np.maximum(px, 0.0)
    neg = np.maximum(-px, 0.0)
    pos_pooled = np.mean(np.maximum(pos, eps) ** p, axis=0) ** (1.0 / p)
    neg_pooled = np.mean(np.maximum(neg, eps) ** p, axis=0) ** (1.0 / p)
    return np.concatenate([pos_pooled, neg_pooled])


def _parcel_percentiles(px: np.ndarray, pcts: tuple[int, ...] = (10, 25, 50, 75, 90)) -> np.ndarray:
    return np.percentile(px, pcts, axis=0).flatten()


def _parcel_center_weighted_mean(px: np.ndarray) -> np.ndarray:
    """Gaussian-weighted mean over pixels. Uses pixel distance from mean position as proxy."""
    if len(px) == 1:
        return px[0]
    # treat pixel indices as coordinates, weight by gaussian from centroid
    idx = np.arange(len(px), dtype=np.float32)
    cx = idx.mean()
    sigma = max(len(px) / 4.0, 1e-6)
    w = np.exp(-((idx - cx) ** 2) / (2 * sigma**2))
    w /= w.sum()
    return (w[:, None] * px).sum(axis=0)


def _parcel_median_iqr(px: np.ndarray) -> np.ndarray:
    median = np.median(px, axis=0)
    q75, q25 = np.percentile(px, [75, 25], axis=0)
    return np.concatenate([median, q75 - q25])


def _parcel_flattened_cov(px: np.ndarray) -> np.ndarray:
    if len(px) < 2:
        D = px.shape[1]
        return np.zeros(D * (D + 1) // 2, dtype=np.float32)
    cov = np.cov(px, rowvar=False)
    return cov[np.triu_indices(px.shape[1])].astype(np.float32)


def _parcel_nd_switch(px: np.ndarray) -> np.ndarray:
    """mean+std when N<D, signed NC-GeM when N>=D."""
    N, D = px.shape
    if N < D:
        return _parcel_mean_std(px)
    return _parcel_signed_non_cancelling_gem(px)


def _parcel_sign_adaptive_nc_gem(px: np.ndarray, p: float = 3.0, eps: float = 1e-6) -> np.ndarray:
    """Signed NC-GeM when N>=D; GeM+std when N<D."""
    N, D = px.shape
    if N >= D:
        return _parcel_signed_non_cancelling_gem(px, p=p, eps=eps)
    px64 = px.astype(np.float64)
    gem = np.mean(np.abs(px64) ** p, axis=0) ** (1.0 / p)
    std = px64.std(axis=0)
    return np.concatenate([gem, std]).astype(np.float32)


def _parcel_signed_sqrt_mean(px: np.ndarray) -> np.ndarray:
    normed = np.sign(px) * np.sqrt(np.abs(px))
    return normed.mean(axis=0).astype(np.float32)


def _parcel_trimmed_mean(px: np.ndarray, trim: float = 0.1) -> np.ndarray:
    lo = np.percentile(px, trim * 100, axis=0)
    hi = np.percentile(px, (1 - trim) * 100, axis=0)
    clipped = np.clip(px, lo, hi)
    return clipped.mean(axis=0).astype(np.float32)



PARCEL_POOL_FNS = {
    "mean": _parcel_mean,
    "std": _parcel_std,
    "max": _parcel_max,
    "mean_std": _parcel_mean_std,
    "stats": _parcel_stats,
    "mean_max": _parcel_mean_max,
    "gem": _parcel_gem,
    "signed_non_cancelling_gem": _parcel_signed_non_cancelling_gem,
    "adaptive_nc_gem": _parcel_adaptive_nc_gem,
    "percentiles": _parcel_percentiles,
    "center_weighted_mean": _parcel_center_weighted_mean,
    "median_iqr": _parcel_median_iqr,
    "flattened_cov": _parcel_flattened_cov,
    "nd_switch": _parcel_nd_switch,
    "sign_adaptive_nc_gem": _parcel_sign_adaptive_nc_gem,
    "signed_sqrt_mean": _parcel_signed_sqrt_mean,
    "trimmed_mean": _parcel_trimmed_mean,
}


# ── Per-patch extraction (used in subprocess) ─────────────────────────────────


def extract_patch_parcels(
    patch_id: int,
    emb_dir: Path,
    ann_dir: Path,
    min_pixels: int = 5,
) -> list[tuple[int, np.ndarray, int]]:
    """Load one patch, return list of (class_id, pixels (N,D), n_pixels) for valid parcels."""
    emb_path = emb_dir / f"S2_{patch_id}.tif"
    pid_path = ann_dir / f"ParcelIDs_{patch_id}.npy"
    tgt_path = ann_dir / f"TARGET_{patch_id}.npy"

    with rasterio.open(emb_path) as src:
        emb = src.read(out_dtype="float32")  # (D, H, W)
    emb = np.moveaxis(emb, 0, -1)  # (H, W, D)
    H, W, D = emb.shape

    pid = np.load(pid_path)  # (H, W) int32 - instance IDs
    target = np.load(tgt_path)  # (3, H, W) uint8
    sem = target[0]  # (H, W) semantic class

    flat_emb = emb.reshape(-1, D)  # (H*W, D)
    flat_pid = pid.ravel()
    flat_sem = sem.ravel()

    parcels: list[tuple[int, np.ndarray]] = []
    for parcel_id in np.unique(flat_pid):
        mask = flat_pid == parcel_id
        classes = np.unique(flat_sem[mask])
        # Majority class (should be one class per parcel)
        cls = int(classes[0]) if len(classes) == 1 else int(
            np.bincount(flat_sem[mask].astype(np.intp)).argmax()
        )
        if cls not in VALID_CLASSES:
            continue
        pixels = flat_emb[mask]  # (N_pixels, D)
        if len(pixels) < min_pixels:
            continue
        parcels.append((cls, pixels, len(pixels)))

    return parcels


def _extract_patch_worker(args: tuple) -> tuple[int, list]:
    patch_id, emb_dir, ann_dir, min_pixels = args
    try:
        parcels = extract_patch_parcels(patch_id, Path(emb_dir), Path(ann_dir), min_pixels)
        return patch_id, parcels
    except Exception as e:
        print(f"  Error patch {patch_id}: {e}")
        return patch_id, []


# ── Main pipeline ─────────────────────────────────────────────────────────────


def save_npz(
    path: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_pixels_train: np.ndarray | None = None,
    n_pixels_test: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = dict(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    if n_pixels_train is not None:
        arrays["n_pixels_train"] = n_pixels_train
    if n_pixels_test is not None:
        arrays["n_pixels_test"] = n_pixels_test
    np.savez_compressed(path, **arrays)


def compute_simple_pools(
    train_parcels: list[tuple[int, np.ndarray, int]],
    test_parcels: list[tuple[int, np.ndarray, int]],
    output_dir: Path,
    methods: list[str],
    overwrite: bool = False,
) -> None:
    methods_todo = [m for m in methods if m in PARCEL_POOL_FNS]
    methods_todo = [m for m in methods_todo if overwrite or not (output_dir / m / "pastis.npz").exists()]
    if not methods_todo:
        return

    print(f"  Computing {len(methods_todo)} simple pool methods...")
    y_train = np.array([cls for cls, _, _ in train_parcels], dtype=np.int32)
    y_test = np.array([cls for cls, _, _ in test_parcels], dtype=np.int32)
    n_pixels_train = np.array([n for _, _, n in train_parcels], dtype=np.int32)
    n_pixels_test = np.array([n for _, _, n in test_parcels], dtype=np.int32)

    for method in methods_todo:
        fn = PARCEL_POOL_FNS[method]
        try:
            x_train = np.stack([fn(px.astype(np.float32)) for _, px, _ in tqdm(train_parcels, desc=f"{method} train", leave=False)])
            x_test = np.stack([fn(px.astype(np.float32)) for _, px, _ in tqdm(test_parcels, desc=f"{method} test", leave=False)])
            out = output_dir / method / "pastis.npz"
            save_npz(out, x_train, y_train, x_test, y_test, n_pixels_train, n_pixels_test)
            print(f"    {method}: train={x_train.shape}, test={x_test.shape} -> {out}")
        except Exception as e:
            print(f"    {method} FAILED: {e}")


def compute_pca_pools(
    train_parcels: list[tuple[int, np.ndarray, int]],
    test_parcels: list[tuple[int, np.ndarray, int]],
    output_dir: Path,
    methods: list[str],
    pca_batch_size: int = 512,
    overwrite: bool = False,
) -> None:
    methods_todo = [m for m in methods if m in PCA_VARIANTS]
    methods_todo = [m for m in methods_todo if overwrite or not (output_dir / m / "pastis.npz").exists()]
    if not methods_todo:
        return

    print("  Mean-pooling for PCA...")
    y_train = np.array([cls for cls, _, _ in train_parcels], dtype=np.int32)
    y_test = np.array([cls for cls, _, _ in test_parcels], dtype=np.int32)
    n_pixels_train = np.array([n for _, _, n in train_parcels], dtype=np.int32)
    n_pixels_test = np.array([n for _, _, n in test_parcels], dtype=np.int32)
    train_mean = np.stack([px.astype(np.float32).mean(axis=0) for _, px, _ in tqdm(train_parcels, desc="mean train", leave=False)])
    test_mean = np.stack([px.astype(np.float32).mean(axis=0) for _, px, _ in tqdm(test_parcels, desc="mean test", leave=False)])

    for method in methods_todo:
        n_components = PCA_VARIANTS[method]
        if n_components > train_mean.shape[1]:
            print(f"    Skipping {method}: n_components={n_components} > dim={train_mean.shape[1]}")
            continue
        print(f"  Fitting IncrementalPCA({n_components})...")
        ipca = IncrementalPCA(n_components=n_components, batch_size=pca_batch_size)
        ipca.fit(train_mean)
        x_train = ipca.transform(train_mean).astype(np.float32)
        x_test = ipca.transform(test_mean).astype(np.float32)
        out = output_dir / method / "pastis.npz"
        save_npz(out, x_train, y_train, x_test, y_test, n_pixels_train, n_pixels_test)
        print(f"    {method}: train={x_train.shape}, test={x_test.shape} -> {out}")


def compute_bovw_pools(
    train_parcels: list[tuple[int, np.ndarray, int]],
    test_parcels: list[tuple[int, np.ndarray, int]],
    output_dir: Path,
    methods: list[str],
    bovw_batch_size: int = 10_000,
    overwrite: bool = False,
) -> None:
    methods_todo = [m for m in methods if m in BOVW_VARIANTS]
    methods_todo = [m for m in methods_todo if overwrite or not (output_dir / m / "pastis.npz").exists()]
    if not methods_todo:
        return

    y_train = np.array([cls for cls, _, _ in train_parcels], dtype=np.int32)
    y_test = np.array([cls for cls, _, _ in test_parcels], dtype=np.int32)
    n_pixels_train = np.array([n for _, _, n in train_parcels], dtype=np.int32)
    n_pixels_test = np.array([n for _, _, n in test_parcels], dtype=np.int32)

    for method in methods_todo:
        n_clusters = BOVW_VARIANTS[method]
        print(f"  Fitting BoVW({n_clusters}) on train pixels (streaming)...")
        bovw = BoVWPooler(n_clusters=n_clusters, batch_size=bovw_batch_size)

        # Fit: stream train parcel pixels in batches
        batch: list[np.ndarray] = []
        batch_pixels = 0
        for _, px, _ in tqdm(train_parcels, desc="bovw fit", leave=False):
            batch.append(px.astype(np.float32))
            batch_pixels += len(px)
            if batch_pixels >= bovw_batch_size:
                flat = np.concatenate(batch, axis=0)
                bovw.kmeans.partial_fit(flat[np.random.permutation(len(flat))])
                batch = []
                batch_pixels = 0
        if batch:
            flat = np.concatenate(batch, axis=0)
            bovw.kmeans.partial_fit(flat)
        bovw._fitted = True  # noqa: SLF001

        # Transform: histogram per parcel
        def _to_hist(px: np.ndarray) -> np.ndarray:
            labels = bovw.kmeans.predict(px.astype(np.float32))
            hist = np.bincount(labels, minlength=n_clusters).astype(np.float32)
            return hist / max(hist.sum(), 1.0)

        x_train = np.stack([_to_hist(px) for _, px, _ in tqdm(train_parcels, desc="bovw train", leave=False)])
        x_test = np.stack([_to_hist(px) for _, px, _ in tqdm(test_parcels, desc="bovw test", leave=False)])
        out = output_dir / method / "pastis.npz"
        save_npz(out, x_train, y_train, x_test, y_test, n_pixels_train, n_pixels_test)
        print(f"    {method}: train={x_train.shape}, test={x_test.shape} -> {out}")


def compute_vlad_pools(
    train_parcels: list[tuple[int, np.ndarray, int]],
    test_parcels: list[tuple[int, np.ndarray, int]],
    output_dir: Path,
    methods: list[str],
    bovw_batch_size: int = 10_000,
    overwrite: bool = False,
) -> None:
    methods_todo = [m for m in methods if m in VLAD_VARIANTS]
    methods_todo = [m for m in methods_todo if overwrite or not (output_dir / m / "pastis.npz").exists()]
    if not methods_todo:
        return

    y_train = np.array([cls for cls, _, _ in train_parcels], dtype=np.int32)
    y_test = np.array([cls for cls, _, _ in test_parcels], dtype=np.int32)
    n_pixels_train = np.array([n for _, _, n in train_parcels], dtype=np.int32)
    n_pixels_test = np.array([n for _, _, n in test_parcels], dtype=np.int32)

    for method in methods_todo:
        n_clusters = VLAD_VARIANTS[method]
        print(f"  Fitting VLAD({n_clusters}) on train pixels (streaming)...")
        vlad = VLADPooler(n_clusters=n_clusters, batch_size=bovw_batch_size)

        batch: list[np.ndarray] = []
        batch_pixels = 0
        for _, px, _ in tqdm(train_parcels, desc="vlad fit", leave=False):
            batch.append(px.astype(np.float32))
            batch_pixels += len(px)
            if batch_pixels >= bovw_batch_size:
                flat = np.concatenate(batch, axis=0)
                vlad.partial_fit(flat[np.random.permutation(len(flat))])
                batch = []
                batch_pixels = 0
        if batch:
            flat = np.concatenate(batch, axis=0)
            vlad.partial_fit(flat)
        vlad._fitted = True  # noqa: SLF001

        x_train = np.stack([vlad.transform_parcel(px) for _, px, _ in tqdm(train_parcels, desc="vlad train", leave=False)])
        x_test = np.stack([vlad.transform_parcel(px) for _, px, _ in tqdm(test_parcels, desc="vlad test", leave=False)])
        out = output_dir / method / "pastis.npz"
        save_npz(out, x_train, y_train, x_test, y_test, n_pixels_train, n_pixels_test)
        print(f"    {method}: train={x_train.shape}, test={x_test.shape} -> {out}")


def compute_whitened_mean_pools(
    train_parcels: list[tuple[int, np.ndarray, int]],
    test_parcels: list[tuple[int, np.ndarray, int]],
    output_dir: Path,
    methods: list[str],
    batch_size: int = 10_000,
    overwrite: bool = False,
) -> None:
    methods_todo = [m for m in methods if m in WHITENED_MEAN_VARIANTS]
    methods_todo = [m for m in methods_todo if overwrite or not (output_dir / m / "pastis.npz").exists()]
    if not methods_todo:
        return

    y_train = np.array([cls for cls, _, _ in train_parcels], dtype=np.int32)
    y_test = np.array([cls for cls, _, _ in test_parcels], dtype=np.int32)
    n_pixels_train = np.array([n for _, _, n in train_parcels], dtype=np.int32)
    n_pixels_test = np.array([n for _, _, n in test_parcels], dtype=np.int32)

    for method in methods_todo:
        print(f"  Fitting WhitenedMean on train pixels (streaming)...")
        D = train_parcels[0][1].shape[1]
        min_batch = max(batch_size, D + 1)
        pooler = WhitenedMeanPooler(batch_size=min_batch)

        batch: list[np.ndarray] = []
        batch_pixels = 0
        for _, px, _ in tqdm(train_parcels, desc="whitened fit", leave=False):
            batch.append(px.astype(np.float32))
            batch_pixels += len(px)
            if batch_pixels >= min_batch:
                flat = np.concatenate(batch, axis=0)
                pooler.partial_fit(flat[np.random.permutation(len(flat))])
                batch = []
                batch_pixels = 0
        if batch:
            flat = np.concatenate(batch, axis=0)
            if flat.shape[0] >= D + 1:
                pooler.partial_fit(flat)
        pooler._fitted = True  # noqa: SLF001

        x_train = np.stack([pooler.transform_parcel(px) for _, px, _ in tqdm(train_parcels, desc="whitened train", leave=False)])
        x_test = np.stack([pooler.transform_parcel(px) for _, px, _ in tqdm(test_parcels, desc="whitened test", leave=False)])
        out = output_dir / method / "pastis.npz"
        save_npz(out, x_train, y_train, x_test, y_test, n_pixels_train, n_pixels_test)
        print(f"    {method}: train={x_train.shape}, test={x_test.shape} -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pool PASTIS parcel embeddings by instance mask")
    parser.add_argument("--dataset-name", required=True, help="e.g. aef or tessera")
    parser.add_argument("--emb-dir", default=None, help="Embedding GeoTIFF dir (default: data/pastis-<name>)")
    parser.add_argument("--ann-dir", default="data/pastis-r/PASTIS-R/ANNOTATIONS", help="PASTIS annotation dir")
    parser.add_argument("--meta", default="data/pastis-r/PASTIS-R/metadata.geojson", help="PASTIS metadata")
    parser.add_argument("--output-dir", default=None, help="Output dir (default: embeddings/pastis-<name>)")
    parser.add_argument("--test-fold", type=int, default=5, help="Which fold to use as test set")
    parser.add_argument("--min-pixels", type=int, default=5, help="Minimum pixels per parcel")
    parser.add_argument("--methods", nargs="+", default=None, help="Pool methods (default: all)")
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel workers for patch loading")
    parser.add_argument("--bovw-batch-size", type=int, default=50_000)
    parser.add_argument("--pca-batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir) if args.emb_dir else Path(f"data/pastis-{args.dataset_name}")
    ann_dir = Path(args.ann_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"embeddings/pastis-{args.dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.methods is None:
        methods = list(PARCEL_POOL_FNS.keys()) + list(PCA_VARIANTS.keys()) + list(BOVW_VARIANTS.keys()) + list(VLAD_VARIANTS.keys()) + list(WHITENED_MEAN_VARIANTS.keys())
    else:
        methods = args.methods

    # ── Load fold assignments ──────────────────────────────────────────────
    with open(args.meta) as f:
        meta = json.load(f)
    fold_by_patch: dict[int, int] = {
        feat["properties"]["ID_PATCH"]: feat["properties"]["Fold"]
        for feat in meta["features"]
    }
    train_patch_ids = [pid for pid, fold in fold_by_patch.items() if fold != args.test_fold]
    test_patch_ids = [pid for pid, fold in fold_by_patch.items() if fold == args.test_fold]
    print(f"Folds: train patches={len(train_patch_ids)}, test patches={len(test_patch_ids)} (fold {args.test_fold})")

    # ── Extract parcels in parallel (deterministic order by patch_id) ─────
    def load_patches(patch_ids: list[int], desc: str) -> list[tuple[int, np.ndarray, int]]:
        parcels: list[tuple[int, np.ndarray, int]] = []
        sorted_ids = sorted(patch_ids)
        worker_args = [(pid, str(emb_dir), str(ann_dir), args.min_pixels) for pid in sorted_ids]
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futs = {ex.submit(_extract_patch_worker, a): a[0] for a in worker_args}
            results: dict[int, list] = {}
            for fut in tqdm(as_completed(futs), total=len(futs), desc=desc):
                pid, patch_parcels = fut.result()
                results[pid] = patch_parcels
        for pid in sorted_ids:
            parcels.extend(results[pid])
        return parcels

    print("\nLoading train parcels...")
    train_parcels = load_patches(train_patch_ids, "train patches")
    print(f"  {len(train_parcels)} train parcels")

    print("\nLoading test parcels...")
    test_parcels = load_patches(test_patch_ids, "test patches")
    print(f"  {len(test_parcels)} test parcels")

    # ── Class distribution ─────────────────────────────────────────────────
    train_classes, train_counts = np.unique([c for c, _, _ in train_parcels], return_counts=True)
    print("\nTrain class distribution:")
    for cls, cnt in zip(train_classes, train_counts):
        print(f"  {cls:2d} {PASTIS_CLASSES.get(cls, '?'):30s}  {cnt:6d}")

    # ── Simple pools ───────────────────────────────────────────────────────
    simple_methods = [m for m in methods if m in PARCEL_POOL_FNS]
    if simple_methods:
        print(f"\nComputing simple pool methods: {simple_methods}")
        compute_simple_pools(train_parcels, test_parcels, output_dir, simple_methods, args.overwrite)

    # ── PCA pools ──────────────────────────────────────────────────────────
    pca_methods = [m for m in methods if m in PCA_VARIANTS]
    if pca_methods:
        print(f"\nComputing PCA pool methods: {pca_methods}")
        compute_pca_pools(train_parcels, test_parcels, output_dir, pca_methods, args.pca_batch_size, args.overwrite)

    # ── BoVW pools ─────────────────────────────────────────────────────────
    bovw_methods = [m for m in methods if m in BOVW_VARIANTS]
    if bovw_methods:
        print(f"\nComputing BoVW pool methods: {bovw_methods}")
        compute_bovw_pools(train_parcels, test_parcels, output_dir, bovw_methods, args.bovw_batch_size, args.overwrite)

    # ── VLAD pools ─────────────────────────────────────────────────────────
    vlad_methods = [m for m in methods if m in VLAD_VARIANTS]
    if vlad_methods:
        print(f"\nComputing VLAD pool methods: {vlad_methods}")
        compute_vlad_pools(train_parcels, test_parcels, output_dir, vlad_methods, args.bovw_batch_size, args.overwrite)

    # ── Whitened mean pools ────────────────────────────────────────────────
    wm_methods = [m for m in methods if m in WHITENED_MEAN_VARIANTS]
    if wm_methods:
        print(f"\nComputing whitened mean pool methods: {wm_methods}")
        compute_whitened_mean_pools(train_parcels, test_parcels, output_dir, wm_methods, args.bovw_batch_size, args.overwrite)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\nDone. Output: {output_dir}")
    saved = list(output_dir.glob("*/pastis.npz"))
    print(f"  {len(saved)} method npz files saved")


if __name__ == "__main__":
    main()
