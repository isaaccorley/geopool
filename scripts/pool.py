"""Generate cached embedding datasets using PyTorch DataLoader for multiproc I/O."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import DataLoader
from tqdm import tqdm

from geopool.data import get_split_paths
from geopool.dataset import EuroSATEmbeddingDataset
from geopool.pool import BOVW_VARIANTS, PCA_VARIANTS, POOL_METHODS, BoVWPooler


def numpy_collate(batch: list[tuple[np.ndarray, int]]) -> tuple[list[np.ndarray], np.ndarray]:
    """Custom collate that keeps embeddings as list (variable spatial dims)."""
    embs, labels = zip(*batch, strict=False)
    return list(embs), np.array(labels, dtype=np.int32)


def load_all(
    split_file: Path,
    data_dir: Path,
    num_workers: int = 8,
    batch_size: int = 64,
    dtype: str = "float32",
) -> tuple[list[np.ndarray], np.ndarray]:
    """Load all embeddings using DataLoader with multiprocessing.

    Returns:
        embeddings: list of (H, W, D) arrays
        labels: (N,) int array
    """
    dataset = EuroSATEmbeddingDataset(split_file, data_dir, dtype=dtype)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    total = len(dataset)
    all_embs: list[np.ndarray | None] = [None] * total
    all_labels = np.empty(total, dtype=np.int32)
    offset = 0

    for embs, labels in tqdm(loader, desc="Loading"):
        batch_size = len(embs)
        all_embs[offset : offset + batch_size] = embs
        all_labels[offset : offset + batch_size] = labels
        offset += batch_size

    return [emb for emb in all_embs if emb is not None], all_labels


def save_npz(
    path: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Save train/test arrays to single compressed npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )


def compute_simple_pools(
    train_embs: list[np.ndarray],
    train_labels: np.ndarray,
    test_embs: list[np.ndarray],
    test_labels: np.ndarray,
    output_dir: Path,
    split_label: str,
    methods: list[str],
    batch_size: int = 256,
    overwrite: bool = False,
) -> None:
    """Compute and save simple (non-PCA) pools."""
    for method in methods:
        out_path = output_dir / method / f"{split_label}.npz"
        if out_path.exists() and not overwrite:
            print(f"  Skipping {method}: {out_path} already exists")
            continue
        pool_fn = POOL_METHODS[method]
        train_count = len(train_embs)
        test_count = len(test_embs)
        train_sample = np.asarray(pool_fn(train_embs[0]), dtype=np.float32)
        test_sample = np.asarray(pool_fn(test_embs[0]), dtype=np.float32)
        if train_sample.shape != test_sample.shape:
            raise ValueError(
                f"{method} output shape mismatch: {train_sample.shape} vs {test_sample.shape}"
            )
        x_train = np.empty((train_count, train_sample.shape[0]), dtype=np.float32)
        x_test = np.empty((test_count, test_sample.shape[0]), dtype=np.float32)
        x_train[0] = train_sample
        x_test[0] = test_sample

        for start in tqdm(range(1, train_count, batch_size), desc=f"{method} train"):
            end = min(start + batch_size, train_count)
            batch = np.stack(train_embs[start:end], axis=0)
            x_train[start:end] = np.asarray(pool_fn(batch), dtype=np.float32)
        for start in tqdm(range(1, test_count, batch_size), desc=f"{method} test"):
            end = min(start + batch_size, test_count)
            batch = np.stack(test_embs[start:end], axis=0)
            x_test[start:end] = np.asarray(pool_fn(batch), dtype=np.float32)

        save_npz(
            out_path,
            x_train,
            train_labels,
            x_test,
            test_labels,
        )
        print(f"  Saved {out_path} [train: {x_train.shape}, test: {x_test.shape}]")


def compute_pca_pools(
    train_embs: list[np.ndarray],
    train_labels: np.ndarray,
    test_embs: list[np.ndarray],
    test_labels: np.ndarray,
    output_dir: Path,
    split_label: str,
    methods: list[str],
    batch_size: int = 512,
    overwrite: bool = False,
) -> None:
    """Compute and save PCA pools (fit on train, transform both).

    Note: PCA is applied to mean-pooled embeddings (D-dimensional), not raw HxWxD.
    This ensures consistent input shape despite variable spatial dimensions.
    """
    # Check which methods need to be computed
    methods_to_compute = []
    for method in methods:
        out_path = output_dir / method / f"{split_label}.npz"
        if out_path.exists() and not overwrite:
            print(f"  Skipping {method}: {out_path} already exists")
        else:
            methods_to_compute.append(method)

    if not methods_to_compute:
        return

    # Mean-pool all embeddings to (D,) vectors
    print("  Mean-pooling embeddings...")
    train_count = len(train_embs)
    test_count = len(test_embs)
    train_dim = train_embs[0].shape[-1]
    test_dim = test_embs[0].shape[-1]
    train_pooled = np.empty((train_count, train_dim), dtype=np.float32)
    test_pooled = np.empty((test_count, test_dim), dtype=np.float32)

    for idx, emb in enumerate(tqdm(train_embs, desc="pool train")):
        train_pooled[idx] = emb.mean(axis=(0, 1), dtype=np.float32)
    for idx, emb in enumerate(tqdm(test_embs, desc="pool test")):
        test_pooled[idx] = emb.mean(axis=(0, 1), dtype=np.float32)

    for method in methods_to_compute:
        n_components = PCA_VARIANTS[method]
        # Skip if n_components > embedding dim
        if n_components > train_pooled.shape[1]:
            print(f"  Skipping {method}: n_components={n_components} > dim={train_pooled.shape[1]}")
            continue

        print(f"  Fitting IncrementalPCA({n_components}) on train...")

        ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        ipca.fit(train_pooled)

        x_train = ipca.transform(train_pooled).astype(np.float32, copy=False)
        x_test = ipca.transform(test_pooled).astype(np.float32, copy=False)

        out_path = output_dir / method / f"{split_label}.npz"
        save_npz(out_path, x_train, train_labels, x_test, test_labels)
        print(f"  Saved {out_path} [train: {x_train.shape}, test: {x_test.shape}]")


def compute_bovw_pools(
    train_embs: list[np.ndarray],
    train_labels: np.ndarray,
    test_embs: list[np.ndarray],
    test_labels: np.ndarray,
    output_dir: Path,
    split_label: str,
    methods: list[str],
    batch_size: int = 10_000,
    image_batch_size: int = 64,
    overwrite: bool = False,
) -> None:
    """Compute and save BoVW (Bag of Visual Words) pools.

    Fits k-means on training pixel embeddings, then represents each patch
    as a histogram of cluster assignments.
    """
    # Check which methods need to be computed
    methods_to_compute = []
    for method in methods:
        out_path = output_dir / method / f"{split_label}.npz"
        if out_path.exists() and not overwrite:
            print(f"  Skipping {method}: {out_path} already exists")
        else:
            methods_to_compute.append(method)

    if not methods_to_compute:
        return

    for method in methods_to_compute:
        n_clusters = BOVW_VARIANTS[method]
        print(f"  Fitting BoVW({n_clusters}) on train pixels...")

        bovw = BoVWPooler(n_clusters=n_clusters, batch_size=batch_size)
        bovw.fit(train_embs, random_state=42, image_batch_size=image_batch_size)

        x_train = bovw.transform_embeddings(
            train_embs, desc=f"{method} train", image_batch_size=image_batch_size
        ).astype(np.float32, copy=False)
        x_test = bovw.transform_embeddings(
            test_embs, desc=f"{method} test", image_batch_size=image_batch_size
        ).astype(np.float32, copy=False)

        out_path = output_dir / method / f"{split_label}.npz"
        save_npz(out_path, x_train, train_labels, x_test, test_labels)
        print(f"  Saved {out_path} [train: {x_train.shape}, test: {x_test.shape}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cached embedding datasets")
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Dataset name suffix (e.g., aef, tessera). Used for default paths.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing embedding GeoTIFFs (default: data/eurosat-<dataset-name>)",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default="data",
        help="Directory containing split files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store cached datasets (default: embeddings/<dataset-name>)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="Pool methods to compute (default: all)",
    )
    parser.add_argument(
        "--split-type",
        type=str,
        choices=["standard", "spatial", "both"],
        default="both",
        help="Which splits to cache",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for DataLoader",
    )
    parser.add_argument(
        "--simple-batch-size",
        type=int,
        default=256,
        help="Batch size for simple pools",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="Data type for embeddings")
    parser.add_argument(
        "--pca-batch-size",
        type=int,
        default=512,
        help="Batch size for IncrementalPCA",
    )
    parser.add_argument(
        "--bovw-batch-size",
        type=int,
        default=10_000,
        help="Mini-batch size for BoVW k-means",
    )
    parser.add_argument(
        "--bovw-image-batch-size",
        type=int,
        default=256,
        help="Number of images per BoVW transform batch",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing npz files (default: skip if exists)",
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name
    data_dir = Path(args.data_dir) if args.data_dir else Path("data") / f"eurosat-{dataset_name}"
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("embeddings") / dataset_name

    if args.methods is None:
        methods = list(POOL_METHODS.keys()) + list(PCA_VARIANTS.keys()) + list(BOVW_VARIANTS.keys())
    else:
        methods = args.methods

    simple_methods = [m for m in methods if m in POOL_METHODS]
    pca_methods = [m for m in methods if m in PCA_VARIANTS]
    bovw_methods = [m for m in methods if m in BOVW_VARIANTS]

    splits: list[str] = ["standard", "spatial"] if args.split_type == "both" else [args.split_type]

    for split in splits:
        print(f"\n=== Processing {split} split ===")
        split_typed = "standard" if split == "standard" else "spatial"
        train_path, _, test_path = get_split_paths(split_dir, split_typed)

        # Load all embeddings with multiprocessing
        print("Loading train embeddings...")
        train_embs, train_labels = load_all(
            train_path, data_dir, args.num_workers, args.batch_size, args.dtype
        )
        print(f"  Loaded {len(train_embs)} train samples")

        print("Loading test embeddings...")
        test_embs, test_labels = load_all(
            test_path, data_dir, args.num_workers, args.batch_size, args.dtype
        )
        print(f"  Loaded {len(test_embs)} test samples")

        # Simple pools
        if simple_methods:
            print("Computing simple pools...")
            compute_simple_pools(
                train_embs,
                train_labels,
                test_embs,
                test_labels,
                output_dir,
                split,
                simple_methods,
                args.simple_batch_size,
                args.overwrite,
            )

        # PCA pools
        if pca_methods:
            print("Computing PCA pools...")
            compute_pca_pools(
                train_embs,
                train_labels,
                test_embs,
                test_labels,
                output_dir,
                split,
                pca_methods,
                args.pca_batch_size,
                args.overwrite,
            )

        # BoVW pools
        if bovw_methods:
            print("Computing BoVW pools...")
            compute_bovw_pools(
                train_embs,
                train_labels,
                test_embs,
                test_labels,
                output_dir,
                split,
                bovw_methods,
                args.bovw_batch_size,
                args.bovw_image_batch_size,
                args.overwrite,
            )


if __name__ == "__main__":
    main()
