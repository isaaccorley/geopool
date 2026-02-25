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


def make_loader(
    split_file: Path,
    data_dir: Path,
    num_workers: int,
    batch_size: int,
    dtype: str,
) -> DataLoader:
    dataset = EuroSATEmbeddingDataset(split_file, data_dir, dtype=dtype)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def mean_pool_batch(embs: list[np.ndarray]) -> np.ndarray:
    batch = np.stack(embs, axis=0)
    return batch.mean(axis=(1, 2), dtype=np.float32)


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
    train_loader: DataLoader,
    test_loader: DataLoader,
    output_dir: Path,
    split_label: str,
    methods: list[str],
    overwrite: bool = False,
) -> None:
    """Compute and save simple (non-PCA) pools."""
    methods_to_compute = []
    for method in methods:
        out_path = output_dir / method / f"{split_label}.npz"
        if out_path.exists() and not overwrite:
            print(f"  Skipping {method}: {out_path} already exists")
        else:
            methods_to_compute.append(method)

    if not methods_to_compute:
        return

    method_count = len(methods_to_compute)
    pool_fns = {method: POOL_METHODS[method] for method in methods_to_compute}
    train_outputs: dict[str, list[np.ndarray]] = {m: [] for m in methods_to_compute}
    test_outputs: dict[str, list[np.ndarray]] = {m: [] for m in methods_to_compute}
    train_labels_list: list[np.ndarray] = []
    test_labels_list: list[np.ndarray] = []

    train_bar = tqdm(train_loader, desc="simple train", total=len(train_loader))
    for embs, labels in train_bar:
        train_labels_list.append(labels)
        batch = np.stack(embs, axis=0)
        for idx, (method, pool_fn) in enumerate(pool_fns.items(), start=1):
            train_bar.set_description(f"simple train {idx}/{method_count}")
            train_outputs[method].append(np.asarray(pool_fn(batch), dtype=np.float32))

    test_bar = tqdm(test_loader, desc="simple test", total=len(test_loader))
    for embs, labels in test_bar:
        test_labels_list.append(labels)
        batch = np.stack(embs, axis=0)
        for idx, (method, pool_fn) in enumerate(pool_fns.items(), start=1):
            test_bar.set_description(f"simple test {idx}/{method_count}")
            test_outputs[method].append(np.asarray(pool_fn(batch), dtype=np.float32))

    y_train = np.concatenate(train_labels_list, axis=0)
    y_test = np.concatenate(test_labels_list, axis=0)
    for method in methods_to_compute:
        x_train = np.concatenate(train_outputs[method], axis=0)
        x_test = np.concatenate(test_outputs[method], axis=0)
        out_path = output_dir / method / f"{split_label}.npz"
        save_npz(out_path, x_train, y_train, x_test, y_test)
        print(f"  Saved {out_path} [train: {x_train.shape}, test: {x_test.shape}]")


def compute_pca_pools(
    train_loader: DataLoader,
    test_loader: DataLoader,
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

    for method in methods_to_compute:
        n_components = PCA_VARIANTS[method]
        ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

        print(f"  Fitting IncrementalPCA({n_components}) on train...")
        train_dim: int | None = None
        for embs, _ in tqdm(train_loader, desc=f"{method} fit"):
            pooled = mean_pool_batch(embs)
            if train_dim is None:
                train_dim = pooled.shape[1]
                if n_components > train_dim:
                    print(f"  Skipping {method}: n_components={n_components} > dim={train_dim}")
                    break
            ipca.partial_fit(pooled)
        if train_dim is None or n_components > train_dim:
            continue

        train_outputs: list[np.ndarray] = []
        test_outputs: list[np.ndarray] = []
        train_labels_list: list[np.ndarray] = []
        test_labels_list: list[np.ndarray] = []

        for embs, labels in tqdm(train_loader, desc=f"{method} train"):
            pooled = mean_pool_batch(embs)
            train_outputs.append(ipca.transform(pooled).astype(np.float32, copy=False))
            train_labels_list.append(labels)
        for embs, labels in tqdm(test_loader, desc=f"{method} test"):
            pooled = mean_pool_batch(embs)
            test_outputs.append(ipca.transform(pooled).astype(np.float32, copy=False))
            test_labels_list.append(labels)

        x_train = np.concatenate(train_outputs, axis=0)
        x_test = np.concatenate(test_outputs, axis=0)
        y_train = np.concatenate(train_labels_list, axis=0)
        y_test = np.concatenate(test_labels_list, axis=0)

        out_path = output_dir / method / f"{split_label}.npz"
        save_npz(out_path, x_train, y_train, x_test, y_test)
        print(f"  Saved {out_path} [train: {x_train.shape}, test: {x_test.shape}]")


def compute_bovw_pools(
    train_loader: DataLoader,
    test_loader: DataLoader,
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
        for embs, _ in tqdm(train_loader, desc=f"{method} fit"):
            bovw.fit(embs, random_state=42, image_batch_size=image_batch_size)

        train_outputs: list[np.ndarray] = []
        test_outputs: list[np.ndarray] = []
        train_labels_list: list[np.ndarray] = []
        test_labels_list: list[np.ndarray] = []

        for embs, labels in tqdm(train_loader, desc=f"{method} train"):
            train_outputs.append(
                bovw.transform_embeddings(
                    embs, desc=None, image_batch_size=image_batch_size
                ).astype(np.float32, copy=False)
            )
            train_labels_list.append(labels)
        for embs, labels in tqdm(test_loader, desc=f"{method} test"):
            test_outputs.append(
                bovw.transform_embeddings(
                    embs, desc=None, image_batch_size=image_batch_size
                ).astype(np.float32, copy=False)
            )
            test_labels_list.append(labels)

        x_train = np.concatenate(train_outputs, axis=0)
        x_test = np.concatenate(test_outputs, axis=0)
        y_train = np.concatenate(train_labels_list, axis=0)
        y_test = np.concatenate(test_labels_list, axis=0)

        out_path = output_dir / method / f"{split_label}.npz"
        save_npz(out_path, x_train, y_train, x_test, y_test)
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
        default=16,
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
        default=32,
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

        train_loader = make_loader(
            train_path, data_dir, args.num_workers, args.batch_size, args.dtype
        )
        test_loader = make_loader(
            test_path, data_dir, args.num_workers, args.batch_size, args.dtype
        )

        # Simple pools
        if simple_methods:
            print("Computing simple pools...")
            simple_train_loader = make_loader(
                train_path, data_dir, args.num_workers, args.simple_batch_size, args.dtype
            )
            simple_test_loader = make_loader(
                test_path, data_dir, args.num_workers, args.simple_batch_size, args.dtype
            )
            compute_simple_pools(
                simple_train_loader,
                simple_test_loader,
                output_dir,
                split,
                simple_methods,
                args.overwrite,
            )

        # PCA pools
        if pca_methods:
            print("Computing PCA pools...")
            compute_pca_pools(
                train_loader,
                test_loader,
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
                train_loader,
                test_loader,
                output_dir,
                split,
                bovw_methods,
                args.bovw_batch_size,
                args.bovw_image_batch_size,
                args.overwrite,
            )


if __name__ == "__main__":
    main()
