"""Run KNN probes on cached embeddings."""

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from geopool.evaluation import evaluate_knn, evaluate_knn_bootstrap, subsample_dataset
from geopool.pool import BOVW_VARIANTS, PCA_VARIANTS, POOL_METHODS

SPLITS = ["standard", "spatial"]


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["x_train"], data["y_train"], data["x_test"], data["y_test"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--embeddings-dir", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 7, 9])
    parser.add_argument(
        "--train-subset",
        type=int,
        nargs="+",
        default=None,
        help="Number of training samples to use. Use -1 for all samples. Multiple values run experiments for each size (e.g., --train-subset 100 500 -1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible subsampling.",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=None,
        help="Number of bootstrap iterations. If set, reports mean, std, and 95%% CI.",
    )
    args = parser.parse_args()

    train_subsets = args.train_subset if args.train_subset else [None]

    dataset_name = args.dataset_name
    embeddings_dir = (
        Path(args.embeddings_dir) if args.embeddings_dir else Path("embeddings") / dataset_name
    )
    results = []

    all_methods = list(POOL_METHODS.keys()) + list(PCA_VARIANTS.keys()) + list(BOVW_VARIANTS.keys())

    for method in all_methods:
        method_dir = embeddings_dir / method
        if not method_dir.exists():
            print(f"Skipping {method} (not found)")
            continue

        for split in SPLITS:
            npz_path = method_dir / f"{split}.npz"
            if not npz_path.exists():
                print(f"Skipping {method}/{split} (file missing)")
                continue

            X_train, y_train, X_test, y_test = load_npz(npz_path)

            for n_train in train_subsets:
                if n_train is not None and n_train > 0:
                    X_train_sub, y_train_sub = subsample_dataset(
                        X_train, y_train, n_train, seed=args.seed
                    )
                    n_actual = len(y_train_sub)
                else:
                    # n_train is None or -1: use all samples
                    X_train_sub, y_train_sub = X_train, y_train
                    n_actual = len(y_train)

                for k in args.k_values:
                    if args.n_bootstraps:
                        metrics = evaluate_knn_bootstrap(
                            X_train_sub,
                            y_train_sub,
                            X_test,
                            y_test,
                            k=k,
                            n_bootstraps=args.n_bootstraps,
                            seed=args.seed,
                        )
                        results.append(
                            {
                                "pool": method,
                                "split_type": split,
                                "probe": f"knn_{k}",
                                "train_samples": n_actual,
                                **metrics,
                            }
                        )
                        subset_str = f"n={n_actual}" if n_train is not None else "full"
                        print(
                            f"{method:10s} {split:10s} {subset_str:10s} k={k} "
                            f"acc={metrics['accuracy_mean']:.4f}±{metrics['accuracy_std']:.4f} "
                            f"f1={metrics['f1_macro_mean']:.4f}±{metrics['f1_macro_std']:.4f}"
                        )
                    else:
                        res = evaluate_knn(X_train_sub, y_train_sub, X_test, y_test, k=k)
                        metrics = cast("dict[str, float]", res["metrics"])
                        results.append(
                            {
                                "pool": method,
                                "split_type": split,
                                "probe": f"knn_{k}",
                                "train_samples": n_actual,
                                **metrics,
                            }
                        )
                        subset_str = f"n={n_actual}" if n_train is not None else "full"
                        print(
                            f"{method:10s} {split:10s} {subset_str:10s} k={k} "
                            f"acc={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f}"
                        )

    df = pd.DataFrame(results)
    output_path = (
        Path(args.output_csv)
        if args.output_csv
        else Path("results") / dataset_name / "knn_results.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
