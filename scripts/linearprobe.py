"""Run linear probes on cached embeddings."""

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from geopool.evaluation import evaluate_linear, evaluate_linear_bootstrap, evaluate_linear_gpu, subsample_dataset
from geopool.pool import BOVW_VARIANTS, PCA_VARIANTS, POOL_METHODS

DEFAULT_SPLITS = ["standard", "spatial"]


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["x_train"], data["y_train"], data["x_test"], data["y_test"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--embeddings-dir", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
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
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Split names to evaluate (default: standard spatial). Use 'pastis' for PASTIS parcels.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU sklearn linear probe (default: GPU via PyTorch when available).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="Restrict to these pool methods (default: all found in embeddings-dir).",
    )
    args = parser.parse_args()

    import torch
    use_gpu = not args.cpu and torch.cuda.is_available()
    print(f"Linear probe: {'GPU (PyTorch L-BFGS)' if use_gpu else 'CPU (sklearn)'}")

    train_subsets = args.train_subset or [None]
    splits = args.splits if args.splits else DEFAULT_SPLITS

    dataset_name = args.dataset_name
    embeddings_dir = (
        Path(args.embeddings_dir) if args.embeddings_dir else Path("embeddings") / dataset_name
    )
    results = []

    all_methods = list(POOL_METHODS.keys()) + list(PCA_VARIANTS.keys()) + list(BOVW_VARIANTS.keys())
    if args.methods:
        all_methods = [m for m in args.methods if m in all_methods or True]  # allow new names

    for method in all_methods:
        method_dir = embeddings_dir / method
        if not method_dir.exists():
            print(f"Skipping {method} (not found)")
            continue

        for split in splits:
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

                if args.n_bootstraps:
                    metrics = evaluate_linear_bootstrap(
                        X_train_sub,
                        y_train_sub,
                        X_test,
                        y_test,
                        n_bootstraps=args.n_bootstraps,
                        seed=args.seed,
                    )
                    results.append(
                        {
                            "pool": method,
                            "split_type": split,
                            "probe": "linear",
                            "train_samples": n_actual,
                            **metrics,
                        }
                    )
                    subset_str = f"n={n_actual}" if n_train is not None else "full"
                    print(
                        f"{method:10s} {split:10s} {subset_str:10s} linear "
                        f"acc={metrics['accuracy_mean']:.4f}±{metrics['accuracy_std']:.4f} "
                        f"f1={metrics['f1_macro_mean']:.4f}±{metrics['f1_macro_std']:.4f}"
                    )
                else:
                    fn = evaluate_linear_gpu if use_gpu else evaluate_linear
                    res = fn(X_train_sub, y_train_sub, X_test, y_test)
                    metrics = cast("dict[str, float]", res["metrics"])
                    results.append(
                        {
                            "pool": method,
                            "split_type": split,
                            "probe": "linear",
                            "train_samples": n_actual,
                            **metrics,
                        }
                    )
                    subset_str = f"n={n_actual}" if n_train is not None else "full"
                    print(
                        f"{method:10s} {split:10s} {subset_str:10s} linear "
                        f"acc={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f} C={metrics['C']}"
                    )

    df = pd.DataFrame(results)
    output_path = (
        Path(args.output_csv)
        if args.output_csv
        else Path("results") / dataset_name / "linear_results.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
