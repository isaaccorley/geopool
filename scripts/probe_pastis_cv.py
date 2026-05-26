"""5-fold CV probes on PASTIS parcels.

Reads embeddings_cv/<dataset>/<method>/pastis_cv.npz produced by
pool_pastis_cv.py and rotates test_fold ∈ {1..5}, reporting per-fold and
mean ± std F1-macro for linear and kNN probes.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from geopool.evaluation import evaluate_knn, evaluate_linear, evaluate_linear_gpu


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--embeddings-dir", default=None)
    p.add_argument("--output-csv", default=None)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--probes", nargs="+", default=["linear", "knn"], choices=["linear", "knn"])
    p.add_argument("--k", type=int, default=5, help="kNN k")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    import torch
    use_gpu = not args.cpu and torch.cuda.is_available()
    print(f"Linear probe: {'GPU' if use_gpu else 'CPU'}")

    emb_dir = Path(args.embeddings_dir) if args.embeddings_dir else Path("embeddings_cv") / f"pastis-{args.dataset_name}"
    method_dirs = sorted([d for d in emb_dir.iterdir() if d.is_dir() and (d / "pastis_cv.npz").exists()])
    if args.methods:
        method_dirs = [d for d in method_dirs if d.name in args.methods]
    if not method_dirs:
        print(f"No pooled embeddings under {emb_dir}")
        return

    results = []
    for mdir in method_dirs:
        method = mdir.name
        data = np.load(mdir / "pastis_cv.npz")
        x, y, fold = data["x"], data["y"], data["fold"]
        folds = sorted(set(fold.tolist()))
        print(f"\n[{method}] x={x.shape} y={y.shape} folds={folds}")

        for f in folds:
            test_mask = fold == f
            train_mask = ~test_mask
            X_tr, y_tr = x[train_mask], y[train_mask]
            X_te, y_te = x[test_mask], y[test_mask]

            for probe in args.probes:
                if probe == "linear":
                    fn = evaluate_linear_gpu if use_gpu else evaluate_linear
                    out = fn(X_tr, y_tr, X_te, y_te)
                    m = out["metrics"]
                else:
                    out = evaluate_knn(X_tr, y_tr, X_te, y_te, k=args.k)
                    m = out["metrics"]
                results.append({
                    "method": method,
                    "probe": probe,
                    "test_fold": int(f),
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "accuracy": float(m["accuracy"]),
                    "f1_macro": float(m["f1_macro"]),
                })
                print(f"  fold {f}  {probe:6s}  acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}")

    df = pd.DataFrame(results)
    out_csv = Path(args.output_csv) if args.output_csv else Path("results") / f"pastis-{args.dataset_name}" / "cv_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved per-fold results: {out_csv}")

    # Aggregate
    agg = df.groupby(["method", "probe"]).agg(
        f1_mean=("f1_macro", "mean"),
        f1_std=("f1_macro", "std"),
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        n_folds=("test_fold", "count"),
    ).reset_index()
    agg_csv = out_csv.with_name("cv_summary.csv")
    agg.to_csv(agg_csv, index=False)
    print(f"Saved CV summary: {agg_csv}")
    print("\n", agg.to_string(index=False))


if __name__ == "__main__":
    main()
